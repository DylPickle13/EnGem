from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from google import genai
from google.genai import types

from api_backoff import call_with_exponential_backoff
from config import (
    GEMINI_EMBEDDING_BATCH_SIZE,
    GEMINI_EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    MINIMAL_MODEL,
    MEMORY_ARCHIVE_DIR,
    MEMORY_FILE_COLLECTION_NAME,
    MEMORY_SEMANTIC_COLLECTION_NAME,
    get_paid_gemini_api_key,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


DEFAULT_DB_PATH = Path(__file__).parent / "memory" / "vector_db"
DEFAULT_COLLECTION_NAME = MEMORY_SEMANTIC_COLLECTION_NAME
DEFAULT_FILE_COLLECTION_NAME = MEMORY_FILE_COLLECTION_NAME
DEFAULT_ARCHIVE_PATH = Path(MEMORY_ARCHIVE_DIR)
MEMORY_RELATED = Path(__file__).parent / "agent_instructions" / "memory_related.md"
MEMORY_SCHEMA_VERSION = "2026-03-11-gemini2"
_SUPPORTED_ATTACHMENT_EMBEDDING_PREFIXES = ("image/", "video/", "audio/")
_SUPPORTED_ATTACHMENT_EMBEDDING_TYPES = {"application/pdf"}

_embedding_service: GeminiEmbeddingService | None = None
_default_store: VectorMemoryStore | None = None
_attachment_store: VectorMemoryStore | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).strip().lower()


def _normalize_embedding(values: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in values))
    if magnitude <= 0:
        return values
    return [component / magnitude for component in values]


def _batch(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (filename or "").strip())
    return cleaned or "attachment"


def _default_attachment_name_for_mime_type(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type == "application/pdf":
        return "document"
    return "attachment"


@dataclass(slots=True)
class MemoryItem:
    """A single memory entry returned from the vector database."""

    memory_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ArchivedAttachment:
    """Metadata describing an archived attachment."""

    file_id: str
    file_name: str
    mime_type: str
    file_digest: str
    file_size_bytes: int
    archive_path: Path


class GeminiEmbeddingService:
    """Explicit Gemini embedding generator for documents, queries, and attachments."""

    def __init__(
        self,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        output_dimensionality: int = GEMINI_EMBEDDING_DIM,
        batch_size: int = GEMINI_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self.batch_size = max(1, batch_size)
        self.client = genai.Client(api_key=get_paid_gemini_api_key())

    def _embed_contents(self, contents: list[Any], task_type: str, description: str) -> list[list[float]]:
        response = call_with_exponential_backoff(
            lambda: self.client.models.embed_content(
                model=self.model_name,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality,
                ),
            ),
            description=description,
        )
        raw_embeddings = getattr(response, "embeddings", None) or []
        return [_normalize_embedding(list(getattr(embedding, "values", []) or [])) for embedding in raw_embeddings]

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        cleaned_texts = [(text or "").strip() for text in texts]
        if not cleaned_texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch in _batch(cleaned_texts, self.batch_size):
            all_embeddings.extend(
                self._embed_contents(
                    contents=batch,
                    task_type=task_type,
                    description=f"Gemini embedding batch ({task_type})",
                )
            )
        return all_embeddings

    def embed_document(self, text: str) -> list[float]:
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("Document text cannot be empty")
        return self.embed_texts([cleaned_text], task_type="RETRIEVAL_DOCUMENT")[0]

    def embed_query(self, text: str) -> list[float]:
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("Query text cannot be empty")
        return self.embed_texts([cleaned_text], task_type="RETRIEVAL_QUERY")[0]

    def supports_attachment_embedding(self, mime_type: str) -> bool:
        normalized_mime = (mime_type or "").strip().lower()
        return normalized_mime.startswith(_SUPPORTED_ATTACHMENT_EMBEDDING_PREFIXES) or normalized_mime in _SUPPORTED_ATTACHMENT_EMBEDDING_TYPES

    def embed_attachment(self, attachment_bytes: bytes, mime_type: str, file_name: str) -> list[float]:
        if not attachment_bytes:
            raise ValueError("Attachment bytes cannot be empty")

        attachment_content = types.Content(
            parts=[
                types.Part(text=f"File name: {file_name}. MIME type: {mime_type}."),
                types.Part.from_bytes(data=attachment_bytes, mime_type=mime_type),
            ],
        )
        response = call_with_exponential_backoff(
            lambda: self.client.models.embed_content(
                model=self.model_name,
                contents=[attachment_content],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimensionality,
                ),
            ),
            description="Gemini attachment embedding",
        )
        raw_embeddings = getattr(response, "embeddings", None) or []
        embeddings = [_normalize_embedding(list(getattr(embedding, "values", []) or [])) for embedding in raw_embeddings]
        if not embeddings:
            raise ValueError("Gemini returned no attachment embeddings")
        return embeddings[0]


class VectorMemoryStore:
    """Persistent memory store backed by ChromaDB with explicit Gemini embeddings."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any | None = None,
        embedding_service: GeminiEmbeddingService | None = None,
    ) -> None:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.embedding_service = embedding_service or get_embedding_service()
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def write_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Write a new memory to the vector DB and return its id."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("Memory text cannot be empty")

        final_id = memory_id or hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        final_metadata = dict(metadata or {})
        final_metadata.setdefault("created_at", _utcnow_iso())
        final_metadata.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
        final_metadata.setdefault("embedding_model", GEMINI_EMBEDDING_MODEL)
        final_metadata.setdefault("embedding_dim", GEMINI_EMBEDDING_DIM)
        final_metadata.setdefault("record_type", "semantic_memory")

        self.collection.upsert(
            ids=[final_id],
            documents=[cleaned_text],
            metadatas=[final_metadata],
            embeddings=[embedding or self.embedding_service.embed_document(cleaned_text)],
        )
        return final_id

    def read_all_memories(
        self,
        limit: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Read all stored memories, optionally limited."""
        result = self.collection.get(include=["documents", "metadatas"], limit=limit, where=where)

        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        memories: list[MemoryItem] = []
        for i, memory_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            memories.append(MemoryItem(memory_id=memory_id, text=text, metadata=metadata))
        return memories

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Semantic search memories by query text."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        if self.count() <= limit:
            return self.read_all_memories(limit=limit, where=where)

        result = self.collection.query(
            query_embeddings=[self.embedding_service.embed_query(cleaned_query)],
            n_results=max(1, limit),
            include=["documents", "metadatas"],
            where=where,
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        memories: list[MemoryItem] = []
        for i, memory_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            memories.append(MemoryItem(memory_id=memory_id, text=text, metadata=metadata))
        return memories

    def count(self) -> int:
        """Return number of items in the collection."""
        return self.collection.count()

    def clear_memories(self) -> int:
        """Delete all stored memories and return how many were removed."""
        result = self.collection.get(include=[])
        ids = result.get("ids") or []

        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)


def get_embedding_service() -> GeminiEmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = GeminiEmbeddingService()
    return _embedding_service


def get_default_store() -> VectorMemoryStore:
    """Convenience factory for the default semantic memory store."""
    global _default_store
    if _default_store is None:
        _default_store = VectorMemoryStore(
            db_path=DEFAULT_DB_PATH,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
    return _default_store


def get_attachment_store() -> VectorMemoryStore:
    """Convenience factory for the attachment/file memory store."""
    global _attachment_store
    if _attachment_store is None:
        _attachment_store = VectorMemoryStore(
            db_path=DEFAULT_DB_PATH,
            collection_name=DEFAULT_FILE_COLLECTION_NAME,
        )
    return _attachment_store


def archive_attachment(
    attachment_bytes: bytes,
    mime_type: str,
    file_name: str,
) -> ArchivedAttachment:
    """Archive raw attachment bytes under a content-addressed path."""
    if not attachment_bytes:
        raise ValueError("Attachment bytes cannot be empty")

    normalized_mime = (mime_type or "").strip().lower() or "application/octet-stream"
    safe_name = _sanitize_filename(file_name or _default_attachment_name_for_mime_type(normalized_mime))
    file_digest = hashlib.sha256(attachment_bytes).hexdigest()
    archive_dir = DEFAULT_ARCHIVE_PATH / file_digest[:2] / file_digest
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / safe_name
    if not archive_path.exists():
        archive_path.write_bytes(attachment_bytes)

    return ArchivedAttachment(
        file_id=file_digest,
        file_name=safe_name,
        mime_type=normalized_mime,
        file_digest=file_digest,
        file_size_bytes=len(attachment_bytes),
        archive_path=archive_path,
    )


def _build_attachment_memory_text(
    archived_attachment: ArchivedAttachment,
    history_file: str,
    extracted_text: str,
    source_type: str,
) -> str:
    sections = [
        f"File: {archived_attachment.file_name}",
        f"MIME type: {archived_attachment.mime_type}",
        f"Source: {source_type}",
        f"History file: {history_file}",
    ]
    cleaned_extracted_text = (extracted_text or "").strip()
    if cleaned_extracted_text:
        sections.append(f"Extracted content: {cleaned_extracted_text}")
    else:
        sections.append("Extracted content: unavailable")
    return "\n".join(sections)


def write_attachment_memory(
    attachment: dict[str, bytes | str],
    history_file: str,
    extracted_text: str = "",
    source_type: str = "generate_response_attachment",
) -> MemoryItem | None:
    """Archive and index an attachment as a file-memory record."""
    attachment_bytes = attachment.get("data")
    if not isinstance(attachment_bytes, bytes) or not attachment_bytes:
        return None

    mime_type = str(attachment.get("mime_type") or "application/octet-stream").strip().lower()
    file_name = str(attachment.get("filename") or _default_attachment_name_for_mime_type(mime_type)).strip()

    archived_attachment = archive_attachment(
        attachment_bytes=attachment_bytes,
        mime_type=mime_type,
        file_name=file_name,
    )
    document_text = _build_attachment_memory_text(
        archived_attachment=archived_attachment,
        history_file=history_file,
        extracted_text=extracted_text,
        source_type=source_type,
    )
    embedding_service = get_embedding_service()
    embedding_error_message = ""
    if embedding_service.supports_attachment_embedding(archived_attachment.mime_type):
        try:
            embedding = embedding_service.embed_attachment(
                attachment_bytes=attachment_bytes,
                mime_type=archived_attachment.mime_type,
                file_name=archived_attachment.file_name,
            )
            extraction_status = "embedded_multimodal"
        except Exception as exc:
            embedding_error_message = str(exc)
            embedding = embedding_service.embed_document(document_text)
            extraction_status = "embedded_text_fallback_after_multimodal_error"
    else:
        embedding = embedding_service.embed_document(document_text)
        extraction_status = "embedded_text_fallback"

    metadata = {
        "record_type": "file_record",
        "schema_version": MEMORY_SCHEMA_VERSION,
        "source_type": source_type,
        "history_file": history_file,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "embedding_dim": GEMINI_EMBEDDING_DIM,
        "file_id": archived_attachment.file_id,
        "file_name": archived_attachment.file_name,
        "mime_type": archived_attachment.mime_type,
        "file_digest": archived_attachment.file_digest,
        "file_size_bytes": archived_attachment.file_size_bytes,
        "archive_path": str(archived_attachment.archive_path),
        "extraction_status": extraction_status if extracted_text.strip() else "no_extracted_text",
        "created_at": _utcnow_iso(),
    }
    if embedding_error_message:
        metadata["embedding_error"] = embedding_error_message[:500]
    get_attachment_store().write_memory(
        text=document_text,
        metadata=metadata,
        memory_id=archived_attachment.file_id,
        embedding=embedding,
    )
    return MemoryItem(memory_id=archived_attachment.file_id, text=document_text, metadata=metadata)


def render_memory_for_prompt(memory_item: MemoryItem) -> str:
    """Render a memory item into prompt-safe text."""
    metadata = dict(memory_item.metadata or {})
    record_type = metadata.get("record_type", "semantic_memory")
    if record_type == "file_record":
        return (
            f"File memory: {memory_item.text}\n"
            f"Metadata: {json.dumps({key: metadata[key] for key in sorted(metadata) if key != 'archive_path'}, ensure_ascii=False)}"
        )
    return (
        f"Memory: {memory_item.text}\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}"
    )


def search_all_memories(query: str, semantic_limit: int = 5, file_limit: int = 3) -> list[MemoryItem]:
    """Query semantic and file memories together."""
    combined: list[MemoryItem] = []
    combined.extend(get_default_store().search_memories(query, limit=semantic_limit))
    combined.extend(get_attachment_store().search_memories(query, limit=file_limit))
    return combined


def write_semantic_memory(text: str, metadata: dict[str, Any] | None = None) -> str | None:
    """Write a semantic memory if it is not an exact normalized duplicate."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None

    existing = get_default_store().search_memories(cleaned_text, limit=5)
    normalized_candidate = _normalize_text(cleaned_text)
    for item in existing:
        if _normalize_text(item.text) == normalized_candidate:
            return item.memory_id

    final_metadata = dict(metadata or {})
    final_metadata.setdefault("record_type", "semantic_memory")
    return get_default_store().write_memory(cleaned_text, metadata=final_metadata)


def read_all_memory_records(limit: int | None = None) -> list[MemoryItem]:
    """Return semantic and file records together for admin surfaces."""
    semantic_records = get_default_store().read_all_memories(limit=limit)
    file_records = get_attachment_store().read_all_memories(limit=limit)
    combined = semantic_records + file_records
    if limit is not None:
        return combined[:limit]
    return combined


def clear_all_memory_stores() -> dict[str, int]:
    """Clear semantic and attachment collections."""
    semantic_cleared = get_default_store().clear_memories()
    file_cleared = get_attachment_store().clear_memories()
    return {
        "semantic": semantic_cleared,
        "files": file_cleared,
        "total": semantic_cleared + file_cleared,
    }


def _select_related_memory_ids(topic: str, candidates: list[MemoryItem]) -> set[str]:
    from importlib import import_module

    llm = import_module("llm")

    candidate_payload = [
        {"id": item.memory_id, "text": item.text}
        for item in candidates
    ]
    prompt = (
        f"Topic: {topic}\n\n"
        f"Candidates JSON:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
    )
    result = llm._run_model_api(
        text=prompt,
        system_instructions=MEMORY_RELATED.read_text(encoding="utf-8"),
        model=MINIMAL_MODEL,
        tool_use_allowed=False,
        force_tool=False,
        temperature=0,
        thinking_level="low",
    )

    raw = (result or "").strip()
    if not raw:
        return set()

    def _try_parse(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    parsed = _try_parse(raw)

    if parsed is None:
        match = re.search(r"(\[.*\])", raw, re.S)
        if match:
            parsed = _try_parse(match.group(1))

    if parsed is None:
        retry_system = (
            MEMORY_RELATED.read_text(encoding="utf-8")
            + "\n\nIf your previous output was not valid JSON, now respond with ONLY a strict JSON array of ids (e.g. [\"id1\", \"id2\"]). No explanation, no markdown."
        )
        try:
            retry = llm._run_model_api(
                text=prompt,
                system_instructions=retry_system,
                model=MINIMAL_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=0,
                thinking_level="low",
            )
            retry_raw = (retry or "").strip()
            parsed = _try_parse(retry_raw)
            if parsed is None:
                match = re.search(r"(\[.*\])", retry_raw, re.S)
                if match:
                    parsed = _try_parse(match.group(1))
        except Exception:
            parsed = None

    if not isinstance(parsed, list):
        return set()

    valid_ids = {item.memory_id for item in candidates}
    return {
        memory_id
        for memory_id in parsed
        if isinstance(memory_id, str) and memory_id in valid_ids
    }


def forget_memories(topic: str) -> str:
    """
    Delete up to 5 semantic memories semantically related to the provided topic.
    """
    normalized_topic = (topic or "").strip()
    if not normalized_topic:
        return "Please provide a memory topic to forget."

    try:
        store = get_default_store()
        matches = store.search_memories(normalized_topic, limit=5)

        if not matches:
            return f"No memories found for topic: '{normalized_topic}'."

        related_ids = _select_related_memory_ids(normalized_topic, matches)
        related_matches = [item for item in matches if item.memory_id in related_ids]

        if not related_matches:
            return (
                f"Found {len(matches)} candidate memory item(s), but none were "
                f"LLM-confirmed as related to topic: '{normalized_topic}'."
            )

        ids_to_delete = [item.memory_id for item in related_matches]
        store.collection.delete(ids=ids_to_delete)

        deleted_preview = "\n".join(
            f"{index + 1}. [ID: {item.memory_id}] {item.text}"
            for index, item in enumerate(related_matches)
        )

        return (
            f"Forgotten {len(ids_to_delete)} LLM-confirmed memory item(s) for topic: '{normalized_topic}'.\n"
            f"{deleted_preview}"
        )
    except Exception as error:
        return f"Failed to forget memories: {error}"