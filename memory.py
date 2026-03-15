from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import shutil
import threading
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
    SKILL_COLLECTION_NAME,
    SKILL_DB_DIR,
    get_paid_gemini_api_key,
)
import history

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


DEFAULT_MEMORY_DB_PATH = Path(__file__).parent / "memory" / "memories_vector_db"
DEFAULT_DB_PATH = DEFAULT_MEMORY_DB_PATH
DEFAULT_SKILL_DB_PATH = Path(SKILL_DB_DIR)
DEFAULT_COLLECTION_NAME = MEMORY_SEMANTIC_COLLECTION_NAME
DEFAULT_FILE_COLLECTION_NAME = MEMORY_FILE_COLLECTION_NAME
DEFAULT_SKILL_COLLECTION_NAME = SKILL_COLLECTION_NAME
DEFAULT_ARCHIVE_PATH = Path(MEMORY_ARCHIVE_DIR)
MEMORY_RELATED = Path(__file__).parent / "agent_instructions" / "memory_related.md"
MEMORY_EXTRACTOR_FILE = Path(__file__).parent / "agent_instructions" / "memory_extractor.md"
SKILL_EXTRACTOR_FILE = Path(__file__).parent / "agent_instructions" / "skill_extractor.md"
SKILLS_DIR = Path(__file__).parent / "skills"
MEMORY_SCHEMA_VERSION = "2026-03-11-gemini2"
_SUPPORTED_ATTACHMENT_EMBEDDING_PREFIXES = ("image/", "video/", "audio/")
_SUPPORTED_ATTACHMENT_EMBEDDING_TYPES = {"application/pdf"}

_embedding_service: GeminiEmbeddingService | None = None
_default_store: VectorMemoryStore | None = None
_attachment_store: VectorMemoryStore | None = None
_skill_store: VectorMemoryStore | None = None
_skill_file_lock = threading.RLock()
_skill_migration_checked = False


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).strip().lower()


def _normalize_embedding(values: list[float]) -> list[float]:
    # Embedding normalization removed — return raw embeddings unchanged.
    return values


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

        # Ensure the collection's embedding dimensionality matches the embedding service.
        # If the existing collection contains vectors with a different dimensionality
        # (e.g. 1536) than the configured Gemini embedding dimension (e.g. 3072),
        # switch to / create a collection suffixed with the required dimension
        # to avoid upsert/query failures due to vector size mismatches.
        try:
            existing_dim: int | None = None
            # Only attempt checks if collection has items
            try:
                existing_count = self.collection.count()
            except Exception:
                existing_count = 0

            if existing_count and existing_count > 0:
                # Prefer explicit per-item metadata if present
                try:
                    meta_res = self.collection.get(include=["metadatas"], limit=1)
                    metadatas = meta_res.get("metadatas") or []
                    if metadatas and isinstance(metadatas[0], dict) and "embedding_dim" in metadatas[0]:
                        existing_dim = int(metadatas[0].get("embedding_dim") or 0)
                except Exception:
                    existing_dim = None

                # Fall back to inspecting stored embedding vector length
                if existing_dim is None:
                    try:
                        emb_res = self.collection.get(include=["embeddings"], limit=1)
                        embeddings = emb_res.get("embeddings") or []
                        if embeddings:
                            emb = embeddings[0]
                            # Handle possible nesting (e.g. [[...]])
                            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                                emb = emb[0]
                            if isinstance(emb, list):
                                existing_dim = len(emb)
                    except Exception:
                        existing_dim = None

            desired_dim = int(getattr(self.embedding_service, "output_dimensionality", 0) or 0)
            if existing_dim and desired_dim and existing_dim != desired_dim:
                suffix = f"_{desired_dim}"
                # Avoid duplicating suffix if already present
                if not str(self.collection_name).endswith(suffix):
                    new_name = f"{self.collection_name}{suffix}"
                else:
                    new_name = self.collection_name
                logging.warning(
                    "Collection '%s' has embedding dim %s but configured dim is %s; switching to '%s'",
                    self.collection_name,
                    existing_dim,
                    desired_dim,
                    new_name,
                )
                self.collection_name = new_name
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception:
            logging.exception("Failed to verify or migrate collection embedding dimensionality")

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
            db_path=DEFAULT_MEMORY_DB_PATH,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
    return _default_store


def get_attachment_store() -> VectorMemoryStore:
    """Convenience factory for the attachment/file memory store."""
    global _attachment_store
    if _attachment_store is None:
        _attachment_store = VectorMemoryStore(
            db_path=DEFAULT_MEMORY_DB_PATH,
            collection_name=DEFAULT_FILE_COLLECTION_NAME,
        )
    return _attachment_store


def get_skill_store() -> VectorMemoryStore:
    """Convenience factory for the reusable skill memory store."""
    global _skill_store, _skill_migration_checked
    if _skill_store is None:
        _skill_store = VectorMemoryStore(
            db_path=DEFAULT_SKILL_DB_PATH,
            collection_name=DEFAULT_SKILL_COLLECTION_NAME,
        )
    if not _skill_migration_checked:
        _migrate_legacy_skill_records()
        _skill_migration_checked = True
    return _skill_store


def _migrate_legacy_skill_records() -> None:
    """Move legacy skill records from semantic store into dedicated skill store."""
    try:
        legacy_records = get_default_store().read_all_memories(where={"record_type": "skill"})
    except Exception:
        return

    if not legacy_records:
        return

    try:
        destination_store = _skill_store or VectorMemoryStore(
            db_path=DEFAULT_SKILL_DB_PATH,
            collection_name=DEFAULT_SKILL_COLLECTION_NAME,
        )
    except Exception:
        return

    try:
        existing_ids = {
            item.memory_id
            for item in destination_store.read_all_memories(where={"record_type": "skill"})
        }
    except Exception:
        existing_ids = set()

    migrated_ids: list[str] = []
    for item in legacy_records:
        if item.memory_id not in existing_ids:
            try:
                destination_store.write_memory(
                    text=item.text,
                    metadata=dict(item.metadata or {}),
                    memory_id=item.memory_id,
                )
            except Exception:
                continue
        migrated_ids.append(item.memory_id)

    if migrated_ids:
        try:
            get_default_store().collection.delete(ids=migrated_ids)
        except Exception:
            pass


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


def build_relevant_memories_text(query: str, semantic_limit: int, file_limit: int) -> str:
    relevant_memories = search_all_memories(
        query,
        semantic_limit=semantic_limit,
        file_limit=file_limit,
    )
    return "\n\n".join(render_memory_for_prompt(item) for item in relevant_memories)


def _slugify_skill_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "skill"


def _coerce_skill_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.7
    return max(0.0, min(1.0, numeric))


def _build_skill_document_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Skill: {skill['name']}",
            f"Summary: {skill['summary']}",
            f"When to use: {skill['when_to_use']}",
            f"Planning pattern: {skill['planning_pattern']}",
            f"Tags: {', '.join(skill['tags']) if skill['tags'] else 'none'}",
            f"Confidence: {skill['confidence']:.2f}",
        ]
    )


def _render_skill_markdown(
    skill: dict[str, Any],
    history_file: str,
    created_at: str,
) -> str:
    cleaned_history_file = (history_file or "default").replace("\n", " ").strip() or "default"
    return (
        f"# {skill['name']}\n\n"
        "## Summary\n"
        f"{skill['summary']}\n\n"
        "## When To Use\n"
        f"{skill['when_to_use']}\n\n"
        "## Planning Pattern\n"
        f"{skill['planning_pattern']}\n\n"
        "## Source\n"
        f"Extracted from conversation history `{cleaned_history_file}` on {created_at}.\n"
    )


def _write_skill_markdown_file(
    skill_id: str,
    skill: dict[str, Any],
    history_file: str,
    created_at: str,
) -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = _slugify_skill_name(skill['name'])

    with _skill_file_lock:
        counter = 0
        while True:
            suffix = f"_{counter}" if counter else ""
            candidate_path = SKILLS_DIR / f"{base_name}{suffix}.md"
            if not candidate_path.exists():
                break
            counter += 1

        candidate_path.write_text(
            _render_skill_markdown(
                skill=skill,
                history_file=history_file,
                created_at=created_at,
            ),
            encoding="utf-8",
        )
    return candidate_path


def _persist_skill_candidate(skill: dict[str, Any], history_file: str) -> dict[str, Any]:
    normalized_name = _normalize_text(skill["name"])
    existing_skill_records = get_skill_store().read_all_memories(where={"record_type": "skill"})
    for item in existing_skill_records:
        existing_name = _normalize_text(str((item.metadata or {}).get("skill_name") or ""))
        if existing_name and existing_name == normalized_name:
            return {
                "status": "duplicate",
                "skill_name": skill["name"],
                "memory_id": item.memory_id,
                "skill_file": str((item.metadata or {}).get("skill_file") or ""),
            }

    created_at = _utcnow_iso()
    document_text = _build_skill_document_text(skill)
    skill_id = hashlib.sha256(f"{history_file}|{document_text}".encode("utf-8")).hexdigest()

    metadata = {
        "record_type": "skill",
        "source_type": "skill_extractor",
        "history_file": history_file,
        "skill_name": skill["name"],
        "skill_status": "draft",
        "skill_confidence": skill["confidence"],
        "skill_tags_json": json.dumps(skill["tags"], ensure_ascii=False),
        "schema_version": MEMORY_SCHEMA_VERSION,
        "created_at": created_at,
    }

    skill_path = _write_skill_markdown_file(
        skill_id=skill_id,
        skill=skill,
        history_file=history_file,
        created_at=created_at,
    )

    skill_file = str(skill_path)
    metadata["skill_file"] = skill_file
    memory_id = get_skill_store().write_memory(
        text=document_text,
        metadata=metadata,
        memory_id=skill_id,
    )

    return {
        "status": "created",
        "skill_name": skill["name"],
        "memory_id": memory_id,
        "skill_file": skill_file,
    }

def _extract_section_from_markdown(text: str, header: str) -> str:
    pattern = rf"^##\s*{re.escape(header)}\s*$\n(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, text, flags=re.M | re.S)
    return (m.group(1).strip() if m else "")


def _sync_skills_from_folder() -> None:
    try:
        store = get_skill_store()
    except Exception:
        logging.exception("Failed to initialize skill store; skipping skill sync.")
        return

    try:
        # Clear existing skill records in the skill collection
        try:
            cleared = store.clear_memories()
            logging.info("Cleared %d existing skill records from skill DB.", cleared)
        except Exception:
            logging.exception("Failed to clear existing skill records; continuing with import.")

        skills_dir = SKILLS_DIR
        if not skills_dir.exists() or not skills_dir.is_dir():
            logging.info("Skills directory not found at %s; skipping skill import.", skills_dir)
            return

        for skill_file in sorted(skills_dir.iterdir()):
            if not skill_file.is_file():
                continue
            if skill_file.suffix.lower() not in (".md", ".markdown", ".txt"):
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                name_match = re.search(r'^\s*#\s*(.+)$', content, flags=re.M)
                skill_name = name_match.group(1).strip() if name_match else skill_file.stem

                summary = _extract_section_from_markdown(content, "Summary")
                when_to_use = _extract_section_from_markdown(content, "When To Use")
                planning_pattern = _extract_section_from_markdown(content, "Planning Pattern")

                tags: list[str] = []
                confidence = 1.0
                skill = {
                    "name": skill_name,
                    "summary": summary or "",
                    "when_to_use": when_to_use or "",
                    "planning_pattern": planning_pattern or "",
                    "tags": tags,
                    "confidence": confidence,
                }

                document_text = _build_skill_document_text(skill)
                metadata = {
                    "record_type": "skill",
                    "source_type": "skill_file_import",
                    "history_file": "skill_files",
                    "skill_name": skill_name,
                    "skill_status": "loaded",
                    "skill_confidence": confidence,
                    "skill_tags_json": json.dumps(tags, ensure_ascii=False),
                    "schema_version": MEMORY_SCHEMA_VERSION,
                    "created_at": _utcnow_iso(),
                    "skill_file": str(skill_file),
                }

                memory_id = hashlib.sha256(f"skill_files|{document_text}".encode("utf-8")).hexdigest()
                try:
                    store.write_memory(text=document_text, metadata=metadata, memory_id=memory_id)
                    logging.info("Imported skill '%s' from %s", skill_name, skill_file)
                except Exception:
                    logging.exception("Failed writing skill '%s' to skill DB", skill_name)
            except Exception:
                logging.exception("Failed reading skill file '%s'", skill_file)
    except Exception:
        logging.exception("Unhandled error syncing skills from folder")

def run_skill_extraction_async(
    history_file: str,
    temperature: float,
    relevant_memories_text: str = "",
    attachment_context_text: str = "",
    history_cache: object | None = None,
) -> None:
    def _worker() -> None:
        try:
            from importlib import import_module

            llm = import_module("llm")
            run_model_api = llm._run_model_api

            extraction_context = "Relevant memories:\n\n" + (relevant_memories_text or "none")
            if attachment_context_text.strip():
                extraction_context += "\n\nRecent attachment memories:\n\n" + attachment_context_text.strip()

            if history_cache is not None:
                extractor_response = run_model_api(
                    text=extraction_context,
                    system_instructions=SKILL_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low",
                    history_cache=history_cache,
                    current_history_text=history_cache.history_text,
                )
            else:
                extraction_input = (
                    "History: "
                    + history.get_conversation_history(history_file=history_file)
                    + "\n\n"
                    + extraction_context
                )
                extractor_response = run_model_api(
                    text=extraction_input,
                    system_instructions=SKILL_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low",
                )

            raw_response = (extractor_response or "").strip()
            parsed_skills = _parse_skill_extraction_response(raw_response)

            if parsed_skills:
                outcomes = [_persist_skill_candidate(skill=skill, history_file=history_file) for skill in parsed_skills]
                history.append_history(
                    role="SkillExtractor",
                    text=json.dumps({"skills": parsed_skills, "outcomes": outcomes}, ensure_ascii=False),
                    history_file=history_file,
                )
            elif raw_response and raw_response != "<NO_SKILL>":
                history.append_history(role="SkillExtractor", text=raw_response, history_file=history_file)
        except Exception as exc:
            print(f"Error generating skill extractor response: {exc}")
        finally:
            if history_cache is not None:
                history_cache.release()

    threading.Thread(target=_worker, daemon=True).start()


def _parse_skill_extraction_response(response_text: str) -> list[dict[str, Any]]:
    cleaned_text = (response_text or "").strip()
    if not cleaned_text or cleaned_text == "<NO_SKILL>":
        return []

    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    parsed = _try_parse_json(cleaned_text)
    if parsed is None:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned_text, re.S)
        if match:
            parsed = _try_parse_json(match.group(1))

    if isinstance(parsed, dict):
        candidates = parsed.get("skills")
    else:
        candidates = parsed

    if not isinstance(candidates, list):
        return []

    normalized_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        name = str(candidate.get("name") or "").strip()
        summary = str(candidate.get("summary") or "").strip()
        when_to_use = str(candidate.get("when_to_use") or "").strip()
        planning_pattern = str(candidate.get("planning_pattern") or "").strip()
        if not all([name, summary, when_to_use, planning_pattern]):
            continue

        raw_tags = candidate.get("tags")
        if not isinstance(raw_tags, list):
            raw_tags = []
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]

        normalized_candidates.append(
            {
                "name": name,
                "summary": summary,
                "when_to_use": when_to_use,
                "planning_pattern": planning_pattern,
                "tags": tags,
                "confidence": _coerce_skill_confidence(candidate.get("confidence")),
            }
        )

    return normalized_candidates


def build_relevant_skills_text(query: str, limit: int = 4) -> str:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return ""

    try:
        skills = get_skill_store().search_memories(
            cleaned_query,
            limit=max(1, limit),
            where={"record_type": "skill"},
        )
    except Exception:
        return ""

    sections: list[str] = []
    for item in skills:
        metadata = dict(item.metadata or {})
        skill_name = str(metadata.get("skill_name") or "Unnamed skill").strip()
        tags_raw = metadata.get("skill_tags_json")
        tags_list: list[str] = []
        if isinstance(tags_raw, str) and tags_raw.strip():
            try:
                parsed_tags = json.loads(tags_raw)
                if isinstance(parsed_tags, list):
                    tags_list = [str(tag).strip() for tag in parsed_tags if str(tag).strip()]
            except Exception:
                tags_list = []
        tags_text = ", ".join(tags_list) if tags_list else "none"
        sections.append(
            "\n".join(
                [
                    f"Skill: {skill_name}",
                    f"Details: {item.text}",
                    f"Tags: {tags_text}",
                    f"Status: {metadata.get('skill_status', 'draft')}",
                ]
            )
        )

    return "\n\n".join(sections)


def build_skill_names_text(query: str, limit: int = 10) -> str:
    """Return top relevant skill file paths for planner prompts using semantic search."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return ""

    try:
        records = get_skill_store().search_memories(
            cleaned_query,
            limit=max(1, limit),
            where={"record_type": "skill"},
        )
    except Exception:
        return ""

    # Preserve relevance order from vector search while deduplicating by file path.
    seen: set[str] = set()
    skill_files: list[str] = []
    for item in records:
        skill_file = str((item.metadata or {}).get("skill_file") or "").strip()
        if not skill_file or skill_file in seen:
            continue
        seen.add(skill_file)
        skill_files.append(skill_file)

    if not skill_files:
        return ""

    return "\n".join(["Available reusable planning skill file paths:", *[f"- {skill_file}" for skill_file in skill_files]])


def run_memory_extraction_async(
    history_file: str,
    temperature: float,
    relevant_memories_text: str = "",
    attachment_context_text: str = "",
    history_cache: object | None = None,
) -> None:
    def _worker() -> None:
        try:
            from importlib import import_module

            llm = import_module("llm")
            run_model_api = llm._run_model_api

            extraction_context = "Relevant memories:\n\n" + relevant_memories_text
            if attachment_context_text.strip():
                extraction_context += "\n\nRecent attachment memories:\n\n" + attachment_context_text.strip()

            if history_cache is not None:
                memory_extractor_response = run_model_api(
                    text=extraction_context,
                    system_instructions=MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low",
                    history_cache=history_cache,
                    current_history_text=history_cache.history_text,
                )
            else:
                extraction_input = (
                    "History: "
                    + history.get_conversation_history(history_file=history_file)
                    + "\n\n"
                    + extraction_context
                )
                memory_extractor_response = run_model_api(
                    text=extraction_input,
                    system_instructions=MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"),
                    model=MINIMAL_MODEL,
                    tool_use_allowed=False,
                    force_tool=False,
                    temperature=temperature,
                    thinking_level="low",
                )

            raw_response = (memory_extractor_response or "").strip()
            parsed_candidates = _parse_memory_extraction_response(raw_response)
            if parsed_candidates:
                for candidate in parsed_candidates:
                    metadata = {
                        "source_type": "memory_extractor",
                        "history_file": history_file,
                        "memory_category": candidate["category"],
                    }
                    if candidate["related_file_ids"]:
                        metadata["related_file_ids_json"] = json.dumps(candidate["related_file_ids"], ensure_ascii=False)
                    write_semantic_memory(candidate["memory"], metadata=metadata)
                history.append_history(
                    role="MemoryExtractor",
                    text=json.dumps({"memories": parsed_candidates}, ensure_ascii=False),
                    history_file=history_file,
                )
            elif raw_response and raw_response != "<NO_MEMORY>":
                write_semantic_memory(
                    raw_response,
                    metadata={
                        "source_type": "memory_extractor_fallback",
                        "history_file": history_file,
                    },
                )
                history.append_history(role="MemoryExtractor", text=raw_response, history_file=history_file)
        except Exception as exc:
            print(f"Error generating memory extractor response: {exc}")
        finally:
            if history_cache is not None:
                history_cache.release()

    threading.Thread(target=_worker, daemon=True).start()


def _parse_memory_extraction_response(response_text: str) -> list[dict[str, Any]]:
    cleaned_text = (response_text or "").strip()
    if not cleaned_text or cleaned_text == "<NO_MEMORY>":
        return []

    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    parsed = _try_parse_json(cleaned_text)
    if parsed is None:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned_text, re.S)
        if match:
            parsed = _try_parse_json(match.group(1))

    if isinstance(parsed, dict):
        candidates = parsed.get("memories")
    else:
        candidates = parsed

    if not isinstance(candidates, list):
        return []

    normalized_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        memory_text = str(candidate.get("memory") or candidate.get("text") or "").strip()
        if not memory_text:
            continue

        category = str(candidate.get("category") or "general").strip() or "general"
        raw_related_file_ids = candidate.get("related_file_ids") or []
        if not isinstance(raw_related_file_ids, list):
            raw_related_file_ids = []

        normalized_candidates.append(
            {
                "memory": memory_text,
                "category": category,
                "related_file_ids": [str(file_id) for file_id in raw_related_file_ids if str(file_id).strip()],
            }
        )

    return normalized_candidates


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
    combined.extend(
        get_default_store().search_memories(
            query,
            limit=semantic_limit,
            where={"record_type": "semantic_memory"},
        )
    )
    combined.extend(get_attachment_store().search_memories(query, limit=file_limit))
    return combined


def write_semantic_memory(text: str, metadata: dict[str, Any] | None = None) -> str | None:
    """Write a semantic memory if it is not an exact normalized duplicate."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None

    existing = get_default_store().search_memories(
        cleaned_text,
        limit=5,
        where={"record_type": "semantic_memory"},
    )
    normalized_candidate = _normalize_text(cleaned_text)
    for item in existing:
        if _normalize_text(item.text) == normalized_candidate:
            return item.memory_id

    final_metadata = dict(metadata or {})
    final_metadata.setdefault("record_type", "semantic_memory")
    return get_default_store().write_memory(cleaned_text, metadata=final_metadata)


def read_all_memory_records(limit: int | None = None) -> list[MemoryItem]:
    """Return semantic, file, and skill records together for admin surfaces."""
    semantic_records = get_default_store().read_all_memories(
        limit=limit,
        where={"record_type": "semantic_memory"},
    )
    file_records = get_attachment_store().read_all_memories(limit=limit)
    skill_records = get_skill_store().read_all_memories(limit=limit)
    combined = semantic_records + file_records + skill_records
    if limit is not None:
        return combined[:limit]
    return combined


def _clear_memory_archive() -> int:
    """Remove all files under DEFAULT_ARCHIVE_PATH and recreate the directory.
    Returns the number of files removed.
    """
    try:
        archive_dir = DEFAULT_ARCHIVE_PATH
        if not archive_dir.exists() or not archive_dir.is_dir():
            return 0
        files_to_remove = sum(1 for p in archive_dir.rglob("*") if p.is_file())
        shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        return files_to_remove
    except Exception:
        logging.exception("Failed clearing memory archive at '%s'", str(DEFAULT_ARCHIVE_PATH))
        return 0


def clear_all_memory_stores() -> dict[str, int]:
    """Clear semantic, attachment, and skill collections."""
    semantic_cleared = get_default_store().clear_memories()
    file_cleared = get_attachment_store().clear_memories()
    skill_cleared = get_skill_store().clear_memories()
    archive_cleared = _clear_memory_archive()
    total_files = file_cleared + archive_cleared
    return {
        "semantic": semantic_cleared,
        "files": total_files,
        "skills": skill_cleared,
        "archives": archive_cleared,
        "total": semantic_cleared + total_files + skill_cleared,
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
        matches = store.search_memories(
            normalized_topic,
            limit=5,
            where={"record_type": "semantic_memory"},
        )

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