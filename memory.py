from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import llm
from config import LOW_MODEL as LOW_MODEL

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


DEFAULT_DB_PATH = Path(__file__).parent / "memory" / "vector_db"
DEFAULT_COLLECTION_NAME = "engem_memory"
MEMORY_RELATED = Path(__file__).parent / "agent_instructions" / "memory_related.md"


def _build_embedding_function() -> Any:
    """Return the preferred embedding function with safe fallback."""
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        pass

    return embedding_functions.DefaultEmbeddingFunction()


@dataclass(slots=True)
class MemoryItem:
    """A single memory entry returned from the vector database."""

    memory_id: str
    text: str
    metadata: dict[str, Any]


class VectorMemoryStore:
    """Persistent memory store backed by ChromaDB."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any | None = None,
    ) -> None:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_function = embedding_function or _build_embedding_function()
        try:
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except ValueError as exc:
            if "Embedding function conflict" not in str(exc):
                raise
            self.collection = self.client.get_collection(name=collection_name)

    def write_memory(self, text: str, metadata: dict[str, Any] | None = None, memory_id: str | None = None) -> str:
        """Write a new memory to the vector DB and return its id."""
        if not text or not text.strip():
            raise ValueError("Memory text cannot be empty")

        final_id = memory_id or str(uuid4())
        final_metadata = dict(metadata or {})
        final_metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())

        self.collection.upsert(
            ids=[final_id],
            documents=[text.strip()],
            metadatas=[final_metadata],
        )
        return final_id

    def read_all_memories(self, limit: int | None = None) -> list[MemoryItem]:
        """Read all stored memories, optionally limited."""
        result = self.collection.get(include=["documents", "metadatas"], limit=limit)

        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        memories: list[MemoryItem] = []
        for i, memory_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            memories.append(MemoryItem(memory_id=memory_id, text=text, metadata=metadata))
        return memories

    def search_memories(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Semantic search memories by query text."""
        if not query or not query.strip():
            return []
        
        if self.count() < limit:
            return self.read_all_memories()

        result = self.collection.query(
            query_texts=[query.strip()],
            n_results=max(1, limit),
            include=["documents", "metadatas"],
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


def get_default_store() -> VectorMemoryStore:
    """Convenience factory for the default project memory store."""
    return VectorMemoryStore(
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
    )


def _select_related_memory_ids(topic: str, candidates: list[MemoryItem]) -> set[str]:
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
        model=LOW_MODEL,
        tool_use_allowed=False,
        force_tool=False,
        temperature=0,
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

    # If direct parse failed, attempt to extract a JSON array substring
    if parsed is None:
        m = re.search(r"(\[.*\])", raw, re.S)
        if m:
            parsed = _try_parse(m.group(1))

    # If still not parsed, retry once asking for strict JSON only
    if parsed is None:
        retry_system = (
            MEMORY_RELATED.read_text(encoding="utf-8")
            + "\n\nIf your previous output was not valid JSON, now respond with ONLY a strict JSON array of ids (e.g. [\"id1\", \"id2\"]). No explanation, no markdown."
        )
        try:
            retry = llm._run_model_api(
                text=prompt,
                system_instructions=retry_system,
                model=LOW_MODEL,
                tool_use_allowed=False,
                force_tool=False,
                temperature=0,
                thinking_level="low"
            )
            retry_raw = (retry or "").strip()
            parsed = _try_parse(retry_raw)
            if parsed is None:
                m2 = re.search(r"(\[.*\])", retry_raw, re.S)
                if m2:
                    parsed = _try_parse(m2.group(1))
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
    Delete up to 5 memories semantically related to the provided topic.
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