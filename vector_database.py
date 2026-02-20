from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions


logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "vector_database"
DEFAULT_COLLECTION_NAME = "jarvis_memory"


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


    def read_memory(self, memory_id: str) -> MemoryItem | None:
        """Read one memory by id."""
        result = self.collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not result.get("ids"):
            return None

        item_id = result["ids"][0]
        text = (result.get("documents") or [""])[0]
        metadata = (result.get("metadatas") or [{}])[0] or {}
        return MemoryItem(memory_id=item_id, text=text, metadata=metadata)

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

    def update_memory(self, memory_id: str, text: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        """Update text and/or metadata for an existing memory."""
        current = self.read_memory(memory_id)
        if current is None:
            return False

        final_text = text.strip() if text is not None else current.text
        if not final_text:
            raise ValueError("Updated memory text cannot be empty")

        final_metadata = dict(current.metadata)
        if metadata:
            final_metadata.update(metadata)
        final_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.collection.upsert(ids=[memory_id], documents=[final_text], metadatas=[final_metadata])
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one memory by id. Returns True if it existed."""
        existing = self.read_memory(memory_id)
        if existing is None:
            return False

        self.collection.delete(ids=[memory_id])
        return True

    def deduplicate_memories(
        self,
        similarity_threshold: float = 0.9999,
    ) -> None:
        """Remove semantic duplicates and keep the oldest memory."""
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        memories = self.read_all_memories()
        if not memories:
            print(
                "Dedup summary:",
                "before=0",
                "removed=0",
                "kept=0",
                f"threshold={similarity_threshold}",
            )
            return

        def _created_at_value(item: MemoryItem) -> datetime:
            raw_created_at = item.metadata.get("created_at")
            if isinstance(raw_created_at, str):
                normalized = raw_created_at.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalized)
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            return datetime.max.replace(tzinfo=timezone.utc)

        memories.sort(key=lambda item: (_created_at_value(item), item.memory_id))
        embeddings: Any = self.embedding_function([item.text for item in memories])
        if not isinstance(embeddings, list) or len(embeddings) != len(memories):
            raise ValueError("Embedding function returned unexpected output")

        def _cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)

        kept_indices: list[int] = []
        removed_ids: list[str] = []
        for index, memory in enumerate(memories):
            is_duplicate = any(
                _cosine_similarity(embeddings[index], embeddings[kept_index]) >= similarity_threshold
                for kept_index in kept_indices
            )
            if is_duplicate:
                removed_ids.append(memory.memory_id)
            else:
                kept_indices.append(index)

        if removed_ids:
            self.collection.delete(ids=removed_ids)

        print(
            "Dedup summary:",
            f"before={len(memories)}",
            f"removed={len(removed_ids)}",
            f"kept={len(memories) - len(removed_ids)}",
            f"threshold={similarity_threshold}",
        )
        if removed_ids:
            print("Removed IDs:", ", ".join(removed_ids))


    def count(self) -> int:
        """Return number of items in the collection."""
        return self.collection.count()


def get_default_store() -> VectorMemoryStore:
    """Convenience factory for the default project memory store."""
    return VectorMemoryStore(
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

if __name__ == "__main__":
    store = get_default_store()
    store.deduplicate_memories()