from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


DEFAULT_DB_PATH = Path(__file__).parent.parent / "memory" / "vector_db"
DEFAULT_COLLECTION_NAME = "picklebot_memory"


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


if __name__ == "__main__":
    store = get_default_store()
    store.clear_memories()