from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions


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
        self.client = chromadb.PersistentClient(path=str(db_path))
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

    def write_memories(self, items: list[dict[str, Any]]) -> list[str]:
        """Write multiple memory entries in one call.

        Expected item format:
        {
            "text": "memory text",
            "metadata": {"source": "chat"},  # optional
            "id": "optional-custom-id"
        }
        """
        if not items:
            return []

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            memory_id = str(item.get("id") or uuid4())
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())

            ids.append(memory_id)
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            return []

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return ids

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
    print(f"Memory store initialized at {store.db_path} with collection '{store.collection_name}'")
    print(f"Current memory count: {store.count()}")