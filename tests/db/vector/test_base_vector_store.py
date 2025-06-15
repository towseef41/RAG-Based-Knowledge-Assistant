import pytest
from typing import List, Dict, Any, Union, Optional
from app.db.vector.base_vector_store import BaseVectorStore


class DummyVectorStore(BaseVectorStore):
    """A mock implementation of BaseVectorStore for testing purposes."""

    def __init__(self):
        self._storage = []

    def store_chunks(
        self,
        document_id: int,
        chunks: List[Union[str, Dict[str, Any]]],
        embeddings: List[List[float]],
    ) -> None:
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            self._storage.append({
                "document_id": document_id,
                "text": chunk_text,
                "metadata": metadata,
                "embedding": embeddings[i],
                "similarity": 1.0  # Dummy similarity score
            })

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Union[str, int]]] = None,
        min_score: float = 0.0,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for chunk in self._storage:
            if filters:
                if not all(chunk["metadata"].get(k) == v for k, v in filters.items()):
                    continue
            if chunk["similarity"] >= min_score:
                results.append(chunk)

        return results[:top_k]


def test_store_and_query_chunks():
    store = DummyVectorStore()

    chunks = [{"text": "chunk one", "metadata": {"section": 1}},
              {"text": "chunk two", "metadata": {"section": 2}}]
    embeddings = [[0.1] * 10, [0.2] * 10]

    store.store_chunks(document_id=1, chunks=chunks, embeddings=embeddings)

    results = store.query(query_embedding=[0.1] * 10)

    assert len(results) == 2
    assert results[0]["text"] == "chunk one"
    assert results[1]["text"] == "chunk two"


def test_query_with_filter():
    store = DummyVectorStore()

    chunks = [
        {"text": "chunk A", "metadata": {"section": 1}},
        {"text": "chunk B", "metadata": {"section": 2}},
    ]
    embeddings = [[0.3] * 10, [0.4] * 10]
    store.store_chunks(document_id=99, chunks=chunks, embeddings=embeddings)

    filtered = store.query(query_embedding=[0.3] * 10, filters={"section": 1})

    assert len(filtered) == 1
    assert filtered[0]["text"] == "chunk A"


def test_query_min_score_threshold():
    store = DummyVectorStore()
    store._storage = [
        {"text": "chunk high", "metadata": {}, "embedding": [0.5]*10, "similarity": 0.9},
        {"text": "chunk low", "metadata": {}, "embedding": [0.1]*10, "similarity": 0.2}
    ]

    results = store.query(query_embedding=[0.5]*10, min_score=0.8)
    assert len(results) == 1
    assert results[0]["text"] == "chunk high"
