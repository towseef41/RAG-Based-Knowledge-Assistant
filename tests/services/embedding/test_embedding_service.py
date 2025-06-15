import pytest
from app.services.embedding.embedding_service import EmbeddingService
from app.services.embedding.base_embedder import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3] if text else []


def test_embedding_service_returns_embedding():
    embedder = MockEmbedder()
    service = EmbeddingService(embedder)
    
    result = service.get_embedding("Test input")
    
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
    assert result == [0.1, 0.2, 0.3]


def test_embedding_service_returns_empty_for_empty_input():
    embedder = MockEmbedder()
    service = EmbeddingService(embedder)
    
    result = service.get_embedding("")
    
    assert isinstance(result, list)
    assert result == []


def test_embedding_service_calls_embedder(monkeypatch):
    """
    Ensure the embedder's get_embedding is actually called.
    """
    called = {"flag": False}

    class SpyEmbedder(BaseEmbedder):
        def get_embedding(self, text: str) -> list[float]:
            called["flag"] = True
            return [1.0]

    service = EmbeddingService(SpyEmbedder())
    result = service.get_embedding("test")

    assert called["flag"] is True
    assert result == [1.0]


def test_embedding_service_propagates_embedder_exception():
    class FailingEmbedder(BaseEmbedder):
        def get_embedding(self, text: str) -> list[float]:
            raise RuntimeError("Embedding failed")

    service = EmbeddingService(FailingEmbedder())

    with pytest.raises(RuntimeError, match="Embedding failed"):
        service.get_embedding("anything")
