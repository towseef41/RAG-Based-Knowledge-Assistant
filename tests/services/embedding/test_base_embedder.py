import pytest
from typing import List
from app.services.embedding.base_embedder import BaseEmbedder


def test_base_embedder_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedder()  # Abstract class should not be directly instantiated


def test_get_embedding_not_implemented():
    """
    Ensure that calling get_embedding on a subclass that doesn't implement it raises NotImplementedError
    """

    class IncompleteEmbedder(BaseEmbedder):
        pass

    with pytest.raises(TypeError):
        IncompleteEmbedder()  # Should raise because it's still abstract


def test_mock_embedder_returns_vector():
    """
    A concrete test of a mock subclass to ensure the structure works
    """

    class MockEmbedder(BaseEmbedder):
        def get_embedding(self, text: str) -> List[float]:
            return [1.0, 2.0, 3.0]

    embedder = MockEmbedder()
    result = embedder.get_embedding("Hello")

    assert isinstance(result, list)
    assert all(isinstance(val, float) for val in result)
    assert result == [1.0, 2.0, 3.0]
