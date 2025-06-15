import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding.local_embedder import LocalEmbedder
import numpy as np

@patch("app.services.embedding.local_embedder.SentenceTransformer")
def test_local_embedder_initialization(mock_sentence_transformer):
    model_mock = MagicMock()
    mock_sentence_transformer.return_value = model_mock

    embedder = LocalEmbedder(model_name="test-model")
    mock_sentence_transformer.assert_called_once_with("test-model")
    assert embedder.model == model_mock

@patch("app.services.embedding.local_embedder.SentenceTransformer")
def test_local_embedder_get_embedding_returns_list(mock_sentence_transformer):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])  # ✅ Return np.array
    mock_sentence_transformer.return_value = mock_model

    embedder = LocalEmbedder()
    result = embedder.get_embedding("Hello world")

    mock_model.encode.assert_called_once_with("Hello world")
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]


@patch("app.services.embedding.local_embedder.SentenceTransformer")
def test_local_embedder_empty_input(mock_sentence_transformer):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([])  # ✅ Return empty np.array
    mock_sentence_transformer.return_value = mock_model

    embedder = LocalEmbedder()
    result = embedder.get_embedding("")

    assert isinstance(result, list)
    assert result == []


def test_local_embedder_real_model_smoke():
    """
    Optional: This test uses the actual model. Use for integration-level confidence.
    Can be skipped if you prefer full mocking for unit tests.
    """
    embedder = LocalEmbedder()
    result = embedder.get_embedding("Quick brown fox")

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)
