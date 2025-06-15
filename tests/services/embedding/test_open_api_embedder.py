import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding.openai_embedder import OpenAIEmbedder


@patch("app.services.embedding.openai_embedder.openai.Embedding.create")
def test_get_embedding_success(mock_create):
    # Arrange: Mock the API response
    mock_create.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]}
        ]
    }

    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")

    # Act
    result = embedder.get_embedding("OpenAI is awesome")

    # Assert
    mock_create.assert_called_once_with(
        input="OpenAI is awesome",
        model="text-embedding-3-small"
    )
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]


@patch("app.services.embedding.openai_embedder.openai.Embedding.create")
def test_get_embedding_empty_string(mock_create):
    mock_create.return_value = {
        "data": [{"embedding": []}]
    }

    embedder = OpenAIEmbedder()
    result = embedder.get_embedding("")

    assert result == []


@patch("app.services.embedding.openai_embedder.openai.Embedding.create")
def test_get_embedding_uses_correct_model(mock_create):
    embedder = OpenAIEmbedder(model_name="text-embedding-ada-002")
    mock_create.return_value = {"data": [{"embedding": [0.4, 0.5]}]}

    result = embedder.get_embedding("test")

    mock_create.assert_called_once_with(
        input="test",
        model="text-embedding-ada-002"
    )
    assert result == [0.4, 0.5]


@patch("app.services.embedding.openai_embedder.openai.Embedding.create")
def test_get_embedding_raises_exception_on_api_failure(mock_create):
    mock_create.side_effect = Exception("API error")

    embedder = OpenAIEmbedder()

    with pytest.raises(Exception, match="API error"):
        embedder.get_embedding("fail case")
