import os
import pytest
from unittest.mock import MagicMock, patch
from app.services.generator.openai_generator import OpenAIGenerator
from app.services.prompt.prompt_manager import PromptManager


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Ensure OPENAI_API_KEY is set for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client and response."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Generated answer"))]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


@patch("app.services.generator.openai_generator.OpenAI")
def test_generate_answer_basic(mock_openai_class, mock_openai_client):
    mock_openai_class.return_value = mock_openai_client
    generator = OpenAIGenerator(model="gpt-4")

    response = generator.generate_answer("What is RAG?")

    assert response == "Generated answer"
    mock_openai_client.chat.completions.create.assert_called_once()
    called_args = mock_openai_client.chat.completions.create.call_args[1]
    assert called_args["model"] == "gpt-4"
    assert called_args["messages"][-1]["content"] == "What is RAG?"


@patch("app.services.generator.openai_generator.OpenAI")
def test_generate_answer_with_context(mock_openai_class, mock_openai_client):
    mock_openai_class.return_value = mock_openai_client
    generator = OpenAIGenerator()
    context = "RAG = Retrieval-Augmented Generation"

    with patch.object(PromptManager, "render", return_value="SYSTEM: RAG intro") as mock_render:
        response = generator.generate_answer("Explain it", context=context)

    assert response == "Generated answer"
    mock_render.assert_called_once_with("rag", context=context)
    called_args = mock_openai_client.chat.completions.create.call_args[1]
    assert called_args["messages"][0]["role"] == "system"
    assert "SYSTEM: RAG intro" in called_args["messages"][0]["content"]


@patch("app.services.generator.openai_generator.OpenAI")
def test_generate_answer_with_chat_history(mock_openai_class, mock_openai_client):
    mock_openai_class.return_value = mock_openai_client
    generator = OpenAIGenerator()
    chat_history = [
        {"role": "user", "content": "What is vector search?"},
        {"role": "assistant", "content": "It's about searching with embeddings."}
    ]

    response = generator.generate_answer("Give example", chat_history=chat_history)

    called_args = mock_openai_client.chat.completions.create.call_args[1]
    assert called_args["messages"][-1]["content"] == "Give example"
    assert called_args["messages"][-2]["content"] == "It's about searching with embeddings."


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        OpenAIGenerator()
