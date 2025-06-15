import pytest
from datetime import datetime
from app.api.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    DocumentChunk,
    SearchResponse,
)


def test_chat_message_valid_roles():
    assert ChatMessage(role="user", content="Hello").role == "user"
    assert ChatMessage(role="assistant", content="Hi").role == "assistant"
    assert ChatMessage(role="system", content="Welcome").role == "system"

def test_chat_message_invalid_role():
    with pytest.raises(ValueError):
        ChatMessage(role="bot", content="Invalid role")


def test_chat_request_defaults():
    request = ChatRequest(query="What's up?")
    assert request.use_knowledge_base is True
    assert request.stream is False
    assert request.temperature == 0.7


def test_chat_response_model():
    msg = ChatMessage(role="assistant", content="Here is your answer.")
    now = datetime.now()
    response = ChatResponse(
        message=msg,
        conversation_id="abc123",
        created_at=now,
        sources=[{"source": "kb"}],
        usage={"prompt_tokens": 10, "completion_tokens": 20}
    )
    assert response.message.role == "assistant"
    assert response.conversation_id == "abc123"
    assert response.sources == [{"source": "kb"}]
    assert response.usage["prompt_tokens"] == 10


def test_search_request_validation():
    req = SearchRequest(query="find this", limit=10, min_score=0.6)
    assert req.query == "find this"
    assert req.limit == 10
    assert req.min_score == 0.6


def test_document_chunk_model():
    chunk = DocumentChunk(chunk_id=1, text="Sample text", metadata={"author": "Alice"})
    assert chunk.chunk_id == 1
    assert chunk.text == "Sample text"
    assert chunk.metadata["author"] == "Alice"


def test_search_response_model():
    chunks = [
        DocumentChunk(chunk_id=1, text="text 1"),
        DocumentChunk(chunk_id=2, text="text 2")
    ]
    response = SearchResponse(query="search this", results=chunks, total_found=2)
    assert response.query == "search this"
    assert len(response.results) == 2
    assert response.total_found == 2
