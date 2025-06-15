import pytest
from unittest.mock import MagicMock
from app.services.rag_service import RagService
from app.db.models import Conversation, Message

@pytest.fixture
def mock_services():
    return {
        "embedding_service": MagicMock(),
        "storage_service": MagicMock(),
        "vector_store_service": MagicMock(),
        "generator_service": MagicMock(),
        "reranking_service": MagicMock(),
    }

@pytest.fixture
def rag_service(mock_services):
    return RagService(
        embedding_service=mock_services["embedding_service"],
        storage_service=mock_services["storage_service"],
        vector_store_service=mock_services["vector_store_service"],
        generator_service=mock_services["generator_service"],
        reranking_service=mock_services["reranking_service"],
    )

def test_chat_new_conversation(rag_service, mock_services):
    query = "What is RAG?"
    embedding = [0.1, 0.2, 0.3]
    chunks = [{"text": "RAG is Retrieval-Augmented Generation."}]
    generated_answer = "RAG stands for Retrieval-Augmented Generation."

    # Mock behavior
    mock_services["embedding_service"].get_embedding.return_value = embedding
    mock_services["vector_store_service"].query.return_value = chunks
    mock_services["generator_service"].generate_answer.return_value = generated_answer

    new_conversation = Conversation(id="conv-id-123", knowledge_base_id="kb-456")
    new_conversation.messages = []

    mock_services["storage_service"].get_conversation_by_id.return_value = None
    mock_services["storage_service"].create_conversation.side_effect = lambda c: setattr(c, "id", "conv-id-123")

    result = rag_service.chat(query=query, knowledge_base_id="kb-456")

    assert result["answer"] == generated_answer
    assert result["context_chunks"] == chunks
    assert result["conversation_id"] == "conv-id-123"

    mock_services["embedding_service"].get_embedding.assert_called_once_with(query)
    mock_services["vector_store_service"].query.assert_called_once()
    mock_services["generator_service"].generate_answer.assert_called_once()
    mock_services["storage_service"].add_message.assert_called()

def test_chat_existing_conversation(rag_service, mock_services):
    query = "Explain transformers."
    embedding = [0.5, 0.6, 0.7]
    chunks = [{"text": "Transformers are attention-based models."}]
    generated_answer = "Transformers use attention mechanisms."

    # Mock behavior
    mock_services["embedding_service"].get_embedding.return_value = embedding
    mock_services["vector_store_service"].query.return_value = chunks
    mock_services["generator_service"].generate_answer.return_value = generated_answer

    existing_conversation = Conversation(id="conv-789", knowledge_base_id="kb-456")
    existing_conversation.messages = [
        Message(role="user", content="What is a model?"),
        Message(role="assistant", content="A model is a mathematical function."),
    ]

    mock_services["storage_service"].get_conversation_by_id.return_value = existing_conversation

    result = rag_service.chat(query=query, conversation_id="conv-789")

    assert result["answer"] == generated_answer
    assert result["context_chunks"] == chunks
    assert result["conversation_id"] == "conv-789"

    mock_services["embedding_service"].get_embedding.assert_called_once_with(query)
    mock_services["vector_store_service"].query.assert_called_once()
    mock_services["generator_service"].generate_answer.assert_called_once()
    mock_services["storage_service"].add_message.assert_called()
