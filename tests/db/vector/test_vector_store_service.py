import pytest
from unittest.mock import MagicMock
from app.db.vector.vector_store_service import VectorStoreService

@pytest.fixture
def mock_vector_store():
    return MagicMock()

@pytest.fixture
def vector_store_service(mock_vector_store):
    return VectorStoreService(vector_store=mock_vector_store)

def test_store_chunks_success(vector_store_service, mock_vector_store):
    document_id = 1
    chunks = [
        {"text": "Chunk 1", "metadata": {"section": "intro"}},
        {"text": "Chunk 2", "metadata": {"section": "body"}},
    ]
    embeddings = [[0.1, 0.2], [0.2, 0.3]]

    vector_store_service.store_chunks(document_id, chunks, embeddings)

    mock_vector_store.store_chunks.assert_called_once_with(document_id, chunks, embeddings)

def test_store_chunks_raises_value_error_on_mismatch(vector_store_service):
    chunks = [{"text": "Chunk 1"}]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Mismatch

    with pytest.raises(ValueError, match="Number of chunks and embeddings must be the same."):
        vector_store_service.store_chunks(1, chunks, embeddings)

def test_query_delegates_to_vector_store(vector_store_service, mock_vector_store):
    query_embedding = [0.1, 0.2, 0.3]
    mock_result = [{"text": "Chunk A", "similarity": 0.95}]
    mock_vector_store.query.return_value = mock_result

    results = vector_store_service.query(
        query_embedding=query_embedding,
        top_k=10,
        knowledge_base_id="kb1",
        filters={"section": "intro"},
        min_score=0.2,
        query_text="example"
    )

    mock_vector_store.query.assert_called_once_with(
        query_embedding=query_embedding,
        top_k=10,
        knowledge_base_id="kb1",
        filters={"section": "intro"},
        min_score=0.2,
        query_text="example"
    )

    assert results == mock_result
