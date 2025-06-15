from unittest.mock import MagicMock
from app.services.reranking.reranking_service import RerankingService


def test_rerank_documents_calls_underlying_reranker():
    # Arrange
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [{"text": "doc3"}, {"text": "doc1"}]

    service = RerankingService(reranker=mock_reranker)

    query = "test query"
    documents = [{"text": "doc1"}, {"text": "doc2"}]

    # Act
    result = service.rerank_documents(query, documents)

    # Assert
    mock_reranker.rerank.assert_called_once_with(query, documents)
    assert result == [{"text": "doc3"}, {"text": "doc1"}]
