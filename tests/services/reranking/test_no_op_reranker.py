import pytest
from app.services.reranking.no_op_reranker import NoOpReranker

def test_no_op_reranker_returns_documents_as_is():
    reranker = NoOpReranker()
    query = "What is RAG?"
    documents = [
        {"text": "Document A"},
        {"text": "Document B"},
        {"text": "Document C"}
    ]

    result = reranker.rerank(query, documents)

    assert result == documents  # Same content
    assert all("text" in doc for doc in result)
