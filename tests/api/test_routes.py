import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

# Import the router and dependency functions
from app.api.routes import router
from app.api.dependencies import (
    get_rag_service,
    get_embedding_service,
    get_vector_store_service,
)

# ----- Step 1: Dummy service implementations for testing -----

class DummyRagService:
    def chat(self, query, conversation_id=None, knowledge_base_id=None, top_k=5, min_score=0.0):
        return {
            "answer": f"Mocked response for: {query}",
            "conversation_id": conversation_id or "dummy-convo-id",
            "context_chunks": [
                {
                    "chunk_id": 1,
                    "text": "Example chunk",
                    "chunk_metadata": {"source": "mock"},
                    "similarity": 0.9,
                }
            ],
        }

class DummyEmbeddingService:
    def get_embedding(self, query):
        return [0.1] * 768  # Simulated embedding vector

class DummyVectorStoreService:
    def query(self, query_embedding, top_k, filters, min_score):
        return [
            {
                "chunk_id": 123,
                "text": "Relevant chunk text",
                "chunk_metadata": {"source": "test"},
                "similarity": 0.95
            }
        ]

# ----- Step 2: Create FastAPI app and override dependencies -----

app = FastAPI()
app.include_router(router)

# Override dependencies
app.dependency_overrides[get_rag_service] = lambda: DummyRagService()
app.dependency_overrides[get_embedding_service] = lambda: DummyEmbeddingService()
app.dependency_overrides[get_vector_store_service] = lambda: DummyVectorStoreService()

client = TestClient(app)

# ----- Step 3: Tests -----

def test_chat_success():
    payload = {
        "query": "What is a RAG pipeline?",
        "conversation_id": "conv123"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"]["content"].startswith("Mocked response for:")
    assert data["conversation_id"] == "conv123"
    assert isinstance(data["sources"], list)
    assert data["sources"][0]["text"] == "Example chunk"

def test_chat_missing_query():
    response = client.post("/chat", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_search_success():
    payload = {
        "query": "find something relevant",
        "limit": 3
    }
    response = client.post("/search", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["query"] == "find something relevant"
    assert len(data["results"]) > 0
    assert data["results"][0]["text"] == "Relevant chunk text"
    assert data["total_found"] == len(data["results"])

def test_search_with_filters():
    payload = {
        "query": "example query",
        "limit": 2,
        "filters": {"source": "test"}
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    assert "results" in response.json()

def test_search_invalid_body():
    response = client.post("/search", json={"limit": 3})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

