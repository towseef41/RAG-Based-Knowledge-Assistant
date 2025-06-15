import pytest
from app.api.dependencies import (
    get_db,
    get_chunking_service,
    get_embedding_service,
    get_vector_store_service,
    get_storage_service,
    get_generator_service,
    get_reranking_service,
    get_rag_service
)
from app.services.embedding.embedding_service import EmbeddingService
from app.services.generator.generator_service import GeneratorService
from app.db.vector.vector_store_service import VectorStoreService
from app.services.reranking.reranking_service import RerankingService
from app.services.storage.storage_service import StorageService
from app.services.chunking.chunking_service import ChunkingService
from app.services.chunking.word_chunker import WordChunker
from app.services.chunking.sentence_chunker import SentenceChunker
from app.services.rag_service import RagService


class DummySession:
    def close(self):
        pass

@pytest.fixture
def dummy_db():
    return DummySession()

def test_get_db():
    gen = get_db()
    db = next(gen)
    assert db is not None
    gen.close()

def test_get_chunking_service_with_word_strategy():
    service = get_chunking_service(strategy="word", chunk_size=100, overlap=20)
    assert isinstance(service, ChunkingService)
    assert isinstance(service.chunker, WordChunker)
    assert service.chunker.chunk_size == 100
    assert service.chunker.overlap == 20

def test_get_chunking_service_with_sentence_strategy():
    service = get_chunking_service(strategy="sentence", chunk_size=100, overlap=20)
    assert isinstance(service, ChunkingService)
    assert isinstance(service.chunker, SentenceChunker)
    assert service.chunker.chunk_size == 100
    assert service.chunker.overlap == 20

def test_get_embedding_service():
    service = get_embedding_service("local")
    assert isinstance(service, EmbeddingService)

def test_get_vector_store_service_inmemory(dummy_db):
    service = get_vector_store_service(strategy="inmemory", db=dummy_db)
    assert isinstance(service, VectorStoreService)

def test_get_vector_store_service_db(dummy_db):
    service = get_vector_store_service(strategy="db", db=dummy_db)
    assert isinstance(service, VectorStoreService)

def test_get_vector_store_service_hybrid_requires_memory_strategy(dummy_db):
    with pytest.raises(ValueError):
        get_vector_store_service(strategy="hybrid", db=dummy_db)

def test_get_vector_store_service_hybrid_with_memory_strategy(dummy_db):
    service = get_vector_store_service(strategy="hybrid", memory_strategy="inmemory", db=dummy_db)
    assert isinstance(service, VectorStoreService)

def test_get_storage_service():
    service = get_storage_service("sqlite")
    assert isinstance(service, StorageService)

def test_get_generator_service(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service = get_generator_service("openai")
    assert isinstance(service, GeneratorService)

def test_get_reranking_service():
    service = get_reranking_service("bge")
    assert isinstance(service, RerankingService)

def test_get_rag_service():
    rag_service = get_rag_service()
    assert isinstance(rag_service, RagService)
