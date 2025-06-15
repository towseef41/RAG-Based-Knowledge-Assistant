import pytest
from app.services.chunking.chunking_service import ChunkingService
from app.services.chunking.base_chunker import BaseChunker

# Dummy chunker for testing
class DummyChunker(BaseChunker):
    def chunk(self, text, doc_metadata=None):
        # Just returns one dummy chunk for simplicity
        return [{"text": text, "metadata": doc_metadata or {}}]

@pytest.fixture
def dummy_chunker():
    return DummyChunker()

@pytest.fixture
def chunking_service(dummy_chunker):
    return ChunkingService(chunker=dummy_chunker)

def test_chunking_service_returns_expected_output(chunking_service):
    text = "This is a test."
    metadata = {"source": "test-doc"}
    
    chunks = chunking_service.chunk_text(text, doc_metadata=metadata)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["metadata"] == metadata

def test_chunking_service_handles_empty_metadata(chunking_service):
    text = "Another test"
    
    chunks = chunking_service.chunk_text(text)

    assert chunks[0]["metadata"] == {}

def test_chunking_service_handles_empty_text(chunking_service):
    chunks = chunking_service.chunk_text("")
    assert chunks[0]["text"] == ""

def test_chunking_service_with_multiple_calls(chunking_service):
    t1 = chunking_service.chunk_text("Hello", doc_metadata={"doc": 1})
    t2 = chunking_service.chunk_text("World", doc_metadata={"doc": 2})
    
    assert t1[0]["metadata"]["doc"] == 1
    assert t2[0]["metadata"]["doc"] == 2
