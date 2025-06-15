import pytest
from app.services.chunking.base_chunker import BaseChunker

# ---------------------------------------------
# 1. Ensure abstract class can't be instantiated
# ---------------------------------------------
def test_base_chunker_is_abstract():
    with pytest.raises(TypeError):
        BaseChunker()  # Cannot instantiate abstract class


# ---------------------------------------------
# 2. Dummy subclass for testing
# ---------------------------------------------
class DummyChunker(BaseChunker):
    def chunk(self, text: str, doc_metadata=None):
        # Returns each word as a chunk with optional metadata
        return [
            {"text": word, "metadata": doc_metadata or {}}
            for word in text.split()
        ]


@pytest.fixture
def dummy_chunker():
    return DummyChunker()


# ---------------------------------------------
# 3. Unit tests for dummy implementation
# ---------------------------------------------
def test_chunk_returns_chunks_with_text(dummy_chunker):
    text = "This is a test"
    chunks = dummy_chunker.chunk(text)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert [c["text"] for c in chunks] == text.split()

def test_chunk_includes_metadata(dummy_chunker):
    text = "metadata check"
    metadata = {"source": "test-doc", "page": 2}
    chunks = dummy_chunker.chunk(text, doc_metadata=metadata)

    for chunk in chunks:
        assert chunk["metadata"] == metadata

def test_chunk_defaults_to_empty_metadata(dummy_chunker):
    chunks = dummy_chunker.chunk("hello world")

    assert all("metadata" in c for c in chunks)
    assert all(c["metadata"] == {} for c in chunks)
