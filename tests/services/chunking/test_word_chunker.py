import pytest
from app.services.chunking.word_chunker import WordChunker

@pytest.fixture
def default_chunker():
    return WordChunker()

def test_chunk_basic_split(default_chunker):
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = default_chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1
    assert chunks[0]["metadata"]["start_word"] == 0
    assert chunks[1]["metadata"]["start_word"] == 120  # 150 - 30 overlap
    assert chunks[0]["metadata"]["word_count"] == 150
    assert chunks[1]["metadata"]["word_count"] == 150
    assert chunks[0]["metadata"]["char_count"] > 0

def test_chunk_with_custom_size_and_overlap():
    chunker = WordChunker(chunk_size=10, overlap=2)
    text = " ".join([f"w{i}" for i in range(25)])
    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0]["metadata"]["start_word"] == 0
    assert chunks[1]["metadata"]["start_word"] == 8
    assert chunks[2]["metadata"]["start_word"] == 16
    assert chunks[-1]["metadata"]["end_word"] == 24

def test_chunk_empty_text(default_chunker):
    chunks = default_chunker.chunk("")
    assert chunks == []

def test_chunk_shorter_than_chunk_size():
    chunker = WordChunker(chunk_size=10, overlap=2)
    text = "word1 word2 word3"
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["word_count"] == 3
    assert chunks[0]["text"] == text

def test_chunk_exactly_chunk_size():
    chunker = WordChunker(chunk_size=5, overlap=0)
    text = "one two three four five"
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["word_count"] == 5

def test_chunk_metadata_propagation():
    chunker = WordChunker(chunk_size=5, overlap=0)
    text = "word " * 10
    meta = {"doc_id": "xyz", "source": "unit_test"}
    chunks = chunker.chunk(text.strip(), doc_metadata=meta)

    for chunk in chunks:
        assert chunk["metadata"]["doc_id"] == "xyz"
        assert chunk["metadata"]["source"] == "unit_test"

def test_chunk_overlap_does_not_exceed_chunk_size():
    with pytest.raises(ValueError):
        WordChunker(chunk_size=5, overlap=10)
