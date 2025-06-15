import pytest
from app.services.chunking.sentence_chunker import SentenceChunker

@pytest.fixture
def default_chunker():
    return SentenceChunker()

def test_chunk_basic(default_chunker):
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five. This is sentence six. This is sentence seven."
    chunks = default_chunker.chunk(text)

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["start_sentence"] == 0
    assert chunks[0]["metadata"]["end_sentence"] == 4
    assert chunks[1]["metadata"]["chunk_index"] == 1
    assert chunks[1]["metadata"]["start_sentence"] == 4
    assert chunks[1]["metadata"]["end_sentence"] == 6
    assert chunks[0]["metadata"]["num_sentences"] == 5
    assert chunks[1]["metadata"]["num_sentences"] == 3  # Remaining sentences

def test_chunk_empty_string(default_chunker):
    assert default_chunker.chunk("") == []

def test_chunk_fewer_sentences_than_chunk_size():
    chunker = SentenceChunker(chunk_size=5, overlap=1)
    text = "Only one sentence here."
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["num_sentences"] == 1

def test_chunk_exact_fit():
    chunker = SentenceChunker(chunk_size=3, overlap=0)
    text = "S1. S2. S3. S4. S5. S6."
    chunks = chunker.chunk(text)

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["num_sentences"] == 3
    assert chunks[1]["metadata"]["num_sentences"] == 3

def test_chunk_overlap_behavior():
    chunker = SentenceChunker(chunk_size=3, overlap=1)
    text = "One. Two. Three. Four. Five. Six. Seven."
    chunks = chunker.chunk(text)

    assert len(chunks) == 4
    assert chunks[0]["text"].startswith("One")
    assert chunks[1]["text"].startswith("Three")
    assert chunks[2]["text"].startswith("Five")

def test_chunk_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError):
        SentenceChunker(chunk_size=3, overlap=5)

def test_metadata_accuracy():
    chunker = SentenceChunker(chunk_size=4, overlap=2)
    text = "S1. S2. S3. S4. S5. S6. S7. S8. S9."
    chunks = chunker.chunk(text)

    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk_index"] == i
        assert "start_sentence" in chunk["metadata"]
        assert "end_sentence" in chunk["metadata"]
        assert "num_sentences" in chunk["metadata"]
        assert chunk["text"].endswith(".")

def test_custom_sentence_endings():
    chunker = SentenceChunker()
    text = "What is this? It's a test! Yes. Another one?"
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["num_sentences"] == 4
