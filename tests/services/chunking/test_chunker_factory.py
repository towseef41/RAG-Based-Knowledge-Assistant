import pytest
from app.services.chunking.chunker_factory import get_chunker
from app.services.chunking.word_chunker import WordChunker
from app.services.chunking.sentence_chunker import SentenceChunker


def test_get_chunker_word_strategy_returns_word_chunker():
    chunker = get_chunker("word", chunk_size=50, overlap=5)
    assert isinstance(chunker, WordChunker)
    assert chunker.chunk_size == 50
    assert chunker.overlap == 5


def test_get_chunker_sentence_strategy_returns_sentence_chunker():
    chunker = get_chunker("sentence")
    assert isinstance(chunker, SentenceChunker)


def test_get_chunker_word_strategy_case_insensitive():
    chunker = get_chunker("WoRd", chunk_size=20, overlap=10)
    assert isinstance(chunker, WordChunker)
    assert chunker.chunk_size == 20


def test_get_chunker_raises_value_error_on_invalid_strategy():
    with pytest.raises(ValueError) as exc:
        get_chunker("invalid-strategy")

    assert "Unsupported chunking strategy" in str(exc.value)


def test_get_chunker_passes_additional_kwargs():
    chunker = get_chunker("word", chunk_size=30, overlap=2)
    assert isinstance(chunker, WordChunker)
    assert chunker.chunk_size == 30
    assert chunker.overlap == 2
