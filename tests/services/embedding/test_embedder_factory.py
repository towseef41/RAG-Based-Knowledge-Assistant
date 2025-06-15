import pytest
from app.services.embedding.embedder_factory import get_embedder
from app.services.embedding.openai_embedder import OpenAIEmbedder
from app.services.embedding.local_embedder import LocalEmbedder


def test_get_embedder_openai():
    embedder = get_embedder("openai")
    assert isinstance(embedder, OpenAIEmbedder)


def test_get_embedder_local():
    embedder = get_embedder("local")
    assert isinstance(embedder, LocalEmbedder)


def test_get_embedder_case_insensitive():
    embedder = get_embedder("LoCaL")
    assert isinstance(embedder, LocalEmbedder)


def test_get_embedder_invalid_backend():
    with pytest.raises(ValueError) as exc_info:
        get_embedder("invalid_backend")
    assert "Unsupported embedding backend" in str(exc_info.value)
