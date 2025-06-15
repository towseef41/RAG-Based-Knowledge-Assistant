import pytest
from unittest.mock import patch
from app.services.generator.generator_factory import get_generator
from app.services.generator.openai_generator import OpenAIGenerator
from app.services.generator.base_generator import BaseGenerator


def test_get_openai_generator_returns_correct_instance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = get_generator(provider="openai")
    assert isinstance(generator, OpenAIGenerator)
    assert isinstance(generator, BaseGenerator)


def test_get_generator_invalid_provider_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        get_generator(provider="unknown")
    assert "Unsupported generator backend" in str(exc_info.value)
