import pytest
from app.services.prompt.prompt_manager import PromptManager


@pytest.fixture
def prompt_manager():
    return PromptManager()


def test_get_existing_prompt(prompt_manager):
    prompt = prompt_manager.get("default_system")
    assert prompt == "You are a helpful assistant."


def test_get_nonexistent_prompt_raises_error(prompt_manager):
    with pytest.raises(ValueError) as excinfo:
        prompt_manager.get("nonexistent")
    assert "Prompt 'nonexistent' not found." in str(excinfo.value)


def test_render_prompt_with_valid_context(prompt_manager):
    context = "The capital of France is Paris."
    rendered = prompt_manager.render("rag", context=context)
    assert "The capital of France is Paris." in rendered
    assert "## Context:" in rendered
    assert "Answer:" in rendered


def test_render_prompt_missing_placeholder_raises_error(prompt_manager):
    with pytest.raises(ValueError) as excinfo:
        prompt_manager.render("rag")  # missing 'context'
    assert "Missing placeholder" in str(excinfo.value)


def test_render_nonexistent_prompt_raises_error(prompt_manager):
    with pytest.raises(ValueError) as excinfo:
        prompt_manager.render("nonexistent", context="test")
    assert "Prompt 'nonexistent' not found." in str(excinfo.value)
