import pytest
from unittest.mock import MagicMock
from app.services.generator.generator_service import GeneratorService
from app.services.generator.base_generator import BaseGenerator

@pytest.fixture
def mock_generator():
    """Fixture to create a mock generator."""
    generator = MagicMock(spec=BaseGenerator)
    generator.generate_answer.return_value = "mocked response"
    return generator

def test_generate_answer_basic(mock_generator):
    service = GeneratorService(generator=mock_generator)
    query = "What is RAG?"
    
    result = service.generate_answer(query)
    
    assert result == "mocked response"
    mock_generator.generate_answer.assert_called_once_with(
        query=query, context=None, chat_history=None
    )

def test_generate_answer_with_context(mock_generator):
    service = GeneratorService(generator=mock_generator)
    query = "Explain embeddings"
    context = "Embeddings represent text as vectors."

    result = service.generate_answer(query, context=context)

    assert result == "mocked response"
    mock_generator.generate_answer.assert_called_once_with(
        query=query, context=context, chat_history=None
    )

def test_generate_answer_with_chat_history(mock_generator):
    service = GeneratorService(generator=mock_generator)
    query = "What is vector search?"
    chat_history = [
        {"role": "user", "content": "Tell me about search algorithms."},
        {"role": "assistant", "content": "There are many, like keyword and vector search."}
    ]

    result = service.generate_answer(query, chat_history=chat_history)

    assert result == "mocked response"
    mock_generator.generate_answer.assert_called_once_with(
        query=query, context=None, chat_history=chat_history
    )

def test_generate_answer_with_all_inputs(mock_generator):
    service = GeneratorService(generator=mock_generator)
    query = "How does RAG work?"
    context = "RAG stands for Retrieval-Augmented Generation."
    chat_history = [{"role": "user", "content": "What are its components?"}]

    result = service.generate_answer(query, context=context, chat_history=chat_history)

    assert result == "mocked response"
    mock_generator.generate_answer.assert_called_once_with(
        query=query, context=context, chat_history=chat_history
    )

def test_generate_answer_with_empty_query(mock_generator):
    service = GeneratorService(generator=mock_generator)
    query = ""

    result = service.generate_answer(query)

    assert result == "mocked response"
    mock_generator.generate_answer.assert_called_once_with(
        query="", context=None, chat_history=None
    )
