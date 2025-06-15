import pytest
from app.services.generator.base_generator import BaseGenerator


class MockGenerator(BaseGenerator):
    def generate_answer(self, query: str, context=None, chat_history=None) -> str:
        return f"Mock response to '{query}' with context '{context}' and history '{chat_history}'"


def test_generate_answer_with_all_arguments():
    generator = MockGenerator()
    query = "What is RAG?"
    context = "RAG stands for Retrieval-Augmented Generation."
    chat_history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

    result = generator.generate_answer(query, context=context, chat_history=chat_history)

    assert isinstance(result, str)
    assert "RAG" in result
    assert "context" in result or "Retrieval-Augmented Generation" in result


def test_generate_answer_with_minimum_arguments():
    generator = MockGenerator()
    result = generator.generate_answer("What is Python?")

    assert isinstance(result, str)
    assert "Python" in result


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        BaseGenerator()
