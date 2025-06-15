import pytest
from typing import List, Dict
from app.services.reranking.base_reranker import BaseReranker


def test_base_reranker_is_abstract():
    """
    Ensure BaseReranker cannot be instantiated directly.
    """
    with pytest.raises(TypeError) as exc_info:
        BaseReranker()  # abstract method not implemented

    assert "Can't instantiate abstract class" in str(exc_info.value)


def test_subclass_must_implement_rerank():
    """
    Ensure subclass without rerank() raises error.
    """

    class BadReranker(BaseReranker):
        pass

    with pytest.raises(TypeError) as exc_info:
        BadReranker()

    assert "Can't instantiate abstract class" in str(exc_info.value)


def test_dummy_reranker_implementation():
    """
    Test a concrete subclass of BaseReranker.
    """

    class DummyReranker(BaseReranker):
        def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
            # Just reverse input as a dummy ranking logic
            return list(reversed(documents))

    reranker = DummyReranker()

    docs = [
        {"text": "doc1"},
        {"text": "doc2"},
        {"text": "doc3"},
    ]

    result = reranker.rerank("any query", docs)

    assert result == [
        {"text": "doc3"},
        {"text": "doc2"},
        {"text": "doc1"},
    ]
