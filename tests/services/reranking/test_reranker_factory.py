import pytest
from app.services.reranking.reranker_factory import get_reranker
from app.services.reranking.bge_raranker import BgeReranker
from app.services.reranking.no_op_reranker import NoOpReranker


def test_get_reranker_bge():
    reranker = get_reranker("bge")
    assert isinstance(reranker, BgeReranker)


def test_get_reranker_none():
    reranker = get_reranker("none")
    assert isinstance(reranker, NoOpReranker)


def test_get_reranker_invalid_strategy():
    with pytest.raises(ValueError) as exc_info:
        get_reranker("invalid")
    assert "Unsupported reranker strategy" in str(exc_info.value)
