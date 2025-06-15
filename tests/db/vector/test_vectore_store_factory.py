import pytest
from unittest.mock import MagicMock

from app.db.vector.in_memory_vector_store import InMemoryVectorStore
from app.db.vector.db_vector_store import DBVectorStore
from app.db.vector.hybrid_vector_store import HybridVectorStore
from app.db.vector.base_vector_store import BaseVectorStore
from app.db.vector.vector_store_factory import get_vector_store


@pytest.fixture
def mock_db_session():
    return MagicMock()


def test_get_vector_store_inmemory_returns_correct_instance(mock_db_session):
    store = get_vector_store("inmemory", mock_db_session)
    assert isinstance(store, InMemoryVectorStore)
    assert isinstance(store, BaseVectorStore)


def test_get_vector_store_db_returns_correct_instance(mock_db_session):
    store = get_vector_store("db", mock_db_session)
    assert isinstance(store, DBVectorStore)
    assert isinstance(store, BaseVectorStore)


def test_get_vector_store_hybrid_with_inmemory_memory_strategy(mock_db_session):
    store = get_vector_store("hybrid", mock_db_session, memory_strategy="inmemory")
    assert isinstance(store, HybridVectorStore)
    # Check inner vector store is InMemoryVectorStore
    assert isinstance(store.vector_store, InMemoryVectorStore)


def test_get_vector_store_hybrid_with_db_memory_strategy(mock_db_session):
    store = get_vector_store("hybrid", mock_db_session, memory_strategy="db")
    assert isinstance(store, HybridVectorStore)
    # Check inner vector store is DBVectorStore
    assert isinstance(store.vector_store, DBVectorStore)


def test_get_vector_store_hybrid_missing_memory_strategy_raises_value_error(mock_db_session):
    with pytest.raises(ValueError) as exc_info:
        get_vector_store("hybrid", mock_db_session)
    assert "memory_strategy is required" in str(exc_info.value)


def test_get_vector_store_hybrid_with_unsupported_memory_strategy_raises_value_error(mock_db_session):
    with pytest.raises(ValueError) as exc_info:
        get_vector_store("hybrid", mock_db_session, memory_strategy="unsupported")
    assert "Unsupported memory strategy" in str(exc_info.value)


def test_get_vector_store_with_unsupported_strategy_raises_value_error(mock_db_session):
    with pytest.raises(ValueError) as exc_info:
        get_vector_store("unsupported_strategy", mock_db_session)
    assert "Unsupported vector store strategy" in str(exc_info.value)
