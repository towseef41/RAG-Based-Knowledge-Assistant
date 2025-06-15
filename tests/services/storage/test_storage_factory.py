import pytest
from app.services.storage.storage_factory import get_storage_backend
from app.services.storage.sqlite_storage import SQLiteStorage


def test_returns_sqlite_storage_by_default():
    storage = get_storage_backend()
    assert isinstance(storage, SQLiteStorage)


def test_returns_sqlite_storage_explicit():
    storage = get_storage_backend("sqlite")
    assert isinstance(storage, SQLiteStorage)


def test_raises_value_error_for_unsupported_backend():
    with pytest.raises(ValueError) as exc_info:
        get_storage_backend("unsupported-backend")
    assert "Unsupported storage backend" in str(exc_info.value)