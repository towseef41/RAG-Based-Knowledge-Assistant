# tests/services/storage/test_base_storage.py
import pytest
from app.services.storage.base_storage import BaseStorage

def test_base_storage_is_abstract():
    with pytest.raises(TypeError):
        BaseStorage()  # Cannot instantiate abstract class with unimplemented methods
