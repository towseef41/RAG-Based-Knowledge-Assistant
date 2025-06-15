import pytest
from app.services.ingestion.base_ingestor import BaseIngestor

def test_cannot_instantiate_base_ingestor():
    with pytest.raises(TypeError):
        BaseIngestor()

def test_requires_load_documents_implementation():
    """
    Ensure that any subclass must implement `load_documents`.
    """
    class IncompleteIngestor(BaseIngestor):
        pass

    with pytest.raises(TypeError):
        IncompleteIngestor()

def test_mock_ingestor_returns_expected_data():
    """
    Use a mock subclass to test the contract.
    """
    class MockIngestor(BaseIngestor):
        def load_documents(self):
            return [("doc1.txt", "This is document 1."), ("doc2.txt", "Another one.")]

    ingestor = MockIngestor()
    result = ingestor.load_documents()

    assert isinstance(result, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
    assert result[0][0] == "doc1.txt"
    assert result[0][1] == "This is document 1."
