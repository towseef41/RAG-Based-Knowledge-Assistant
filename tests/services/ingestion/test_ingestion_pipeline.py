import os
import pytest
from unittest.mock import patch, MagicMock
import logging

from app.services.ingestion.ingestion_pipeline import IngestionPipeline

@pytest.fixture
def mock_services():
    return {
        "chunking_service": MagicMock(),
        "embedding_service": MagicMock(),
        "storage_service": MagicMock()
    }


@pytest.fixture
def pipeline(tmp_path, mock_services):
    # Create a temp directory to simulate folder with files
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello World")

    return IngestionPipeline(
        folder_path=str(tmp_path),
        chunking_service=mock_services["chunking_service"],
        embedding_service=mock_services["embedding_service"],
        storage_service=mock_services["storage_service"]
    )


@patch("app.services.ingestion.ingestion_pipeline.get_ingestor_for_extension")
def test_ingestion_success(mock_get_ingestor, tmp_path, mock_services):
    # 📝 Create dummy .txt file
    test_file = tmp_path / "example.txt"
    test_file.write_text("Hello world")

    # 🧪 Mock the ingestor
    mock_ingestor_class = MagicMock()
    mock_ingestor_instance = MagicMock()
    mock_ingestor_instance.load_documents.return_value = [
        ("example.txt", "Some content", {"source": "unit"})
    ]
    mock_ingestor_class.return_value = mock_ingestor_instance
    mock_get_ingestor.return_value = mock_ingestor_class

    # 🧪 Setup mock services
    mock_services["chunking_service"].chunk_text.return_value = [{"text": "chunk1"}]
    mock_services["embedding_service"].get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_doc = MagicMock()
    mock_doc.id = "doc-1"
    mock_services["storage_service"].store_document.return_value = mock_doc

    # 🔁 Run the pipeline
    pipeline = IngestionPipeline(
        folder_path=str(tmp_path),
        chunking_service=mock_services["chunking_service"],
        embedding_service=mock_services["embedding_service"],
        storage_service=mock_services["storage_service"],
    )
    pipeline.run()

    # ✅ Assertions
    mock_get_ingestor.assert_called_once_with(".txt")
    mock_ingestor_class.assert_called_once_with(file_path=str(test_file))
    mock_ingestor_instance.load_documents.assert_called_once()



@patch("app.services.ingestion.ingestor_factory.get_ingestor_for_extension")
def test_unsupported_file_type(mock_get_ingestor, tmp_path, mock_services):
    # Arrange: create a dummy file with unsupported extension
    unsupported_file = tmp_path / "notes.xyz"
    unsupported_file.write_text("Some unsupported content")

    mock_get_ingestor.return_value = None

    pipeline = IngestionPipeline(
        folder_path=str(tmp_path),
        chunking_service=mock_services["chunking_service"],
        embedding_service=mock_services["embedding_service"],
        storage_service=mock_services["storage_service"]
    )

    with patch.object(logging.getLogger("app.services.ingestion.ingestion_pipeline"), 'warning') as mock_warning:
        pipeline.run()

        # Assert that the warning logger was called with the correct message
        assert any("Skipping unsupported file type" in args[0] for args, _ in mock_warning.call_args_list)

@patch("app.services.ingestion.ingestion_pipeline.get_ingestor_for_extension")
def test_ingestion_skips_existing_document(mock_get_ingestor, tmp_path, mock_services):
    test_file = tmp_path / "example.txt"
    test_file.write_text("Hello world")

    mock_ingestor_class = MagicMock()
    mock_ingestor_instance = MagicMock()
    mock_ingestor_instance.load_documents.return_value = [
        ("example.txt", "Some content", {"source": "unit"})
    ]
    mock_ingestor_class.return_value = mock_ingestor_instance
    mock_get_ingestor.return_value = mock_ingestor_class

    mock_services["chunking_service"].chunk_text.return_value = [{"text": "chunk1"}]
    mock_services["embedding_service"].get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_doc = MagicMock()
    mock_doc.id = "doc-1"
    mock_services["storage_service"].store_document.return_value = mock_doc

    mock_services["storage_service"].document_exists.return_value = True

    pipeline = IngestionPipeline(
        folder_path=str(tmp_path),
        chunking_service=mock_services["chunking_service"],
        embedding_service=mock_services["embedding_service"],
        storage_service=mock_services["storage_service"],
    )
    pipeline.run()

    mock_get_ingestor.assert_called_once_with(".txt")
    mock_ingestor_class.assert_called_once_with(file_path=str(test_file))
    mock_ingestor_instance.load_documents.assert_called_once()

    mock_services["storage_service"].document_exists.assert_called_once_with("example.txt")
    mock_services["chunking_service"].chunk_text.assert_not_called()
    mock_services["embedding_service"].get_embedding.assert_not_called()
    mock_services["storage_service"].store_document.assert_not_called()
    mock_services["storage_service"].store_chunks.assert_not_called()


@patch("app.services.ingestion.ingestor_factory.get_ingestor_for_extension")
def test_ingestion_failure_handling(mock_get_ingestor, pipeline):
    mock_ingestor = MagicMock()
    mock_ingestor_instance = mock_ingestor.return_value
    mock_ingestor_instance.load_documents.side_effect = Exception("Load failed")

    mock_get_ingestor.return_value = mock_ingestor

    logger = logging.getLogger("app.services.ingestion.ingestion_pipeline")

    with patch.object(logger, "error") as mock_error:
        pipeline.run()
        assert any("Failed to ingest" in call.args[0] for call in mock_error.call_args_list)
