import os
import fitz
import pytest
from pathlib import Path
from app.services.ingestion.pdf_ingestor import PDFIngestor


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Creates a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF World")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_pdf_ingestor_success(sample_pdf):
    """Test that PDFIngestor correctly extracts text and metadata from a valid PDF."""
    ingestor = PDFIngestor(file_path=str(sample_pdf))
    documents = ingestor.load_documents()

    assert isinstance(documents, list)
    assert len(documents) == 1

    filename, content, metadata = documents[0]

    assert filename == "test_doc.pdf"
    assert "Hello PDF World" in content
    assert isinstance(metadata, dict)


def test_pdf_ingestor_empty_file(tmp_path):
    """Test that PDFIngestor returns an empty list for an empty PDF file."""
    empty_pdf = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page
    doc.save(empty_pdf)
    doc.close()

    ingestor = PDFIngestor(file_path=str(empty_pdf))
    documents = ingestor.load_documents()

    assert isinstance(documents, list)
    assert len(documents) == 0  # Text is empty but still has a page


def test_pdf_ingestor_invalid_file(tmp_path):
    """Test that PDFIngestor handles a non-PDF file gracefully."""
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("This is not a valid PDF file")

    ingestor = PDFIngestor(file_path=str(bad_file))
    documents = ingestor.load_documents()

    assert documents == []  # Should gracefully return an empty list
