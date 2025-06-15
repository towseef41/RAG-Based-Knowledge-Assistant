import os
import pytest
from pathlib import Path
from app.services.ingestion.txt_ingestor import TXTIngestor


def test_txt_ingestor_single_file(tmp_path):
    """Test ingestion of a single valid .txt file."""
    test_file = tmp_path / "example.txt"
    test_file.write_text("This is a test file.")

    ingestor = TXTIngestor(folder_path=str(tmp_path))
    documents = ingestor.load_documents()

    assert len(documents) == 1
    assert documents[0][0] == "example.txt"
    assert documents[0][1] == "This is a test file."


def test_txt_ingestor_ignores_non_txt(tmp_path):
    """Ensure non-.txt files are ignored."""
    (tmp_path / "data.csv").write_text("not relevant")
    (tmp_path / "image.jpg").write_text("irrelevant binary data")
    (tmp_path / "notes.txt").write_text("Only this matters")

    ingestor = TXTIngestor(folder_path=str(tmp_path))
    documents = ingestor.load_documents()

    assert len(documents) == 1
    assert documents[0][0] == "notes.txt"
    assert "Only this matters" in documents[0][1]


def test_txt_ingestor_skips_empty_files(tmp_path):
    """Ensure empty text files are skipped."""
    (tmp_path / "empty.txt").write_text("   \n")
    (tmp_path / "valid.txt").write_text("Hello World")

    ingestor = TXTIngestor(folder_path=str(tmp_path))
    documents = ingestor.load_documents()

    assert len(documents) == 1
    assert documents[0][0] == "valid.txt"
    assert documents[0][1] == "Hello World"


def test_txt_ingestor_handles_read_error(tmp_path, monkeypatch):
    """Ensure ingestor handles file read errors gracefully."""

    bad_file = tmp_path / "corrupt.txt"
    bad_file.write_text("can't read me")

    # Monkeypatch open to raise IOError only when trying to open "corrupt.txt"
    original_open = open

    def mock_open(path, *args, **kwargs):
        if "corrupt.txt" in str(path):
            raise IOError("Simulated read error")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    ingestor = TXTIngestor(folder_path=str(tmp_path))
    documents = ingestor.load_documents()

    assert documents == []  # Should skip unreadable files without crashing
