from .base_ingestor import BaseIngestor
import fitz
import os
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class PDFIngestor(BaseIngestor):
    """
    Ingestor for a single PDF file.
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): Full path to the PDF file.
        """
        self.file_path = file_path

    def load_documents(self) -> List[Tuple[str, str, dict]]:
        """
        Extracts text from the PDF.

        Returns:
            List[Tuple[str, str, dict]]: List containing a single (filename, content, metadata) tuple.
        """
        try:
            with fitz.open(self.file_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
                metadata = doc.metadata
                if text:
                    logger.info(f"Extracted text from PDF file: {self.file_path} ({len(text)} characters)")
                    return [(os.path.basename(self.file_path), text, metadata)]
                else:
                    logger.warning(f"No text found in PDF file: {self.file_path}")
                    return []
        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF file {self.file_path}: {e}", exc_info=True)
            return []
