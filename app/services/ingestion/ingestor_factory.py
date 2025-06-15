"""
Ingestor Factory Module

This module provides a registry and factory method to dynamically select the appropriate
document ingestor based on file extension. It enables a plug-and-play architecture
to support multiple document types such as PDF, TXT, DOCX, etc.
"""

import logging
from typing import Optional, Type
from app.services.ingestion.base_ingestor import BaseIngestor
from app.services.ingestion.pdf_ingestor import PDFIngestor
from app.services.ingestion.txt_ingestor import TXTIngestor

logger = logging.getLogger(__name__)

# Map of file extensions to their corresponding Ingestor classes
INGESTOR_MAP: dict[str, Type[BaseIngestor]] = {
    ".pdf": PDFIngestor,
    ".txt": TXTIngestor,
}

def get_ingestor_for_extension(extension: str) -> Optional[Type[BaseIngestor]]:
    """
    Retrieves the appropriate ingestor class based on the file extension.

    Args:
        extension (str): The file extension (e.g., ".pdf", ".txt").

    Returns:
        Optional[Type[BaseIngestor]]: The ingestor class corresponding to the extension,
        or None if the extension is not supported.

    Example:
        >>> get_ingestor_for_extension(".pdf")
        <class 'PDFIngestor'>
    """
    ext = extension.lower()
    ingestor_class = INGESTOR_MAP.get(ext)
    if ingestor_class:
        logger.debug(f"Found ingestor for extension '{ext}': {ingestor_class.__name__}")
    else:
        logger.warning(f"No ingestor found for extension '{ext}'")
    return ingestor_class
