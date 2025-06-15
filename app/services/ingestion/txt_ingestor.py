"""
TXT Ingestor

This module defines the TXTIngestor class that extracts text content
from all `.txt` files in a given folder.
"""

import os
import logging
from typing import List, Tuple
from .base_ingestor import BaseIngestor

logger = logging.getLogger(__name__)

class TXTIngestor(BaseIngestor):
    """
    Ingestor for .txt files.
    
    Scans a folder for .txt files and loads their content.
    """

    def __init__(self, folder_path: str = "sample_data"):
        """
        Args:
            folder_path (str): Directory path to scan for text files.
        """
        self.folder_path = folder_path

    def load_documents(self) -> List[Tuple[str, str]]:
        """
        Loads all .txt files in the folder and extracts their content.

        Returns:
            List[Tuple[str, str]]: List of (filename, content) tuples.
        """
        documents = []

        try:
            files = os.listdir(self.folder_path)
        except Exception as e:
            logger.error(f"Failed to list directory {self.folder_path}: {e}", exc_info=True)
            return documents

        for file_name in files:
            if file_name.lower().endswith(".txt"):
                file_path = os.path.join(self.folder_path, file_name)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            documents.append((file_name, content))
                            logger.info(f"Loaded text file: {file_name} ({len(content)} characters)")
                        else:
                            logger.warning(f"Text file is empty: {file_name}")
                except Exception as e:
                    logger.error(f"Error reading {file_name}: {e}", exc_info=True)

        return documents
