"""
Ingestion Pipeline

This module defines the core ingestion pipeline that:
- Detects supported file types in a folder
- Uses the appropriate ingestor to load document contents
- Chunks text, generates embeddings, and stores both
"""

import os
import traceback
import logging

from app.services.chunking.chunking_service import ChunkingService
from app.services.embedding.embedding_service import EmbeddingService
from app.services.storage.storage_service import StorageService
from app.services.ingestion.ingestor_factory import get_ingestor_for_extension

# Module-level logger
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the ingestion process from raw files to persisted embeddings.

    This pipeline:
    - Detects supported file types in the specified folder
    - Uses the appropriate ingestor to load document contents
    - Chunks the text using a strategy (e.g., word, sentence)
    - Generates embeddings for each chunk
    - Stores documents, chunks, and embeddings via the storage service
    """

    def __init__(
        self,
        folder_path: str,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        storage_service: StorageService
    ):
        """
        Initializes the pipeline with all required services.

        Parameters
        ----------
        folder_path : str
            Path to the directory containing input files.
        chunking_service : ChunkingService
            The service responsible for chunking the input text.
        embedding_service : EmbeddingService
            The service used to generate vector embeddings for text chunks.
        storage_service : StorageService
            The service used to store documents and chunks with embeddings.
        """
        self.folder_path = folder_path
        self.chunker = chunking_service
        self.embedder = embedding_service
        self.storage = storage_service

        logger.info(f"IngestionPipeline initialized for folder: {self.folder_path}")

    def run(self) -> None:
        logger.info("Starting ingestion pipeline...")

        for file_name in os.listdir(self.folder_path):
            ext = os.path.splitext(file_name)[-1].lower()
            IngestorClass = get_ingestor_for_extension(ext)

            if not IngestorClass:
                logger.warning(f"Skipping unsupported file type: {file_name}")
                continue

            file_path = os.path.join(self.folder_path, file_name)
            logger.info(f"Processing file: {file_path}")

            try:
                ingestor = IngestorClass(file_path=file_path)
                documents = ingestor.load_documents()
                logger.debug(f"Loaded {len(documents)} document(s) from {file_name}")

                for doc_name, content, metadata in documents:
                    if self.storage.document_exists(doc_name):
                        logger.info(f"⚠️ Document '{doc_name}' already exists. Skipping.")
                        continue

                    logger.debug(f"Chunking document: {doc_name}")
                    chunks = self.chunker.chunk_text(content, metadata)
                    logger.debug(f"Generated {len(chunks)} chunks")

                    logger.debug("Generating embeddings...")
                    embeddings = [self.embedder.get_embedding(c['text']) for c in chunks]

                    doc = self.storage.store_document(
                        name=doc_name,
                        document_metadata=metadata,
                        path=file_path
                    )
                    self.storage.store_chunks(doc.id, chunks, embeddings)

                    logger.info(f"✅ Ingested {doc_name} ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"❌ Failed to ingest {file_name}: {e}")
                traceback.print_exc()
