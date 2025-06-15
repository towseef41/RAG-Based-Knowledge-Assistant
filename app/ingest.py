"""
Ingestion Script for RAG Pipeline

This script initializes and runs the IngestionPipeline to process documents
from the `sample_data` folder. It performs the following tasks:

1. Loads supported document files (e.g., .pdf, .txt)
2. Extracts text content using appropriate ingestors
3. Splits the text into chunks
4. Generates vector embeddings for each chunk
5. Stores documents, chunks, and embeddings into the database

Usage:
    python ingest.py

Note:
    - Make sure your document files are placed in the `sample_data/` folder.
    - Supported file types are registered via the `IngestorFactory`.
    - Ensure dependencies are installed:
        pip install -r requirements.txt
"""

import logging
from app.logging_config import setup_logging
from app.services.ingestion.ingestion_pipeline import IngestionPipeline
from app.api.dependencies import get_chunking_service, get_embedding_service, get_storage_service

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    logger.info("Starting ingestion script...")

    try:
        pipeline = IngestionPipeline(
            folder_path="sample_data",
            chunking_service=get_chunking_service(),
            embedding_service=get_embedding_service(),
            storage_service=get_storage_service()
        )
        logger.info("IngestionPipeline initialized successfully.")

        pipeline.run()
        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error running ingestion pipeline: {e}", exc_info=True)
