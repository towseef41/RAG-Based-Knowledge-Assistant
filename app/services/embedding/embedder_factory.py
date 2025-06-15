"""
Factory module for creating embedding service instances based on the backend type.
"""

import logging
from app.services.embedding.openai_embedder import OpenAIEmbedder
from app.services.embedding.local_embedder import LocalEmbedder

# Configure module-level logger
logger = logging.getLogger(__name__)


def get_embedder(backend: str = "openai"):
    """
    Returns an embedder instance based on the specified backend.

    This factory function allows switching between different embedding providers
    (e.g., OpenAI, local models) without changing the code that uses the embedder.

    Args:
        backend (str): The name of the embedding backend to use. Supported values:
                       - "openai" : Uses OpenAI's API-based embedding service.
                       - "local"  : Uses a locally hosted embedding model via sentence-transformers.

    Returns:
        An instance of a class implementing `BaseEmbedder`.

    Raises:
        ValueError: If the specified backend is not supported.

    Example:
        embedder = get_embedder("openai")
        vector = embedder.get_embedding("Your input text here")
    """
    backend = backend.lower()
    logger.info(f"Initializing embedder with backend: '{backend}'")

    if backend == "openai":
        logger.debug("Returning OpenAIEmbedder instance")
        return OpenAIEmbedder()
    elif backend == "local":
        logger.debug("Returning LocalEmbedder instance")
        return LocalEmbedder()
    else:
        logger.error(f"Unsupported embedding backend requested: {backend}")
        raise ValueError(f"Unsupported embedding backend: {backend}")
