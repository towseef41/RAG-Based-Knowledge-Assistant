import logging
from app.services.embedding.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A unified service for generating vector embeddings from text using a pluggable embedder.

    This class delegates the actual embedding generation to an implementation of the `BaseEmbedder`
    interface, allowing for backend flexibility (e.g., OpenAI, local models).

    Attributes
    ----------
    embedder : BaseEmbedder
        An instance of a backend embedder that implements the embedding logic.

    Example
    -------
    >>> from app.dependencies import get_openai_embedder
    >>> service = EmbeddingService(embedder=get_openai_embedder())
    >>> embedding = service.get_embedding("sample text")
    """

    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize the embedding service with a specific embedder implementation.

        Parameters
        ----------
        embedder : BaseEmbedder
            The embedding backend that will handle vector generation.
        """
        self.embedder = embedder
        logger.info(f"EmbeddingService initialized with embedder: {type(embedder).__name__}")

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for the input text.

        Parameters
        ----------
        text : str
            The input string to convert into a vector embedding.

        Returns
        -------
        list[float]
            A list of float values representing the text embedding.
        """
        logger.debug(f"Generating embedding for text of length {len(text)}")
        embedding = self.embedder.get_embedding(text)
        logger.debug(f"Generated embedding of dimension {len(embedding)}")
        return embedding
