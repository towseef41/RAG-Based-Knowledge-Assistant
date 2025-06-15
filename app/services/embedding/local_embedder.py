import logging
from sentence_transformers import SentenceTransformer
from app.services.embedding.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class LocalEmbedder(BaseEmbedder):
    """
    Embedding generator using a local transformer model from sentence-transformers.

    This embedder loads a pre-trained model and performs inference locally
    without requiring external API calls. Suitable for offline use cases or
    when you want more control over latency and cost.

    Attributes:
        model (SentenceTransformer): The loaded transformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedder with a specific sentence-transformers model.

        Args:
            model_name (str): The name of the pre-trained model to load.
                              Default is "all-MiniLM-L6-v2".
        """
        logger.info(f"Initializing LocalEmbedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Local model loaded successfully.")

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for the given text using the loaded model.

        Args:
            text (str): The input string to embed.

        Returns:
            list[float]: A list of floats representing the embedding vector.
        """
        logger.debug(f"Encoding text of length {len(text)}")
        embedding = self.model.encode(text).tolist()
        logger.debug(f"Generated embedding of dimension {len(embedding)}")
        return embedding
