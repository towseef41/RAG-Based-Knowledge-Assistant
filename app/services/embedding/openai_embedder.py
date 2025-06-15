"""
OpenAIEmbedder implementation using OpenAI's text embedding API.
"""

import os
import openai
import logging
from typing import List
from app.services.embedding.base_embedder import BaseEmbedder

# Configure logging
logger = logging.getLogger(__name__)

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding generator that uses OpenAI's hosted embedding models.

    This embedder interacts with OpenAI's API to generate vector embeddings.
    Ideal for high-accuracy tasks and production environments with reliable internet access.

    Attributes:
        model (str): The name of the OpenAI embedding model to use.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedder with a specific OpenAI model.

        Args:
            model_name (str): The name of the OpenAI embedding model.
                              Defaults to "text-embedding-3-small".
        """
        self.model = model_name
        logger.info(f"OpenAIEmbedder initialized with model: {self.model}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for the given input text using OpenAI's API.

        Args:
            text (str): The input string to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.

        Raises:
            openai.OpenAIError: If the API call fails.
        """
        logger.debug(f"Requesting embedding for text of length {len(text)} using model '{self.model}'")

        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            embedding = response["data"][0]["embedding"]
            logger.debug(f"Received embedding of dimension {len(embedding)}")
            return embedding
        except openai.OpenAIError as e:
            logger.error(f"OpenAI embedding API call failed: {e}")
            raise
