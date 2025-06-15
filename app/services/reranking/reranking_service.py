import logging
from typing import List, Dict
from app.services.reranking.base_reranker import BaseReranker

logger = logging.getLogger(__name__)

class RerankingService:
    """
    Service to rerank a list of documents based on their relevance to a query
    using an injected reranker strategy.

    Attributes
    ----------
    reranker : BaseReranker
        The underlying reranking implementation to reorder documents.
    """

    def __init__(self, reranker: BaseReranker):
        """
        Initialize the reranking service with a specific reranker.

        Parameters
        ----------
        reranker : BaseReranker
            An implementation of the BaseReranker interface.
        """
        self.reranker = reranker
        logger.info(f"Initialized RerankingService with reranker: {type(reranker).__name__}")

    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank a list of documents based on their relevance to the given query.

        Parameters
        ----------
        query : str
            The user query string used for reranking.
        documents : List[Dict]
            A list of documents, each represented as a dictionary.

        Returns
        -------
        List[Dict]
            The reranked list of documents, ordered by relevance.

        Logs
        ----
        - Logs the start of the reranking process at DEBUG level.
        - Logs the completion and count of reranked documents at INFO level.
        - Logs any errors encountered during reranking at ERROR level with stack trace.
        """
        logger.debug(f"Starting reranking for query: {query} with {len(documents)} documents")
        try:
            reranked_docs = self.reranker.rerank(query, documents)
            logger.info(f"Reranking completed: returned {len(reranked_docs)} documents")
            return reranked_docs
        except Exception as e:
            logger.error(f"Error during reranking for query '{query}': {e}", exc_info=True)
            raise
