from typing import List, Dict
from app.services.reranking.base_reranker import BaseReranker
import logging

logger = logging.getLogger(__name__)

class NoOpReranker(BaseReranker):
    """
    No-op reranker that returns documents in the original order without any reranking.

    This class implements the BaseReranker interface but performs no changes to the input list,
    useful as a default or placeholder reranker.

    Methods
    -------
    rerank(query: str, documents: List[Dict]) -> List[Dict]
        Returns the input documents unchanged.
    """

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Return documents unchanged, performing no reranking.

        Parameters
        ----------
        query : str
            The query string (ignored in this implementation).
        documents : List[Dict]
            List of documents to (not) rerank.

        Returns
        -------
        List[Dict]
            The original list of documents in the same order.

        Logs
        ----
        Logs the number of documents received and that no reranking was performed.
        """
        logger.info(f"NoOpReranker called with {len(documents)} documents; returning original order without changes.")
        return documents
