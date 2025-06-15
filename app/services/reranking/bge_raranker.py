import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from app.services.reranking.base_reranker import BaseReranker

logger = logging.getLogger(__name__)

class BgeReranker(BaseReranker):
    """
    Reranker implementation using the BGE reranker model from Hugging Face Transformers.

    This class reranks a list of documents based on their relevance to a given query
    by scoring each query-document pair using a pretrained sequence classification model.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        Tokenizer loaded from the pretrained model.
    model : AutoModelForSequenceClassification
        Pretrained sequence classification model used for scoring.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize the reranker by loading the pretrained model and tokenizer.

        Parameters
        ----------
        model_name : str, optional
            The Hugging Face model name to load. Defaults to "BAAI/bge-reranker-base".
        """
        logger.info(f"Loading BGE reranker model and tokenizer from '{model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank the documents based on their relevance scores computed against the query.

        Parameters
        ----------
        query : str
            The user query string.
        documents : List[Dict]
            List of documents to be reranked. Each document must contain a "text" field.

        Returns
        -------
        List[Dict]
            The list of documents with an added "score" field, sorted by descending score.

        Logs
        ----
        - Logs number of documents to rerank at DEBUG level.
        - Logs completion of scoring and reranking at INFO level.
        """
        logger.debug(f"Reranking {len(documents)} documents for query: '{query}'")

        pairs = [(query, doc["text"]) for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        reranked = [
            {**doc, "score": score.item()} for doc, score in zip(documents, scores)
        ]
        reranked.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Reranking complete: {len(reranked)} documents scored and sorted")
        return reranked
