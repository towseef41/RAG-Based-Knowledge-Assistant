import logging
from app.services.reranking.base_reranker import BaseReranker
from app.services.reranking.bge_raranker import BgeReranker
from app.services.reranking.no_op_reranker import NoOpReranker

logger = logging.getLogger(__name__)

def get_reranker(strategy: str = "bge") -> BaseReranker:
    """
    Factory function to instantiate a reranker strategy.

    Parameters
    ----------
    strategy : str, optional
        The reranker strategy to use. Supported values are:
        - "bge": Uses the BgeReranker (default)
        - "none": Uses the NoOpReranker that performs no reranking

    Returns
    -------
    BaseReranker
        An instance of the selected reranker implementation.

    Raises
    ------
    ValueError
        If an unsupported strategy string is provided.

    Logs
    ----
    - Logs the chosen strategy and the created instance at INFO level.
    - Logs an error before raising ValueError if strategy is unsupported.
    """
    strategy = strategy.lower()

    if strategy == "bge":
        reranker = BgeReranker()
        logger.info(f"Reranker strategy '{strategy}' selected: {type(reranker).__name__} instance created.")
        return reranker

    elif strategy == "none":
        reranker = NoOpReranker()
        logger.info(f"Reranker strategy '{strategy}' selected: {type(reranker).__name__} instance created.")
        return reranker

    else:
        logger.error(f"Unsupported reranker strategy requested: '{strategy}'")
        raise ValueError(f"Unsupported reranker strategy: {strategy}")
