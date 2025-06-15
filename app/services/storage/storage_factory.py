import logging
from app.services.storage.sqlite_storage import SQLiteStorage
# from app.services.storage.supabase_storage import SupabaseStorage (future)
# from app.services.storage.qdrant_storage import QdrantStorage (future)

logger = logging.getLogger(__name__)

def get_storage_backend(backend: str = "sqlite", **kwargs):
    """
    Factory function to retrieve the appropriate storage backend.

    This function provides a unified interface to initialize and return a
    specific storage implementation (e.g., SQLite, Supabase, Qdrant) based
    on the given `backend` argument.

    Args:
        backend (str): The name of the storage backend to use.
                       Supported values: "sqlite" (default), "supabase", "qdrant".
        **kwargs: Additional keyword arguments to pass to the backend constructor.

    Returns:
        BaseStorage: An instance of a storage class that implements the BaseStorage interface.

    Raises:
        ValueError: If the specified backend is not supported.
    """
    backend = backend.lower()
    logger.info(f"Requested storage backend: {backend}")

    if backend == "sqlite":
        logger.info("Initializing SQLiteStorage backend")
        return SQLiteStorage()
    # elif backend == "supabase":
    #     logger.info("Initializing SupabaseStorage backend")
    #     return SupabaseStorage(**kwargs)
    # elif backend == "qdrant":
    #     logger.info("Initializing QdrantStorage backend")
    #     return QdrantStorage(**kwargs)
    else:
        logger.error(f"Unsupported storage backend requested: {backend}")
        raise ValueError(f"Unsupported storage backend: {backend}")
