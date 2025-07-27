import logging
from core.common.config import VECTOR_STORE_TYPE
from .vectorstore_manager import ChromaDBManager
from .pgvector_manager import PGVectorManager
from langchain_core.vectorstores import VectorStore

# Initialize logger
logger = logging.getLogger(__name__)

def get_vectorstore() -> VectorStore:
    """
    Factory function that returns the appropriate vector store instance
    based on the application's configuration.
    """
    store_type = VECTOR_STORE_TYPE
    logger.info(f"Vector store type configured as: '{store_type}'")

    if store_type == "pgvector":
        logger.info("Initializing vector store using PGVectorManager.")
        return PGVectorManager.get_vectorstore()
    elif store_type == "chroma":
        logger.info("Initializing vector store using ChromaDBManager.")
        return ChromaDBManager.get_vectorstore()
    else:
        logger.error(f"Invalid VECTOR_STORE_TYPE: '{store_type}'. Supported types are 'chroma' and 'pgvector'.")
        raise ValueError(f"Unsupported vector store type: {store_type}")

def get_vectorstore_chunk_count() -> int:
    """
    Factory function that returns the chunk count from the appropriate
    vector store based on the application's configuration.
    """
    store_type = VECTOR_STORE_TYPE
    if store_type == "pgvector":
        return PGVectorManager.get_chunk_count()
    elif store_type == "chroma":
        return ChromaDBManager.get_chunk_count()
    else:
        logger.error(f"Invalid VECTOR_STORE_TYPE: '{store_type}'.")
        return 0

def get_all_documents_metadata() -> list:
    """
    Factory function that returns all document metadata from the appropriate
    vector store based on the application's configuration.
    """
    store_type = VECTOR_STORE_TYPE
    if store_type == "pgvector":
        return PGVectorManager.get_all_metadata()
    elif store_type == "chroma":
        return ChromaDBManager.get_all_metadata()
    else:
        logger.error(f"Invalid VECTOR_STORE_TYPE: '{store_type}'.")
        return []

def delete_document(document_id: str) -> bool:
    """Factory function to delete a document from the configured vector store."""
    store_type = VECTOR_STORE_TYPE
    if store_type == "pgvector":
        return PGVectorManager.delete_document(document_id)
    elif store_type == "chroma":
        return ChromaDBManager.delete_document(document_id)
    else:
        logger.error(f"Invalid VECTOR_STORE_TYPE: '{store_type}'.")
        return False

def update_document_metadata(document_id: str, updates: dict) -> bool:
    """Factory function to update a document's metadata in the configured vector store."""
    store_type = VECTOR_STORE_TYPE
    if store_type == "pgvector":
        return PGVectorManager.update_document_metadata(document_id, updates)
    elif store_type == "chroma":
        return ChromaDBManager.update_document_metadata(document_id, updates)
    else:
        logger.error(f"Invalid VECTOR_STORE_TYPE: '{store_type}'.")
        return False 