# vector/embedding_manager.py
import logging
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from threading import Lock
from utils.logger_config import logger

class EmbeddingsManager:
    _instance: Optional[HuggingFaceEmbeddings] = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> HuggingFaceEmbeddings:
        # First check without a lock for performance
        if cls._instance:
            return cls._instance

        # If not initialized, acquire a lock
        with cls._lock:
            # Double-check if another thread initialized it while we were waiting for the lock
            if cls._instance is None:
                logger.info("Initializing embeddings model for the first time...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    cls._instance = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': device}
                    )
                    logger.info(f"HuggingFace embeddings model loaded successfully on device: {device}")
                except Exception as e:
                    logger.critical(f"Failed to load embeddings model: {e}", exc_info=True)
                    raise
        
        # The instance is guaranteed to be non-None here, but mypy doesn't know that.
        assert cls._instance is not None
        return cls._instance

def get_embeddings_instance() -> HuggingFaceEmbeddings:
    return EmbeddingsManager.get_instance() 