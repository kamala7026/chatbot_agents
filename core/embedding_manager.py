# core/embedding_manager.py
import logging
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from utils.logger_config import logger

class EmbeddingManager:
    """
    Manages the singleton instance of the HuggingFace embeddings model.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        """
        Returns the singleton instance of the HuggingFaceEmbeddings model,
        loading it if it doesn't exist.
        """
        if cls._instance is None:
            logger.info("EmbeddingManager: No cached embeddings instance found. Loading model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                cls._instance = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': device}
                )
                logger.info(f"EmbeddingManager: HuggingFace embeddings model loaded successfully on device: {device}")
            except Exception as e:
                logger.critical(f"EmbeddingManager: Failed to load embeddings model: {e}", exc_info=True)
                raise
        return cls._instance

# --- Public Access Function ---
def get_embeddings_instance():
    """Provides access to the singleton embeddings model instance."""
    return EmbeddingManager.get_instance() 