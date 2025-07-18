import chromadb
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import logging
from typing import List, Dict, Any, Union
import tempfile
import shutil
import uuid

from langchain_core.vectorstores import VectorStore

# Import singleton managers
from .embedding_manager import get_embeddings_instance
# Use the new factory functions directly
from vector.chromavector_manager import get_vectorstore, get_all_documents_metadata, delete_document, update_document_metadata 
from utils.logger_config import logger
from core.semantic_document_processor import SemanticDocumentProcessor

# --- Constants ---
# No longer needed here, defined in the manager
# CHROMA_PERSIST_DIR = "./chroma_db_docs"

# Ensure ChromaDB telemetry is disabled if it's not handled globally in main.py
# If main.py already sets this in os.environ, it's redundant here but harmless.
# os.environ['CHROMA_ANALYTICS'] = 'False'

class DocumentManager:
    """
    Manages documents stored in the vector store.
    It now gets a singleton instance of the vector store from a centralized manager.
    """

    def __init__(self):
        # The vectorstore is now retrieved from the factory when needed.
        self.document_processor = SemanticDocumentProcessor()
        logger.info("DocumentManager instance initialized (stateless).")

    def add_document(self, filename: str, file_contents: Any, metadata: Dict[str, Any]) -> bool:
        """
        Processes and adds a document to the configured vector store.
        It saves the uploaded file temporarily to handle file-path-based loaders.
        """
        # Generate a unique document ID and add essential metadata
        document_id = str(uuid.uuid4())
        metadata['document_id'] = document_id
        metadata['filename'] = filename
        metadata['source'] = filename  # Often, the source is the filename

        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            shutil.copyfileobj(file_contents, tmp)
            tmp_path = tmp.name
        
        try:
            logger.info(f"Processing and adding document: {filename} from temp path: {tmp_path}")
            vectorstore = get_vectorstore()
            
            # Determine file type and load content from the temporary file path
            file_type = filename.split('.')[-1].lower()
            texts = self.document_processor.load_document(tmp_path, file_type)
            
            # Chunk the documents
            chunks = self.document_processor.chunk_documents(texts, metadata)
            
            vectorstore.add_documents(chunks)
            logger.info(f"Successfully added document {filename} to the vector store.")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}", exc_info=True)
            return False
        finally:
            # Ensure the temporary file is deleted after processing
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Removed temporary file: {tmp_path}")

    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieves metadata for all unique documents using the factory.
        The consolidation logic is now handled by the vector store managers.
        """
        try:
            logger.info("Fetching all document metadata via factory.")
            # The factory now returns clean, consolidated data directly.
            return get_all_documents_metadata()
        except Exception as e:
            logger.error(f"Error fetching documents from vector store: {e}", exc_info=True)
            return []

    def delete_document(self, document_id: str) -> bool:
        """Deletes all chunks associated with a given document_id via the factory."""
        logger.info(f"Attempting to delete document {document_id} using vector store factory.")
        return delete_document(document_id)

    def update_document_metadata(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Updates metadata for a document via the factory."""
        logger.info(f"Attempting to update metadata for document {document_id} using vector store factory.")
        return update_document_metadata(document_id, updates)

    # The original update_document_metadata_field and update_document_access methods
    # are now redundant if update_document_metadata handles multiple fields.
    # I've kept them commented out but recommend removing them for cleaner code.
    # If you still want single-field update methods, they should call the more general one.
    # def update_document_metadata_field(self, document_id: str, field_name: str, new_value: str) -> bool:
    #     logger.info(f"DocumentManager: Updating single field '{field_name}' for {document_id}.")
    #     return self.update_document_metadata(document_id, {field_name: new_value})

    # def update_document_access(self, document_id: str, new_access: str) -> bool:
    #     logger.info(f"DocumentManager: Updating access for {document_id}.")
    #     return self.update_document_metadata(document_id, {"access": new_access})