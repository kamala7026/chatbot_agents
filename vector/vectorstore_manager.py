# core/vectorstore_manager.py
import os
import chromadb
import logging
from langchain_community.vectorstores import Chroma
from typing import Optional, List, Dict, Any

# Import singleton managers and configuration
from core.embedding_manager import get_embeddings_instance
from core.config import CHROMA_PERSIST_DIR, COLLECTION_NAME

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

# --- Constants ---
# Moved to core/config.py

# --- Singleton Instance ---
_vectorstore_instance = None

class ChromaDBManager:
    """
    Manages the singleton instance of the Chroma vector store.
    """
    _vectorstore: Optional[Chroma] = None

    @classmethod
    def get_vectorstore(cls) -> Chroma:
        """
        Returns the singleton instance of the Chroma vector store.
        """
        if cls._vectorstore is None:
            logger.info("Vectorstore instance not found, creating a new one.")
            try:
                embeddings = get_embeddings_instance()
                if not embeddings:
                    raise RuntimeError("Could not get embeddings instance for the vector store.")

                if not os.path.exists(CHROMA_PERSIST_DIR):
                    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                    logger.info(f"Created ChromaDB persistence directory: {CHROMA_PERSIST_DIR}")

                # Use PersistentClient for a robust connection
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                
                # get_or_create_collection is safer and ensures the collection exists
                collection = client.get_or_create_collection(COLLECTION_NAME)
                logger.info(f"Successfully connected to ChromaDB collection '{collection.name}'.")

                cls._vectorstore = Chroma(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR  # Good practice to be explicit
                )
                
                count = cls._vectorstore._collection.count()
                logger.info(f"Vectorstore initialized. Document chunks currently in DB: {count}")

            except Exception as e:
                logger.critical(f"Fatal error initializing ChromaDB vector store: {e}", exc_info=True)
                # Reset instance on failure so next call can retry
                cls._vectorstore = None
                raise RuntimeError(f"Fatal error initializing vector store: {e}") from e
        
        logger.info(f"Returning ChromaDB instance with ID: {id(cls._vectorstore)}") # DEBUGGING
        return cls._vectorstore

    @classmethod
    def get_chunk_count(cls) -> int:
        """Returns the total number of chunks in the ChromaDB store."""
        store = cls.get_vectorstore()
        if store:
            return store._collection.count()
        return 0

    @classmethod
    def get_all_metadata(cls) -> List[Dict[str, Any]]:
        """Returns all metadata from the ChromaDB store."""
        store = cls.get_vectorstore()
        if store:
            try:
                # The 'get()' method in ChromaDB returns all documents.
                results = store.get()
                logger.info(f"ChromaDB raw get() result: {results}") # DEBUGGING
                
                if not results or not results.get('metadatas'):
                    logger.info("ChromaDB: No documents found in the vector store.")
                    return []

                all_metadata = results.get('metadatas', [])
                if not all_metadata:
                    return []
                
                unique_documents = {}
                for metadata in all_metadata:
                    doc_id = metadata.get("document_id")
                    if not doc_id:
                        continue

                    if doc_id not in unique_documents:
                        unique_documents[doc_id] = { "metadata": {}, "total_chunks": 0 }
                    
                    unique_documents[doc_id]['metadata'].update(metadata)
                    unique_documents[doc_id]['total_chunks'] += 1

                consolidated_metadata = []
                for doc_id, data in unique_documents.items():
                    final_meta = {
                        "document_id": doc_id,
                        "filename": data['metadata'].get("filename", "Unknown"),
                        "description": data['metadata'].get("description", "No description"),
                        "status": data['metadata'].get("status", "Inactive"),
                        "access": data['metadata'].get("access", "Private"),
                        "category": data['metadata'].get("category", "General"),
                        "total_chunks": data['total_chunks']
                    }
                    consolidated_metadata.append(final_meta)
                
                logger.info(f"ChromaDB: Successfully retrieved and consolidated metadata for {len(consolidated_metadata)} unique documents.")
                return consolidated_metadata

            except Exception as e:
                logger.error(f"ChromaDB: Error retrieving all metadata: {e}", exc_info=True)
        return []

    @classmethod
    def delete_document(cls, document_id: str) -> bool:
        """Deletes all chunks associated with a document_id from ChromaDB."""
        store = cls.get_vectorstore()
        if store and hasattr(store, '_collection'):
            try:
                store._collection.delete(where={"document_id": document_id})
                logger.info(f"ChromaDB: Successfully deleted document with ID: {document_id}")
                return True
            except Exception as e:
                logger.error(f"ChromaDB: Error deleting document {document_id}: {e}", exc_info=True)
                return False
        return False

    @classmethod
    def update_document_metadata(cls, document_id: str, updates: Dict[str, Any]) -> bool:
        """Updates metadata for all chunks of a document in ChromaDB."""
        store = cls.get_vectorstore()
        if store and hasattr(store, '_collection'):
            try:
                # First, find the IDs of the chunks that belong to the document
                chunks_to_update = store._collection.get(
                    where={"document_id": document_id},
                    include=[]  # Only need IDs
                )
                ids_to_update = chunks_to_update.get('ids', [])
                if not ids_to_update:
                    logger.warning(f"ChromaDB: No chunks found for document_id {document_id} to update.")
                    return False
                
                # Create a metadata list for the update operation
                metadatas_for_update = [updates for _ in ids_to_update]
                store._collection.update(ids=ids_to_update, metadatas=metadatas_for_update) # type: ignore
                logger.info(f"ChromaDB: Successfully updated metadata for document {document_id}.")
                return True
            except Exception as e:
                logger.error(f"ChromaDB: Error updating metadata for document {document_id}: {e}", exc_info=True)
                return False
        return False 