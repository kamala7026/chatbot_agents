import chromadb
import torch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import logging
from typing import List, Dict, Any, Union

# Import the configured logger from your utils module
from utils.logger_config import logger

# --- Constants (should match your main app.py and core/chatbot.py) ---
CHROMA_PERSIST_DIR = "./chroma_db_docs"

# Ensure ChromaDB telemetry is disabled if it's not handled globally in main.py
# If main.py already sets this in os.environ, it's redundant here but harmless.
os.environ['CHROMA_ANALYTICS'] = 'False'

class DocumentManager:
    """
    Manages documents stored in the ChromaDB vector store.
    Assumes the vectorstore is already populated and persistent.
    It connects to the existing ChromaDB collection to retrieve, delete, and update metadata.
    """

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self._initialize_db()
        logger.info("DocumentManager instance initialized.")

    def _initialize_db(self):
        """Initializes the ChromaDB connection."""
        try:
            # Ensure the persistence directory exists
            if not os.path.exists(CHROMA_PERSIST_DIR):
                logger.warning(f"ChromaDB persistence directory '{CHROMA_PERSIST_DIR}' does not exist. No documents found to manage.")
                self.vectorstore = None
                return

            # Initialize embeddings (must be the same model as used for ingestion)
            # Use 'cpu' for management tasks unless GPU is specifically required and available
            device = "cuda" if torch.cuda.is_available() else "cpu" # Assuming torch is imported from chatbot
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            logger.info(f"DocumentManager: Embeddings model loaded on device: {device}.")

            # Connect to the persistent client
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            collection_name = "documents"

            try:
                # Attempt to get the existing collection.
                # If it doesn't exist, this will raise an exception.
                collection = client.get_collection(collection_name)
                logger.info(f"DocumentManager: Found existing ChromaDB collection '{collection_name}'.")
            except Exception as e:
                # Catch the error if collection is not found.
                logger.warning(f"DocumentManager: Collection '{collection_name}' not found in '{CHROMA_PERSIST_DIR}'. Error: {e}")
                self.vectorstore = None
                return

            self.vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR # Explicitly set persist_directory
            )
            count = self.vectorstore._collection.count()
            logger.info(f"DocumentManager: Connected to ChromaDB at {CHROMA_PERSIST_DIR}. Current document chunks in DB: {count}")

        except Exception as e:
            logger.error(f"DocumentManager: Fatal error initializing ChromaDB connection: {e}", exc_info=True)
            self.vectorstore = None  # Ensure vectorstore is None on failure

    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieves metadata for all unique documents (based on document_id).
        Returns a list of dictionaries, each representing a document with its chunk count.
        """
        if not self.vectorstore:
            logger.warning("DocumentManager: Vectorstore not initialized. Cannot retrieve document metadata.")
            return []

        try:
            logger.info("DocumentManager: Fetching all document metadata from ChromaDB.")
            # Fetch all items from the collection
            all_chunks = self.vectorstore._collection.get(
                ids=None,  # Get all IDs
                include=['metadatas']  # Only need metadata
            )

            unique_documents_map = {}  # To store one entry per document_id
            for metadata in all_chunks['metadatas']:
                doc_id = metadata.get("document_id")
                if doc_id:
                    if doc_id not in unique_documents_map:
                        # Store the first chunk's metadata as representative
                        unique_documents_map[doc_id] = {
                            "document_id": doc_id,
                            "filename": metadata.get("filename", "N/A"),
                            "category": metadata.get("category", "N/A"),
                            "description": metadata.get("description", "N/A"),
                            "status": metadata.get("status", "N/A"),
                            "access": metadata.get("access", "N/A"),
                            "total_chunks": 0  # Initialize chunk count
                        }
                    unique_documents_map[doc_id]["total_chunks"] += 1

            doc_list = list(unique_documents_map.values())
            logger.info(f"DocumentManager: Retrieved metadata for {len(doc_list)} unique documents.")
            return doc_list
        except Exception as e:
            logger.error(f"DocumentManager: Error fetching documents from ChromaDB: {e}", exc_info=True)
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes all chunks associated with a given document_id.
        """
        if not self.vectorstore:
            logger.warning("DocumentManager: Vectorstore not initialized. Cannot delete document.")
            return False
        try:
            logger.info(f"DocumentManager: Attempting to delete document with ID: {document_id}")
            # ChromaDB's delete works with a where clause on metadata
            self.vectorstore._collection.delete(
                where={"document_id": document_id}
            )
            self.vectorstore.persist()  # Ensure changes are written to disk
            logger.info(f"DocumentManager: Successfully deleted document_id: {document_id}")
            return True
        except Exception as e:
            logger.error(f"DocumentManager: Error deleting document {document_id}: {e}", exc_info=True)
            return False

    def update_document_metadata(self, document_id: str, updates: Dict[str, str]) -> bool:
        """
        Updates multiple metadata fields for all chunks of a given document_id.
        'updates' is a dictionary where keys are field names (e.g., 'access', 'status')
        and values are the new values for those fields.
        """
        if not self.vectorstore:
            logger.warning("DocumentManager: Vectorstore not initialized. Cannot update document metadata.")
            return False
        if not updates:
            logger.info("DocumentManager: No metadata updates provided.")
            return True  # No changes requested, so consider it a success

        try:
            logger.info(f"DocumentManager: Attempting to update metadata for document ID: {document_id} with {updates}")
            chunks_to_update = self.vectorstore._collection.get(
                where={"document_id": document_id},
                include=[]  # Only need IDs for update
            )
            ids_to_update = chunks_to_update.get('ids', [])

            if not ids_to_update:
                logger.warning(f"DocumentManager: No chunks found for document_id: {document_id}. Cannot update metadata.")
                return False

            # ChromaDB's update method expects a list of IDs and corresponding lists of new metadatas/documents/embeddings.
            # When updating metadata, you provide a list of dictionaries, where each dictionary specifies the metadata
            # to be merged with the existing metadata for the corresponding ID.
            # Here, we want to apply the same `updates` dictionary to all chunks of the document.
            metadatas_for_update = [updates for _ in ids_to_update]

            self.vectorstore._collection.update(
                ids=ids_to_update,
                metadatas=metadatas_for_update
            )

            updated_fields_str = ", ".join([f"'{k}' to '{v}'" for k, v in updates.items()])
            logger.info(f"DocumentManager: Successfully updated metadata for document_id {document_id}: {updated_fields_str}")
            self.vectorstore.persist()
            return True
        except Exception as e:
            logger.error(f"DocumentManager: Error updating document {document_id} metadata: {e}", exc_info=True)
            return False

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