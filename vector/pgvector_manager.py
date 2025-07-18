import logging
from typing import Optional
import sqlalchemy
from langchain_community.vectorstores.pgvector import PGVector
from core.embedding_manager import get_embeddings_instance # Use the singleton manager
from core.config import PGVECTOR_CONNECTION_STRING, COLLECTION_NAME
from langchain.schema import Document
import psycopg # Added for direct database connection

logger = logging.getLogger("aviator_chatbot")

class PGVectorManager:
    """
    Manages the connection and interactions with the PGVector vector store.
    This class follows a singleton pattern to ensure only one connection pool is created.
    """
    _vectorstore: Optional[PGVector] = None
    _collection_name: Optional[str] = None
    _connection_string: Optional[str] = None
    _connection: Optional[psycopg.Connection] = None # Add this line

    @classmethod
    def get_vectorstore(cls) -> PGVector:
        """
        Initializes and returns the singleton PGVector instance.

        This method checks if a connection has already been established.
        If not, it attempts to connect to the PGVector database using the
        connection string from the application's configuration.

        Returns:
            The initialized PGVector instance.

        Raises:
            ConnectionError: If the PGVECTOR_CONNECTION_STRING is not set or
                             if the connection to the database fails.
        """
        if cls._vectorstore is None:
            logger.info("PGVector store not initialized. Attempting to connect...")
            if not PGVECTOR_CONNECTION_STRING:
                logger.error("PGVECTOR_CONNECTION_STRING is not set in the environment.")
                raise ConnectionError("PGVector connection string not configured.")
            
            try:
                # Get the singleton embeddings instance
                embeddings = get_embeddings_instance()
                
                # Store the connection string for other managers to use
                cls._connection_string = PGVECTOR_CONNECTION_STRING

                # This method will create the collection if it doesn't exist,
                # ensuring the schema is correct. We pass an empty list of documents
                # because we are only using this for initialization.
                cls._vectorstore = PGVector.from_documents(
                    embedding=embeddings,
                    documents=[], # Pass empty list to ensure creation without adding docs
                    collection_name=COLLECTION_NAME,
                    connection_string=cls._connection_string,
                    # This pre-deletes the collection to ensure a clean slate.
                    # Set to False if you want to preserve data between runs.
                    pre_delete_collection=False 
                )
                
                logger.info("PGVector connection initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize PGVector connection: {e}", exc_info=True)
                raise ConnectionError(f"Could not connect to PGVector database: {e}")
        
        return cls._vectorstore

    @classmethod
    def get_db_connection(cls) -> Optional[psycopg.Connection]:
        """
        Returns a direct, separate connection to the PostgreSQL database.
        This is used for non-vector-store operations like managing chat history.
        """
        if cls._connection is None or cls._connection.closed:
            logger.info("Database connection for history not available or closed. Creating new one.")
            try:
                # Use the connection string established by get_vectorstore
                if not cls._connection_string:
                    # Ensure the vector store is initialized first to get the conn string
                    cls.get_vectorstore()
                
                if not cls._connection_string:
                     raise ConnectionError("Connection string not available after vector store init.")

                # Clean the connection string for raw psycopg
                conn_str = cls._connection_string.replace("+psycopg", "")
                cls._connection = psycopg.connect(conn_str)

            except Exception as e:
                logger.error(f"Failed to create a separate database connection: {e}", exc_info=True)
                return None # Return None on failure
        
        return cls._connection

    @classmethod
    def get_chunk_count(cls) -> int:
        """
        Returns the total number of chunks (embeddings) in the PGVector store
        for the configured collection.
        """
        vectorstore = cls.get_vectorstore()
        if not hasattr(vectorstore, "EmbeddingStore"):
            logger.warning("PGVector store is not in a state to count chunks.")
            return 0

        try:
            with vectorstore._make_session() as session:
                collection = vectorstore.get_collection(session)
                if not collection:
                    logger.warning(f"Collection '{vectorstore.collection_name}' not found. Cannot get chunk count.")
                    return 0
                
                count = (
                    session.query(vectorstore.EmbeddingStore)
                    .filter(vectorstore.EmbeddingStore.collection_id == collection.uuid)
                    .count()
                )
                logger.info(f"Found {count} chunks in collection '{vectorstore.collection_name}'.")
                return count
        except Exception as e:
            logger.error(f"Error counting chunks in PGVector: {e}", exc_info=True)
            return 0

    @classmethod
    def get_all_metadata(cls) -> list:
        """
        Retrieves consolidated metadata for each unique document in PGVector.
        It fetches all chunks and then aggregates them by document_id.
        """
        store = cls.get_vectorstore()
        if store:
            try:
                # This fetches all chunks; we need to consolidate them.
                results = store.similarity_search(query="", k=10000)
                
                if not results:
                    logger.info("PGVector: No documents found in the vector store.")
                    return []

                unique_documents = {}
                for doc in results:
                    doc_id = doc.metadata.get("document_id")
                    if not doc_id:
                        continue

                    if doc_id not in unique_documents:
                        # Initialize the document entry with a counter
                        unique_documents[doc_id] = {
                            "metadata": {},
                            "total_chunks": 0
                        }
                    
                    # Always update metadata from the current chunk, ensuring the most
                    # complete version is retained. Then, increment the chunk count.
                    unique_documents[doc_id]['metadata'].update(doc.metadata)
                    unique_documents[doc_id]['total_chunks'] += 1

                # Now, build the final list, ensuring defaults for any missing keys
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

                logger.info(f"PGVector: Successfully retrieved and consolidated metadata for {len(consolidated_metadata)} unique documents.")
                return consolidated_metadata

            except Exception as e:
                logger.error(f"PGVector: Error retrieving all metadata: {e}", exc_info=True)
                return []
        return []

    @classmethod
    def delete_document(cls, document_id: str) -> bool:
        """Deletes all chunks associated with a document_id from PGVector."""
        vectorstore = cls.get_vectorstore()
        if not hasattr(vectorstore, "EmbeddingStore"):
            logger.error("PGVector store is not in a state to delete document.")
            return False
        
        try:
            with vectorstore._make_session() as session:
                collection = vectorstore.get_collection(session)
                if not collection:
                    logger.warning(f"Collection '{vectorstore.collection_name}' not found.")
                    return False
                
                # Find the UUIDs of the embeddings to delete
                stmt = sqlalchemy.select(vectorstore.EmbeddingStore.uuid).where(
                    vectorstore.EmbeddingStore.collection_id == collection.uuid,
                    vectorstore.EmbeddingStore.cmetadata["document_id"].astext == document_id
                )
                ids_to_delete = session.execute(stmt).scalars().all()

                if not ids_to_delete:
                    logger.warning(f"PGVector: No documents found with document_id {document_id} to delete.")
                    return False

                # Delete the embeddings by their primary key (uuid)
                delete_stmt = sqlalchemy.delete(vectorstore.EmbeddingStore).where(
                    vectorstore.EmbeddingStore.uuid.in_(ids_to_delete)
                )
                
                session.execute(delete_stmt)
                session.commit()
                
                logger.info(f"PGVector: Successfully deleted {len(ids_to_delete)} chunks for document {document_id}.")
                return True
        except Exception as e:
            logger.error(f"PGVector: Error deleting document {document_id}: {e}", exc_info=True)
            return False

    @classmethod
    def update_document_metadata(cls, document_id: str, updates: dict) -> bool:
        """
        Updates metadata for all chunks of a document in PGVector.
        """
        vectorstore = cls.get_vectorstore()
        if not hasattr(vectorstore, "EmbeddingStore"):
            logger.error("PGVector store is not in a state to update metadata.")
            return False
        
        try:
            with vectorstore._make_session() as session:
                collection = vectorstore.get_collection(session)
                if not collection:
                    logger.warning(f"Collection '{vectorstore.collection_name}' not found.")
                    return False
                
                # Find the embeddings to update
                stmt = sqlalchemy.select(vectorstore.EmbeddingStore).where(
                    vectorstore.EmbeddingStore.collection_id == collection.uuid,
                    vectorstore.EmbeddingStore.cmetadata["document_id"].astext == document_id
                )
                
                docs_to_update = session.execute(stmt).scalars().all()
                
                if not docs_to_update:
                    logger.warning(f"PGVector: No documents found with document_id {document_id} to update.")
                    return False

                # Update metadata for each document
                for doc in docs_to_update:
                    updated_metadata = doc.cmetadata.copy()
                    updated_metadata.update(updates)
                    doc.cmetadata = updated_metadata
                
                session.commit()
                
                logger.info(f"PGVector: Successfully updated metadata for {len(docs_to_update)} chunks of document {document_id}.")
                return True
        except Exception as e:
            logger.error(f"PGVector: Error updating metadata for document {document_id}: {e}", exc_info=True)
            return False 