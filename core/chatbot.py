import os
import uuid
import tempfile
import logging
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from vector.embedding_manager import get_embeddings_instance
from vector.chromavector_manager import get_vectorstore, get_vectorstore_chunk_count
from vector.semantic_document_processor import SemanticDocumentProcessor
from core.common.prompt_manager import PromptManager
from agent.agent_manager import AgentManager  # Import the new manager
from utils.logger_config import logger

class RAGChatbot:
    """Main RAG Chatbot class, focused on chat orchestration and document management."""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.document_processor = SemanticDocumentProcessor()
        self.llm = None
        self.agent_manager = None  # Add a placeholder for the agent manager
        self.google_api_key = None
        self.initialized = False
        logger.info("RAGChatbot instance created (now stateless).")

    def _ensure_initialized(self):
        """
        Private helper to ensure all components are initialized.
        If not, it will attempt to 'self-heal' by re-initializing.
        Raises an exception if it cannot recover.
        """
        if self.initialized and self.agent_manager:
            return  # Already initialized, do nothing

        logger.warning("Chatbot components are not initialized. Attempting to self-heal...")
        if not self.google_api_key:
            raise Exception("Fatal: Cannot re-initialize chatbot, Google API key is missing.")
        
        if not self.initialize_components(self.google_api_key):
            raise Exception("Fatal: Could not re-initialize chatbot components.")

    def initialize_components(self, google_api_key: str) -> bool:
        """Initialize LLM, embeddings, vector store, and the new agent manager."""
        self.google_api_key = google_api_key
        logger.info("Initializing chatbot components...")
        try:
            self.embeddings = get_embeddings_instance()
            self.vectorstore = get_vectorstore()
            logger.info("Embeddings model and vector store successfully loaded via managers.")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.7
            )
            logger.info("Google Generative AI LLM initialized.")

            if self.vectorstore is None:
                raise Exception("Vector store failed to initialize after all attempts.")

            # Initialize the new AgentManager
            self.agent_manager = AgentManager(self.llm, self.vectorstore)
            logger.info("AgentManager initialized successfully.")

            logger.info("Chatbot components initialized successfully.")
            self.initialized = True
            return True

        except Exception as e:
            logger.critical(f"Error during chatbot initialization: {e}", exc_info=True)
            self.embeddings = None
            self.llm = None
            self.vectorstore = None
            self.agent_manager = None
            self.initialized = False
            return False

    def chat(self, message: str, user_type: str, chat_history: List[Dict[str, str]]) -> str:
        """Handles chat by delegating agent creation to the AgentManager."""
        logger.info(f"Received chat message from user_type '{user_type}'. Message: '{message}'")
        try:
            self._ensure_initialized()
            assert self.agent_manager is not None
            agent_executor = self.agent_manager.get_agent_executor(user_type, chat_history)
            response = agent_executor.invoke({"input": message})
            output = response.get("output", "Sorry, I had trouble generating a response.")
            logger.info(f"Agent generated response: '{output}'")
            return output
        except Exception as e:
            logger.error(f"Error during chat execution: {e}", exc_info=True)
            return f"An error occurred: {e}"

    def generate_chat_title(self, user_message: str) -> str:
        """
        Generates a concise title for a new chat based on the user's first message.
        """
        self._ensure_initialized()
        assert self.llm is not None

        prompt_text = PromptManager.get_chat_title_prompt()
        prompt = PromptTemplate.from_template(prompt_text)
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"message": user_message})
            content = response.content
            if isinstance(content, str):
                title = content.strip().strip('"')
            else:
                title = str(content).strip().strip('"')

            logger.info(f"Generated chat title: '{title}'")
            return title if title else "Untitled Chat"
        except Exception as e:
            logger.error(f"Error generating chat title: {e}", exc_info=True)
            return "Untitled Chat"
    
    def add_document(self, file_contents: bytes, filename: str, category: str, description: str, status: str, access: str) -> bool:
        """Add document to vector store, now taking file content and name directly."""
        try:
            self._ensure_initialized()
            assert self.vectorstore is not None
            assert self.embeddings is not None

            logger.info(f"Attempting to add document: {filename}")
            # Create a temporary file to write the content to
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_contents)
                tmp_file_path = tmp_file.name
            logger.debug(f"File '{filename}' saved temporarily to {tmp_file_path}")

            file_type = filename.split('.')[-1].lower()

            # Step 1: Load the document to extract text
            extracted_texts = self.document_processor.load_document(tmp_file_path, file_type)
            if not extracted_texts:
                logger.warning(f"Could not extract any text from document: {filename}")
                return False
            logger.debug(f"Extracted {len(extracted_texts)} text blocks from {filename}.")

            # Step 2: Chunk the extracted text with metadata
            document_id = str(uuid.uuid4())
            common_metadata = {
                "source": filename,
                "document_id": document_id,
                "filename": filename,
                "category": category,
                "description": description,
                "status": status,
                "access": access
            }

            chunks = self.document_processor.chunk_documents(extracted_texts, common_metadata)
            if not chunks:
                logger.warning(f"No chunks created from document: {filename}")
                return False
            logger.info(f"Created {len(chunks)} chunks for {filename}.")

            # Batch add chunks to ChromaDB
            # Use a sensible default batch size. The dynamic check was causing issues.
            max_chroma_batch_size = 100 
            logger.info(f"Using batch size of {max_chroma_batch_size} for adding documents.")

            for i in range(0, len(chunks), max_chroma_batch_size):
                batch = chunks[i:i + max_chroma_batch_size]
                logger.info(f"Adding batch {i // max_chroma_batch_size + 1} of {len(batch)} chunks for '{filename}' to ChromaDB.")
                self.vectorstore.add_documents(batch)

            logger.info(f"Successfully added all chunks of '{filename}' to vector database.")
            return True

        except Exception as e:
            logger.error(f"Error adding document '{filename}': {e}", exc_info=True)
            return False
        finally:
            # Clean up the temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                logger.debug(f"Removed temporary file: {tmp_file_path}")

    def get_vectorstore_chunk_count(self) -> int:
        """Returns the total number of chunks in the vector store by using the factory."""
        self._ensure_initialized()
        return get_vectorstore_chunk_count()