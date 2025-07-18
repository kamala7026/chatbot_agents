import os
import uuid
import tempfile
import logging
import torch
import chromadb

from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStore # Use the generic type

# Import core components
from .embedding_manager import get_embeddings_instance
from vector.chromavector_manager import get_vectorstore, get_vectorstore_chunk_count # Use the new factory
from .prompt_manager import PromptManager
from core.semantic_document_processor import SemanticDocumentProcessor
from core.rag_retriever import RAGRetriever  # The actual retriever logic
#from core.internet_search import InternetSearchTool as CoreInternetSearch  # The actual Tavily client

# Import tool wrappers
from tools.ticket_create_agent import TicketCreationTool
from tools.rag_tool import RAGTool  # New
from tools.internet_search_tool import InternetSearchTool as AgentInternetSearchTool  # New, avoid name collision
from tools.default_response_tool import DefaultResponseTool  # New

from utils.logger_config import logger  # Import the configured logger


# DEFAULT_CHROMA_MAX_BATCH_SIZE = 5000

# Configure ChromaDB telemetry globally
# os.environ['CHROMA_ANALYTICS'] = 'False'
# TAVILY_API_KEY is now loaded from the .env file by core/config.py
# os.environ['TAVILY_API_KEY'] = "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"


class RAGChatbot:
    """Main RAG Chatbot class, orchestrating RAG, tools, and LLM interactions."""

    def __init__(self):
        # Components are now lazy-loaded or retrieved from singletons.
        # The actual initialization happens in initialize_components or on first use.
        self.embeddings = None
        self.vectorstore = None
        self.document_processor = SemanticDocumentProcessor()
        self.llm = None
        #self.core_internet_search_client = CoreInternetSearch()
        self.agent_prompt_template = None
        self.google_api_key = None # To store the key for re-initialization
        self.initialized = False # The new flag
        logger.info("RAGChatbot instance created (now stateless).")

    def _ensure_initialized(self):
        """
        Private helper to ensure all components are initialized.
        If not, it will attempt to 'self-heal' by re-initializing.
        Raises an exception if it cannot recover.
        """
        if self.llm and self.vectorstore and self.agent_prompt_template:
            return  # Already initialized, do nothing

        logger.warning("Chatbot components are not initialized. Attempting to self-heal...")
        if not self.google_api_key:
            raise Exception("Fatal: Cannot re-initialize chatbot, Google API key is missing.")
        
        if not self.initialize_components(self.google_api_key):
            raise Exception("Fatal: Could not re-initialize chatbot components.")

    def initialize_components(self, google_api_key: str) -> bool:
        """Initialize LLM, embeddings, vector store, and agent prompt."""
        # Store the API key in case re-initialization is needed
        self.google_api_key = google_api_key
        logger.info("Initializing chatbot components...")
        try:
            # Retrieve singleton instances from their managers
            self.embeddings = get_embeddings_instance()
            self.vectorstore = get_vectorstore() # Use the factory
            logger.info("Embeddings model and vector store successfully loaded via managers.")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.7
            )
            logger.info("Google Generative AI LLM initialized.")

            if self.vectorstore is None:
                raise Exception("Vector store failed to initialize after all attempts.")

            # Pre-compile the agent prompt template by fetching it from the manager
            self.agent_prompt_template = PromptManager.get_agent_prompt()

            logger.info("Chatbot components initialized successfully.")
            self.initialized = True # Set the flag on success
            return True

        except Exception as e:
            logger.critical(f"Error during chatbot initialization: {e}", exc_info=True)
            self.embeddings = None
            self.llm = None
            self.vectorstore = None
            self.initialized = False # Ensure flag is false on failure
            return False

    def get_agent_executor(self, user_type: str, chat_history: List[Any]) -> AgentExecutor:
        """Dynamically creates a LangChain agent with the given history."""
        self._ensure_initialized()

        # Add assertions to satisfy the linter after the check
        assert self.llm is not None
        assert self.vectorstore is not None
        assert self.agent_prompt_template is not None

        retriever_logic = RAGRetriever(self.vectorstore, self.llm, user_type)

        # Create a temporary memory for this specific conversation turn
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=15 # Allow a larger buffer for context
        )
        # Load the provided history into the temporary memory
        for message in chat_history:
            if message['role'] == 'user':
                memory.chat_memory.add_user_message(message['content'])
            elif message['role'] == 'assistant':
                memory.chat_memory.add_ai_message(message['content'])

        def get_current_chat_history() -> List[Any]:
            return memory.load_memory_variables({})['chat_history']

        tools = [
            RAGTool(retriever_logic).get_tool(),
            DefaultResponseTool().get_tool(),
            TicketCreationTool(self.llm, chat_history_provider=get_current_chat_history).get_tool()
        ]
        
        tool_names = [tool.name for tool in tools]
        logger.debug(f"Agent created with tools: {tool_names}")

        # The agent prompt is now retrieved from the cached instance variable
        prompt = self.agent_prompt_template

        return AgentExecutor(
            agent=create_react_agent(self.llm, tools, prompt),
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=120
        )

    def chat(self, message: str, user_type: str, chat_history: List[Dict[str, str]]) -> str:
        """Handles chat by creating a temporary agent with the provided history."""
        logger.info(f"Received chat message from user_type '{user_type}'. Message: '{message}'")
        try:
            agent_executor = self.get_agent_executor(user_type, chat_history)
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

        prompt_text = (
            "Summarize the following user's first message into a short, descriptive title of no more than 5 words. "
            "For example, 'Tell me about the Q2 earnings report' could be 'Q2 Earnings Report'.\n\n"
            "USER MESSAGE: '{message}'\n\n"
            "TITLE:"
        )
        prompt = PromptTemplate.from_template(prompt_text)
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"message": user_message})
            # The response object from langchain has a 'content' attribute
            content = response.content
            if isinstance(content, str):
                title = content.strip().strip('"')
            else:
                # Handle cases where content is not a direct string
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