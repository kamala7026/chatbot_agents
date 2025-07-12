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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Import core components
from core.document_processor import DocumentProcessor
from core.rag_retriever import RAGRetriever  # The actual retriever logic
from core.internet_search import InternetSearchTool as CoreInternetSearch  # The actual Tavily client

# Import tool wrappers
from tools.ticket_create_agent import TicketCreationTool
from tools.rag_tool import RAGTool  # New
from tools.internet_search_tool import InternetSearchTool as AgentInternetSearchTool  # New, avoid name collision
from tools.default_response_tool import DefaultResponseTool  # New

from utils.logger_config import logger  # Import the configured logger

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_PERSIST_DIR = "./chroma_db_docs"
# Define a conservative default for Chroma batch size if dynamic retrieval fails
DEFAULT_CHROMA_MAX_BATCH_SIZE = 5000

# Configure ChromaDB telemetry globally
os.environ['CHROMA_ANALYTICS'] = 'False'
# TAVILY_API_KEY should ideally be set in environment variables before running the app.
# For demonstration purposes, I'm leaving it here as in your original code.
os.environ['TAVILY_API_KEY'] = "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"


def load_embeddings_model_cached():
    """Loads and caches the HuggingFace embeddings model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading HuggingFace embeddings model on device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    logger.info("Embeddings model loaded.")
    return embeddings


class RAGChatbot:
    """Main RAG Chatbot class, orchestrating RAG, tools, and LLM interactions."""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        # DocumentProcessor will now handle multi-modal extraction
        self.document_processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
        self.llm = None
        # Initialize the core internet search client here, not the LangChain tool wrapper
        self.core_internet_search_client = CoreInternetSearch()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )
        self.agent_executor = None
        logger.info("RAGChatbot instance created.")

    def initialize_components(self, google_api_key: str) -> bool:
        """Initialize LLM, embeddings, and vector store."""
        logger.info("Initializing chatbot components...")
        try:
            self.embeddings = load_embeddings_model_cached()
            logger.info("Embeddings model successfully loaded.")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.7
            )
            logger.info("Google Generative AI LLM initialized.")

            if not os.path.exists(CHROMA_PERSIST_DIR):
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                logger.info(f"Created ChromaDB persistence directory: {CHROMA_PERSIST_DIR}")

            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name="documents"
                )
                count = self.vectorstore._collection.count()
                logger.info(f"Vector store initialized from existing collection. Document count: {count}")
            except Exception as chroma_error:
                logger.warning(
                    f"ChromaDB direct initialization failed: {chroma_error}. Attempting persistent client fallback.")
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                # It's safer to get_or_create_collection to ensure it exists
                collection = client.get_or_create_collection("documents", embedding_function=self.embeddings)
                self.vectorstore = Chroma(
                    client=client,
                    collection_name="documents",
                    embedding_function=self.embeddings  # Still pass here for LangChain wrapper
                )
                count = self.vectorstore._collection.count()
                logger.info(f"ChromaDB initialized via PersistentClient fallback. Document count: {count}")

            if self.vectorstore is None:
                raise Exception("Vector store failed to initialize after all attempts.")

            logger.info("Chatbot components initialized successfully.")
            return True

        except Exception as e:
            logger.critical(f"Error during chatbot initialization: {e}", exc_info=True)
            self.embeddings = None
            self.llm = None
            self.vectorstore = None
            return False

    def setup_agent(self, user_type: str = "non-support"):
        """Setup LangChain agent with tools."""
        if not self.llm or not self.vectorstore:
            logger.error("LLM or Vectorstore not initialized. Cannot set up agent.")
            return

        logger.info(f"Setting up LangChain agent for user_type: {user_type}")

        # Instantiate core components that tools will use
        retriever_logic = RAGRetriever(self.vectorstore, self.llm, user_type)

        # self.core_internet_search_client is already initialized in __init__

        # Define a helper to pass chat history to stateful tools
        def get_current_chat_history() -> List[Any]:
            return self.memory.load_memory_variables({})['chat_history']

        # Instantiate LangChain Tool wrappers
        rag_tool_instance = RAGTool(retriever_logic)
        internet_search_tool_instance = AgentInternetSearchTool(self.core_internet_search_client)
        default_response_tool_instance = DefaultResponseTool()
        ticket_tool_instance = TicketCreationTool(self.llm, chat_history_provider=get_current_chat_history)

        tools = [
            rag_tool_instance.get_tool(),
            #internet_search_tool_instance.get_tool(),
            default_response_tool_instance.get_tool(),
            ticket_tool_instance.get_tool()
        ]
        logger.debug(f"Registered tools: {[tool.name for tool in tools]}")

        # The prompt remains the same, as it refers to tool names, not their internal functions
        # I've slightly adjusted the RAG tool's instruction based on the multi-modal content
        prompt = PromptTemplate.from_template("""
                        You are an AI assistant that helps users by answering questions based on documents.
                        You use a ReAct agent with the following tools:
                        Your name is Aviator
                        - rag_retriever: Use this tool to search uploaded documents for specific information. Don't summarize always, try to give the exactly mention in the documents always means response in proper structure data and also maintain the markdown as if source content has. response content can be large. Don't say page number, Source like Story.pdf and Category in final answer. 
                            sometimes you may need to use rag_retriever tool multiple times to get more details of each section or topics of previous results to provide more accurate answer to user.
                        - default_response: Use this tool for greetings or casual conversation (e.g., "hi", "hello", "thank you"). When this tool is used, its output is often the final answer.
                        - create_support_ticket: Use this tool when the user explicitly asks to 'create a ticket', 'raise an issue', 'open a support request', or book a ticket or book an issue or want to have a ticket or issue or similar. This tool will manage the collection of required details.

                        IMPORTANT RULES FOR RESPONDING:
                        - You must follow the ReAct format strictly. Each response must contain either:
                          A) A Thought, then an Action and Action Input, IF you need to use a tool.
                          B) A Thought, then a Final Answer, IF you have a complete answer to the user's request WITHOUT needing further tool use.
                        - YOU MUST NOT output both Action/Action Input AND Final Answer in the same response. Choose one path.
                        - ONLY use Final Answer when your response is fully complete and directly answers the user's last input.
                        - If the user's input is a simple greeting (like "hi", "hello"), a thank you, or other casual conversational remark, **you must use the `default_response` tool**. The `Observation` from this tool will typically be your `Final Answer` in the *next turn*.
                        - For any factual or information-seeking question, prioritize `rag_retriever` first. Don't summarize the content if the content is structured data like list of points or tabular data or actual format of lines.
                        - If the user expresses an intent to create a ticket, regardless of whether they provide details upfront, you must use the `create_support_ticket` tool. This tool will then handle extracting information and asking for missing details.
                        - **When a tool (especially `create_support_ticket` or `default_response`) provides a complete, user-facing answer in its `Observation`, your *next step* should be `Final Answer: <Observation Content>`. Do NOT attempt to take another Action if the tool's output is meant to be the final response to the user's current request.**

                        EXAMPLES:

                        User: Hello!
                        Thought: The user greeted me. I should use the default_response tool for a friendly greeting.
                        Action: default_response
                        Action Input: hello
                        Observation: Hi there! How can I help you today?
                        Final Answer: Hi there! How can I help you today?

                        User: What is the AIC policy?
                        Thought: The user is asking a question about a specific policy, which is likely in the documents. I should use the rag_retriever tool.
                        Action: rag_retriever
                        Action Input: What is the AIC policy?
                        Observation: Context from documents: ... (document content) ...
                        Thought: I have retrieved information about the AIC policy. I should now provide a summary as the final answer.
                        Final Answer: Based on the AIC policy document, ... (summarized answer) ...

                        User: I want to create a ticket.
                        Thought: The user explicitly stated intent to create a ticket. I should use the create_support_ticket tool to begin the process of gathering details.
                        Action: create_support_ticket
                        Action Input: I want to create a ticket.
                        Observation: To create the ticket, I need a few more details: department (IT, HR, or PS), issue severity (High, Medium, or Low), impacted client name, impacted time (e.g., 'yesterday', 'today 2 PM'). Please provide them.
                        Final Answer: To create the ticket, I need a few more details: department (IT, HR, or PS), issue severity (High, Medium, or Low), impacted client name, impacted time (e.g., 'yesterday', 'today 2 PM'). Please provide them.

                        User: Can you raise an issue? It's for HR, high severity, for John Doe, and it happened yesterday.
                        Thought: The user wants to raise an issue and provided all necessary details. I should use the create_support_ticket tool to create the ticket.
                        Action: create_support_ticket
                        Action Input: Can you raise an issue? It's for HR, high severity, for John Doe, and it happened yesterday.
                        Observation: Ticket **TICKET-XXXX** created successfully with the following details: ... Is there anything else I can help you with?
                        Final Answer: Ticket **TICKET-XXXX** created successfully with the following details: ... Is there anything else I can help you with?

                        Available tools:
                        {tools}

                        Tool names: {tool_names}

                        Previous conversation:
                        {chat_history}

                        Human: {input}

                        Thought: {agent_scratchpad}
                        """)

        self.agent_executor = AgentExecutor(
            agent=create_react_agent(self.llm, tools, prompt),
            tools=tools,
            memory=self.memory,
            verbose=True,  # Keep verbose for debugging in console/logs
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=120
        )
        logger.info("LangChain agent executor setup complete.")

    def add_document(self, file, category: str, description: str, status: str, access: str) -> bool:
        """Add document to vector store, now handling multi-modal content and batching."""
        if self.vectorstore is None or self.embeddings is None:
            logger.error("Chatbot components (vector store or embeddings) are not initialized for document addition.")
            return False

        tmp_file_path = None
        try:
            logger.info(f"Attempting to add document: {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            logger.debug(f"File '{file.name}' saved temporarily to {tmp_file_path}")

            file_type = file.name.split('.')[-1].lower()

            extracted_elements = self.document_processor.load_document(tmp_file_path, file_type)
            if not extracted_elements:
                logger.warning(f"Could not extract any elements from document: {file.name}")
                return False
            logger.debug(f"Extracted {len(extracted_elements)} elements from {file.name}.")

            common_metadata = {
                "filename": file.name,
                "category": category,
                "description": description,
                "status": status,
                "access": access,
                "document_id": str(uuid.uuid4())
            }

            documents = self.document_processor.chunk_documents(extracted_elements, common_metadata)
            if not documents:
                logger.warning(f"No chunks created from document: {file.name}")
                return False
            logger.info(f"Created {len(documents)} chunks for {file.name}.")

            # --- Start NEW: Batching logic for ChromaDB ---
            chroma_client = self.vectorstore._client
            max_chroma_batch_size = DEFAULT_CHROMA_MAX_BATCH_SIZE  # Use the default constant

            try:
                # Attempt to get the max_batch_size from the client if available
                if hasattr(chroma_client, 'max_batch_size'):
                    max_chroma_batch_size = chroma_client.max_batch_size
                    logger.info(f"Retrieved ChromaDB max_batch_size dynamically: {max_chroma_batch_size}")
                else:
                    logger.warning(
                        f"ChromaDB client does not expose 'max_batch_size'. Using default: {DEFAULT_CHROMA_MAX_BATCH_SIZE}")
            except Exception as e:
                logger.warning(
                    f"Error retrieving ChromaDB max_batch_size dynamically ({e}). Using default: {DEFAULT_CHROMA_MAX_BATCH_SIZE}")

            # Ensure the batch size is not too large
            # Use the dynamically retrieved size if successful, otherwise the default.
            batch_size = max_chroma_batch_size  # Set batch_size to the determined max_chroma_batch_size

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Adding batch {i // batch_size + 1} of {len(batch)} chunks for '{file.name}' to ChromaDB.")
                self.vectorstore.add_documents(batch)

            self.vectorstore.persist()  # Persist once all batches are added
            logger.info(f"Successfully added all chunks of '{file.name}' to vector database.")
            # --- End NEW: Batching logic ---

            return True

        except Exception as e:
            logger.error(f"Error adding document '{file.name}': {e}", exc_info=True)
            return False
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                logger.debug(f"Temporary file {tmp_file_path} cleaned up.")

    def chat(self, message: str) -> str:
        """Process chat message."""
        if not self.agent_executor:
            logger.error("Chatbot agent not initialized. Cannot process message.")
            return "Please initialize the chatbot first!"

        logger.info(f"Processing chat message: '{message}'")
        try:
            response = self.agent_executor.invoke({"input": message})
            output = response.get("output", "Sorry, I couldn't process your request.")
            logger.info(f"Chat message processed. Response: {output[:100]}...")
            return output
        except Exception as e:
            logger.error(f"Error processing chat message '{message}': {e}", exc_info=True)
            return f"Error processing message: {str(e)}"