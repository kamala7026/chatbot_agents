import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from dotenv import load_dotenv


# Import the configured logger
from utils.logger_config import logger, setup_logging

# --- Pre-computation and Setup ---
# Set up logging first to capture messages from all modules
setup_logging()

# Load environment variables from .env file
load_dotenv()

from .dependencies import get_chatbot_logic, get_doc_manager_instance
from core.services.history_service import HistoryService
# No longer need to import these here, they are managed by the dependency injectors
# from core.chatbot import RAGChatbot
# from core.services.document_service import DocumentService
# from core.services.history_service import HistoryService # Import the new manager

# Import API routers
from . import chat_api, documents_api, history_api, auth_api

# Initialize the History Database
HistoryService.initialize_database()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Chatbot API",
    description="API for the AI Chatbot with document management and chat history.",
    version="1.0.0"
)

# --- Middleware ---
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the shared, stateless chatbot logic instance.
    """
    logger.info("API server starting up...")
    
    # Get the singleton instance of the chatbot
    chatbot = get_chatbot_logic()
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.critical("GOOGLE_API_KEY not set. Please create a .env file with the key.")
    else:
        # Initialize the components of the singleton instance
        if not chatbot.initialize_components(google_api_key):
            logger.critical("Failed to initialize chatbot components. The application may not function correctly.")
            # Depending on requirements, you might want to raise a RuntimeError here
            # raise RuntimeError("Failed to initialize chatbot components.")
    
    logger.info("Chatbot logic initialized via singleton.")
    logger.info("DocumentService is ready (managed by singleton).")
    logger.info("API is ready to accept requests.")

# --- Include Routers ---
app.include_router(chat_api.router, prefix="/chat", tags=["Chat"])
app.include_router(documents_api.documents_router, prefix="/documents", tags=["Documents"])
app.include_router(history_api.history_router, prefix="/history", tags=["History"])
app.include_router(auth_api.auth_router, tags=["Authentication"])


# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Aviator Chatbot API"} 