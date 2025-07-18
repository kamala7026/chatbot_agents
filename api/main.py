import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from dotenv import load_dotenv


# Import the configured logger
from utils.logger_config import logger, setup_logging

# Load environment variables from .env file
load_dotenv()

# Import the singleton instances
from api.dependencies import get_chatbot_logic, get_doc_manager_instance
from core.chatbot import RAGChatbot
from core.document_management import DocumentManager
from core.history_manager import HistoryManager # Import the new manager

# Import API routers
from api import chat_api, documents_api, history_api

# Initialize the History Database
HistoryManager.initialize_database()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Aviator Chatbot API - Multi-Session",
    description="A unified API for chat and document management with session handling.",
    version="1.1.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming requests, including method, path, and processing time."""
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request: {request.method} {request.url.path} from {client_host}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} for {request.url.path} processed in {process_time:.2f}ms")
    
    return response

# --- App Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the shared, stateless chatbot logic instance.
    """
    # Configure logging before any other operations
    setup_logging()
    
    logger.info("API server starting up...")
    
    # Create and initialize the single chatbot logic instance and doc manager
    app.state.chatbot_logic = RAGChatbot()
    app.state.doc_manager = DocumentManager()
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.critical("GOOGLE_API_KEY not set. Please create a .env file with the key.")
        raise RuntimeError("GOOGLE_API_KEY not set. Please create a .env file with the key.")
    
    # The key is now passed to the chatbot instance, which will store it for potential re-initialization.
    if not app.state.chatbot_logic.initialize_components(google_api_key=google_api_key):
        logger.critical("Failed to initialize chatbot components.")
        raise RuntimeError("Failed to initialize chatbot components.")
    
    logger.info("Chatbot logic initialized successfully.")
    logger.info("DocumentManager is ready (now fully stateless).")
    logger.info("API is ready to accept requests.")

# --- Include Routers ---
app.include_router(chat_api.router)
app.include_router(documents_api.router)
app.include_router(history_api.router)

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Aviator Chatbot API"} 