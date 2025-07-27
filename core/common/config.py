# core/config.py
"""
Centralized configuration for the core application logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- ChromaDB/Vectorstore Configuration ---
# The directory where the persistent ChromaDB vector store is saved.
CHROMA_PERSIST_DIR = "./data/chroma_db_docs"

# The name of the collection within ChromaDB where document chunks are stored.
COLLECTION_NAME = "chatbot_documents"

# --- Logging Configuration ---
# The directory where application logs will be stored.
# By default, this is a hidden directory in the user's home folder.
# This can be changed to any absolute path on the system (e.g., r"C:\Logs\Chatbot" or "/var/log/chatbot").
# LOG_DIR = Path.home() / ".chatbot_agents" / "logs" 
LOG_DIR = r"C:\logs"

# --- Chunking and Batching Configuration ---
# These values are loaded from the environment, with sensible defaults.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DEFAULT_CHROMA_MAX_BATCH_SIZE = int(os.getenv("DEFAULT_CHROMA_MAX_BATCH_SIZE", 100))
CHROMA_ANALYTICS = os.getenv("CHROMA_ANALYTICS", False)
# TAVILY_API_KEY should ideally be set in environment variables before running the app.
# For demonstration purposes, I'm leaving it here as in your original code.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING","postgresql+psycopg://postgres:admin@localhost:5432/vectordb")
#PGVECTOR_CONNECTION_STRING="postgresql+psycopg://postgres:admin@localhost:5432/vectordb"
# --- Application Settings ---
# Database
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pgvector").lower() # 'chroma' or 'pgvector'
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

# --- Document Metadata Options ---
# These lists are used for dropdowns in the UI and for metadata validation.
CATEGORIES = ["TGO", "LENS", "AO", "AIC"]
STATUS_OPTIONS = ["Active", "Inactive"]
ACCESS_OPTIONS = ["Internal", "External"]

# --- Adservice API Configuration ---
# The base URL for the external Lifecycles API endpoint.
ADSERVICE_API_URL = "http://qtotcdc.otxlab.net:8080/adservices/v1/search/lifecycles"
