# api/dependencies.py
import os
from fastapi import Request
from core.chatbot import RAGChatbot
from core.document_management import DocumentManager

# --- Dependency Injection Functions ---

def get_chatbot_logic(request: Request) -> RAGChatbot:
    """Dependency injector that provides the single RAGChatbot instance from the app state."""
    return request.app.state.chatbot_logic

def get_doc_manager_instance(request: Request) -> DocumentManager:
    """Dependency injector that provides the single DocumentManager instance from the app state."""
    return request.app.state.doc_manager 