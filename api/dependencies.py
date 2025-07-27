"""
Dependencies for the FastAPI application.
These functions provide singleton instances for dependency injection.
"""

from core.chatbot import RAGChatbot
from core.services.document_service import DocumentService
from core.services.history_service import HistoryService
from core.services.feedback_service import get_feedback_service, FeedbackService

# Singleton instances
_chatbot_instance = None
_doc_manager_instance = None

def get_chatbot_logic() -> RAGChatbot:
    """
    Returns the singleton instance of the RAGChatbot.
    """
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot()
    return _chatbot_instance

def get_doc_manager_instance() -> DocumentService:
    """
    Returns the singleton instance of the DocumentService.
    """
    global _doc_manager_instance
    if _doc_manager_instance is None:
        _doc_manager_instance = DocumentService()
    return _doc_manager_instance

def get_history_manager() -> HistoryService:
    """
    Returns the singleton instance of the HistoryService.
    """
    return HistoryService()

def get_feedback_service_dependency() -> FeedbackService:
    """
    Returns the singleton instance of the FeedbackService.
    """
    return get_feedback_service() 