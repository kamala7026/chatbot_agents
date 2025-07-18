# api/history_api.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from core.history_manager import HistoryManager
from utils.logger_config import logger

router = APIRouter(
    prefix="/history",
    tags=["Chat History"],
)

@router.get("/{username}", response_model=List[Dict[str, Any]])
def get_user_history(username: str):
    """Retrieves the list of chat histories for a specific user."""
    logger.info(f"API request to get history list for user: {username}")
    history_list = HistoryManager.get_history_list(username)
    if history_list is None:
        logger.error(f"History retrieval returned None for user: {username}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")
    return history_list

@router.get("/{username}/{chat_id}", response_model=List[Dict[str, Any]])
def get_chat_messages(username: str, chat_id: str):
    """Retrieves all messages for a specific chat session."""
    logger.info(f"API request to get messages for user '{username}', chat '{chat_id}'.")
    # Note: We don't strictly need the username for this query if chat_ids are unique,
    # but it's good practice for authorization in the future.
    messages = HistoryManager.get_messages_for_chat(chat_id)
    if messages is None:
        # This case handles potential DB errors, though get_messages_for_chat returns [] on error.
        raise HTTPException(status_code=500, detail="Failed to retrieve messages for the chat session.")
    return messages

@router.post("/new/{username}", response_model=Dict[str, str])
def create_new_chat_session(username: str):
    """Creates a new chat session for a user and returns the new chat_id."""
    logger.info(f"API request to create new chat for user: {username}")
    chat_id = HistoryManager.create_new_chat(username)
    if not chat_id:
        logger.error(f"Failed to create new chat session for user: {username}")
        raise HTTPException(status_code=500, detail="Failed to create new chat session.")
    return {"chat_id": chat_id}

@router.post("/update_title")
def update_chat_title_api(chat_id: str, new_title: str):
    """Updates the title of a specific chat session."""
    logger.info(f"API request to update title for chat_id: {chat_id}")
    success = HistoryManager.update_chat_title(chat_id, new_title)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update chat title.")
    return {"message": "Chat title updated successfully."} 