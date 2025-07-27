# api/history_api.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from uuid import UUID

from core.services.history_service import HistoryService
from utils.logger_config import logger
from api.dependencies import get_history_manager

history_router = APIRouter()

@history_router.get("/user_history/{username}", response_model=List[Dict[str, Any]])
def get_user_history(username: str):
    """Retrieves the list of chat histories for a specific user."""
    logger.info(f"API request to get history list for user: {username}")
    history_list = HistoryService.get_history_list(username)
    if history_list is None:
        logger.error(f"History retrieval returned None for user: {username}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")
    return history_list

@history_router.get("/{username}/{chat_id}", response_model=List[Dict[str, Any]])
def get_chat_messages(username: str, chat_id: str):
    """Retrieves all messages for a specific chat session with feedback information."""
    try:
        # Validate that the provided chat_id is a valid UUID.
        UUID(chat_id)
    except ValueError:
        logger.warning(f"Invalid UUID format for chat_id: {chat_id}")
        raise HTTPException(status_code=400, detail=f"Invalid chat ID format. '{chat_id}' is not a valid UUID.")
    
    logger.info(f"API request to get messages for user '{username}', chat '{chat_id}' (with feedback).")
    messages = HistoryService.get_messages_for_chat(chat_id, username)
    if messages is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve messages for the chat session.")
    
    # Debug: Log message structure
    logger.info(f"Returning {len(messages)} messages. Sample message structure: {messages[0] if messages else 'No messages'}")
    for i, msg in enumerate(messages):
        if 'feedback' in msg:
            logger.info(f"Message {i}: role={msg['role']}, feedback={msg['feedback']}")
        else:
            logger.info(f"Message {i}: role={msg['role']}, NO FEEDBACK FIELD")
    
    return messages

@history_router.post("/new/{username}", response_model=Dict[str, str])
def create_new_chat_session(username: str):
    """Creates a new chat session for a user and returns the new chat_id."""
    logger.info(f"API request to create new chat for user: {username}")
    chat_id = HistoryService.create_new_chat(username)
    if not chat_id:
        logger.error(f"Failed to create new chat session for user: {username}")
        raise HTTPException(status_code=500, detail="Failed to create new chat session.")
    return {"chat_id": chat_id}

@history_router.post("/update_title")
def update_chat_title_api(chat_id: str, new_title: str):
    """Updates the title of a specific chat session."""
    try:
        UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format.")

    logger.info(f"API request to update title for chat_id: {chat_id}")
    success = HistoryService.update_chat_title(chat_id, new_title)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update chat title.")
    return {"message": "Chat title updated successfully."}