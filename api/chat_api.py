from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the configured logger
from utils.logger_config import logger

from core.chatbot import RAGChatbot
from api.dependencies import get_chatbot_logic
from core.history_manager import HistoryManager # Import the new manager

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    username: str
    chat_id: str
    user_input: str

class ChatResponse(BaseModel):
    response: str

# --- API Router ---
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    chatbot: RAGChatbot = Depends(get_chatbot_logic)
):
    """
    Handles a user's chat message, gets a response, and saves the conversation.
    """
    logger.info(f"Chat request for user '{request.username}' in chat '{request.chat_id}'.")
    
    # Load previous messages from the database
    chat_history = HistoryManager.get_messages_for_chat(request.chat_id)

    # If this is the first message in the chat, generate a title in the background
    is_first_message = not chat_history

    # Get the response from the chatbot using the correct method name
    response_text = chatbot.chat(
        message=request.user_input,
        user_type="User", # Providing a default user_type
        chat_history=chat_history
    )

    if not response_text:
        raise HTTPException(status_code=500, detail="Failed to get a response from the chatbot.")

    # --- Background Tasks ---
    # Always save the user message and the assistant response
    background_tasks.add_task(
        HistoryManager.add_message_to_history, 
        request.chat_id, 'user', request.user_input
    )
    background_tasks.add_task(
        HistoryManager.add_message_to_history,
        request.chat_id, 'assistant', response_text
    )

    # If it was the first message, add a task to update the title
    if is_first_message:
        def generate_and_update_title():
            new_title = chatbot.generate_chat_title(request.user_input)
            HistoryManager.update_chat_title(request.chat_id, new_title)
        
        background_tasks.add_task(generate_and_update_title)

    # Return the response to the user immediately
    return ChatResponse(response=response_text) 