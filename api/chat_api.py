from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from typing import List, Dict, Any

# Import the configured logger
from utils.logger_config import logger

from core.chatbot import RAGChatbot
from core.services.history_service import HistoryService # Import the new manager
from core.services.feedback_service import get_feedback_service, FeedbackService
from api.dependencies import get_chatbot_logic, get_history_manager, get_feedback_service_dependency
from api.schemas import FeedbackRequest, FeedbackResponse, ChatRequest, ChatResponse



# --- API Router ---
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    chatbot: RAGChatbot = Depends(get_chatbot_logic),
    history_manager: HistoryService = Depends(get_history_manager)
):
    """
    Handles a user's chat message, gets a response, and saves the conversation.
    """
    logger.info(f"Chat request for user '{request.username}' in chat '{request.chat_id}'.")
    
    # If chat_id is not provided, create a new chat session
    if not request.chat_id:
        request.chat_id = history_manager.create_new_chat(request.username)
        if not request.chat_id:
            raise HTTPException(status_code=500, detail="Failed to create a new chat session.")

    # Load previous messages from the database
    chat_history = history_manager.get_messages_for_chat(request.chat_id)

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
        history_manager.add_message_to_history, 
        request.chat_id, 'user', request.user_input
    )
    background_tasks.add_task(
        history_manager.add_message_to_history,
        request.chat_id, 'assistant', response_text
    )

    # If it was the first message, add a task to update the title
    if is_first_message:
        def generate_and_update_title():
            assert request.chat_id is not None, "chat_id should not be None here"
            new_title = chatbot.generate_chat_title(request.user_input)
            history_manager.update_chat_title(request.chat_id, new_title)
        
        background_tasks.add_task(generate_and_update_title)

    # Return the response to the user immediately
    return ChatResponse(response=response_text, chat_id=request.chat_id)

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service_dependency)
):
    """
    Submit feedback (like/dislike) for an AI response.
    """
    logger.info(f"Feedback submission: {request.feedback_type} from user '{request.username}' for chat '{request.chat_id}'")
    
    # Validate feedback type
    if request.feedback_type not in ['liked', 'disliked']:
        raise HTTPException(status_code=400, detail="Invalid feedback type. Must be 'liked' or 'disliked'.")
    
    try:
        feedback_id = feedback_service.submit_feedback(
            username=request.username,
            chat_id=request.chat_id,
            message_index=request.message_index,
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            feedback_type=request.feedback_type
        )
        
        logger.info(f"Feedback submitted successfully with ID: {feedback_id}")
        
        return FeedbackResponse(
            message=f"Feedback '{request.feedback_type}' submitted successfully",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@router.get("/feedback/stats/{username}")
async def get_user_feedback_stats(
    username: str,
    feedback_service: FeedbackService = Depends(get_feedback_service_dependency)
):
    """
    Get feedback statistics for a specific user.
    """
    try:
        stats = feedback_service.get_feedback_stats(username)
        return {
            "username": username,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get feedback stats for user {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback statistics")

@router.get("/feedback/history/{username}")
async def get_user_feedback_history(
    username: str,
    page: int = 1,
    limit: int = 10,
    feedback_service: FeedbackService = Depends(get_feedback_service_dependency)
):
    """
    Get feedback history for a specific user with pagination.
    """
    try:
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 10
            
        offset = (page - 1) * limit
        feedback_history, total_count = feedback_service.get_user_feedback_history_paginated(username, offset, limit)
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "username": username,
            "feedback_history": feedback_history,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_items": total_count,
                "items_per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
    except Exception as e:
        logger.error(f"Failed to get feedback history for user {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback history") 