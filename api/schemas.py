from pydantic import BaseModel
from typing import Optional, Dict, Any

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    description: str
    status: str
    access: str
    category: str
    total_chunks: int

class DocumentMetadataUpdate(BaseModel):
    """Schema for updating document metadata. All fields are optional."""
    description: Optional[str] = None
    status: Optional[str] = None
    access: Optional[str] = None
    category: Optional[str] = None

class StandardResponse(BaseModel):
    """A standard response model for simple success messages."""
    message: str

class FeedbackRequest(BaseModel):
    """Schema for message feedback (like/dislike)."""
    username: str
    chat_id: str
    message_index: int
    user_message: str
    assistant_message: str
    feedback_type: str  # 'liked' or 'disliked'

class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    message: str
    feedback_id: str

# --- Authentication Schemas ---

class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Response model for successful login."""
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None

class UserInfoResponse(BaseModel):
    """Response model for user information."""
    id: int
    username: str
    user_type: str

# --- Chat Schemas ---

class ChatRequest(BaseModel):
    """Request model for chat messages."""
    username: str
    chat_id: str | None = None
    user_input: str

class ChatResponse(BaseModel):
    """Response model for chat messages."""
    response: str
    chat_id: str 