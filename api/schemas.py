from pydantic import BaseModel
from typing import Optional

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