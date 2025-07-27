"""
Core Data Models and Schemas

This module contains all Pydantic models and data structures used throughout
the core module of the Aviator Chatbot Backend.

Centralized location for:
- Service-specific data models
- Business logic schemas
- Internal data structures
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID

# --- Ticket Creation Models ---

class TicketDetails(BaseModel):
    """Data model for holding the details of a support ticket."""
    department: str = Field(..., description="The department for the ticket (e.g., IT, HR, PS).")
    severity: str = Field(..., description="The severity of the issue (e.g., High, Medium, Low).")
    client_name: str = Field(..., description="The name of the impacted client.")
    impacted_time: str = Field(..., description="The time the issue occurred (e.g., 'yesterday', 'today 2 PM').")

class ExtractionResponse(BaseModel):
    """Model for the data extracted by the LLM."""
    mentioned_fields: List[str] = Field(description="List of fields explicitly mentioned in the user's last message.")
    extracted_data: Dict[str, Any] = Field(description="The actual data extracted for the mentioned fields.")

# --- Document Service Models ---

class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    doc_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename of the document")
    upload_date: datetime = Field(default_factory=datetime.now, description="When the document was uploaded")
    file_type: str = Field(..., description="Type/extension of the document")
    size_bytes: int = Field(..., description="Size of the document in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# --- User Service Models ---

class UserCredentials(BaseModel):
    """Model for user authentication credentials."""
    username: str = Field(..., description="Username for authentication")
    password_hash: str = Field(..., description="Hashed password")
    salt: str = Field(..., description="Salt used for password hashing")

class UserProfile(BaseModel):
    """Model for user profile information."""
    user_id: UUID = Field(..., description="Unique identifier for the user")
    username: str = Field(..., description="Username")
    email: str = Field(None, description="User's email address")
    role: str = Field(..., description="User's role in the system")
    created_at: datetime = Field(default_factory=datetime.now, description="When the user was created")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

# --- Feedback Service Models ---

class FeedbackEntry(BaseModel):
    """Model for user feedback on AI responses."""
    feedback_id: UUID = Field(..., description="Unique identifier for the feedback")
    user_id: UUID = Field(..., description="User who provided the feedback")
    chat_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID that received feedback")
    feedback_type: str = Field(..., description="Type of feedback (like/dislike)")
    comment: Optional[str] = Field(None, description="Optional comment with the feedback")
    created_at: datetime = Field(default_factory=datetime.now, description="When the feedback was created")

# --- History Service Models ---

class ChatMessage(BaseModel):
    """Model for chat message history."""
    message_id: str = Field(..., description="Unique identifier for the message")
    chat_id: str = Field(..., description="Chat session ID")
    user_id: UUID = Field(..., description="User who sent/received the message")
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Role of the sender (user/assistant)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")

class ChatSession(BaseModel):
    """Model for chat session information."""
    chat_id: str = Field(..., description="Unique identifier for the chat session")
    user_id: UUID = Field(..., description="User who owns the session")
    title: Optional[str] = Field(None, description="Optional title for the chat session")
    created_at: datetime = Field(default_factory=datetime.now, description="When the session was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last activity in the session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata") 