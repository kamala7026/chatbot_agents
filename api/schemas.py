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