from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any

from api.schemas import DocumentMetadata, DocumentMetadataUpdate
from core.services.document_service import DocumentService
from api.dependencies import get_doc_manager_instance

documents_router = APIRouter()

@documents_router.post("/upload", summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(...),
    category: str = Form(...),
    status: str = Form(...),
    access: str = Form(...),
    doc_manager: DocumentService = Depends(get_doc_manager_instance)
):
    """
    Uploads a document and its metadata.
    """
    try:
        content = await file.read()
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty.")

        metadata = {
            "description": description,
            "category": category,
            "status": status,
            "access": access
        }
        success = doc_manager.add_document(
            file_contents=content,
            filename=filename,
            metadata=metadata
        )
        if success:
            return {"message": "Document uploaded successfully", "filename": file.filename}
        else:
            raise HTTPException(status_code=500, detail="Failed to process and add document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@documents_router.get("/", response_model=Dict[str, Any], summary="Get all documents with pagination")
async def get_all_documents(
    page: int = 1,
    limit: int = 10,
    doc_manager: DocumentService = Depends(get_doc_manager_instance)
):
    """
    Retrieves metadata for all documents in the vector store with pagination.
    """
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 10
        
    offset = (page - 1) * limit
    documents, total_count = doc_manager.get_all_documents_paginated(offset, limit)
    
    total_pages = (total_count + limit - 1) // limit
    
    return {
        "documents": documents,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total_count,
            "items_per_page": limit,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
    }

@documents_router.delete("/{document_id}", summary="Delete a document")
async def delete_document(
    document_id: str,
    doc_manager: DocumentService = Depends(get_doc_manager_instance)
):
    """
    Deletes a document from the vector store by its ID.
    """
    if doc_manager.delete_document(document_id):
        return {"message": f"Document {document_id} deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")

@documents_router.patch("/{document_id}", summary="Update document metadata")
async def update_document_metadata(
    document_id: str,
    update_data: DocumentMetadataUpdate,
    doc_manager: DocumentService = Depends(get_doc_manager_instance)
):
    """
    Updates the metadata for a specific document.
    """
    if doc_manager.update_document_metadata(document_id, update_data.dict(exclude_unset=True)):
        return {"message": f"Metadata for document {document_id} updated successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.") 