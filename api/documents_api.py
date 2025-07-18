from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Optional

from .schemas import DocumentMetadata, StandardResponse, DocumentMetadataUpdate
from api.dependencies import get_doc_manager_instance
from utils.logger_config import logger
from core.document_management import DocumentManager
from vector.chromavector_manager import get_vectorstore_chunk_count

router = APIRouter(
    prefix="/documents",
    tags=["Document Management"],
)

@router.post("/upload", response_model=StandardResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(...),
    category: str = Form(...),
    status: str = Form(...),
    access: str = Form(...),
    doc_manager: DocumentManager = Depends(get_doc_manager_instance)
):
    """
    Uploads a document and adds it to the vector store.
    """
    if not doc_manager:
        raise HTTPException(status_code=503, detail="Document manager not available.")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    metadata = {
        "description": description,
        "category": category,
        "status": status,
        "access": access
    }
    
    success = doc_manager.add_document(file.filename, file.file, metadata)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process and add document.")
        
    return StandardResponse(message="Document uploaded and processed successfully.")

@router.get("/", response_model=List[DocumentMetadata])
def get_all_documents(
    doc_manager: DocumentManager = Depends(get_doc_manager_instance)
):
    """
    Retrieves metadata for all documents in the vector store.
    """
    try:
        # Log the total number of chunks to help diagnose empty responses
        total_chunks = get_vectorstore_chunk_count()
        logger.info(f"Total chunks in vector store: {total_chunks}")

        docs = doc_manager.get_all_documents_metadata()
        if docs is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve documents from the vector store.")
        
        logger.info(f"Retrieved metadata for {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving all document metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}", response_model=StandardResponse)
def delete_document_by_id(
    document_id: str,
    doc_manager: DocumentManager = Depends(get_doc_manager_instance)
):
    """Deletes a document and all its associated chunks from the vector store."""
    logger.info(f"API request to delete document with ID: {document_id}")
    if not doc_manager.delete_document(document_id):
        logger.error(f"API delete failed for document ID: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found or could not be deleted.")
    return StandardResponse(message="Document deleted successfully.")

@router.patch("/{document_id}/metadata", response_model=StandardResponse)
def update_document_metadata_by_id(
    document_id: str,
    metadata_update: DocumentMetadataUpdate,
    doc_manager: DocumentManager = Depends(get_doc_manager_instance)
):
    """Updates the metadata for a specific document."""
    logger.info(f"API request to update metadata for document ID: {document_id}")
    updates = metadata_update.dict(exclude_unset=True)
    if not doc_manager.update_document_metadata(document_id, updates):
        logger.error(f"API metadata update failed for document ID: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found or metadata could not be updated.")
    return StandardResponse(message="Document metadata updated successfully.") 