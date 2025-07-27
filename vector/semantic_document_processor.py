import PyPDF2
import docx
import logging
from typing import List, Dict, Any
from datetime import datetime

# LangChain components for semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

# Import the shared embedding model instance
from vector.embedding_manager import get_embeddings_instance

# Initialize logger
logger = logging.getLogger("aviator_chatbot")


class SemanticDocumentProcessor:
    """
    Handle document loading and processing with semantic chunking.
    This processor uses an embedding model to split text based on semantic meaning,
    which can produce more coherent and contextually relevant chunks.
    """

    def __init__(self):
        """
        Initializes the SemanticDocumentProcessor.
        It retrieves a shared instance of the embeddings model and sets up the
        SemanticChunker.
        """
        embeddings = get_embeddings_instance()
        if not embeddings:
            logger.critical("SemanticDocumentProcessor: Could not get embeddings instance. Semantic chunking will fail.")
            raise RuntimeError("Failed to initialize embeddings for SemanticDocumentProcessor.")

        # Using the percentile threshold can be more robust to different document types.
        # It determines splits based on the distribution of semantic similarity scores.
        self.text_splitter = SemanticChunker(
            embeddings, breakpoint_threshold_type="percentile"
        )
        logger.info("SemanticDocumentProcessor initialized with SemanticChunker.")

    def load_document(self, file_path: str, file_type: str) -> List[str]:
        """Load document based on file type."""
        logger.info(f"Loading document: {file_path} (Type: {file_type})")

        # --- FIX: Ensure extracted content is always valid before returning ---
        raw_text_content = []
        if file_type == "pdf":
            raw_text_content = self._load_pdf(file_path)
        elif file_type == "docx":
            raw_text_content = self._load_docx(file_path)
        elif file_type == "txt":
            raw_text_content = self._load_txt(file_path)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        # Filter out any empty or whitespace-only strings from the loaded content
        valid_text_content = [t for t in raw_text_content if t and t.strip()]
        if not valid_text_content:
            logger.warning(f"No valid text content extracted from {file_path}. Returning empty list.")
        return valid_text_content

    def _load_pdf(self, file_path: str) -> List[str]:
        """Load PDF document."""
        try:
            full_text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    # --- FIX: Only add text if it's not empty or just whitespace ---
                    if page_text and page_text.strip():
                        # Optionally add page numbers for context, even in this simple setup
                        # text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
                        full_text += page_text.strip() + "\n"  # Continue with original style, just stripped
                logger.info(f"Successfully loaded PDF: {file_path}")
            # --- FIX: Return empty list if no meaningful text was extracted ---
            return [full_text] if full_text.strip() else []
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

    def _load_docx(self, file_path: str) -> List[str]:
        """Load DOCX document."""
        try:
            doc = docx.Document(file_path)
            full_text = ""
            for paragraph in doc.paragraphs:
                # --- FIX: Only add text if it's not empty or just whitespace ---
                if paragraph.text and paragraph.text.strip():
                    full_text += paragraph.text.strip() + "\n"
            logger.info(f"Successfully loaded DOCX: {file_path}")
            # --- FIX: Return empty list if no meaningful text was extracted ---
            return [full_text] if full_text.strip() else []
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise

    def _load_txt(self, file_path: str) -> List[str]:
        """Load TXT document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully loaded TXT: {file_path}")
            # --- FIX: Return empty list if no meaningful text was extracted ---
            return [text] if text.strip() else []
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise

    def chunk_documents(self, texts: List[str], metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk documents using the SemanticChunker and enrich with metadata.
        
        The SemanticChunker's `create_documents` method is used here, which directly
        returns Langchain `Document` objects. This method then iterates through
        these chunks to add the necessary document-level and chunk-specific metadata.
        """
        all_docs = []
        # The SemanticChunker works on a list of raw texts.
        # It's often best to pass each document's full text as a single element.
        for text_content in texts:
            if not text_content or not text_content.strip():
                logger.warning("Skipping empty or whitespace-only text input for chunking.")
                continue

            # `create_documents` returns a list of Document objects, each representing a chunk.
            chunks = self.text_splitter.create_documents([text_content])
            logger.info(f"Semantically chunked a text block into {len(chunks)} pieces.")

            for i, chunk_doc in enumerate(chunks):
                # The chunk_doc already has page_content. We just need to add/update metadata.
                if not chunk_doc.page_content or not chunk_doc.page_content.strip():
                    logger.debug(f"Skipping empty or whitespace-only chunk (index: {i}).")
                    continue
                
                # Merge the base metadata with chunk-specific info
                final_metadata = metadata.copy()
                final_metadata['chunk_id'] = i
                final_metadata['timestamp'] = datetime.now().isoformat()
                
                # If the chunker added any of its own metadata, preserve it.
                chunk_doc.metadata.update(final_metadata)
                
                # Ensure content is stripped of leading/trailing whitespace
                chunk_doc.page_content = chunk_doc.page_content.strip()
                
                all_docs.append(chunk_doc)

        logger.info(f"Total {len(all_docs)} document chunks created with metadata.")
        return all_docs 