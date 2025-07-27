# streamlit_ui/document_upload_ui.py
import streamlit as st
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .config import API_BASE_URL
from utils.logger_config import logger # Import the configured logger

# Import centralized configuration for metadata options
from core.common.config import CATEGORIES, STATUS_OPTIONS, ACCESS_OPTIONS


# --- Configuration ---
UPLOAD_ENDPOINT = f"{API_BASE_URL}/documents/upload"
SUCCESS_MESSAGE = "Document uploaded successfully!"
FAILURE_MESSAGE = "Failed to upload document. Please check the logs."

# --- API Helper Functions ---
def get_headers():
    """Returns the required headers for API calls."""
    return {}

# The get_db_overview function is removed as it's no longer needed.

def upload_document(file, description, category, status, access):
    """Uploads a document to the backend."""
    logger.info(f"Preparing to upload document '{file.name}'.")
    form_data = {
        'description': description,
        'category': category,
        'status': status,
        'access': access
    }
    try:
        with st.spinner(f"Uploading {file.name}..."):
            response = requests.post(
                UPLOAD_ENDPOINT,
                files={'file': (file.name, file.getvalue(), file.type)},
                data=form_data,
                headers=get_headers()
            )
            response.raise_for_status()
            logger.info(f"Successfully uploaded document '{file.name}'.")
            return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error during file upload: {e}", exc_info=True)
        st.error(f"Error uploading file: {e}")
        return None

# --- UI Rendering ---
def render_document_upload_ui():
    """Renders the document upload interface."""
    st.header("📤 Document Upload")
    
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'txt', 'md'],
            help="Upload a PDF, TXT, or Markdown file."
        )
        
        st.markdown("---")
        st.subheader("Document Metadata")

        description = st.text_input("Description", help="A brief summary of the document's content.")
        category = st.selectbox("Category", options=CATEGORIES, help="Select the document category.")
        status = st.selectbox("Status", options=STATUS_OPTIONS, help="Set the initial status.")
        access = st.selectbox("Access Level", options=ACCESS_OPTIONS, help="Set the access level.")
        
        submitted = st.form_submit_button("Upload Document")

        if submitted:
            if uploaded_file is not None and description:
                result = upload_document(uploaded_file, description, category, status, access)
                if result:
                    st.success("✅ Document uploaded successfully!")
                    logger.info("Document upload form submitted and processed successfully.")
                    # The new caching mechanism in the management UI makes this redundant.
                else:
                    st.error("❌ Failed to upload document.")
            else:
                st.warning("⚠️ Please select a file and provide a description.")
                logger.warning("Upload form submitted but file or description was missing.")

    # The Database Overview section has been removed. 