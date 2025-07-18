# streamlit_ui/document_management_ui.py
import streamlit as st
import requests
import pandas as pd
import time
import os
import sys
from config import API_BASE_URL
from utils.logger_config import logger # Import the configured logger

# Import centralized configuration
from core.config import CATEGORIES, STATUS_OPTIONS, ACCESS_OPTIONS

# --- Configuration ---
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/documents"

# --- API Helper Functions ---
def get_headers():
    """Returns the required headers with the session ID."""
    return {}

def get_all_documents():
    """Fetches all document metadata from the API."""
    logger.info("Requesting all document metadata from API.")
    try:
        response = requests.get(DOCUMENTS_ENDPOINT, headers=get_headers())
        response.raise_for_status()
        documents = response.json()
        logger.info(f"Successfully fetched metadata for {len(documents)} documents.")
        return documents
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error fetching documents: {e}", exc_info=True)
        st.error(f"Error fetching documents: {e}")
        return []

def update_document_metadata(doc_id: str, updates: dict):
    """Updates a document's metadata via the API."""
    logger.info(f"Requesting metadata update for doc_id '{doc_id}' with data: {updates}")
    try:
        # Corrected the endpoint to match the API definition
        response = requests.patch(f"{DOCUMENTS_ENDPOINT}/{doc_id}/metadata", json=updates, headers=get_headers())
        response.raise_for_status()
        logger.info(f"Successfully updated metadata for doc_id '{doc_id}'.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error updating document {doc_id}: {e}", exc_info=True)
        st.error(f"Failed to update document {doc_id}: {e}")
        return False

def delete_document(doc_id: str):
    """Deletes a document via the API."""
    logger.info(f"Requesting deletion of doc_id '{doc_id}'.")
    try:
        response = requests.delete(f"{DOCUMENTS_ENDPOINT}/{doc_id}", headers=get_headers())
        response.raise_for_status()
        logger.info(f"Successfully deleted doc_id '{doc_id}'.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error deleting document {doc_id}: {e}", exc_info=True)
        st.error(f"Failed to delete document {doc_id}: {e}")
        return False

# --- UI Rendering ---
@st.cache_data(ttl=60) # Cache the data for 60 seconds
def get_cached_documents():
    """Cached function to fetch documents to avoid constant API calls."""
    return get_all_documents()

def render_document_management_ui():
    """Renders the document management interface."""
    st.header("📋 Manage Documents")

    all_docs = get_cached_documents()

    if not all_docs:
        st.info("No documents found. Upload documents in the 'Document Upload' tab.")
        return

    docs_df = pd.DataFrame(all_docs)
    
    if 'category' not in docs_df.columns:
        docs_df['category'] = CATEGORIES[0] if CATEGORIES else "General"
    docs_df['category'] = docs_df['category'].fillna(CATEGORIES[0] if CATEGORIES else "General")
    
    # Use the DataFrame's hash as a key for session state to store the original DataFrame
    df_hash = hash(docs_df.to_json())
    if f"original_docs_df_{df_hash}" not in st.session_state:
        st.session_state[f"original_docs_df_{df_hash}"] = docs_df.copy()

    original_df = st.session_state[f"original_docs_df_{df_hash}"]

    docs_df['Delete?'] = False
    
    column_config = {
        "document_id": st.column_config.Column("ID", disabled=True),
        "filename": st.column_config.Column("Filename", disabled=True),
    }
    if 'description' in docs_df.columns:
        column_config['description'] = st.column_config.Column("Description", disabled=True)
    if 'total_chunks' in docs_df.columns:
        column_config['total_chunks'] = st.column_config.NumberColumn("Chunks", disabled=True)
    if 'access' in docs_df.columns:
        column_config['access'] = st.column_config.SelectboxColumn("Access", options=ACCESS_OPTIONS, required=True)
    if 'status' in docs_df.columns:
        column_config['status'] = st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, required=True)
    if 'category' in docs_df.columns:
        column_config['category'] = st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True)
    
    column_config["Delete?"] = st.column_config.CheckboxColumn("Delete?", default=False)

    edited_df = st.data_editor(
        docs_df,
        column_config=column_config,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key="doc_editor"
    )

    if st.button("Apply Changes", type="primary"):
        logger.info("'Apply Changes' button clicked in document management.")
        changes_applied = False
        with st.spinner("Applying changes..."):
            docs_to_delete = edited_df[edited_df['Delete?']]
            for _, row in docs_to_delete.iterrows():
                doc_id = str(row['document_id'])
                if delete_document(doc_id):
                    st.success(f"Deleted document: {row['filename']}")
                    changes_applied = True
                else:
                    st.error(f"Failed to delete: {row['filename']}")
            
            for index, edited_row in edited_df.iterrows():
                original_row_series = original_df[original_df['document_id'] == edited_row['document_id']]
                if original_row_series.empty:
                    continue
                original_row = original_row_series.iloc[0]

                updates = {}
                if 'access' in edited_row and edited_row['access'] != original_row.get('access'):
                    updates['access'] = edited_row['access']
                if 'status' in edited_row and edited_row['status'] != original_row.get('status'):
                    updates['status'] = edited_row['status']
                if 'category' in edited_row and edited_row['category'] != original_row.get('category'):
                    updates['category'] = edited_row['category']
                
                if updates:
                    doc_id = str(edited_row['document_id'])
                    if update_document_metadata(doc_id, updates):
                        st.success(f"Updated metadata for: {edited_row['filename']}")
                        changes_applied = True
                    else:
                        st.error(f"Failed to update: {edited_row['filename']}")
        
        if changes_applied:
            logger.info("Changes applied, clearing cache and rerunning.")
            get_cached_documents.clear() # Clear the cache to force a refresh
            st.rerun()
        else:
            st.info("No changes were detected.")

    if st.button("Refresh List"):
        logger.info("'Refresh List' button clicked.")
        get_cached_documents.clear() # Clear the cache to force a refresh
        st.rerun() 