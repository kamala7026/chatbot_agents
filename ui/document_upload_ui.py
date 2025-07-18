import streamlit as st
import uuid
import logging

# Import centralized configuration for metadata options
from core.config import CATEGORIES, STATUS_OPTIONS, ACCESS_OPTIONS

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

def render_document_upload_ui(chatbot, doc_manager):
    """Renders the document upload interface."""
    st.header("📄 Document Upload")

    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the chatbot first!")
        st.info("Steps:\n1. Enter Google API Key in sidebar\n2. Select user type\n3. Click 'Initialize Chatbot'")
        logger.info("Document upload UI not rendered: Chatbot not initialized.")
        return
    elif chatbot.vectorstore is None:
        st.error("❌ Vector database not available!")
        st.info("Please re-initialize the chatbot using the sidebar.")
        logger.warning("Document upload UI: Vector database not available.")
        return

    with st.expander("🔍 Debug Information (Document Upload)"):
        st.write("Vectorstore object initialized:", chatbot.vectorstore is not None)
        st.write("Embeddings object initialized:", chatbot.embeddings is not None)
        try:
            count = chatbot.vectorstore._collection.count()
            st.write("Current document chunk count:", count)
        except Exception as e:
            st.write("Error getting count:", str(e))
        logger.debug("Displayed debug info for document upload.")

    with st.form("document_upload"):
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT (max 200MB)"
        )

        if uploaded_file:
            st.info(f"📁 Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
            logger.info(f"User selected file: {uploaded_file.name}")

        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("📂 Category", CATEGORIES, help="Document category")
            status = st.selectbox("🔄 Status", STATUS_OPTIONS, help="Active documents are searchable")
        with col2:
            access = st.selectbox("🔐 Access Level", ACCESS_OPTIONS,
                                  help="Internal: Support users only, External: All users")
            description = st.text_area("📝 Description",
                                       placeholder="Brief description of the document content...",
                                       help="This helps with document discovery")

        submitted = st.form_submit_button("🚀 Upload Document", type="primary")

        if submitted:
            logger.info("Document upload form submitted.")
            if uploaded_file is not None:
                if description.strip():
                    with st.status(f"Processing document: **{uploaded_file.name}**", expanded=True) as doc_status:
                        try:
                            doc_status.write("Starting document processing...")
                            success = chatbot.add_document(
                                uploaded_file, category, description, status, access
                            )
                            if success:
                                doc_status.update(label=f"Successfully added {uploaded_file.name}", state="complete", expanded=False)
                                st.session_state.doc_manager = doc_manager.__class__() # Refresh doc_manager
                                logger.info(f"Document {uploaded_file.name} successfully added.")
                                st.rerun()
                            else:
                                doc_status.update(label=f"Failed to add {uploaded_file.name}", state="error", expanded=True)
                                logger.error(f"Failed to add document {uploaded_file.name}.")
                        except Exception as e:
                            doc_status.error(f"An error occurred during upload: {e}")
                            doc_status.update(label=f"Failed to add {uploaded_file.name}", state="error", expanded=True)
                            logger.critical(f"Critical error during document upload of {uploaded_file.name}: {e}", exc_info=True)
                else:
                    st.error("❌ Please provide a description for the document.")
                    logger.warning("Document upload failed: No description provided.")
            else:
                st.error("❌ Please select a file to upload.")
                logger.warning("Document upload failed: No file selected.")