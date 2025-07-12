# main.py
import streamlit as st
import os
import logging

# Import refactored modules
from utils.logger_config import setup_logging, logger
from core.chatbot import RAGChatbot
from core.document_management import DocumentManager # Ensure this import is correct
from ui.sidebar_ui import render_sidebar
from ui.chat_ui import render_chat_ui
from ui.document_upload_ui import render_document_upload_ui
from ui.document_management_ui import render_document_management_ui
from ui.styles import inject_styles

# Set up logging for the application
setup_logging()

# Configure Streamlit page
st.set_page_config(
    page_title="Aviator Chatbot",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_styles()

def main():
    logger.info("Starting Aviator Chatbot application.")

    # Initialize session state variables if not already present
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        logger.info("Initialized RAGChatbot in session state.")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        logger.info("Chatbot initialization status set to False.")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Chat history initialized.")
    if 'current_user_type' not in st.session_state:
        st.session_state.current_user_type = "non-support"
        logger.info("Default user type set to 'non-support'.")
    if 'doc_manager' not in st.session_state:
        # THIS IS IMPORTANT: Ensure doc_manager is created here once per session
        st.session_state.doc_manager = DocumentManager()
        logger.info("Initialized DocumentManager in session state.")
    if 'pending_delete_doc_id' not in st.session_state:
        st.session_state.pending_delete_doc_id = None
    if 'processing_message' not in st.session_state:
        st.session_state.processing_message = False
    if 'current_user_input' not in st.session_state:
        st.session_state.current_user_input = None

    # Render sidebar UI
    render_sidebar(st.session_state.chatbot, st.session_state.doc_manager)
    st.markdown('<hr style="border: 0.5px solid #eee; margin: 10px 0;">', unsafe_allow_html=True)
    # Render main content tabs
    #tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Document Upload", "📋 Document Management"])
    render_chat_ui(st.session_state.chatbot)
    #with tab1:
        #render_document_management_ui(st.session_state.chatbot, st.session_state.doc_manager)
       # render_chat_ui(st.session_state.chatbot)
    #with tab2:
    render_document_upload_ui(st.session_state.chatbot, st.session_state.doc_manager)
   # with tab3:
      #  render_document_management_ui(st.session_state.chatbot, st.session_state.doc_manager)

    logger.debug("Main application loop completed.")

if __name__ == "__main__":
    main()