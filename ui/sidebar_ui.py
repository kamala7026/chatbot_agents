import streamlit as st
import logging

# Import the necessary classes from their respective modules
from core.chatbot import RAGChatbot
from core.document_management import DocumentManager # Assuming document_management.py is at the root

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

def render_sidebar(chatbot, doc_manager):
    """Renders the Streamlit sidebar for configuration."""
    with st.sidebar:
        st.header("Configuration")

        google_api_key = st.text_input("Google API Key", type="password")
        user_type = st.selectbox("User Type", ["support", "non-support"])

        # Check if Google API Key is provided
        if google_api_key:
            # Logic for initial chatbot initialization
            if not st.session_state.initialized:
                if st.button("🚀 Initialize Chatbot", type="primary"):
                    logger.info("Initialize Chatbot button clicked.")
                    with st.status("Initializing Chatbot Components...", expanded=True, state="running") as status:
                        try:
                            # Use the existing chatbot instance, it should be valid here (not None)
                            if chatbot.initialize_components(google_api_key):
                                status.write("✅ Chatbot components loaded.")
                                # Setup agent on the *existing* chatbot instance
                                chatbot.setup_agent(user_type)
                                st.session_state.initialized = True
                                st.session_state.current_user_type = user_type
                                status.update(label="Chatbot Ready!", state="complete", expanded=False)
                                st.balloons()
                                logger.info("Chatbot successfully initialized.")
                                st.rerun() # Rerun to update the main UI
                            else:
                                status.error("❌ Chatbot initialization failed.")
                                logger.error("Chatbot initialization failed via sidebar button.")
                        except Exception as e:
                            status.error(f"An unexpected error occurred during initialization: {e}")
                            logger.critical(f"Unexpected error during chatbot initialization: {e}", exc_info=True)
            # Logic for when chatbot is already initialized
            else:
                # Option to update user type (re-sets up agent with new type)
                if user_type != st.session_state.current_user_type:
                    if st.button("🔄 Update User Type"):
                        logger.info(f"Update User Type button clicked. New type: {user_type}")
                        # Ensure LLM is available before setting up agent
                        if chatbot.llm:
                            chatbot.setup_agent(user_type)
                            st.session_state.current_user_type = user_type
                            st.success(f"Updated to **{user_type}** user!")
                            logger.info(f"User type updated to {user_type}.")
                        else:
                            st.error("LLM not initialized. Please re-initialize chatbot.")
                            st.session_state.initialized = False # Force re-init if LLM is gone
                            logger.warning("Attempted to update user type but LLM was not initialized. Forcing re-initialization.")
                            st.rerun()

                # Option to re-initialize the entire chatbot (resets everything)
                if st.button("🔄 Re-initialize Chatbot"):
                    logger.info("Re-initialize Chatbot button clicked.")
                    st.session_state.initialized = False
                    # FIX: Create a fresh RAGChatbot instance instead of setting to None
                    st.session_state.chatbot = RAGChatbot()
                    st.session_state.chat_history = []
                    # Re-create DocumentManager to reflect potential underlying DB changes
                    st.session_state.doc_manager = DocumentManager()
                    # Reset data editor key counter if it exists (from previous attempts)
                    if 'doc_editor_key_counter' in st.session_state:
                         st.session_state.doc_editor_key_counter = 0

                    st.session_state.current_user_input = None
                    st.session_state.processing_message = False
                    logger.info("Chatbot state reset for re-initialization and new instance created.")
                    st.rerun() # Trigger a rerun to re-evaluate app state
        else:
            # Warning if API key is missing
            st.warning("⚠️ Please enter your Google API Key first!")
            logger.info("Google API Key not provided, prompting user.")

        # Display chatbot status based on session state
        if st.session_state.initialized:
            st.success("✅ Chatbot Ready!")
            st.info(f"👤 Current user: **{st.session_state.current_user_type}**")

            # Display Vector Database connection status and stats
            if chatbot.vectorstore:
                st.success("🗃️ Vector Database: Connected")
                st.subheader("Database Stats")
                try:
                    # Access collection count through the chatbot's vectorstore
                    count = chatbot.vectorstore._collection.count()
                    st.metric("Total Document Chunks", count)
                    if count > 0:
                        st.success("📚 Documents ready for queries!")
                    else:
                        st.info("📝 Upload documents to get started!")
                    logger.info(f"Sidebar: Vector store connected with {count} chunks.")
                except Exception as e:
                    st.warning(f"Stats unavailable: {str(e)}")
                    logger.error(f"Error getting database stats in sidebar: {e}")
            else:
               st.error("🗃️ Vector Database: Not Connected")
               st.error("Please re-initialize the chatbot!")
               logger.warning("Sidebar: Vector store not connected.")
        else:
            st.info("🤖 Chatbot not initialized")
            st.info("Enter your Google API key and click 'Initialize Chatbot'")
            logger.info("Sidebar: Chatbot not initialized yet.")