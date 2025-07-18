# streamlit_ui/chat_ui.py
import streamlit as st
import requests
import time
import re
import os
import sys
from config import API_BASE_URL
from utils.logger_config import logger # Import the configured logger

# --- Configuration ---
HISTORY_API = f"{API_BASE_URL}/history"
CHAT_API = f"{API_BASE_URL}/chat"

# --- API Helper Functions ---
def get_history_list(username: str):
    """Fetches the list of chat histories for a user."""
    logger.info(f"Fetching chat history list for user: {username}")
    try:
        response = requests.get(f"{HISTORY_API}/{username}")
        response.raise_for_status()
        history = response.json()
        logger.debug(f"Successfully fetched {len(history)} history items for {username}.")
        return history
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error fetching history list for {username}: {e}", exc_info=True)
        st.error(f"Error fetching history list: {e}")
        return []

def get_messages(username: str, chat_id: str):
    """Fetches the messages for a specific chat."""
    logger.info(f"Fetching messages for user: {username}, chat_id: {chat_id}")
    try:
        response = requests.get(f"{HISTORY_API}/{username}/{chat_id}")
        response.raise_for_status()
        messages = response.json()
        logger.debug(f"Successfully fetched {len(messages)} messages for chat_id: {chat_id}")
        return messages
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error fetching messages for {chat_id}: {e}", exc_info=True)
        st.error(f"Error fetching messages: {e}")
        return []

def create_new_chat(username: str):
    """Asks the API to create a new chat thread."""
    logger.info(f"Requesting new chat for user: {username}")
    try:
        response = requests.post(f"{HISTORY_API}/new/{username}")
        response.raise_for_status()
        chat_id = response.json().get("chat_id")
        logger.info(f"Successfully created new chat with chat_id: {chat_id} for user: {username}")
        return chat_id
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error creating new chat for {username}: {e}", exc_info=True)
        st.error(f"Error creating new chat: {e}")
        return None

def send_chat_message(username: str, chat_id: str, user_input: str):
    """Sends a message to the chat API and returns the full response."""
    logger.info(f"Sending message for user: {username}, chat_id: {chat_id}, input: '{user_input[:50]}...'")
    try:
        payload = {"username": username, "chat_id": chat_id, "user_input": user_input}
        response = requests.post(CHAT_API, json=payload)
        response.raise_for_status()
        api_response = response.json().get("response")
        logger.debug(f"Got API response for chat_id {chat_id}: '{api_response[:50]}...'")
        return api_response
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error sending message for {chat_id}: {e}", exc_info=True)
        st.error(f"Error sending message: {e}")
        return "Sorry, an error occurred while communicating with the server."

# --- UI Rendering ---
def render_history_panel(username: str, container):
    """Renders the chat history in a styled, right-hand panel."""
    with container:
        with st.container(border=True):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown('<h4>Chat History</h4>', unsafe_allow_html=True)
            with col2:
                if st.button("➕", key="new_chat_icon", help="Start a new chat", use_container_width=True):
                    logger.info(f"User '{username}' clicked 'New Chat' icon.")
                    st.session_state.active_chat_id = "new_chat"
                    st.session_state.messages = []
                    st.rerun()

            st.divider()

            if "history_list" not in st.session_state:
                st.session_state.history_list = get_history_list(username)

            history_list = st.session_state.history_list
            
            with st.container(height=400):
                if not history_list:
                    st.caption("No chat history yet.")
                else:
                    for history_item in history_list:
                        timestamp = history_item.get('timestamp', '').split('T')[0]
                        title = history_item.get('title', 'Chat')
                        button_label = f"📜 {title} ({timestamp})"
                        if st.button(button_label, key=history_item['id'], use_container_width=False):
                            logger.info(f"User '{username}' selected chat from history. ID: {history_item['id']}")
                            st.session_state.active_chat_id = history_item['id']
                            st.session_state.messages = get_messages(username, st.session_state.active_chat_id)
                            st.rerun()

def render_chat_ui():
    """Renders the main chat interface, including messages and input form."""
    logger.info("Rendering the main chat UI.")
    username = st.session_state.get("username")
    if not username:
        st.warning("Please log in to use the chat.")
        return

    # --- Initialization on first load ---
    if "active_chat_id" not in st.session_state:
        initialize_chat_session(username)

    # After a response is streamed, a flag is set. On the next run, refresh history if needed.
    if st.session_state.get("needs_history_refresh"):
        if "history_list" in st.session_state:
            del st.session_state.history_list
        del st.session_state.needs_history_refresh
        logger.info("History list invalidated for refresh.")
        st.rerun()

    # Hide Streamlit's default UI elements
    st.markdown("""
        <style>
            .st-emotion-cache-16txtl3 {
                display: none;
            }
            .st-emotion-cache-h4xjwg {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Main Layout ---
    st.title("Aviator Chat")
    
    col1, col2 = st.columns([0.7, 0.3])

    # --- Render History Panel first ---
    render_history_panel(username, col2)

    with col1:
        # This container will hold the chat messages
        chat_container = st.container(height=500, border=True)

        # --- Callback for feedback buttons ---
        def handle_feedback(message_index, feedback_type):
            """Updates the feedback state for a given message."""
            message = st.session_state.messages[message_index]
            logger.info(f"User '{username}' provided feedback ('{feedback_type}') for message index {message_index} in chat {st.session_state.active_chat_id}.")
            if message.get('feedback') == feedback_type:
                message['feedback'] = None
            else:
                message['feedback'] = feedback_type

        LOADING_ANIMATION_HTML = """
        <div class="loading-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
        """
        LOADING_PLACEHOLDER = "‎" * 3 

        # --- Display existing messages ---
        with chat_container:
            for i, message in enumerate(st.session_state.get("messages", [])):
                avatar = "👤" if message["role"] == "user" else "✈️"
                with st.chat_message(message["role"], avatar=avatar):
                    # If this is the last message and it's a placeholder, stream the response here.
                    if message["content"] == LOADING_PLACEHOLDER and i == len(st.session_state.messages) - 1:
                        placeholder = st.empty()
                        placeholder.markdown(LOADING_ANIMATION_HTML, unsafe_allow_html=True)
                        
                        user_prompt = st.session_state.messages[i-1]["content"]
                        full_response = send_chat_message(username, st.session_state.active_chat_id, user_prompt)

                        def response_generator(response_str: str):
                            chunks = re.split(r'(\s+)', response_str)
                            for chunk in chunks:
                                yield chunk
                                time.sleep(0.02)
                        
                        streamed_response = placeholder.write_stream(response_generator(full_response))
                        
                        # Update the state with the final response
                        st.session_state.messages[i] = {"role": "assistant", "content": streamed_response}
                        
                        # Set flag to refresh history on the next run if needed
                        if st.session_state.get("just_created_chat"):
                            st.session_state.needs_history_refresh = True
                            del st.session_state.just_created_chat
                        
                        st.rerun() # Rerun to finalize the display and potentially trigger history refresh
                    else:
                        st.markdown(message["content"])
                        if message["role"] == "assistant":
                            if "feedback" not in message:
                                message["feedback"] = None
                            btn_col1, btn_col2, _ = st.columns([0.07, 0.07, 0.86])
                            with btn_col1:
                                is_liked = message["feedback"] == "liked"
                                st.button("👍", key=f"like_{i}", type="primary" if is_liked else "secondary", on_click=handle_feedback, args=(i, "liked"))
                            with btn_col2:
                                is_disliked = message["feedback"] == "disliked"
                                st.button("👎", key=f"dislike_{i}", type="primary" if is_disliked else "secondary", on_click=handle_feedback, args=(i, "disliked"))

        # --- Handle new input ---
        if prompt := st.chat_input("Ask me something..."):
            logger.info(f"User '{username}' submitted a new prompt.")

            if st.session_state.get("active_chat_id") == "new_chat":
                logger.info("This is the first message in a new chat. Creating session on the backend.")
                new_chat_id = create_new_chat(username)
                if new_chat_id:
                    st.session_state.active_chat_id = new_chat_id
                    st.session_state.just_created_chat = True
                else:
                    st.error("Could not create a new chat session. Please try again.")
                    st.stop()

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": LOADING_PLACEHOLDER})
            st.rerun()

def initialize_chat_session(username: str):
    """Initializes the chat session, loading the most recent chat or starting a new one."""
    logger.info(f"No active chat for '{username}'. Initializing chat session.")
    history_list = get_history_list(username)
    if history_list:
        # If there's history, load the most recent chat
        st.session_state.active_chat_id = history_list[0]['id']
        st.session_state.messages = get_messages(username, st.session_state.active_chat_id)
    else:
        # If no history, prepare a new transient chat screen
        st.session_state.active_chat_id = "new_chat"
        st.session_state.messages = []
    logger.info(f"Initialized UI with active_chat_id: {st.session_state.active_chat_id}")
    st.rerun()