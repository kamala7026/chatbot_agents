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
    if not username or not isinstance(username, str):
        logger.warning("get_history_list called with invalid username.")
        return []
    logger.info(f"Fetching chat history list for user: {username}")
    try:
        response = requests.get(f"{HISTORY_API}/user_history/{username}")
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

def send_chat_message(username: str, chat_id: str | None, user_input: str):
    """Sends a message to the chat API and returns the full response."""
    logger.info(f"Sending message for user: {username}, chat_id: {chat_id}, input: '{user_input[:50]}...'")
    try:
        payload = {"username": username, "user_input": user_input}
        if chat_id and chat_id != "new_chat":
            payload["chat_id"] = chat_id
            
        response = requests.post(f"{CHAT_API}/", json=payload)
        response.raise_for_status()
        
        api_response_data = response.json()
        
        # If this was a new chat, update the active_chat_id from the response
        if not chat_id or chat_id == "new_chat":
            new_chat_id = api_response_data.get("chat_id")
            if new_chat_id:
                st.session_state.active_chat_id = new_chat_id
                st.session_state.needs_history_refresh = True # Flag for refresh
                logger.info(f"New chat created. New Chat ID: {new_chat_id}. Flagging for history refresh.")

        api_response = api_response_data.get("response")
        logger.debug(f"Got API response for chat_id {st.session_state.active_chat_id}: '{api_response[:50]}...'")
        return api_response
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error sending message for {chat_id}: {e}", exc_info=True)
        st.error(f"Error sending message: {e}")
        return None

def handle_new_chat_response(full_response, message_index):
    """Handles the response for a new chat, updating messages and triggering a refresh."""
    # This function will be called after the response is streamed.
    # It finalizes the message and then triggers a rerun, which will then handle the history refresh.
    st.session_state.messages[message_index] = {"role": "assistant", "content": full_response}
    st.session_state.needs_history_refresh = True
    # No rerun here; the caller will handle it.

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

            # Only fetch history if it's not already in the session state.
            if "history_list" not in st.session_state:
                st.session_state.history_list = get_history_list(username)

            history_list = st.session_state.get("history_list", []) # Use .get for safety
            
            with st.container(height=400):
                if not history_list:
                    st.caption("No chat history yet.")
                else:
                    for history_item in history_list:
                        timestamp = history_item.get('timestamp', '').split('T')[0]
                        title = history_item.get('title', 'Chat')
                        button_label = f"📜 {title} ({timestamp})"
                        # Ensure the key is a string and unique
                        history_id = str(history_item.get('id', ''))
                        if not history_id:
                            continue
                        
                        if st.button(button_label, key=history_id, use_container_width=False):
                            logger.info(f"User '{username}' selected chat from history. ID: {history_id}")
                            st.session_state.active_chat_id = history_id
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
        st.session_state.history_list = get_history_list(username) # Refresh the list
        st.session_state.pop("needs_history_refresh", None) # Clear the flag
        logger.info("History list refreshed.")
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

                        if full_response is None:
                            # Error is already handled in send_chat_message, just stop the placeholder
                            placeholder.empty()
                            st.session_state.messages.pop() # Remove placeholder
                            st.rerun()
                            return
                        
                        def response_generator(response_str: str):
                            chunks = re.split(r'(\s+)', response_str)
                            for chunk in chunks:
                                yield chunk
                                time.sleep(0.02)
                        
                        streamed_response = placeholder.write_stream(response_generator(full_response))
                        
                        # Update the state with the final response
                        st.session_state.messages[i] = {"role": "assistant", "content": streamed_response}
                        
                        # The history refresh is now triggered at the top of the script run,
                        # so we no longer need the 'just_created_chat' flag here.
                        
                        st.rerun() # Rerun to finalize the display
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

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": LOADING_PLACEHOLDER})
            st.rerun()

def initialize_chat_session(username: str):
    """Initializes the chat session, loading the most recent chat or starting a new one."""
    logger.info(f"No active chat for '{username}'. Initializing chat session.")
    history_list = get_history_list(username)
    st.session_state.history_list = history_list

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