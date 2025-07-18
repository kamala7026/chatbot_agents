import streamlit as st
import logging
import uuid
import time # Import time for simulating delays

# The content_renderer import is removed as the file is deleted.

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

# Initialize session state variables at the top level to prevent KeyErrors
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_message" not in st.session_state:
    st.session_state.processing_message = False
if "current_user_input" not in st.session_state:
    st.session_state.current_user_input = None # Not strictly used in this non-streaming test, but kept for consistency
if "prompt_to_process" not in st.session_state:
    st.session_state.prompt_to_process = None # This will hold the prompt for the loading dots phase
# Assuming initialized is set elsewhere, or set it here for testing:
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# --- Dummy Chatbot (NON-STREAMING for this test) ---
class NonStreamingDummyChatbot:
    def chat(self, user_input):
        logger.info(f"NonStreamingDummyChatbot received: {user_input}")
        # Simulate a blocking "thinking" time here.
        # The app will freeze for 2 seconds, during which the loading dots should be visible.
        time.sleep(2)
        return f"Hello! I received your message: '{user_input}'. This is a static response after thinking for a bit."

    def clear(self):
        logger.info("NonStreamingDummyChatbot memory cleared.")
        pass

# Instantiate the dummy chatbot for testing
non_streaming_chatbot = NonStreamingDummyChatbot()


def render_chat_ui(chatbot_instance): # Renamed 'chatbot' to 'chatbot_instance' for clarity
    """Renders the chat interface without streaming, focusing on loading icon display."""

    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    header_cols = st.columns([0.1, 0.8, 0.1])
    with header_cols[0]:
        st.markdown('<span style="font-size: 2em; color: #6495ED;">✈️</span>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<h3 style="margin:0; padding:0; color:#333;">Aviator</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.9em; color:#666; margin:0; padding:0;">General Knowledge</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border: 0.5px solid #eee; margin: 10px 0;">', unsafe_allow_html=True)

    if not st.session_state.get("initialized", False):
        st.warning("Please initialize the chatbot using the sidebar first.")
        logger.info("Chat UI not rendered: Chatbot not initialized.")
        return

    chat_messages_container = st.container(height=400, border=True)

    with chat_messages_container:
        # Display all COMPLETED messages from chat history directly
        for message in st.session_state.chat_history:
            avatar_icon = "👤" if message["role"] == "user" else "✈️"
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"])

        # --- MODIFIED CSS FOR ALIGNMENT ---
        if st.session_state.get("processing_message", False) and st.session_state.get("prompt_to_process"):
            # Display loading dots first
            with st.chat_message("assistant", avatar="✈️"):
                st.markdown("""
                    <div class="loading-dots-container">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                """, unsafe_allow_html=True)

            # --- Critical Change for Non-Streaming Test ---
            # Call the chatbot.chat method. Since it's now non-streaming and contains time.sleep(2),
            # the app will freeze here for 2 seconds, during which the loading dots should be visible.
            full_response = chatbot_instance.chat(st.session_state.prompt_to_process)
            
            # To fix the single-line issue, we forcefully insert newlines before each bullet.
            # The '■ ' with a space is important to avoid breaking words that might contain the symbol.
            sanitized_response = full_response.replace('■ ', '\n\n* ').replace('• ', '\n\n* ')

            st.session_state.chat_history.append({"role": "assistant", "content": sanitized_response})
            logger.info("Chatbot non-streaming response completed and added to history.")

            # Reset processing flags and trigger rerun to show the final message
            st.session_state.processing_message = False
            st.session_state.prompt_to_process = None
            st.session_state.current_user_input = None # Reset for good measure
            st.rerun() # This will cause Streamlit to render the final response

        # --- REMOVED THE ELIF BLOCK FOR STREAMING ---
        # The entire elif block that handled current_user_input and streaming is gone.


    # Chat input and clear chat button
    prompt = st.chat_input("Ask me something...", disabled=st.session_state.get("processing_message", False))

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        if st.button("Clear Chat", key="clear_chat_button_main"):
            logger.info("Clear Chat History button clicked.")
            st.session_state.chat_history = []
            if chatbot_instance and hasattr(chatbot_instance, 'memory') and chatbot_instance.memory:
                chatbot_instance.memory.clear()
            elif chatbot_instance and hasattr(chatbot_instance, 'clear'):
                chatbot_instance.clear()

            st.session_state.processing_message = False
            st.session_state.current_user_input = None
            st.session_state.prompt_to_process = None
            st.rerun()
    with col2:
        pass

    # Logic to trigger message processing when prompt is entered
    if prompt and not st.session_state.get("processing_message", False):
        logger.info(f"User input received: '{prompt}'")
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.processing_message = True
        st.session_state.prompt_to_process = prompt # Store the prompt here
        st.session_state.current_user_input = None # Ensure this is None initially
        st.rerun() # Trigger a rerun to display the loading dots (first pass)

# --- Main execution ---
if __name__ == "__main__":
    # Use the non-streaming dummy chatbot for this test
    render_chat_ui(non_streaming_chatbot)