# streamlit_ui/app.py
import streamlit as st
import sys
import os

# Add the project root to the Python path at the very beginning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from login_ui import render_login_ui # Import the login UI

# --- Page Configuration ---
st.set_page_config(
    page_title="Aviator Chat",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Function to load CSS ---
def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {file_name}")

# Load custom CSS - Updated to the new consolidated stylesheet
load_css("streamlit_ui/style.css")

# --- Main App Logic ---
def main():
    """Main function to run the Streamlit app."""
    # Load custom CSS
    # Load custom CSS

    # Initialize session state for login status if it doesn't exist
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # --- Login Gate ---
    if not st.session_state.get("logged_in"):
        render_login_ui()
        return # Stop execution if not logged in

    # --- Main Application UI ---
    st.sidebar.title("Control Center")
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
    if st.sidebar.button("Logout"):
        # Clear the entire session state on logout
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()

    # --- Page Navigation ---
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    
    if st.sidebar.button("💬 Chat", use_container_width=True):
        st.session_state.page = "Chat"
    if st.sidebar.button("📋 Document Management", use_container_width=True):
        st.session_state.page = "Document Management"
    if st.sidebar.button("📄 Document Upload", use_container_width=True):
        st.session_state.page = "Document Upload"

    # Render Page based on Session State
    if st.session_state.page == "Chat":
        from chat_ui import render_chat_ui
        render_chat_ui()
    elif st.session_state.page == "Document Management":
        from document_management_ui import render_document_management_ui
        render_document_management_ui()
    elif st.session_state.page == "Document Upload":
        from document_upload_ui import render_document_upload_ui
        render_document_upload_ui()

if __name__ == "__main__":
    main() 