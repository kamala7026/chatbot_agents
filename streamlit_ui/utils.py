# streamlit_ui/utils.py
import streamlit as st
import uuid

def get_session_id() -> str:
    """
    Retrieves the session ID from Streamlit's session state.
    If it doesn't exist, it generates a new UUID, stores it, and returns it.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id 