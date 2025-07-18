# streamlit_ui/login_ui.py
import streamlit as st

# --- Hardcoded Credentials ---
# In a real application, this would be a database call.
VALID_CREDENTIALS = {
    "testuser": "password123",
    "user2": "abc",
    "kamala": "admin"
}

def render_login_ui():
    """Renders the login interface and handles authentication."""
    st.title("Welcome to Aviator Chat")
    #st.subheader("Please log in to continue")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun() # Rerun to show the main app
            else:
                st.error("Invalid username or password") 