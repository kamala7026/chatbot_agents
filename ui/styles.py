import streamlit as st
import os

def load_css(file_path):
    """Loads a CSS file and returns its content as a string."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"CSS file not found at {file_path}")
        return ""

def inject_styles():
    """
    Injects all custom CSS into the Streamlit application.
    This function now loads styles from ui/styles.css.
    """
    css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
    custom_css = load_css(css_file_path)
    
    if custom_css:
        st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)