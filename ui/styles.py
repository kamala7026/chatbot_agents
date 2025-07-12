import streamlit as st

def inject_styles():
    """Injects custom CSS for Streamlit UI customization."""
    st.markdown("""
    <style>
/* General body and main content adjustments */
.main {
    padding-top: 20px; /* Adjust top padding */
    padding-left: 20px;
    padding-right: 20px;
    padding-bottom: 20px;
}

/* Specific styling for chat messages for a more compact look */
.stChatMessage {
    margin-bottom: 5px; /* Reduce space between messages */
    padding: 5px 10px; /* Reduce internal padding of message bubbles */
    border-radius: 10px; /* Slightly more rounded corners */
}

.stChatMessage [data-testid="chatMessageUser"] {
    background-color: #e0f2f7; /* Light blue for user messages */
    text-align: right; /* Align user message content to right */
    border-bottom-right-radius: 0; /* Make bottom-right corner sharp */
}

.stChatMessage [data-testid="chatMessageAssistant"] {
    background-color: #f0f2f6; /* Light grey for assistant messages */
    text-align: left; /* Align assistant message content to left */
    border-bottom-left-radius: 0; /* Make bottom-left corner sharp */
}

/* Avatar styling */
.stChatMessage [data-testid="stChatMessageAvatar"] {
    width: 30px; /* Smaller avatar size */
    height: 30px;
    font-size: 18px; /* Adjust font size for emojis */
    line-height: 30px; /* Center emoji vertically */
    border-radius: 50%; /* Make avatar perfectly round */
    background-color: #cce0ff; /* Light blue background for avatars */
    display: flex;
    justify-content: center;
    align-items: center;
    color: #333; /* Darker text for emojis */
}

/* Reduce padding around the chat input area */
.stChatInputContainer {
    padding: 10px 0;
}

/* Hide Streamlit header, footer, and default sidebar toggle for a cleaner look */
header.st-emotion-cache-zt5ig8 { /* Specific class for the top header */
    visibility: hidden;
    height: 0%;
}
.st-emotion-cache-cio0dv { /* Specific class for the main menu button */
    visibility: hidden;
}
.st-emotion-cache-ysnmbv { /* Specific class for footer */
    visibility: hidden;
}
/* Adjust main content padding when sidebar is hidden */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
}

/* Target the main app container to constrain its width */
.css-1r6dm1u { /* You might need to inspect this class name in your browser if it changes */
    max-width: 800px; /* Set a max width for the chat window, adjust as needed */
    margin-left: auto;
    margin-right: auto;
    border: 1px solid #ddd; /* Add a subtle border */
    box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Add a subtle shadow */
    border-radius: 10px; /* Rounded corners for the entire chat window */
    overflow: hidden; /* Hide overflow content from border-radius */
}

/* Sidebar styling for custom header */
[data-testid="stSidebarContent"] {
    padding-top: 20px; /* Add some padding at the top of the sidebar */
}

/* Adjust the chat input background */
div.st-emotion-cache-1c7y2kl:has(div.st-emotion-cache-fybftg) {
    background-color: #f0f2f6; /* Lighter background for the input area */
    border-radius: 15px; /* Rounded input field */
    border: 1px solid #ddd;
    padding: 8px 15px; /* Padding inside the input container */
}

/* Specific targeting for the text input within the chat input */
div.st-emotion-cache-1c7y2kl input[data-testid="stTextInput"] {
    background-color: transparent !important; /* Transparent background for the actual text input */
    border: none !important; /* Remove border */
    box-shadow: none !important; /* Remove shadow */
    padding: 0; /* Remove internal padding if any */
}

/* --- Loading Dots Animation --- */
.loading-dots-container {
    display: flex;
    align-items: center;
    justify-content: flex-start; /* Align dots to the left */
    padding: 5px 0; /* Adjust padding as needed */
    min-height: 25px; /* Ensure space for dots */
}

.loading-dot {
    width: 8px;
    height: 8px;
    background-color: #6495ED; /* Color of the dots */
    border-radius: 50%;
    margin: 0 3px; /* Space between dots */
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dot:nth-child(1) { animation-delay: -0.32s; }
.loading-dot:nth-child(2) { animation-delay: -0.16s; }
.loading-dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

</style>
""", unsafe_allow_html=True)