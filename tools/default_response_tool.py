import logging
from typing import Any
from langchain_core.tools import Tool

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

class DefaultResponseTool:
    """
    A LangChain Tool for handling general greetings or casual conversation.
    It provides a direct, friendly response without needing complex processing.
    """
    def __init__(self):
        logger.info("DefaultResponseTool initialized.")

    def _default_response_action(self, query: str = "") -> str:
        """
        Handles general greetings or casual conversation.
        This is the actual function that will be called by the LangChain agent.
        The `query` parameter is optional as it might not be strictly needed for greetings.

        Args:
            query: The user's input (ignored for generic greetings).
        Returns:
            A fixed friendly response.
        """
        logger.info(f"DefaultResponseTool: Executing _default_response_action for query: '{query[:50]}...'")
        return "Hi there! How can I help you today?"

    def get_tool(self) -> Tool:
        """
        Returns the LangChain Tool instance for default responses.
        """
        return Tool(
            name="default_response",
            func=self._default_response_action,
            description="Use this when the user says hello, hi, thank you, or any casual greeting or closing. It provides a direct friendly response."
        )