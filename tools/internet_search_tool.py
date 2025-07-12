import logging
from typing import Any
from langchain_core.tools import Tool

# Import InternetSearchTool from core since it contains the actual search logic
from core.internet_search import InternetSearchTool as CoreInternetSearch

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

class InternetSearchTool:
    """
    A LangChain Tool wrapper for the CoreInternetSearch (Tavily).
    It encapsulates the logic for performing internet searches.
    """
    def __init__(self, core_internet_search: CoreInternetSearch):
        """
        Initializes the InternetSearchTool.

        Args:
            core_internet_search: An instance of the CoreInternetSearch class.
        """
        self.core_internet_search = core_internet_search
        logger.info("InternetSearchTool (wrapper) initialized.")

    def _perform_internet_search_action(self, query: str) -> str:
        """
        Perform an internet search for general knowledge using the underlying search client.
        This is the actual function that will be called by the LangChain agent.

        Args:
            query: The search query.
        Returns:
            A string containing the search results.
        """
        logger.info(f"InternetSearchTool: Executing _perform_internet_search_action with query: '{query}'")
        try:
            result = self.core_internet_search.search(query)
            logger.debug("InternetSearchTool: Internet search completed.")
            return result
        except Exception as e:
            logger.error(f"InternetSearchTool: Error during internet search for query '{query}': {e}", exc_info=True)
            return f"Error performing internet search: {str(e)}"

    def get_tool(self) -> Tool:
        """
        Returns the LangChain Tool instance for internet search.
        """
        return Tool(
            name="internet_search",
            func=self._perform_internet_search_action,
            description="Search the internet when information is not found in documents or for general knowledge questions."
        )