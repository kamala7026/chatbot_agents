import logging
import os
from tavily import TavilyClient
from core.common.config import TAVILY_API_KEY

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

class InternetSearchService:
    """
    A tool for performing internet searches using the Tavily API.
    It can provide concise answers and lists of relevant source links.
    """
    def __init__(self):
        if not TAVILY_API_KEY:
            logger.critical("Tavily API Key (TAVILY_API_KEY) not found in environment variables.")
            raise ValueError("Tavily API Key (TAVILY_API_KEY) not found in environment variables. Get it from tavily.com")
        
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("InternetSearchTool initialized.")

    def search(self, query: str, num_results: int = 3) -> str:
        """
        Search the internet using Tavily API.
        Returns a concise answer and top N relevant source links.
        """
        logger.info(f"Performing internet search for query: '{query}' with {num_results} results.")
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results,
                include_answer=True,
                include_raw_content=False,
            )

            answer = response.get('answer', 'No concise answer found.')
            sources = response.get('results', [])
            formatted_sources = []
            if sources:
                for i, source in enumerate(sources):
                    title = source.get('title', 'N/A')
                    url = source.get('url', 'N/A')
                    formatted_sources.append(f"{i + 1}. {title}\n   URL: {url}\n")
            else:
                formatted_sources.append("No relevant sources found.")

            output = f"Tavily Answer: {answer}\n\nSources:\n" + "\n".join(formatted_sources)
            logger.info(f"Internet search for '{query}' successful.")
            logger.debug(f"Tavily output: {output[:200]}...") # Log first 200 chars
            return output

        except Exception as e:
            logger.error(f"Error performing internet search with Tavily for query '{query}': {e}", exc_info=True)
            return f"Error performing internet search with Tavily: {str(e)}"