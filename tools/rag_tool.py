import logging
from typing import List, Any
from langchain_core.tools import Tool
from langchain.schema import Document as LangchainDocument

# Import RAGRetriever from core since it's a core component for retrieval logic
from core.rag_retriever import RAGRetriever

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

class RAGTool:
    """
    A LangChain Tool wrapper for the RAGRetriever.
    It encapsulates the logic for searching information within uploaded documents.
    """
    def __init__(self, retriever: RAGRetriever):
        """
        Initializes the RAGTool.

        Args:
            retriever: An instance of RAGRetriever responsible for document fetching.
        """
        self.retriever = retriever
        logger.info("RAGTool initialized.")

    def _rag_search_action(self, query: str) -> str:
        """
        Search for information in uploaded documents using the RAGRetriever.
        This is the actual function that will be called by the LangChain agent.

        Args:
            query: The user's query or question.
        Returns:
            A string containing the retrieved context and sources, or a message indicating no documents were found.
        """
        logger.info(f"RAGTool: Executing rag_search_action with query: '{query}'")
        try:
            docs: List[LangchainDocument] = self.retriever.get_relevant_documents(query)
            if not docs:
                logger.info("RAGTool: No relevant documents found for the query.")
                return "No relevant documents found in the database."

            context_parts = []
            source_info_parts = set() # Use a set to avoid duplicate source entries if multiple chunks from same doc are returned

            for doc in docs:
                context_parts.append(doc.page_content)
                filename = doc.metadata.get('filename', 'Unknown')
                category = doc.metadata.get('category', 'Unknown')
                source_info_parts.add(f"Source: {filename} (Category: {category})")

            context = "\n\n".join(context_parts)
            source_info = "\n".join(sorted(list(source_info_parts))) # Sort for consistent output

            response = f"Context from documents:\n{context}\n\nSources:\n{source_info}"
            logger.debug(f"RAGTool: Successfully retrieved context from {len(docs)} document chunks.")
            return response
        except Exception as e:
            logger.error(f"RAGTool: Error during rag_search_action for query '{query}': {e}", exc_info=True)
            return f"Error searching documents: {str(e)}"

    def get_tool(self) -> Tool:
        """
        Returns the LangChain Tool instance for the RAG Retriever.
        """
        return Tool(
            name="rag_retriever",
            func=self._rag_search_action,
            description="Search for information in uploaded documents. Use this first for any query that is likely to be answered by documents."
        )