import logging
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.schema import Document as LangchainDocument
from langchain_core.vectorstores import VectorStore # Use generic VectorStore

# Import the new prompt manager
from .prompt_manager import PromptManager

# Initialize logger
logger = logging.getLogger("aviator_chatbot")

class RAGRetriever:
    """Simple RAG retriever with metadata filtering and multi-query."""

    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, user_type: str = "non-support"):
        self.vectorstore = vectorstore
        self.user_type = user_type
        self.llm = llm
        # Get the prompt from the centralized manager
        self.query_gen_prompt = PromptManager.get_query_generation_prompt()
        logger.info(f"RAGRetriever initialized for user_type: {user_type}")

    def get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Retrieve relevant documents with metadata filtering and multi-query."""
        logger.info(f"Retrieving documents for query: '{query}' (User Type: {self.user_type})")
        query_variations = self._generate_query_variations(query)
        logger.debug(f"Generated query variations: {query_variations}")

        all_results = []
        seen_content = set()

        for q in query_variations:
            try:
                results = self.vectorstore.similarity_search(q, k=30)
                logger.debug(f"Found {len(results)} raw results for variation: '{q}'")
                filtered_results = self._filter_by_metadata(results)

                for doc in filtered_results:
                    if doc.page_content not in seen_content:
                        all_results.append(doc)
                        seen_content.add(doc.page_content)
            except Exception as e:
                logger.error(f"Error in similarity search for '{q}': {e}")
                continue

        final_results = all_results[:20]
        logger.info(f"Returning {len(final_results)} relevant unique documents.")
        return final_results

    def _generate_query_variations(self, original_query: str) -> List[str]:
        """Generate query variations for multi-query retrieval using the LLM."""
        try:
            query_generator_chain = self.query_gen_prompt | self.llm.bind(stop=["\n\n"]) | (lambda x: x.content.split('\n'))
            variations = query_generator_chain.invoke({"question": original_query})
            variations = [v.strip() for v in variations if v.strip()]
            if original_query not in variations:
                variations.insert(0, original_query)
            logger.debug(f"Query variations generated: {variations}")
            return variations
        except Exception as e:
            logger.warning(f"Error generating query variations: {e}. Falling back to original query only.")
            return [original_query]

    def _filter_by_metadata(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Filter documents based on metadata (status and access)."""
        filtered = []
        for doc in documents:
            metadata = doc.metadata
            doc_status = metadata.get("status", "").lower()
            doc_access = metadata.get("access", "").lower()

            if doc_status != "active":
                logger.debug(f"Filtering out document '{metadata.get('filename')}' due to inactive status ('{doc_status}').")
                continue

            if self.user_type == "non-support" and doc_access != "external":
                logger.debug(f"Filtering out document '{metadata.get('filename')}' for 'non-support' user due to internal access ('{doc_access}').")
                continue
            filtered.append(doc)
        logger.debug(f"Filtered {len(documents)} documents down to {len(filtered)} based on metadata for user_type '{self.user_type}'.")
        return filtered