import logging
import json
import uuid
from typing import Dict, Any, List, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.services.ticket_creation_service import TicketCreationService, ExtractionResponse
from utils.logger_config import logger

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

class TicketCreationTool:
    """
    A LangChain tool for creating support tickets. It manages the conversational
    flow and uses the TicketCreationService for core logic.
    """
    def __init__(self, llm: BaseChatModel, chat_history_provider: Callable[[], List[Any]]):
        self.chat_history_provider = chat_history_provider
        
        # --- FIX: The service requires an LLM that can handle structured output. ---
        if isinstance(llm, ChatGoogleGenerativeAI):
            # The service will internally create the structured output chain.
            self.service = TicketCreationService(llm)
        else:
            logger.error("The provided LLM is not a ChatGoogleGenerativeAI instance. Ticket creation may fail.")
            self.service = TicketCreationService(llm)

        self.required_fields = ["department", "severity", "client_name", "impacted_time"]
        self.current_ticket_details: Dict[str, Any] = {}
        logger.info("TicketCreationTool initialized.")

    def _get_chat_history_str(self, turn_limit: int = 5) -> str:
        """Formats a limited portion of the chat history for context."""
        history = self.chat_history_provider()
        filtered_history = [msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))]
        relevant_history = filtered_history[-turn_limit * 2:]
        return "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in relevant_history])

    def _ask_for_missing_details(self, missing_fields: List[str]) -> str:
        """Constructs a user-friendly prompt for missing details."""
        field_map = {
            "department": "department (e.g., IT, HR, PS)",
            "severity": "issue severity (e.g., High, Medium, Low)",
            "client_name": "impacted client name",
            "impacted_time": "impacted time (e.g., 'yesterday', 'today 2 PM')"
        }
        prompts = [field_map.get(field, field) for field in missing_fields]
        return "I need a few more details. Please provide:\n\n" + "\n\n".join(prompts)

    def _simulate_ticket_creation(self) -> str:
        """Simulates ticket creation and formats a confirmation message."""
        ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
        details_list = [f"- {k.replace('_', ' ').title()}: **{v}**" 
                        for k, v in self.current_ticket_details.items() if v and str(v).lower() != "n/a"]
        details_str = "\n".join(details_list)
        final_message = f"Ticket **{ticket_id}** created successfully with:\n{details_str}"
        
        logger.info(f"Ticket '{ticket_id}' simulated creation. Details: {self.current_ticket_details}")
        self.current_ticket_details = {}  # Reset state
        return final_message

    def create_ticket_action(self, user_input: str) -> str:
        """The main callable function for the LangChain Tool."""
        logger.info(f"TicketCreationTool invoked with user input: '{user_input}'")
        
        chat_history_str = self._get_chat_history_str()
        extraction_result = self.service.extract_details(user_input, chat_history_str)

        # The service returns a dict, so we access it with .get()
        mentioned_fields = extraction_result.get("mentioned_fields", [])
        extracted_data = extraction_result.get("extracted_data", {})

        for field_name in mentioned_fields:
            if field_name in self.required_fields and extracted_data.get(field_name):
                self.current_ticket_details[field_name] = extracted_data[field_name]
        
        for field in self.required_fields:
            if field not in self.current_ticket_details:
                self.current_ticket_details[field] = 'N/A'

        missing_fields = self.service.validate_fields(self.current_ticket_details)
        if not missing_fields:
            return self._simulate_ticket_creation()
        else:
            return self._ask_for_missing_details(missing_fields)

    def get_tool(self) -> Tool:
        """Returns the configured LangChain Tool object."""
        return Tool(
            name="create_support_ticket",
            func=self.create_ticket_action,
            description=(
                "Use this tool when the user explicitly asks to 'create a ticket', 'raise an issue', "
                "'open a support request', or similar. This tool will guide the user to collect all "
                "necessary details (department, severity, client name, impacted time) and confirm once complete."
            )
        )