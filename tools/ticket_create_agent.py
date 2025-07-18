import logging
import json
import uuid
from typing import Dict, Any, List, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage # Required for chat history type hinting

# Import the new prompt manager
from core.prompt_manager import PromptManager

# Initialize logger for this module
logger = logging.getLogger("aviator_chatbot")

class TicketCreationTool:
    """
    A LangChain tool for creating support tickets.
    It's designed to be stateful within the agent's memory for multi-turn conversations.
    It gathers necessary information (department, severity, client name, impacted time)
    and simulates ticket creation.
    """
    def __init__(self, llm: BaseChatModel, chat_history_provider: Callable[[], List[Any]]):
        """
        Initializes the TicketCreationTool.

        Args:
            llm: The Language Model to use for extracting details.
            chat_history_provider: A callable (function/method) that returns
                                   the current chat history as a list of LangChain messages.
        """
        self.llm = llm
        self.chat_history_provider = chat_history_provider
        # Define the fields required for a ticket. These are case-sensitive as used internally.
        self.required_fields = ["department", "severity", "client_name", "impacted_time"]
        
        # Define allowed values for specific fields to enable robust validation.
        # Values are lowercase for case-insensitive comparison.
        self.allowed_values = {
            "department": ["it", "hr", "ps", "network support", "payroll", "general support"],
            "severity": ["high", "medium", "low", "critical", "urgent"]
        }

        # This dictionary holds the details gathered during the multi-turn interaction
        self.current_ticket_details: Dict[str, Any] = {}
        
        # Get the prompt from the centralized manager
        self.extraction_prompt = PromptManager.get_ticket_extraction_prompt()
        logger.info("TicketCreationTool initialized.")

    def _get_chat_history_str(self, turn_limit: int = 5) -> str:
        """
        Retrieves and formats a limited portion of the chat history as a string
        to provide context to the LLM for detail extraction.

        Args:
            turn_limit: The number of recent conversational turns (human-AI pairs) to include.
        Returns:
            A string representation of the chat history.
        """
        history = self.chat_history_provider()
        formatted_history = []
        # Get last 'turn_limit' exchanges (each exchange is a human message + an AI message)
        # Ensure we only process actual HumanMessage and AIMessage instances
        # First, filter the history to get only the message types we care about.
        filtered_history = [
            msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))
        ]

        # Now, get the last 'turn_limit' exchanges. Python's slicing handles cases
        # where the list is shorter than the slice size by simply returning the entire list.
        relevant_history = filtered_history[-turn_limit * 2:]

        for msg in relevant_history:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"AI: {msg.content}")
        logger.debug(f"Formatted chat history for extraction: {formatted_history}")
        return "\n".join(formatted_history)

    def _extract_details_with_llm(self, user_input: str) -> Dict[str, str]:
        """
        Uses the LLM to extract ticket details from conversation history and current input.
        It's crucial that the LLM is instructed to return a valid JSON object.

        Args:
            user_input: The current input from the user.
        Returns:
            A dictionary containing extracted ticket details. Returns an empty dict on parsing failure.
        """
        chat_history_str = self._get_chat_history_str()
        llm_output_content = "" # Initialize for logging full output on error
        try:
            response_chain = self.extraction_prompt | self.llm
            llm_response = response_chain.invoke({
                "chat_history_str": chat_history_str,
                "user_input": user_input
            })
            llm_output_content = llm_response.content
            logger.debug(f"Raw LLM extraction output: {llm_output_content}")

            # Ensure the LLM output is a string before processing
            if not isinstance(llm_output_content, str):
                logger.error(f"LLM extraction returned a non-string value: {llm_output_content}")
                return {}

            # The LLM might include markdown (e.g., ```json\n...\n```), so strip it.
            json_str = llm_output_content.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "", 1)
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip() # Final strip after removing markdown

            extracted_data = json.loads(json_str)
            logger.info(f"Successfully extracted details with LLM: {extracted_data}")
            return extracted_data
        except json.JSONDecodeError as jde:
            logger.error(f"JSON parsing error after LLM extraction: {jde}. LLM Output was: {llm_output_content}", exc_info=True)
            return {} # Return empty on parsing failure
        except Exception as e:
            logger.error(f"Error during LLM detail extraction: {e}. LLM Output: {llm_output_content}", exc_info=True)
            return {} # Return empty on general failure

    def _get_missing_fields(self) -> List[str]:
        """
        Returns a list of required fields that are still missing or invalid.
        This now uses a robust validation logic based on allowed values.
        """
        missing = []
        for field in self.required_fields:
            value = self.current_ticket_details.get(field)
            is_missing = False

            # 1. Check if the value is fundamentally missing or an invalid placeholder.
            if not value or str(value).strip().lower() in ["n/a", "na", "unknown", "", "missing"]:
                is_missing = True
            else:
                clean_value = str(value).strip().lower()
                # 2. If the field has a predefined list of allowed values, validate against it.
                if field in self.allowed_values:
                    if clean_value not in self.allowed_values[field]:
                        logger.warning(f"Validation failed for field '{field}'. Value '{value}' not in allowed list: {self.allowed_values[field]}")
                        is_missing = True
                # 3. For free-text fields, perform a basic sanity check.
                elif isinstance(value, str) and len(value.strip()) < 2:
                    logger.warning(f"Validation failed for free-text field '{field}'. Value '{value}' is too short.")
                    is_missing = True
            
            if is_missing:
                missing.append(field)
        
        logger.debug(f"Checked for missing/invalid fields. Found: {missing}")
        return missing

    def _ask_for_missing_details(self, missing_fields: List[str]) -> str:
        """
        Constructs a user-friendly prompt to ask the user for specific missing details.

        Args:
            missing_fields: A list of field names that are still needed.
        Returns:
            A human-readable string asking for the missing information.
        """
        field_map = {
            "department": "department: e.g., IT, HR, or PS",
            "severity": "issue severity: e.g., High, Medium, or Low",
            "client_name": "impacted client name: e.g., Opentext",
            "impacted_time": "impacted time: e.g., 'yesterday', 'today 2 PM', 'last Monday'"
        }
        
        # Build the formatted string with each missing item on a new line
        prompts = [field_map.get(field, field) for field in missing_fields]
        # Use double newlines to ensure markdown renders separate paragraphs.
        response_message = "I need a few more details. Please provide them.\n\n" + "\n\n".join(prompts)
        
        logger.info(f"Asking user for missing ticket details: {missing_fields}")
        return response_message

    def _simulate_ticket_creation(self) -> str:
        """
        Simulates the ticket creation process with the gathered details.
        Generates a unique ticket ID and formats a confirmation message.
        Crucially, it clears `self.current_ticket_details` after successful "creation".

        Returns:
            A confirmation message with ticket details.
        """
        # Generate a unique and traceable ticket ID
        ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"

        details_list = []
        for k, v in self.current_ticket_details.items():
            if v and v.strip().lower() != "n/a":
                details_list.append(f"- {k.replace('_', ' ').title()}: **{v}**")

        details_str = "\n".join(details_list) if details_list else "No specific details captured."

        final_message = (f"Ticket **{ticket_id}** created successfully with the following details:\n"
                         f"{details_str}\n")

        logger.info(f"Ticket '{ticket_id}' simulated creation. Details: {self.current_ticket_details}")
        # Reset the state for the next ticket creation request
        self.current_ticket_details = {}
        logger.debug("Cleared current ticket details for next interaction.")
        return final_message

    def create_ticket_action(self, user_input: str) -> str:
        """
        The main callable function for the LangChain Tool.
        Manages the state of ticket creation, extracts info, and prompts or confirms.

        Args:
            user_input: The current input from the user.
        Returns:
            A message indicating whether more information is needed or if the ticket is created.
        """
        logger.info(f"TicketCreationTool invoked with user input: '{user_input}'")

        extraction_result = self._extract_details_with_llm(user_input)

        # Validate the structure of the LLM's response.
        if not isinstance(extraction_result, dict) or "mentioned_fields" not in extraction_result or "extracted_data" not in extraction_result:
            logger.warning("LLM extraction result was malformed or empty. Re-prompting for missing info.")
            return self._ask_for_missing_details(self._get_missing_fields())
        
        mentioned_fields = extraction_result.get("mentioned_fields", [])
        extracted_data = extraction_result.get("extracted_data", {})

        logger.info(f"LLM identified explicitly mentioned fields: {mentioned_fields}")

        # Intelligent Merge: Only update details for fields the LLM confirmed were in the LAST user message.
        for field_name in mentioned_fields:
            if field_name in self.required_fields:
                new_value = extracted_data.get(field_name)
                # Only update if the new value is a valid, non-empty string.
                if isinstance(new_value, str) and new_value.strip().lower() not in ["", "n/a"]:
                    self.current_ticket_details[field_name] = new_value
                    logger.info(f"State updated for '{field_name}' with new value: '{new_value}'")
        
        # Initialize any required fields that have not been collected yet.
        for field in self.required_fields:
            if field not in self.current_ticket_details:
                self.current_ticket_details[field] = 'N/A'

        logger.debug(f"Current collected ticket details after merge: {self.current_ticket_details}")

        # Check if all required fields are collected and valid.
        missing_fields = self._get_missing_fields()
        if not missing_fields:
            logger.info("All required ticket details collected. Simulating ticket creation.")
            return self._simulate_ticket_creation()
        else:
            logger.info(f"Ticket details incomplete. Asking for: {missing_fields}")
            return self._ask_for_missing_details(missing_fields)

    def get_tool(self) -> Tool:
        """
        Returns the LangChain Tool object that wraps the `create_ticket_action` method.
        """
        return Tool(
            name="create_support_ticket",
            func=self.create_ticket_action, # Use the renamed action method
            description=(
                "Use this tool when the user explicitly asks to 'create a ticket', 'raise an issue', "
                "'open a support request', or similar. This tool is designed for multi-turn conversations: "
                "it will guide the user to collect all necessary details (department, severity, client name, impacted time) "
                "and, once complete, will provide the final answer for ticket creation. "
                "The agent should NOT attempt to generate a Final Answer or another Action after this tool's output "
                "if the ticket is created successfully by the tool (i.e., when all information is gathered and the tool confirms ticket creation)."
            )
        )