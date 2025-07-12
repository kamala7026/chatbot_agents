import logging
import json
import uuid
from typing import Dict, Any, List, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage # Required for chat history type hinting

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
        # This dictionary holds the details gathered during the multi-turn interaction
        self.current_ticket_details: Dict[str, Any] = {}
        logger.info("TicketCreationTool initialized.")

        # Define a more specific prompt for the LLM to extract ticket details
        # This prompt is critical for the LLM's performance in extraction.
        self.extraction_prompt = PromptTemplate.from_template("""
        You are an AI assistant designed to extract information for creating a support ticket.
        You need to identify the following details from the user's conversation history or their current input.
        If a detail is not found or unclear, indicate 'N/A'.
        Be smart about inferring. For example, if the user mentions 'payroll issue', the department is likely HR.
        If they mention 'system down' or 'unable to log in', severity is likely High.
        Try to be as precise as possible, but use 'N/A' if truly no information can be inferred.

        Required fields:
        - department (Mandatory value and Allowed values: IT, HR, PS, Network Support, Payroll, General Support)
        - severity (Mandatory value and Allowed values: High, Medium, Low, Critical, Urgent)
        - client_name (Mandatory value: Name of the client or user impacted, e.g., 'John Doe', 'Acme Corp')
        - impacted_time (Mandatory value: Time when the issue occurred, e.g., 'yesterday', 'today 2 PM', 'last Monday', 'now', 'this morning')
        Rule:
         - If user tell like cancel creating ticket or ignore or stop during followup these above inputs then you have to cancel booking and return some response like Accepted your request and canceling booking ticket. You can ask for further any other help.
         - Any 'Required fields' should not be missed, you have to ask user one field value at a time.
         - Don't use previously already created ticket values. 
                    
        Conversation history (most recent last, up to 5 turns):
        {chat_history_str}

        Current User Input: {user_input}

        Extract the following as a JSON object. Ensure the JSON is perfectly valid.
        {{
            "department": "extracted value or N/A",
            "severity": "extracted value or N/A",
            "client_name": "extracted value or N/A",
            "impacted_time": "extracted value or N/A"
        }}
        """)


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
        relevant_history = [
            msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))
        ][-turn_limit*2:] if len(history) > turn_limit*2 else history

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
        Returns a list of required fields that are still missing or marked as 'N/A'
        in the current_ticket_details.
        """
        missing = [
            field for field in self.required_fields
            if not self.current_ticket_details.get(field) or self.current_ticket_details.get(field).strip().lower() == "n/a"
        ]
        logger.debug(f"Missing fields: {missing}")
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
            "department": "department (e.g., IT, HR, or PS)",
            "severity": "issue severity (e.g., High, Medium, or Low)",
            "client_name": "impacted client name",
            "impacted_time": "impacted time (e.g., 'yesterday', 'today 2 PM', 'last Monday')"
        }
        prompts = [field_map.get(field, field) for field in missing_fields] # Use .get with fallback
        response_message = f"To create the ticket, I need a few more details: **{', '.join(prompts)}**. Please provide them."
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

        # Always try to extract details from the current input and history
        extracted_from_llm = self._extract_details_with_llm(user_input)

        # Update `current_ticket_details` with newly extracted information.
        # This allows multi-turn collection and overwrites 'N/A' or old values.
        for field in self.required_fields:
            extracted_value = extracted_from_llm.get(field)
            if extracted_value and extracted_value.strip().lower() != "n/a":
                self.current_ticket_details[field] = extracted_value
            elif field not in self.current_ticket_details:
                # Initialize any unextracted required fields as 'N/A' to track them
                self.current_ticket_details[field] = "N/A"

        logger.debug(f"Current collected ticket details: {self.current_ticket_details}")

        # Check if all required fields are collected
        missing_fields = self._get_missing_fields()
        if not missing_fields:
            # All details collected, proceed to simulate ticket creation
            logger.info("All required ticket details collected. Simulating ticket creation.")
            return self._simulate_ticket_creation()
        else:
            # Still missing details, ask the user for them
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