# ticket_agent.py
from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage # Required for chat history type hinting

class TicketCreationTool:
    """
    A tool to handle the creation of support tickets. It gathers necessary information
    (department, severity, client name, impacted time) and simulates ticket creation.
    """
    def __init__(self, llm: BaseChatModel, chat_history_provider):
        """
        Initializes the TicketCreationTool.

        Args:
            llm: The Language Model to use for extracting details.
            chat_history_provider: A callable (function/method) that returns
                                   the current chat history as a list of LangChain messages.
        """
        self.llm = llm
        self.chat_history_provider = chat_history_provider # Callback to get chat history
        self.required_fields = ["department", "severity", "client_name", "impacted_time"]
        self.current_ticket_details: Dict[str, Any] = {}
        # self.conversation_turn_started_ticket = 0 # Not directly used in the current logic, can be removed if not needed for future features

        # Define a more specific prompt for the LLM to extract ticket details
        self.extraction_prompt = PromptTemplate.from_template("""
        You are an AI assistant designed to extract information for creating a support ticket.
        You need to identify the following details from the user's conversation history or their current input:
        - department (e.g., IT, HR, PS, or a more specific team like "Network Support", "Payroll")
        - severity (e.g., High, Medium, Low, Critical, Urgent)
        - client_name (the name of the client or user impacted)
        - impacted_time (e.g., 'yesterday', 'today 2 PM', 'last Monday', 'now')

        If a detail is not found or unclear, indicate 'N/A'.
        Be smart about inferring. For example, if the user mentions 'payroll issue', the department is likely HR.
        If they mention 'system down' or 'unable to log in', severity is likely High.
        Try to be as precise as possible, but use 'N/A' if truly no information can be inferred.

        Conversation history (most recent last):
        {chat_history_str}

        Current User Input: {user_input}

        Extract the following as a JSON object:
        {{
            "department": "N/A or extracted value",
            "severity": "N/A or extracted value",
            "client_name": "N/A or extracted value",
            "impacted_time": "N/A or extracted value"
        }}
        """)

    def _get_chat_history_str(self, turn_limit: int = 5) -> str:
        """
        Retrieves and formats a limited portion of the chat history as a string.

        Args:
            turn_limit: The number of recent conversational turns (human-AI pairs) to include.
        Returns:
            A string representation of the chat history.
        """
        history = self.chat_history_provider()
        formatted_history = []
        # Get last 'turn_limit' exchanges (each exchange is a human message + an AI message)
        relevant_history = history[-turn_limit*2:] if len(history) > turn_limit*2 else history

        for msg in relevant_history:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"AI: {msg.content}")
        return "\n".join(formatted_history)

    def _extract_details_with_llm(self, user_input: str) -> Dict[str, str]:
        """
        Uses the LLM to extract ticket details from conversation history and current input.

        Args:
            user_input: The current input from the user.
        Returns:
            A dictionary containing extracted ticket details.
        """
        chat_history_str = self._get_chat_history_str()
        llm_output = "" # Initialize for error reporting
        try:
            response_chain = self.extraction_prompt | self.llm
            llm_output = response_chain.invoke({
                "chat_history_str": chat_history_str,
                "user_input": user_input
            }).content
            # Attempt to parse the JSON string from the LLM's output
            import json
            # The LLM might include markdown (e.g., ```json\n...\n```), so strip it.
            json_str = llm_output.strip().replace("```json\n", "").replace("\n```", "")
            return json.loads(json_str)
        except json.JSONDecodeError as jde:
            print(f"JSON parsing error: {jde}. LLM Output was: {llm_output}")
            return {} # Return empty on parsing failure
        except Exception as e:
            print(f"Error extracting details with LLM: {e}. LLM Output: {llm_output}")
            return {} # Return empty on general failure to prevent crashes

    def _get_missing_fields(self) -> List[str]:
        """
        Returns a list of required fields that are still missing or marked as 'N/A'.
        """
        return [
            field for field in self.required_fields
            if not self.current_ticket_details.get(field) or self.current_ticket_details.get(field) == "N/A"
        ]

    def _ask_for_missing_details(self, missing_fields: List[str]) -> str:
        """
        Constructs a prompt to ask the user for specific missing details.

        Args:
            missing_fields: A list of field names that are still needed.
        Returns:
            A human-readable string asking for the missing information.
        """
        field_map = {
            "department": "department (IT, HR, or PS)",
            "severity": "issue severity (High, Medium, or Low)",
            "client_name": "impacted client name",
            "impacted_time": "impacted time (e.g., 'yesterday', 'today 2 PM')"
        }
        prompts = [field_map[field] for field in missing_fields]
        return f"To create the ticket, I need a few more details: **{', '.join(prompts)}**. Please provide them."

    def _simulate_ticket_creation(self) -> str:
        """
        Simulates the ticket creation process with the gathered details.
        Clears current ticket details after generating the final message.

        Returns:
            A confirmation message with ticket details.
        """
        details = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in self.current_ticket_details.items()])
        # Generate a simple hash-based ticket ID
        ticket_id = f"TICKET-{hash(frozenset(self.current_ticket_details.items())) % 10000:04d}"

        # Formulate the final response
        final_message = (f"Ticket **{ticket_id}** created successfully with the following details:\n"
                         f"{details}\n")

        # Crucially, clear details *after* the message is formulated but *before* returning
        self.current_ticket_details = {}
        return final_message

    def create_ticket(self, user_input: str) -> str:
        """
        Main function for the tool. Manages the state of ticket creation.
        It uses the LLM to extract information from the conversation history and current input.

        Args:
            user_input: The current input from the user.
        Returns:
            A message indicating whether more information is needed or if the ticket is created.
        """
        # Always try to extract details from the current input and history
        extracted_from_llm = self._extract_details_with_llm(user_input)
        # print(f"LLM extracted: {extracted_from_llm}") # Debugging LLM extraction

        # Update current ticket details with new information, preferring new info
        for field in self.required_fields:
            if extracted_from_llm.get(field) and extracted_from_llm[field] != "N/A":
                self.current_ticket_details[field] = extracted_from_llm[field]
            # If the field is not in extracted_from_llm or is 'N/A', and it's also not yet in current_ticket_details,
            # initialize it as 'N/A' to keep track of missing fields.
            elif field not in self.current_ticket_details:
                 self.current_ticket_details[field] = "N/A" # Ensure all are initialized

        # Check for completeness
        missing_fields = self._get_missing_fields()
        if not missing_fields:
            # All details collected, simulate ticket creation
            return self._simulate_ticket_creation()
        else:
            # Ask for missing details
            return self._ask_for_missing_details(missing_fields)

    def get_tool(self):
        """
        Returns the LangChain Tool object for the agent to use.
        """
        return Tool(
            name="create_support_ticket",
            func=self.create_ticket,
            description=(
                "Use this tool when the user explicitly asks to 'create a ticket', 'raise an issue', "
                "'open a support request', or similar. This tool will guide the user to collect "
                "all necessary details for ticket creation (department, severity, client name, impacted time) "
                "and, once complete, will provide the final answer for ticket creation. "
                "The agent should NOT attempt to generate a Final Answer or another Action after this tool's output "
                "if the ticket is created successfully by the tool."
            )
        )