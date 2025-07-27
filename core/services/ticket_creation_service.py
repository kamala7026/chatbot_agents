# core/ticket_creation_service.py
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from core.common.prompt_manager import PromptManager
from utils.logger_config import logger


# --- Pydantic Models for Ticket Creation ---
class TicketDetails(BaseModel):
    """Data model for holding the details of a support ticket."""
    department: str = Field(..., description="The department for the ticket (e.g., IT, HR, PS).")
    severity: str = Field(..., description="The severity of the issue (e.g., High, Medium, Low).")
    client_name: str = Field(..., description="The name of the impacted client.")
    impacted_time: str = Field(..., description="The time the issue occurred (e.g., 'yesterday', 'today 2 PM').")

class ExtractionResponse(BaseModel):
    """Model for the data extracted by the LLM."""
    mentioned_fields: List[str] = Field(description="List of fields explicitly mentioned in the user's last message.")
    extracted_data: Dict[str, Any] = Field(description="The actual data extracted for the mentioned fields.")


# --- Service Class for Core Logic ---
class TicketCreationService:
    """Handles the core logic of extracting and validating ticket details."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.extraction_prompt = PromptManager.get_ticket_extraction_prompt()
        self._create_extraction_chain()
        logger.info("TicketCreationService initialized.")

    def _create_extraction_chain(self):
        """Creates the LangChain runnable for extracting ticket parameters."""
        self.extraction_chain = self.extraction_prompt | self.llm | JsonOutputParser()

    def extract_details(self, user_input: str, chat_history_str: str) -> Dict[str, Any]:
        """
        Uses the LLM to extract ticket details from conversation history and current input.
        
        Args:
            user_input: The current input from the user.
            chat_history_str: A string representation of the recent conversation.
            
        Returns:
            A dictionary containing the extracted details, conforming to ExtractionResponse.
        """
        try:
            response = self.extraction_chain.invoke({
                "chat_history_str": chat_history_str,
                "user_input": user_input
            })
            # --- FIX: The output is now a dict from the JsonOutputParser ---
            logger.debug(f"LLM extraction successful. Raw response: {response}")
            
            # Since the output is already a dictionary, we can perform basic validation
            # and then pass it to the Pydantic model for robust parsing.
            if isinstance(response, dict):
                validated_response = ExtractionResponse(**response)
                return validated_response.dict()
            else:
                logger.error(f"Expected a dict from the extraction chain, but got {type(response)}.")
                return {"mentioned_fields": [], "extracted_data": {}}

        except Exception as e:
            logger.error(f"Error during LLM detail extraction in service: {e}", exc_info=True)
            return {"mentioned_fields": [], "extracted_data": {}}

    @staticmethod
    def validate_fields(ticket_details: Dict[str, Any]) -> List[str]:
        """
        Performs robust validation on the collected ticket details.
        
        Args:
            ticket_details: A dictionary of the currently collected ticket details.
        
        Returns:
            A list of required fields that are still missing or invalid.
        """
        required_fields = ["department", "severity", "client_name", "impacted_time"]
        allowed_values = {
            "department": ["it", "hr", "ps", "network support", "payroll", "general support"],
            "severity": ["high", "medium", "low", "critical", "urgent"]
        }
        
        missing = []
        for field in required_fields:
            value = ticket_details.get(field)
            is_missing = False

            if not value or str(value).strip().lower() in ["n/a", "na", "unknown", "", "missing"]:
                is_missing = True
            else:
                clean_value = str(value).strip().lower()
                if field in allowed_values and clean_value not in allowed_values[field]:
                    is_missing = True
            
            if is_missing:
                missing.append(field)
        
        logger.debug(f"Validation in service found missing fields: {missing}")
        return missing 