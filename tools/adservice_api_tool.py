import requests
import json
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain.agents import Tool

from utils.logger_config import logger
from core.common.config import ADSERVICE_API_URL
from core.common.prompt_manager import PromptManager
from core.services.adservice_api_service import AdserviceApiService


# --- Pydantic Models for Robust Chains ---

class ConversationState(BaseModel):
    """Tracks all gathered parameters for the lifecycle API call."""
    node_id: Optional[str] = Field(None, description="The 'Node ID' for the client application.")
    buids: Optional[str] = Field(None, description="The 'Requested BUIDs' for the search.")
    file_id: Optional[str] = Field(None, description="The 'File ID' for the content file.")
    start_date: Optional[str] = Field("0", description="Optional start date (Unix timestamp). Defaults to 0.")
    end_date: Optional[str] = Field("0", description="Optional end date (Unix timestamp). Defaults to 0.")

    def is_complete(self) -> bool:
        """Checks if all mandatory fields are present."""
        return all([self.node_id, self.buids, self.file_id])

class ExtractedParams(BaseModel):
    """Model for data extracted by the LLM."""
    node_id: Optional[str] = Field(None, description="The extracted 'Node ID'.")
    buids: Optional[str] = Field(None, description="The extracted 'BUIDs'.")
    file_id: Optional[str] = Field(None, description="The extracted 'File ID'.")


# --- The Main Tool Class ---

class AdserviceApiTool:
    """A self-contained tool to manage the conversation for fetching lifecycle data."""
    
    name: str = "get_lifecycles_data"
    description: str = (
        "Use this tool when the user asks for 'lifecycle details', 'File details', 'Transaction details', or 'File search'. "
        "This tool will handle the entire process of gathering the required parameters (Node ID, BUIDs, File ID) from the user "
        "and then calling the API to fetch the data."
    )

    def __init__(self, llm: BaseChatModel, chat_history_provider):
        self.llm = llm
        self.get_chat_history = chat_history_provider
        self.state = ConversationState(node_id=None, buids=None, file_id=None, start_date='0', end_date='0')
        self.core_api = AdserviceApiService()  # Use the new manager
        self._create_extraction_chain()
        logger.info("AdserviceApiTool initialized.")

    def _create_extraction_chain(self):
        """Creates the LangChain runnable for extracting parameters."""
        system_prompt = PromptManager.get_adservice_extraction_prompt()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is the conversation history:\n\n{history}\n\n"
                     "And here is the current user input:\n\n{input}")
        ])
        
        self.extraction_chain = prompt | self.llm.with_structured_output(ExtractedParams)

    def _reset_state(self):
        """Resets the internal state for a new request."""
        self.state = ConversationState(node_id=None, buids=None, file_id=None, start_date='0', end_date='0')
        logger.info("AdserviceApiTool state has been reset.")

    def get_lifecycle_details(self, user_input: str) -> str:
        """The main entry point for the tool."""
        chat_history = self.get_chat_history()
        logger.debug(f"Running extraction chain with history and input: '{user_input}'")
        extracted: ExtractedParams = self.extraction_chain.invoke({
            "history": chat_history, 
            "input": user_input
        })  # type: ignore

        if extracted.node_id: self.state.node_id = extracted.node_id
        if extracted.buids: self.state.buids = extracted.buids
        if extracted.file_id: self.state.file_id = extracted.file_id

        logger.debug(f"Tool state updated: Node ID='{self.state.node_id}', BUIDs='{self.state.buids}', File ID='{self.state.file_id}'")

        if self.state.is_complete():
            # The is_complete check ensures these are not None.
            assert self.state.node_id is not None
            assert self.state.buids is not None
            assert self.state.file_id is not None
            assert self.state.start_date is not None
            assert self.state.end_date is not None
            
            result = self.core_api.get_lifecycles_data(
                node_id=self.state.node_id,
                buids=self.state.buids,
                file_id=self.state.file_id,
                start_date=self.state.start_date,
                end_date=self.state.end_date,
            )
            self._reset_state()
            return result
        else:
            missing_params = []
            if not self.state.node_id: missing_params.append("Node ID")
            if not self.state.buids: missing_params.append("Requested BUIDs")
            if not self.state.file_id: missing_params.append("File ID")
            
            return f"I can help with that. To get the lifecycle details, I need the following information: {', '.join(missing_params)}. Please provide the missing details."

    def get_tool(self) -> Tool:
        """Returns the configured LangChain tool."""
        return Tool(
            name=self.name,
            func=self.get_lifecycle_details,
            description=self.description,
        )
