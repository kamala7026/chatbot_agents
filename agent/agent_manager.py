from typing import List, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from core.common.prompt_manager import PromptManager
from core.services.rag_retrive_service import RAGRetrieveService
from tools.adservice_api_tool import AdserviceApiTool
from tools.default_response_tool import DefaultResponseTool
from tools.rag_tool import RAGTool
from tools.ticket_create_tool import TicketCreationTool
from utils.logger_config import logger


class AgentManager:
    """Manages the creation and configuration of the LangChain agent."""

    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.agent_prompt_template = PromptManager.get_agent_prompt()
        logger.info("AgentManager initialized.")

    def get_agent_executor(self, user_type: str, chat_history: List[Any]) -> AgentExecutor:
        """Dynamically creates a LangChain agent with the given history."""
        # Add assertions to satisfy the linter
        assert self.llm is not None
        assert self.vectorstore is not None
        assert self.agent_prompt_template is not None

        retriever_logic = RAGRetrieveService(self.vectorstore, self.llm, user_type)

        # Create a temporary memory for this specific conversation turn
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=15  # Allow a larger buffer for context
        )
        # Load the provided history into the temporary memory
        for message in chat_history:
            if message['role'] == 'user':
                memory.chat_memory.add_user_message(message['content'])
            elif message['role'] == 'assistant':
                memory.chat_memory.add_ai_message(message['content'])

        def get_current_chat_history() -> List[Any]:
            return memory.load_memory_variables({})['chat_history']

        tools = [
            RAGTool(retriever_logic).get_tool(),
            DefaultResponseTool().get_tool(),
            TicketCreationTool(self.llm, chat_history_provider=get_current_chat_history).get_tool(),
            AdserviceApiTool(self.llm, chat_history_provider=get_current_chat_history).get_tool()
        ]

        tool_names = [tool.name for tool in tools]
        logger.debug(f"Agent created with tools: {tool_names}")

        # The agent prompt is now retrieved from the cached instance variable
        prompt = self.agent_prompt_template

        return AgentExecutor(
            agent=create_react_agent(self.llm, tools, prompt),
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=120
        ) 