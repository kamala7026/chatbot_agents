# core/history_manager.py
import logging
from typing import List, Dict, Any, Optional
import uuid
import psycopg

from vector.pgvector_manager import PGVectorManager

logger = logging.getLogger("aviator_chatbot")

class HistoryService:
    """
    Manages chat history operations, including loading, saving, and creating new sessions.
    This class is designed to be stateless; it operates on a directory structure.
    """
    
    @staticmethod
    def _get_db_connection() -> Optional[psycopg.Connection]:
        """Gets a connection to the database."""
        return PGVectorManager.get_db_connection()

    @staticmethod
    def initialize_database():
        """
        Creates the chat_history and chat_messages tables if they don't exist.
        """
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error("Failed to get database connection.")
                return

            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        chat_id UUID UNIQUE NOT NULL,
                        username VARCHAR(255) NOT NULL,
                        title TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        chat_id UUID NOT NULL REFERENCES chat_history(chat_id) ON DELETE CASCADE,
                        role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
                        content TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_username ON chat_history (username);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages (chat_id);")
                conn.commit()
            logger.info("Database initialized for HistoryManager (history and messages).")
        except Exception as e:
            logger.error(f"Error initializing chat tables: {e}", exc_info=True)
            if conn:
                conn.rollback()

    @staticmethod
    def create_new_chat(username: str, title: str = "New Chat") -> Optional[str]:
        """Creates a new chat record and returns the chat_id."""
        chat_id = uuid.uuid4()
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for creating chat.")
                return None

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history (chat_id, username, title) VALUES (%s, %s, %s)",
                    (str(chat_id), username, title)
                )
                conn.commit()
            logger.info(f"Created new chat for user '{username}' with chat_id '{chat_id}'.")
            return str(chat_id)
        except Exception as e:
            logger.error(f"Failed to create new chat for user '{username}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return None

    @staticmethod
    def get_history_list(username: str) -> List[Dict[str, Any]]:
        """Retrieves a list of all chat histories for a given user, sorted by last update."""
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for fetching history.")
                return []

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chat_id, title, updated_at FROM chat_history WHERE username = %s ORDER BY updated_at DESC",
                    (username,)
                )
                history = cur.fetchall()
            
            # Format the data into the structure the UI expects
            history_list = [
                {"id": row[0], "title": row[1], "timestamp": row[2].isoformat()}
                for row in history
            ]
            logger.info(f"Retrieved {len(history_list)} history items for user '{username}'.")
            return history_list
        except Exception as e:
            logger.error(f"Failed to get history list for user '{username}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return []

    @staticmethod
    def add_message_to_history(chat_id: str, role: str, content: str):
        """Adds a single message to the chat_messages table."""
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error(f"Failed to get DB connection for adding message to chat {chat_id}.")
                return

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (chat_id, role, content) VALUES (%s, %s, %s)",
                    (chat_id, role, content)
                )
                # Also update the updated_at timestamp for the parent chat
                cur.execute(
                    "UPDATE chat_history SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = %s",
                    (chat_id,)
                )
                conn.commit()
            logger.debug(f"Added message from '{role}' to chat_id '{chat_id}'.")
        except Exception as e:
            logger.error(f"Failed to add message to history for chat {chat_id}: {e}", exc_info=True)
            if conn:
                conn.rollback()

    @staticmethod
    def get_messages_for_chat(chat_id: str, username: str = None) -> List[Dict[str, Any]]:
        """Retrieves all messages for a specific chat_id, ordered by creation time, with feedback information."""
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error(f"Failed to get DB connection for fetching messages for chat {chat_id}.")
                return []

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM chat_messages WHERE chat_id = %s ORDER BY created_at ASC",
                    (chat_id,)
                )
                messages = cur.fetchall()

            history = [{"role": row[0], "content": row[1]} for row in messages]
            
            # Initialize all messages with feedback field set to None
            for message in history:
                message['feedback'] = None
            
            # Add feedback information if username is provided
            if username:
                try:
                    from core.services.feedback_service import get_feedback_service
                    feedback_service = get_feedback_service()
                    feedback_map = feedback_service.get_chat_feedback(username, chat_id)
                    
                    logger.debug(f"Retrieved feedback map: {feedback_map}")
                    
                    # Add feedback to assistant messages (message_index is for assistant responses)
                    assistant_message_index = 0
                    for i, message in enumerate(history):
                        if message['role'] == 'assistant':
                            feedback = feedback_map.get(assistant_message_index)
                            if feedback:
                                history[i]['feedback'] = feedback
                                logger.debug(f"Set feedback for assistant message {assistant_message_index}: {feedback}")
                            assistant_message_index += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to get feedback for chat {chat_id}: {e}")
                    # Feedback field is already set to None for all messages above

            logger.debug(f"Retrieved {len(history)} messages for chat_id '{chat_id}'.")
            return history
        except Exception as e:
            logger.error(f"Failed to get messages for chat '{chat_id}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return []

    @staticmethod
    def update_chat_title(chat_id: str, new_title: str) -> bool:
        """Updates the title of a specific chat and its updated_at timestamp."""
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for updating title.")
                return False

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_history SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE chat_id = %s",
                    (new_title, chat_id)
                )
                conn.commit()
            logger.info(f"Updated title for chat_id '{chat_id}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to update title for chat_id '{chat_id}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False

    @staticmethod
    def get_chat_title(chat_id: str) -> Optional[str]:
        """Retrieves the title of a specific chat."""
        try:
            conn = HistoryService._get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for fetching title.")
                return None

            with conn.cursor() as cur:
                cur.execute("SELECT title FROM chat_history WHERE chat_id = %s", (chat_id,))
                result = cur.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get title for chat_id '{chat_id}': {e}", exc_info=True)
            if conn:
                conn.rollback()
            return None 