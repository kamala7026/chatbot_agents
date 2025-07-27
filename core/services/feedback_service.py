"""
Feedback Service for handling user feedback (like/dislike) on AI responses.
This service stores feedback data in the PostgreSQL database for analytics.
"""

import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
import os
from utils.logger_config import logger
from core.common.config import PGVECTOR_CONNECTION_STRING

class FeedbackService:
    """Service for managing user feedback on AI responses."""
    
    def __init__(self):
        """Initialize the feedback service with database connection."""
        self.connection_string = PGVECTOR_CONNECTION_STRING
        self.engine = None
        self.feedback_table = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables if needed."""
        try:
            self.engine = create_engine(self.connection_string)
            
            # Create feedback table if it doesn't exist
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        username VARCHAR(255) NOT NULL,
                        chat_id VARCHAR(255) NOT NULL,
                        message_index INTEGER NOT NULL,
                        user_message TEXT NOT NULL,
                        assistant_message TEXT NOT NULL,
                        feedback_type VARCHAR(10) NOT NULL CHECK (feedback_type IN ('liked', 'disliked')),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """))
                
                # Create index for efficient queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_user_feedback_chat 
                    ON user_feedback(username, chat_id)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_user_feedback_type 
                    ON user_feedback(feedback_type)
                """))
                
                conn.commit()
                logger.info("Feedback database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
            raise

    def submit_feedback(
        self, 
        username: str, 
        chat_id: str, 
        message_index: int,
        user_message: str, 
        assistant_message: str, 
        feedback_type: str
    ) -> str:
        """
        Submit user feedback for an AI response.
        
        Args:
            username: The user providing feedback
            chat_id: The chat session ID
            message_index: Index of the message in the conversation
            user_message: The original user query
            assistant_message: The AI response being rated
            feedback_type: 'liked' or 'disliked'
            
        Returns:
            str: The feedback ID
        """
        try:
            feedback_id = str(uuid.uuid4())
            
            with self.engine.connect() as conn:
                # Check if feedback already exists for this message
                result = conn.execute(text("""
                    SELECT id FROM user_feedback 
                    WHERE username = :username 
                    AND chat_id = :chat_id 
                    AND message_index = :message_index
                """), {
                    "username": username,
                    "chat_id": chat_id,
                    "message_index": message_index
                })
                
                existing_feedback = result.fetchone()
                
                if existing_feedback:
                    # Update existing feedback
                    conn.execute(text("""
                        UPDATE user_feedback 
                        SET feedback_type = :feedback_type, 
                            updated_at = NOW()
                        WHERE username = :username 
                        AND chat_id = :chat_id 
                        AND message_index = :message_index
                    """), {
                        "feedback_type": feedback_type,
                        "username": username,
                        "chat_id": chat_id,
                        "message_index": message_index
                    })
                    feedback_id = str(existing_feedback[0])
                    logger.info(f"Updated feedback for user {username}, chat {chat_id}, message {message_index}")
                else:
                    # Insert new feedback
                    conn.execute(text("""
                        INSERT INTO user_feedback 
                        (id, username, chat_id, message_index, user_message, assistant_message, feedback_type)
                        VALUES (:id, :username, :chat_id, :message_index, :user_message, :assistant_message, :feedback_type)
                    """), {
                        "id": feedback_id,
                        "username": username,
                        "chat_id": chat_id,
                        "message_index": message_index,
                        "user_message": user_message,
                        "assistant_message": assistant_message,
                        "feedback_type": feedback_type
                    })
                    logger.info(f"Created new feedback for user {username}, chat {chat_id}, message {message_index}")
                
                conn.commit()
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise

    def get_feedback_stats(self, username: Optional[str] = None) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            username: Optional username to filter stats
            
        Returns:
            Dict with feedback statistics
        """
        try:
            with self.engine.connect() as conn:
                if username:
                    result = conn.execute(text("""
                        SELECT 
                            feedback_type,
                            COUNT(*) as count
                        FROM user_feedback 
                        WHERE username = :username
                        GROUP BY feedback_type
                    """), {"username": username})
                else:
                    result = conn.execute(text("""
                        SELECT 
                            feedback_type,
                            COUNT(*) as count
                        FROM user_feedback 
                        GROUP BY feedback_type
                    """))
                
                stats = {"liked": 0, "disliked": 0, "total": 0}
                
                for row in result:
                    stats[row[0]] = row[1]
                    stats["total"] += row[1]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"liked": 0, "disliked": 0, "total": 0}

    def get_user_feedback_history(self, username: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get feedback history for a specific user.
        
        Args:
            username: The username to get feedback for
            limit: Maximum number of feedback records to return
            
        Returns:
            List of feedback records
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        id, chat_id, message_index, user_message, 
                        assistant_message, feedback_type, created_at
                    FROM user_feedback 
                    WHERE username = :username
                    ORDER BY created_at DESC
                    LIMIT :limit
                """), {"username": username, "limit": limit})
                
                feedback_history = []
                for row in result:
                    feedback_history.append({
                        "id": str(row[0]),
                        "chat_id": row[1],
                        "message_index": row[2],
                        "user_message": row[3],
                        "assistant_message": row[4],
                        "feedback_type": row[5],
                        "created_at": row[6].isoformat() if row[6] else None
                    })
                
                return feedback_history
                
        except Exception as e:
            logger.error(f"Failed to get user feedback history: {e}")
            return []

    def get_user_feedback_history_paginated(self, username: str, offset: int = 0, limit: int = 10) -> tuple[List[Dict[str, Any]], int]:
        """
        Get paginated feedback history for a specific user.
        
        Args:
            username: The username to get feedback for
            offset: Number of records to skip
            limit: Maximum number of feedback records to return
            
        Returns:
            Tuple of (feedback_records, total_count)
        """
        try:
            with self.engine.connect() as conn:
                # Get total count
                count_result = conn.execute(text("""
                    SELECT COUNT(*) FROM user_feedback WHERE username = :username
                """), {"username": username})
                total_count = count_result.scalar() or 0
                
                # Get paginated results
                result = conn.execute(text("""
                    SELECT 
                        id, chat_id, message_index, user_message, 
                        assistant_message, feedback_type, created_at
                    FROM user_feedback 
                    WHERE username = :username
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """), {"username": username, "limit": limit, "offset": offset})
                
                feedback_history = []
                for row in result:
                    feedback_history.append({
                        "id": str(row[0]),
                        "chat_id": row[1],
                        "message_index": row[2],
                        "user_message": row[3],
                        "assistant_message": row[4],
                        "feedback_type": row[5],
                        "created_at": row[6].isoformat() if row[6] else None
                    })
                
                return feedback_history, total_count
                
        except Exception as e:
            logger.error(f"Failed to get paginated user feedback history: {e}")
            return [], 0

    def get_chat_feedback(self, username: str, chat_id: str) -> Dict[int, str]:
        """
        Get feedback status for all messages in a specific chat.
        Returns a dictionary mapping message_index to feedback_type.
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT message_index, feedback_type 
                    FROM user_feedback 
                    WHERE username = :username AND chat_id = :chat_id
                    ORDER BY message_index
                """)
                result = conn.execute(query, {
                    "username": username,
                    "chat_id": chat_id
                })
                
                # Create a mapping of message_index to feedback_type
                feedback_map = {}
                for row in result:
                    feedback_map[row.message_index] = row.feedback_type
                
                logger.debug(f"Retrieved feedback for {len(feedback_map)} messages in chat {chat_id}")
                return feedback_map
                
        except Exception as e:
            logger.error(f"Database error in get_chat_feedback: {e}")
            return {}

# Singleton instance
_feedback_service_instance = None

def get_feedback_service() -> FeedbackService:
    """Get the singleton feedback service instance."""
    global _feedback_service_instance
    if _feedback_service_instance is None:
        _feedback_service_instance = FeedbackService()
    return _feedback_service_instance 