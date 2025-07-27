"""
User service for handling authentication and user management.
"""
import logging
import hashlib
import os
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from core.common.schemas import UserCredentials, UserProfile

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class UserService:
    """Service class for user authentication and management."""
    
    def __init__(self):
        """Initialize the UserService with database connection."""
        # Get database configuration from environment variables
        self.db_url = os.getenv("PGVECTOR_CONNECTION_STRING", "postgresql+psycopg://postgres:admin@localhost:5432/vectordb")
        self.engine = None
        self._initialize_database()
    
    def _get_engine(self):
        """Get or create database engine."""
        if self.engine is None:
            try:
                self.engine = create_engine(self.db_url)
                logger.info("Database engine created successfully for UserService")
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise
        return self.engine
    
    def _initialize_database(self):
        """Initialize the users table if it doesn't exist."""
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                # Create users table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(64) NOT NULL,
                        user_type VARCHAR(20) NOT NULL CHECK (user_type IN ('Support', 'Client', 'Tester')),
                        full_name VARCHAR(100),
                        email VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """))
                
                # Create index on username for faster lookups
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
                """))
                
                # Insert default users if table is empty
                result = conn.execute(text("SELECT COUNT(*) FROM users"))
                count = result.scalar()
                
                if count == 0:
                    logger.info("Inserting default users...")
                    default_users = [
                        {
                            'username': 'admin',
                            'password': 'admin123',
                            'user_type': 'Support',
                            'full_name': 'System Administrator',
                            'email': 'admin@aviator.com'
                        },
                        {
                            'username': 'kamala',
                            'password': 'kamala123',
                            'user_type': 'Support',
                            'full_name': 'Kamala Harris',
                            'email': 'kamala@aviator.com'
                        },
                        {
                            'username': 'tester',
                            'password': 'test123',
                            'user_type': 'Tester',
                            'full_name': 'Test User',
                            'email': 'tester@aviator.com'
                        },
                        {
                            'username': 'client1',
                            'password': 'client123',
                            'user_type': 'Client',
                            'full_name': 'Client User',
                            'email': 'client1@aviator.com'
                        }
                    ]
                    
                    for user in default_users:
                        password_hash = self._hash_password(user['password'])
                        conn.execute(text("""
                            INSERT INTO users (username, password_hash, user_type, full_name, email)
                            VALUES (:username, :password_hash, :user_type, :full_name, :email)
                        """), {
                            'username': user['username'],
                            'password_hash': password_hash,
                            'user_type': user['user_type'],
                            'full_name': user['full_name'],
                            'email': user['email']
                        })
                    
                    conn.commit()
                    logger.info("Default users inserted successfully")
                
                conn.commit()
                logger.info("Users table initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize users database: {e}")
            raise
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials and return user information.
        
        Args:
            username: Username to authenticate
            password: Plain text password
            
        Returns:
            User dict if authentication successful, None otherwise
        """
        try:
            password_hash = self._hash_password(password)
            engine = self._get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, username, user_type, full_name, email, is_active
                    FROM users 
                    WHERE username = :username AND password_hash = :password_hash AND is_active = TRUE
                """), {
                    'username': username,
                    'password_hash': password_hash
                })
                
                row = result.fetchone()
                if row:
                    user_data = {
                        'id': row[0],
                        'username': row[1],
                        'user_type': row[2],
                        'full_name': row[3],
                        'email': row[4],
                        'is_active': row[5]
                    }
                    logger.info(f"User authenticated successfully: {username} ({user_data['user_type']})")
                    return user_data
                else:
                    logger.warning(f"Authentication failed for user: {username}")
                    return None
                    
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by username.
        
        Args:
            username: Username to lookup
            
        Returns:
            User dict if found, None otherwise
        """
        try:
            engine = self._get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, username, user_type, full_name, email, is_active, created_at
                    FROM users 
                    WHERE username = :username
                """), {'username': username})
                
                row = result.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'user_type': row[2],
                        'full_name': row[3],
                        'email': row[4],
                        'is_active': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
            return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users.
        
        Returns:
            List of user dictionaries
        """
        try:
            engine = self._get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, username, user_type, full_name, email, is_active, created_at
                    FROM users 
                    ORDER BY created_at DESC
                """))
                
                users = []
                for row in result:
                    users.append({
                        'id': row[0],
                        'username': row[1],
                        'user_type': row[2],
                        'full_name': row[3],
                        'email': row[4],
                        'is_active': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })
                
                return users
                
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """
        Update user information.
        
        Args:
            username: Username to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            engine = self._get_engine()
            
            # Build dynamic update query
            set_clauses = []
            params = {'username': username}
            
            for field, value in updates.items():
                if field in ['user_type', 'full_name', 'email', 'is_active']:
                    set_clauses.append(f"{field} = :{field}")
                    params[field] = value
                elif field == 'password':
                    set_clauses.append("password_hash = :password_hash")
                    params['password_hash'] = self._hash_password(value)
            
            if not set_clauses:
                logger.warning("No valid fields to update")
                return False
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            
            with engine.connect() as conn:
                query = f"UPDATE users SET {', '.join(set_clauses)} WHERE username = :username"
                result = conn.execute(text(query), params)
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"User {username} updated successfully")
                    return True
                else:
                    logger.warning(f"No user found with username: {username}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating user {username}: {e}")
            return False


# Singleton instance
_user_service_instance = None

def get_user_service() -> UserService:
    """Get singleton instance of UserService."""
    global _user_service_instance
    if _user_service_instance is None:
        _user_service_instance = UserService()
    return _user_service_instance

def get_user_service_dependency() -> UserService:
    """Dependency injection for FastAPI."""
    return get_user_service() 