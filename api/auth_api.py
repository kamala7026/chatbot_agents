"""
Authentication API endpoints.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from core.services.user_service import UserService, get_user_service_dependency
from api.schemas import LoginRequest, LoginResponse, UserInfoResponse

logger = logging.getLogger(__name__)

# Router for authentication endpoints
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

@auth_router.post("/login", response_model=LoginResponse, summary="Authenticate user")
async def login(
    request: LoginRequest,
    user_service: UserService = Depends(get_user_service_dependency)
):
    """
    Authenticate user credentials and return user information.
    
    Args:
        request: Login credentials (username and password)
        user_service: Injected UserService instance
        
    Returns:
        LoginResponse with success status and user information
        
    Raises:
        HTTPException: If authentication fails or server error occurs
    """
    try:
        logger.info(f"Login attempt for user: {request.username}")
        
        # Validate input
        if not request.username or not request.password:
            raise HTTPException(
                status_code=400, 
                detail="Username and password are required"
            )
        
        # Authenticate user
        user_data = user_service.authenticate_user(request.username, request.password)
        
        if user_data:
            logger.info(f"Login successful for user: {request.username} ({user_data['user_type']})")
            return LoginResponse(
                success=True,
                message="Login successful",
                user=user_data
            )
        else:
            logger.warning(f"Login failed for user: {request.username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Login error for user {request.username}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during authentication"
        )

@auth_router.get("/user/{username}", response_model=UserInfoResponse, summary="Get user information")
async def get_user_info(
    username: str,
    user_service: UserService = Depends(get_user_service_dependency)
):
    """
    Get user information by username.
    
    Args:
        username: Username to lookup
        user_service: Injected UserService instance
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found or server error occurs
    """
    try:
        logger.info(f"Getting user info for: {username}")
        
        user_data = user_service.get_user_by_username(username)
        
        if user_data:
            return UserInfoResponse(
                id=user_data['id'],
                username=user_data['username'],
                user_type=user_data['user_type'],
                full_name=user_data.get('full_name'),
                email=user_data.get('email'),
                is_active=user_data['is_active']
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"User '{username}' not found"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting user info for {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving user information"
        )

@auth_router.post("/validate", summary="Validate user session")
async def validate_session(
    request: LoginRequest,
    user_service: UserService = Depends(get_user_service_dependency)
):
    """
    Validate user session/credentials (can be used for session refresh).
    
    Args:
        request: User credentials
        user_service: Injected UserService instance
        
    Returns:
        User information if valid
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        logger.info(f"Validating session for user: {request.username}")
        
        user_data = user_service.authenticate_user(request.username, request.password)
        
        if user_data:
            return {
                "valid": True,
                "user": user_data
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid session"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session validation error for user {request.username}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during session validation"
        ) 