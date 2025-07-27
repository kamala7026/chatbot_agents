#!/usr/bin/env python3
"""
FastAPI Server Runner for Aviator Chatbot Backend

This script starts the FastAPI application using uvicorn.
Run this from the chatbot_agents root directory.
"""

import uvicorn
import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    """
    Main entry point for running the FastAPI application.
    Uses uvicorn to serve the application defined in 'api.main'.
    """
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8001,
        log_level="info",
        reload=True  # Enable auto-reload for development
    ) 