# run_api.py
from api.main import app
import uvicorn

if __name__ == "__main__":
    """
    This is the main entry point for running the FastAPI application.
    It uses uvicorn to serve the application defined in 'api.main'.
    """
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        log_level="debug", # Ensure DEBUG logs are processed by uvicorn
        reload=False      # Disable reloader for stability
    ) 