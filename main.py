import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Entry point for running the FastAPI application.
    Configures and starts the uvicorn server with the FastAPI app.
    """
    # Hardcoded values for testing
    host = "localhost"
    port = 8000
    reload = True
    
    # Start the uvicorn server
    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()  # Removed the period