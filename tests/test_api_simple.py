# tests/test_api_simple.py
from fastapi.testclient import TestClient
from app.api import app
import pytest

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_simple_evaluation():
    """Test a simple evaluation request without mocking."""
    test_input = {
        "app_name": "Simple Test",
        "user": "Test User",
        "app_actual_response": "Simple response.",
        "expected_response": "Expected response.",
        "context": "Some context."
    }
    
    response = client.post("/evaluate/", json=test_input)
    assert response.status_code == 200
    result = response.json()
    assert result["App Name"] == "Simple Test"
    # We're not testing the mock here, just that the endpoint works