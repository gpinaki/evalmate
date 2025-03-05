# tests/test_api.py - Simplified test for basic functionality
import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_modes_endpoint():
    """Test the modes endpoint."""
    response = client.get("/modes/")
    assert response.status_code == 200
    data = response.json()
    assert "available_modes" in data
    assert "standard" in data["available_modes"]

def test_estimate_endpoint():
    """Test the estimate endpoint."""
    response = client.get("/estimate/?mode=quick")
    assert response.status_code == 200
    data = response.json()
    assert "estimated_api_calls" in data
    assert data["mode"] == "quick"