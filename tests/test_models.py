# tests/test_models.py
from app.models import EvaluationRequest

def test_evaluation_request_model():
    # Test valid input
    valid_data = {
        "app_name": "Test App",
        "user": "Test User",
        "app_actual_response": "AI is the future.",
        "expected_response": "Expected response",
        "context": "Context information"
    }
    
    request = EvaluationRequest(**valid_data)
    assert request.app_name == "Test App"
    assert request.user == "Test User"
    assert request.app_actual_response == "AI is the future."