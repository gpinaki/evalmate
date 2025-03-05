# tests/test_evaluation.py
from app.evaluation import LLMEvaluator 

# tests/test_evaluation.py - update the test to check the right fields
def test_llm_evaluator():
    evaluator = LLMEvaluator()
    
    result = evaluator.evaluate_response(
        app_name="Test App",
        user="Test User",
        app_actual_response="AI is the future.",
        expected_response="Expected response",
        context="Context information"
    )
    
    # Basic field checks
    assert result["App Name"] == "Test App"
    assert result["User"] == "Test User"
    
    # Check for existence of metric fields
    expected_metrics = [
        "Contextual Precision Score",
        "Contextual Recall Score",
        "Contextual Relevancy Score",
        "Answer Relevancy Score",
        "Faithfulness Score",
        "Hallucination Score"
    ]
    
    for metric in expected_metrics:
        assert metric in result, f"Missing metric: {metric}"
        assert isinstance(result[metric], (int, float)), f"Metric {metric} is not a number"