# test_all_metrics.py
import os
import logging
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric, 
    ContextualRecallMetric, 
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    HallucinationMetric
)

from langchain_openai import OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_metric(metric_name, metric_instance, test_case):
    """Test a single metric with a given test case."""
    logger.info(f"Testing {metric_name}...")
    
    try:
        # Run evaluation with just this metric
        results = evaluate(test_cases=[test_case], metrics=[metric_instance])
        
        # Check results
        if hasattr(results, 'test_results') and results.test_results:
            for result in results.test_results:
                if hasattr(result, 'metrics_data') and result.metrics_data:
                    for metric_data in result.metrics_data:
                        logger.info(f"  - Score: {metric_data.score}")
                        logger.info(f"  - Success: {metric_data.success}")
                        logger.info(f"  - Reason: {metric_data.reason}")
            return True
        else:
            logger.warning(f"No results for {metric_name}")
            return False
    except Exception as e:
        logger.error(f"Error testing {metric_name}: {str(e)}", exc_info=True)
        return False

def main():
    """Test all metrics one by one."""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not set!")
        return
    
    # Sample data for test case
    context_text = "Artificial Intelligence (AI) enables machines to perform cognitive tasks. Machine learning is a subset of AI that allows systems to learn from data."
    
    # Create a complete test case with all possible fields
    test_case = LLMTestCase(
        input="What is AI?",
        actual_output="AI is artificial intelligence which allows computers to perform tasks that typically require human intelligence.",
        expected_output="Artificial Intelligence is a field of computer science that enables machines to perform tasks requiring human intelligence.",
        context=[context_text],  # Some metrics might use this
        retrieval_context=[context_text],  # Some metrics might use this instead
    )
    
    # Common parameters for all metrics
    metric_params = {
        "threshold": 0.5,
        "include_reason": True,
        "model": "gpt-4o-mini"
    }
    
    # Test each metric individually
    metrics_to_test = [
        ("Contextual Precision", ContextualPrecisionMetric(**metric_params)),
        ("Contextual Recall", ContextualRecallMetric(**metric_params)),
        ("Contextual Relevancy", ContextualRelevancyMetric(**metric_params)),
        ("Answer Relevancy", AnswerRelevancyMetric(**metric_params)),
        ("Faithfulness", FaithfulnessMetric(**metric_params)),
        ("Hallucination", HallucinationMetric(**metric_params))
    ]
    
    results = {}
    for name, metric in metrics_to_test:
        success = test_metric(name, metric, test_case)
        results[name] = "Success" if success else "Failed"
    
    # Summary
    logger.info("\n--- TEST RESULTS SUMMARY ---")
    for name, status in results.items():
        logger.info(f"{name}: {status}")

if __name__ == "__main__":
    main()