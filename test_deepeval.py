# test_deepeval_fixed.py
import os
import logging
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_deepeval_minimal():
    """Test a minimal DeepEval example using the proper TestCase class."""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not set!")
        return False
    
    logger.info(f"Using API key: {OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")
    
    # Create a simple metric
    metric = ContextualRelevancyMetric(
        threshold=0.5,
        include_reason=True,
        model="gpt-3.5-turbo"  # Using 3.5 as it's more reliable for testing
    )
    
    # Create a proper LLMTestCase object
    try:
        # The context text we'll use
        context_text = "Artificial Intelligence (AI) enables machines to perform cognitive tasks."
        
        # Create test case using DeepEval's TestCase class
        # Note: We need to use retrieval_context for the ContextualRelevancyMetric
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence.",
            expected_output="Artificial Intelligence is a field of computer science.",
            retrieval_context=[context_text]  # This is required for ContextualRelevancyMetric
        )
        
        logger.info("Created test case successfully")
        
        # Run evaluation with the proper test case
        logger.info("Running evaluation...")
        results = evaluate(test_cases=[test_case], metrics=[metric])
        
        # Log results
        logger.info(f"Results type: {type(results)}")
        logger.info(f"Results: {results}")
        
        # Check test results
        if hasattr(results, 'test_results') and results.test_results:
            logger.info(f"Number of test results: {len(results.test_results)}")
            for i, result in enumerate(results.test_results):
                logger.info(f"Test result {i+1}:")
                if hasattr(result, 'metrics_data') and result.metrics_data:
                    for metric_data in result.metrics_data:
                        logger.info(f"  - Metric: {metric_data.name}, Score: {metric_data.score}")
                        logger.info(f"    Reason: {metric_data.reason}")
            return True
        else:
            logger.warning("No test results returned!")
            return False
            
    except Exception as e:
        logger.error(f"Error running DeepEval: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_deepeval_minimal()
    if success:
        print("DeepEval test successful!")
    else:
        print("DeepEval test failed!")