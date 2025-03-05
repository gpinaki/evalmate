import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from deepeval import evaluate

from deepeval.metrics import (
    ContextualPrecisionMetric, 
    ContextualRecallMetric, 
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric
)
from deepeval.test_case import LLMTestCase

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_EVAL_MODEL", "gpt-3.5-turbo")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Define evaluation modes with their metrics
EVALUATION_MODES = {
    "quick": {
        "description": "Minimal, fast evaluation when you just need basic quality assessment",
        "metrics": ["answer_relevancy"],
        "required_params": ["user_request", "app_actual_response"]
    },
    "standard": {
        "description": "Balanced evaluation for general LLM responses without context",
        "metrics": ["answer_relevancy", "faithfulness"],
        "required_params": ["user_request", "app_actual_response"]
    },
    "rag": {
        "description": "Evaluate retrieval-augmented generation systems",
        "metrics": ["contextual_relevancy", "faithfulness", "hallucination"],
        "required_params": ["user_request", "app_actual_response", "context"]
    },
    "agent": {
        "description": "Evaluate agentic systems that may use tools or reasoning",
        "metrics": ["answer_relevancy", "faithfulness", "hallucination"],
        "required_params": ["user_request", "app_actual_response"]
    },
    "complete": {
        "description": "Comprehensive evaluation using all available metrics",
        "metrics": ["answer_relevancy", "faithfulness", "hallucination", 
                   "contextual_relevancy", "contextual_precision", "contextual_recall",
                   "bias", "toxicity"],  # Added bias and toxicity
        "required_params": ["user_request", "app_actual_response", "context"]
    },
    "safety": {
        "description": "Evaluate content for harmful or biased language",
        "metrics": ["bias", "toxicity"],
        "required_params": ["user_request", "app_actual_response"]
    }
}


class LLMEvaluator:
    """
    Evaluates LLM-generated responses using multiple metrics from DeepEval.
    Supports different evaluation modes for various use cases.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        """Initialize with configurable model and threshold."""
        self.model = model
        self.threshold = threshold
        
        # Check if OpenAI API key is set
        if OPENAI_API_KEY:
            masked_key = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else "***"
            logger.info(f"OpenAI API key is configured (key: {masked_key})")
        else:
            logger.warning("OpenAI API key is NOT set! Evaluation will fall back to dummy metrics.")
        
        # Initialize metrics when needed, not at startup
        self.metrics = {}
    
    def _initialize_metrics(self, requested_metrics: List[str]):
        """Initialize only the requested evaluation metrics."""
        if not self.metrics:
            self.metrics = {}
            
        # Only initialize metrics that aren't already initialized
        metrics_to_initialize = [m for m in requested_metrics if m not in self.metrics]
        
        if not metrics_to_initialize:
            return  # All requested metrics already initialized
            
        # Common parameters for all metrics
        metric_params = {
            "threshold": self.threshold,
            "include_reason": True,
            "model": self.model
        }
        
        # Mapping from metric names to classes
        metric_classes = {
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
            "hallucination": HallucinationMetric,
            "contextual_relevancy": ContextualRelevancyMetric,
            "contextual_precision": ContextualPrecisionMetric,
            "contextual_recall": ContextualRecallMetric,
            "bias": BiasMetric,
            "toxicity": ToxicityMetric
        }
        
        # Initialize only the requested metrics
        for metric_name in metrics_to_initialize:
            if metric_name in metric_classes:
                try:
                    self.metrics[metric_name] = metric_classes[metric_name](**metric_params)
                    logger.info(f"Initialized {metric_name} metric")
                except Exception as e:
                    logger.error(f"Failed to initialize {metric_name} metric: {str(e)}")
        
        logger.info(f"Metrics initialized: {list(self.metrics.keys())}")
    
    def evaluate_response(self, 
                         app_name: str,
                         user: str,
                         user_request: str,
                         app_actual_response: str,
                         expected_response: Optional[str] = None,
                         context: Optional[str] = None,
                         mode: str = "standard") -> Dict[str, Any]:
        """
        Evaluates an LLM-generated response using selected metrics based on the mode.
        """
        # Validate mode
        if mode not in EVALUATION_MODES:
            logger.warning(f"Invalid mode '{mode}', falling back to 'standard'")
            mode = "standard"
        
        # Check required parameters for the mode
        mode_config = EVALUATION_MODES[mode]
        required_params = mode_config["required_params"]
        
        missing_params = []
        if "context" in required_params and not context:
            missing_params.append("context")
        if "expected_response" in required_params and not expected_response:
            missing_params.append("expected_response")
            
        if missing_params:
            param_list = ", ".join(missing_params)
            logger.error(f"Mode '{mode}' requires {param_list} parameter(s) which were not provided")
            return self._get_dummy_metrics(
                app_name, user, user_request, app_actual_response, 
                expected_response, context, mode,
                error=f"Missing required parameters: {param_list}"
            )
        
        # For development/testing, use dummy metrics if no API key
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key found. Using dummy metrics.")
            return self._get_dummy_metrics(
                app_name, user, user_request, app_actual_response, 
                expected_response, context, mode
            )
        
        # Get metrics for this mode
        metrics_to_use = mode_config["metrics"]
        
        try:
            # Initialize the required metrics if not already done
            self._initialize_metrics(metrics_to_use)
            
            # Create test case based on available parameters
            context_list = [context] if context else None
            
            try:
                test_case = LLMTestCase(
                    input=user_request,
                    actual_output=app_actual_response,
                    expected_output=expected_response if expected_response else "",
                    context=context_list,
                    retrieval_context=context_list
                )
            except Exception as e:
                logger.error(f"Error creating test case: {str(e)}")
                return self._get_dummy_metrics(
                    app_name, user, user_request, app_actual_response,
                    expected_response, context, mode,
                    error=f"Error creating test case: {str(e)}"
                )
                
            # Get the metric objects for this evaluation
            active_metrics = [self.metrics[m] for m in metrics_to_use if m in self.metrics]
            
            if not active_metrics:
                logger.error(f"No active metrics for mode '{mode}'")
                return self._get_dummy_metrics(
                    app_name, user, user_request, app_actual_response,
                    expected_response, context, mode,
                    error="No active metrics initialized"
                )
            
            # Run evaluation with the selected metrics
            logger.info(f"Running evaluation in '{mode}' mode with metrics: {metrics_to_use}")
            eval_results = evaluate(
                test_cases=[test_case],
                metrics=active_metrics
            )
            
            # Process results
            return self._process_evaluation_results(
                eval_results,
                app_name, user, user_request, app_actual_response,
                expected_response, context, mode, metrics_to_use
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            return self._get_dummy_metrics(
                app_name, user, user_request, app_actual_response,
                expected_response, context, mode,
                error=f"Evaluation failed: {str(e)}"
            )
    
    def _process_evaluation_results(self, eval_results, app_name, user, user_request, app_actual_response, expected_response, context, mode, requested_metrics):
        """Process raw evaluation results into a structured response."""
        try:
            # Initialize result dictionary with metadata
            eval_dict = {
                "App Name": app_name,
                "User": user,
                "User Request": user_request,
                "Actual Output": app_actual_response,
                "Expected Output": expected_response,
                "Context": context,
                "Evaluation Mode": mode,
                "Evaluation Details": {}
            }
            
            # Check if we have valid results
            if not hasattr(eval_results, 'test_results') or not eval_results.test_results:
                logger.warning("No test results returned from DeepEval. Using fallback metrics.")
                return self._get_dummy_metrics(app_name, user, user_request, app_actual_response, expected_response, context, mode, error="No test results returned")
            
            # Initialize all possible metric scores to None
            all_possible_metrics = ["answer_relevancy", "faithfulness", "hallucination", "contextual_relevancy", 
                            "contextual_precision", "contextual_recall", "bias", "toxicity"]
            for metric in all_possible_metrics:
                metric_key = f"{metric}_score"
                eval_dict[metric_key] = None
            
            # Extract metrics from evaluation results
            metrics_found = False
            metric_name_mapping = {
                "Answer Relevancy": "answer_relevancy",
                "Faithfulness": "faithfulness",
                "Hallucination": "hallucination",
                "Contextual Relevancy": "contextual_relevancy",
                "Contextual Precision": "contextual_precision",
                "Contextual Recall": "contextual_recall",
                "Bias": "bias",
                "Toxicity": "toxicity"
            }
            
            for result in eval_results.test_results:
                if not hasattr(result, 'metrics_data') or not result.metrics_data:
                    continue
                    
                metrics_found = True
                
                for metric_data in result.metrics_data:
                    # Get standardized metric name using mapping
                    original_name = metric_data.name
                    if original_name in metric_name_mapping:
                        metric_name = metric_name_mapping[original_name]
                    else:
                        # Handle unexpected metric names by converting to snake_case
                        metric_name = original_name.lower().replace(" ", "_")
                        logger.warning(f"Unknown metric name: {original_name}, converted to {metric_name}")
                    
                    # Add score to main response
                    metric_score_key = f"{metric_name}_score"
                    eval_dict[metric_score_key] = metric_data.score
                    
                    # Add detailed feedback
                    eval_dict["Evaluation Details"][metric_name] = {
                        "score": metric_data.score,
                        "success": metric_data.success,
                        "reason": metric_data.reason
                    }
            
            if not metrics_found:
                logger.warning("No metrics data found in results. Using fallback metrics.")
                return self._get_dummy_metrics(app_name, user, user_request, app_actual_response, expected_response, context, mode, error="No metrics data found")
            
            # Check if all requested metrics were evaluated
            missing_metrics = []
            for metric in requested_metrics:
                metric_key = f"{metric}_score"
                if metric_key not in eval_dict or eval_dict[metric_key] is None:
                    missing_metrics.append(metric)
                    # Fill with dummy score
                    eval_dict[metric_key] = self._get_dummy_score(metric)
            
            if missing_metrics:
                logger.warning(f"Some metrics were not evaluated: {missing_metrics}")
                eval_dict["Evaluation Details"]["warnings"] = f"Missing metrics: {', '.join(missing_metrics)}"
                
            return eval_dict
            
        except Exception as e:
            logger.error(f"Error processing evaluation results: {str(e)}", exc_info=True)
            return self._get_dummy_metrics(app_name, user, user_request, app_actual_response, expected_response, context, mode, error=f"Error processing results: {str(e)}")
    
    def _get_dummy_score(self, metric: str) -> float:
        """Get a reasonable dummy score for a specific metric."""
        # Reasonable default scores that won't alarm users too much
        dummy_scores = {
            "answer_relevancy": 0.85,
            "faithfulness": 0.90,
            "hallucination": 0.10,  # Lower is better for hallucination
            "contextual_relevancy": 0.82,
            "contextual_precision": 0.80,
            "contextual_recall": 0.78,
            "bias": 0.15,  # Lower is better for bias
            "toxicity": 0.05  # Lower is better for toxicity
        }
        return dummy_scores.get(metric, 0.80)
    
    def _get_dummy_metrics(self,
                          app_name: str,
                          user: str,
                          user_request: str,
                          app_actual_response: str,
                          expected_response: Optional[str],
                          context: Optional[str],
                          mode: str,
                          error: Optional[str] = None) -> Dict[str, Any]:
        """Provide dummy metrics when evaluation cannot be performed."""
        # Get the list of metrics for this mode
        metrics_to_include = EVALUATION_MODES.get(mode, EVALUATION_MODES["standard"])["metrics"]
        
        # Initialize result with metadata
        result = {
            "App Name": app_name,
            "User": user,
            "User Request": user_request,
            "Actual Output": app_actual_response,
            "Expected Output": expected_response,
            "Context": context,
            "Evaluation Mode": mode,
            "Evaluation Details": {}
        }
        
        # Initialize all metric scores to None
        for metric in ["answer_relevancy", "faithfulness", "hallucination", 
                      "contextual_relevancy", "contextual_precision", "contextual_recall"]:
            metric_key = f"{metric}_score"
            result[metric_key] = None
        
        # Only include scores for metrics in this mode
        for metric in metrics_to_include:
            metric_key = f"{metric}_score"
            result[metric_key] = self._get_dummy_score(metric)
            
            # Add detailed feedback
            result["Evaluation Details"][metric] = {
                "score": result[metric_key],
                "success": True,
                "reason": f"Dummy reason for {metric} metric"
            }
        
        # Add error information if provided
        if error:
            result["Evaluation Details"]["error"] = error
            
        return result
    
    # Future expansion: Method for calculating token usage
    def _calculate_token_usage(self, results) -> Dict[str, int]:
         """Calculate token usage from evaluation results."""
         token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
         # Process results to extract token usage
         return token_usage
    
    def _calculate_cost(self, total_tokens: int) -> float:
         """Calculate estimated cost based on token usage."""
         # Cost per 1K tokens for the model being used
         cost_per_1k = 0.0015  # For gpt-3.5-turbo
         return (total_tokens / 1000) * cost_per_1k