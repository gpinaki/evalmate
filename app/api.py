# app/api.py
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse  # Add this line
from app.models import EvaluationRequest, EvaluationResponse
from app.evaluation import LLMEvaluator, EVALUATION_MODES
import logging
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="LLM Evaluation API",
    description="API for evaluating LLM-generated responses using various metrics",
    version="1.0.0",
)

# Create evaluator dependency
def get_evaluator() -> LLMEvaluator:
    """Dependency function to get LLMEvaluator instance."""
    model = os.getenv("DEFAULT_EVAL_MODEL", "gpt-3.5-turbo")
    threshold = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    return LLMEvaluator(model=model, threshold=threshold)

@app.post(
    "/evaluate/", 
    response_model=EvaluationResponse,
    summary="Evaluate an LLM response",
    description="Evaluates an LLM-generated response using various metrics based on the selected mode."
)
async def evaluate_response(
    request: EvaluationRequest, 
    evaluator: LLMEvaluator = Depends(get_evaluator)
):
    """
    Evaluate an LLM response with metrics based on the specified mode.
    
    Different modes require different parameters and evaluate different aspects:
    - quick: Basic evaluation with minimal LLM calls
    - standard: General evaluation without context (default)
    - rag: Evaluation for retrieval-augmented generation systems (requires context)
    - agent: Evaluation for agentic systems
    - complete: Comprehensive evaluation using all metrics (requires context)
    - safety: Evaluation for harmful or biased content
    """
    try:
        logger.info(f"Received evaluation request for app: {request.app_name}, mode: {request.mode}")
        
        # Validate mode
        if request.mode not in EVALUATION_MODES:
            logger.warning(f"Invalid mode '{request.mode}', falling back to 'standard'")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid mode: {request.mode}", "available_modes": list(EVALUATION_MODES.keys())}
            )
            
        # Validate required parameters
        mode_config = EVALUATION_MODES[request.mode]
        required_params = mode_config["required_params"]
        
        missing_params = []
        if "context" in required_params and not request.context:
            missing_params.append("context")
        if "expected_response" in required_params and not request.expected_response:
            missing_params.append("expected_response")
            
        if missing_params:
            param_list = ", ".join(missing_params)
            return JSONResponse(
                status_code=400,
                content={"error": f"Mode '{request.mode}' requires {param_list} parameter(s) which were not provided"}
            )
        
        result = evaluator.evaluate_response(
            app_name=request.app_name,
            user=request.user,
            user_request=request.user_request,
            app_actual_response=request.app_actual_response,
            expected_response=request.expected_response,
            context=request.context,
            mode=request.mode
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error evaluating response: {str(e)}"}
        )

@app.get(
    "/modes/", 
    summary="Get available evaluation modes",
    description="Returns information about the available evaluation modes and their required parameters."
)
async def get_evaluation_modes():
    """Get information about available evaluation modes."""
    return {
        "available_modes": {
            mode: {
                "description": info["description"],
                "metrics": info["metrics"],
                "required_parameters": info["required_params"]
            } for mode, info in EVALUATION_MODES.items()
        },
        "default_mode": "standard"
    }

@app.get(
    "/estimate/", 
    summary="Estimate API calls for a mode",
    description="Estimates the number of LLM API calls required for a specific evaluation mode."
)
async def estimate_api_calls(
    mode: str = Query("standard", description="Evaluation mode")
):
    """
    Estimate the number of LLM API calls and approximate cost for a given mode.
    This helps users understand the resource implications of their evaluation choice.
    """
    if mode not in EVALUATION_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    
    metrics = EVALUATION_MODES[mode]["metrics"]
    
    # Approximate API calls per metric with optimized multi-task prompting
    api_calls_per_metric = {
        "answer_relevancy": 2,
        "faithfulness": 2,
        "hallucination": 2,
        "contextual_relevancy": 2,
        "contextual_precision": 2,
        "contextual_recall": 2,
        "bias": 2,
        "toxicity": 2
    }
    
    # Count the total calls - with optimization for certain combinations
    # We can reduce total calls by combining certain metrics
    total_calls = 0
    
    # Basic optimizations based on related metrics
    if set(["answer_relevancy", "faithfulness"]).issubset(metrics):
        # These can be combined into fewer calls
        total_calls += 3  # Instead of 2+2=4
        # Remove these from individual counting
        metrics = [m for m in metrics if m not in ["answer_relevancy", "faithfulness"]]
    
    if set(["contextual_relevancy", "hallucination"]).issubset(metrics):
        # These can be combined into fewer calls
        total_calls += 3  # Instead of 2+2=4
        # Remove these from individual counting
        metrics = [m for m in metrics if m not in ["contextual_relevancy", "hallucination"]]
    
    # Add remaining metrics
    for metric in metrics:
        total_calls += api_calls_per_metric.get(metric, 2)
    
    # Estimate cost
    token_estimate = 800  # Approximate tokens per API call
    cost_per_1k_tokens = 0.0015  # Cost for gpt-3.5-turbo, adjust for other models
    estimated_cost = (total_calls * token_estimate * cost_per_1k_tokens) / 1000
    
    return {
        "mode": mode,
        "metrics": EVALUATION_MODES[mode]["metrics"],
        "estimated_api_calls": total_calls,
        "estimated_tokens": total_calls * token_estimate,
        "estimated_cost_usd": round(estimated_cost, 4)
    }

@app.get("/health", summary="Health check endpoint")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy"}