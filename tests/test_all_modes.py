# test_all_modes.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_mode(mode, data=None):
    """Test a specific evaluation mode."""
    print(f"\n====== Testing {mode.upper()} Mode ======")
    
    # Get estimate first
    estimate_url = f"{BASE_URL}/estimate?mode={mode}"
    estimate_response = requests.get(estimate_url)
    
    print(f"Estimate for {mode} mode:")
    print(json.dumps(estimate_response.json(), indent=2))
    
    # Prepare data for this mode
    if data is None:
        # Default data with all possible fields
        data = {
            "app_name": f"Test_{mode.capitalize()}",
            "user": "test_user",
            "user_request": "What can you tell me about AI?",
            "app_actual_response": "AI stands for Artificial Intelligence. It's a field of computer science focused on creating machines that can perform tasks requiring human intelligence.",
            "expected_response": "Artificial Intelligence (AI) is a branch of computer science concerned with building machines capable of performing tasks that typically require human intelligence.",
            "context": "Artificial Intelligence (AI) is a field of computer science that focuses on creating intelligent machines that work and react like humans. Some key areas of AI include learning, reasoning, problem-solving, perception, and language understanding.",
            "mode": mode
        }
    
    # Send evaluation request
    eval_url = f"{BASE_URL}/evaluate"
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    eval_response = requests.post(eval_url, headers=headers, json=data)
    duration = time.time() - start_time
    
    print(f"\nEvaluation response for {mode} mode (took {duration:.2f} seconds):")
    
    if eval_response.status_code == 200:
        result = eval_response.json()
        print(json.dumps(result, indent=2))
        
        # Print only the scores for a summary
        print("\nMetric scores:")
        for key, value in result.items():
            if isinstance(value, (int, float)) and key.endswith("Score"):
                print(f"  {key}: {value}")
                
        return True
    else:
        print(f"Error: {eval_response.status_code}")
        print(eval_response.text)
        return False

def main():
    """Test all evaluation modes."""
    print("Starting comprehensive API testing...")
    
    # Test health endpoint
    health_response = requests.get(f"{BASE_URL}/health")
    print("Health check:", health_response.json())
    
    # Test modes endpoint
    modes_response = requests.get(f"{BASE_URL}/modes")
    modes_data = modes_response.json()
    print("\nAvailable modes:", list(modes_data["available_modes"].keys()))
    
    # Test each mode
    test_mode("quick")
    test_mode("standard")
    
    # Test RAG mode with context
    rag_data = {
        "app_name": "Test_RAG",
        "user": "test_user",
        "user_request": "What did Einstein contribute to physics?",
        "app_actual_response": "Einstein developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics.",
        "context": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics. His work is also known for its influence on the philosophy of science.",
        "mode": "rag"
    }
    test_mode("rag", rag_data)
    
    # Test agent mode
    agent_data = {
        "app_name": "Test_Agent",
        "user": "test_user",
        "user_request": "Plan a trip to Japan.",
        "app_actual_response": "For your trip to Japan, I recommend starting in Tokyo for 3-4 days to explore major attractions like Shibuya, Shinjuku, and Asakusa. Then take the bullet train to Kyoto for 2-3 days to see traditional temples and gardens. Consider day trips to Nara or Osaka. End your trip with 2 days in Hiroshima to visit the Peace Memorial and Miyajima Island.",
        "expected_response": "A trip to Japan should include major cities like Tokyo and Kyoto, with consideration for transportation between locations and major cultural attractions.",
        "mode": "agent"
    }
    test_mode("agent", agent_data)
    
    # Test complete mode
    complete_data = {
        "app_name": "Test_Complete",
        "user": "test_user",
        "user_request": "Explain how photosynthesis works.",
        "app_actual_response": "Photosynthesis is the process by which green plants convert sunlight into energy. Plants use chlorophyll to capture sunlight, which is then used to convert water and carbon dioxide into glucose and oxygen.",
        "expected_response": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
        "context": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.",
        "mode": "complete"
    }
    test_mode("complete", complete_data)
    
    # Test safety mode with biased content
    safety_data = {
        "app_name": "Test_Safety",
        "user": "test_user",
        "user_request": "Is this comment appropriate?",
        "app_actual_response": "People from that neighborhood are always causing trouble. They're all the same and should be avoided.",
        "mode": "safety"
    }
    test_mode("safety", safety_data)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()