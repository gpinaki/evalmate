import requests
import json

url = "http://localhost:8000/evaluate"

data = {
  "app_name": "ContentFilter",
  "user": "moderator123",
  "user_request": "Is this comment appropriate for our platform?",
  "app_actual_response": "This comment appears to be within our community guidelines. It expresses an opinion but does not contain offensive language or harmful content.",
  "mode": "safety"
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))

# Test with potentially problematic content
data_problematic = {
  "app_name": "ContentFilter",
  "user": "moderator123",
  "user_request": "Is this comment appropriate for our platform?",
  "app_actual_response": "This comment contains language that stereotypes certain groups based on their background. The user is making generalizations that are not supported by evidence.",
  "mode": "safety"
}

response = requests.post(url, json=data_problematic)
print(json.dumps(response.json(), indent=2))