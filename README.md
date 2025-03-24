
# LLM Evaluation API: EvalMate

A FastAPI-based API for evaluating LLM-generated responses using configurable quality and safety metrics.

## About This Project

This open-source project is built on top of [DeepEval](https://github.com/confident-ai/deepeval), an evaluation framework for LLM applications. It wraps DeepEval’s core functionality in a structured, developer-friendly API designed for experimentation, prototyping, and integration.

Note: This API is a standalone personal project and not associated with any client or organization. All code and examples are generic and domain-neutral.

Key Enhancements
Clean REST API for easy integration into apps and pipelines
Multiple evaluation modes tailored to various LLM use cases
Token usage and cost estimation support
Extended support for custom safety metrics

⚠️ **Disclaimer:** This project is a personal initiative developed independently for learning and open-source contribution. It is not affiliated with, sponsored by, or representative of the views, work, or intellectual property of any current or former employer or client. All examples and content are generic and domain-neutral.


## Features

- Support for multiple evaluation modes: quick, standard, rag, agent, complete, and safety
- Rich set of evaluation metrics: relevance, faithfulness, hallucination, bias, toxicity, and more
- Token usage estimation to assist in cost transparency
- Modular, extensible architecture designed for future customization
- Optimized evaluation through selective metric invocation

## Metrics Explained

| Metric | Description | Score Range | Lower is Better? |
|--------|-------------|-------------|-----------------|
| Answer Relevancy | Measures how well the response addresses the query | 0.0-1.0 | No |
| Faithfulness | Evaluates if the response is faithful to provided information | 0.0-1.0 | No |
| Hallucination | Detects fabricated information in the response | 0.0-1.0 | Yes |
| Contextual Relevancy | Assesses if the response uses relevant context | 0.0-1.0 | No |
| Contextual Precision | Measures precision of context usage | 0.0-1.0 | No |
| Contextual Recall | Evaluates how much relevant context is included | 0.0-1.0 | No |
| Bias | Detects biased language or viewpoints | 0.0-1.0 | Yes |
| Toxicity | Identifies harmful or inappropriate content | 0.0-1.0 | Yes |

## Evaluation Modes

| Mode | Description | Metrics Used | Required Parameters |
|------|-------------|--------------|---------------------|
| quick | Minimal, fast evaluation | Answer Relevancy | user_request, app_actual_response |
| standard | Balanced evaluation without context | Answer Relevancy, Faithfulness | user_request, app_actual_response |
| rag | For retrieval-augmented generation | Contextual Relevancy, Faithfulness, Hallucination | user_request, app_actual_response, context |
| agent | For AI agents and tools | Answer Relevancy, Faithfulness, Hallucination | user_request, app_actual_response |
| complete | Comprehensive evaluation | All metrics | user_request, app_actual_response, context |
| safety | Content moderation evaluation | Bias, Toxicity | user_request, app_actual_response |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /evaluate/ | POST | Evaluate an LLM response using specified mode |
| /estimate/ | GET | Estimate API calls and cost for an evaluation mode |
| /modes/ | GET | Get information about available evaluation modes |
| /token-tracking-info/ | GET | Information about token usage estimation |
| /health | GET | Check if the API is running |

## Project Structure
```
llm-eval-demo/
│── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic data models for API validation
│   ├── evaluation.py      # LLM evaluation logic
│   ├── api.py             # FastAPI application and routes
│── tests/
│   ├── __init__.py
│   ├── test_api.py        # API endpoint tests
│── main.py                # Entry point to run FastAPI
│── requirements.txt       # Dependencies
│── README.md              # Documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   DEFAULT_EVAL_MODEL=gpt-3.5-turbo
   DEFAULT_THRESHOLD=0.5
   ```

## Running the API

```bash
python main.py
```

Or use uvicorn directly:

```bash
uvicorn app.api:app --reload
```

## API Documentation

Once running, access the Swagger UI at:
http://localhost:8000/docs

## Evaluation Modes

- **quick**: Minimal evaluation with just Answer Relevancy
- **standard** (default): General evaluation with Answer Relevancy and Faithfulness
- **rag**: For RAG systems, with Contextual Relevancy, Faithfulness, and Hallucination
- **agent**: For AI agents, with Answer Relevancy, Faithfulness, and Hallucination
- **complete**: Comprehensive evaluation with all available metrics

## Token Usage Estimation
The API includes approximate token tracking and cost estimates based on metric usage patterns. This is useful for developers looking to manage LLM evaluation costs.

## Future Roadmap

- **Accurate Token Tracking**: Implement precise token usage tracking for all metrics
- **Provider Alternatives**: Support for multiple LLM providers beyond OpenAI
- **Custom Metrics**: Allow users to define and use their own evaluation metrics
- **Evaluation History**: Store and analyze evaluation results over time
- **Web Dashboard**: Visual interface for evaluation management and results analysis

## Example API Request

### Quick Mode
```
{
  "app_name": "QuickChat",
  "user": "user123",
  "user_request": "What is the capital of France?",
  "app_actual_response": "The capital of France is Paris.",
  "mode": "quick"
}
```

### Standard Mode (Default)
```
{
  "app_name": "CustomerService",
  "user": "client456",
  "user_request": "How do I reset my password?",
  "app_actual_response": "To reset your password, go to the login page and click on 'Forgot Password'. Follow the instructions sent to your email.",
  "expected_response": "You can reset your password through the 'Forgot Password' link on our login page.",
  "mode": "standard"
}
```

### RAG Mode 
```
{
  "app_name": "LegalAssistant",
  "user": "lawyer789",
  "user_request": "What are the key points in the 2023 privacy regulation update?",
  "app_actual_response": "The 2023 privacy regulation update includes stricter data breach notification requirements, increased fines for non-compliance, and new rules for AI systems processing personal data.",
  "context": "The 2023 privacy regulation update introduces three major changes: 1) Companies must report data breaches within 48 hours; 2) Maximum fines increased to 5% of global revenue; 3) AI systems processing personal data require explicit consent and documentation of training data sources.",
  "mode": "rag"
}
```

### Agent Mode 
```
{
  "app_name": "TravelPlanner",
  "user": "traveler321",
  "user_request": "Plan a 3-day itinerary for Tokyo.",
  "app_actual_response": "Here's your 3-day Tokyo itinerary: Day 1: Visit Senso-ji Temple in Asakusa, explore the Tokyo Skytree, and enjoy dinner in Shinjuku. Day 2: Morning at the Tsukiji Outer Market, afternoon in Harajuku, and evening in Shibuya. Day 3: Day trip to Kamakura to see the Great Buddha, then return to Tokyo for final shopping in Ginza.",
  "expected_response": "A 3-day Tokyo itinerary should include major attractions like Senso-ji Temple, Tokyo Skytree, Shibuya Crossing, Harajuku, and nearby areas of interest.",
  "mode": "agent"
}
```

### Complete Mode 
```
{
  "app_name": "MedicalAdvisor",
  "user": "patient567",
  "user_request": "What are the symptoms of seasonal allergies?",
  "app_actual_response": "Common seasonal allergy symptoms include sneezing, runny or stuffy nose, watery and itchy eyes, itchy throat or ears, and postnasal drip. Some people may also experience fatigue due to poor quality sleep caused by symptoms.",
  "expected_response": "Seasonal allergies typically present with symptoms such as sneezing, nasal congestion, runny nose, itchy or watery eyes, itchy throat or ears, and sometimes fatigue.",
  "context": "Seasonal allergies, also known as allergic rhinitis or hay fever, occur when your immune system overreacts to outdoor allergens like pollen from trees, grasses, and weeds. Symptoms include sneezing, itching of the nose, eyes, or roof of the mouth, runny nose, stuffy nose, red and watery eyes, and sometimes fatigue. The timing of these symptoms depends on which pollens you're allergic to and geographic location. Many people take antihistamines, decongestants, or use nasal sprays to manage symptoms.",
  "mode": "complete"
}
```

### Safety Mode
```
{
  "app_name": "ContentModeration",
  "user": "demo_user",
  "user_request": "Can you review this response for a customer support system?",
  "app_actual_response": "I'm sorry, but that's a stupid question. Go figure it out yourself or maybe read the manual for once instead of wasting everyone's time.",
  "mode": "safety"
}
```

## Example API Response

```
{
  "App Name": "Example",
  "User": "test_user",
  "User Request": "What is AI?",
  "Actual Output": "AI is artificial intelligence...",
  "Expected Output": "Artificial Intelligence is...",
  "Context": "AI refers to...",
  "Evaluation Mode": "standard",
  "Answer Relevancy Score": 0.92,
  "Faithfulness Score": 0.88,
  "Evaluation Details": {
    "answer_relevancy": {
      "score": 0.92,
      "success": true,
      "reason": "The response directly addresses the question..."
    },
    "faithfulness": {
      "score": 0.88,
      "success": true,
      "reason": "The response aligns with the provided information..."
    }
  }
}
```

### Error Handling
```
{
  "error": "Mode 'rag' requires context parameter(s) which were not provided"
}
```


## Running Tests

```bash
pytest tests/
```

## Development Process

This project was developed using a combination of:
- Manual coding and domain expertise
- AI assistance with code generation and documentation using Claude (Anthropic)
- Test-driven development to ensure reliability

AI-assisted development helped expedite the implementation while maintaining code quality and best practices. All generated code was reviewed, tested, and integrated to meet project requirements.

## Disclaimer

This repository and its contents are entirely personal and do not represent the views of any company or organization. No proprietary code, confidential data, or client-specific implementation details are included. All work has been developed independently for open-source experimentation and public sharing.

## License

MIT

## Acknowledgments

- [DeepEval](https://github.com/confident-ai/deepeval) for providing the foundational evaluation metrics
- OpenAI for their API that powers the evaluation capabilities
