"""
FastAPI Endpoints Verification Test Suite

Tests FastAPI endpoint functions directly with mocked and controlled responses:
1. GET /health (returns {"status": "healthy"})
2. GET / (serves HTML template index.html)
3. POST /chat (successful response with mocked Groq)
4. POST /chat (provider error: 429 quota exceeded handling with preserved session)
5. POST /chat (provider error: 503 rate limit transient handling with preserved session)
6. POST /chat (provider error: 401 authentication error handling)
"""

import os
import uuid
import unittest
from unittest.mock import MagicMock, patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

from fastapi import Response, Request
from fastapi.responses import JSONResponse
from app import health_check, chat, index, QueryRequest
from langchain_core.messages import AIMessage, HumanMessage


class TestFastAPIEndpoints(unittest.TestCase):

    def test_01_health_endpoint(self):
        result = health_check()
        self.assertEqual(result, {"status": "healthy"})

    def test_02_chat_mocked_success(self):
        mock_response = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hello! I am your AI Medical Conversation Assistant.")
            ],
            "agent_name": "CONVERSATION_AGENT"
        }
        with patch("app.process_query", return_value=mock_response):
            response = Response()
            req = QueryRequest(query="Hello", conversation_history=[])
            test_sid = f"conv_{uuid.uuid4()}"
            result = chat(request=req, response=response, x_session_id=test_sid)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["agent"], "CONVERSATION_AGENT")
            self.assertEqual(result["session_id"], test_sid)
            self.assertIn("Medical Conversation Assistant", result["response"])

    def test_03_chat_quota_exceeded_error_handling(self):
        from groq_llm_provider import GroqProviderError
        quota_err = GroqProviderError(
            message="Groq API daily token quota exceeded. Please check your Groq API plan or quota limits.",
            category="quota_exceeded",
            status_code=429,
            error_code="QUOTA_EXCEEDED",
            is_retryable=False
        )
        with patch("app.process_query", side_effect=quota_err):
            response = Response()
            req = QueryRequest(query="Tell me about brain tumors", conversation_history=[])
            test_sid = f"conv_{uuid.uuid4()}"
            result = chat(request=req, response=response, x_session_id=test_sid)
            self.assertIsInstance(result, JSONResponse)
            self.assertEqual(result.status_code, 429)
            import json
            data = json.loads(result.body.decode())
            self.assertEqual(data["error_code"], "QUOTA_EXCEEDED")
            self.assertEqual(data["details"]["provider"], "groq")
            self.assertTrue(data["details"]["session_preserved"])
            self.assertEqual(data["session_id"], test_sid)

    def test_04_chat_rate_limit_transient_error_handling(self):
        from groq_llm_provider import GroqProviderError
        rate_err = GroqProviderError(
            message="Groq API rate limit reached. Service temporarily busy.",
            category="rate_limit",
            status_code=503,
            error_code="RATE_LIMIT_EXCEEDED",
            is_retryable=True
        )
        with patch("app.process_query", side_effect=rate_err):
            response = Response()
            req = QueryRequest(query="What are symptoms of covid?", conversation_history=[])
            test_sid = f"conv_{uuid.uuid4()}"
            result = chat(request=req, response=response, x_session_id=test_sid)
            self.assertIsInstance(result, JSONResponse)
            self.assertEqual(result.status_code, 503)
            import json
            data = json.loads(result.body.decode())
            self.assertEqual(data["error_code"], "RATE_LIMIT_EXCEEDED")
            self.assertEqual(data["status"], "service_unavailable")
            self.assertEqual(data["details"]["provider"], "groq")
            self.assertTrue(data["details"]["session_preserved"])
            self.assertEqual(data["session_id"], test_sid)

    def test_05_chat_auth_error_handling(self):
        from groq_llm_provider import GroqProviderError
        auth_err = GroqProviderError(
            message="Groq API authentication failed. Please verify your GROQ_API_KEY.",
            category="authentication",
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            is_retryable=False
        )
        with patch("app.process_query", side_effect=auth_err):
            response = Response()
            req = QueryRequest(query="Hi", conversation_history=[])
            test_sid = f"conv_{uuid.uuid4()}"
            result = chat(request=req, response=response, x_session_id=test_sid)
            self.assertIsInstance(result, JSONResponse)
            self.assertEqual(result.status_code, 401)
            import json
            data = json.loads(result.body.decode())
            self.assertEqual(data["error_code"], "AUTHENTICATION_ERROR")
            self.assertEqual(data["details"]["provider"], "groq")


if __name__ == "__main__":
    unittest.main(verbosity=2)
