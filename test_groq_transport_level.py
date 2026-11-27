"""
Transport-Level Verification Test Suite for Groq HTTP Requests and Quota Handling
Task: Verify Real Groq HTTP Requests, Quota Handling, and Token Accounting

Tests:
1. Transport-level interception: ChatGroq -> groq.Groq -> SyncHttpxClientWrapper -> MockTransport
2. Verify 1 model invocation produces exactly 1 outgoing HTTP transport request (no hidden retries)
3. Verify transient 429 RPM rate limit triggers exact retry sequence at HTTP transport level
4. Verify permanent 429 TPD daily quota exhaustion halts retries immediately (exactly 1 HTTP attempt)
5. Verify permanent 401 Authentication error halts retries immediately (exactly 1 HTTP attempt)
6. Verify token accounting: tokens extracted directly from response usage metadata, not estimated
7. Verify failed requests do not increment token counts
8. Verify rate limit headers parsing (retry-after, x-ratelimit-*)
"""

import json
import os
import unittest
from unittest.mock import patch
import httpx

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

from groq_llm_provider import (
    GroqLLMProvider,
    GroqProviderError,
    GroqUsageTracker,
    create_groq_provider,
    usage_tracker
)
from groq_rate_limiter import GroqRateLimiter, get_groq_limiter


def make_mock_groq_response(
    content: str = "Test response content",
    prompt_tokens: int = 15,
    completion_tokens: int = 8,
    model: str = "qwen/qwen3.8-27b",
    status_code: int = 200,
    headers: dict = None
) -> httpx.Response:
    """Create a realistic HTTP response matching Groq's official API schema."""
    resp_headers = {
        "content-type": "application/json",
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-requests": "29",
        "x-ratelimit-reset-requests": "2s",
        "x-ratelimit-limit-tokens": "6000",
        "x-ratelimit-remaining-tokens": "5900",
        "x-ratelimit-reset-tokens": "1.5s",
    }
    if headers:
        resp_headers.update(headers)

    body = {
        "id": "chatcmpl-transport-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    return httpx.Response(status_code, headers=resp_headers, json=body)


def make_mock_groq_error_response(
    status_code: int,
    message: str,
    error_type: str = "rate_limit_exceeded",
    error_code: str = "rate_limit_exceeded",
    headers: dict = None
) -> httpx.Response:
    """Create a realistic error HTTP response matching Groq's official API schema."""
    resp_headers = {
        "content-type": "application/json",
    }
    if headers:
        resp_headers.update(headers)

    body = {
        "error": {
            "message": message,
            "type": error_type,
            "code": error_code
        }
    }
    return httpx.Response(status_code, headers=resp_headers, json=body)


class TestGroqTransportLevel(unittest.TestCase):

    def setUp(self):
        usage_tracker.reset_query_metrics()

    # 1. Verify 1 model invocation produces exactly 1 outgoing HTTP transport request
    def test_01_single_invocation_single_http_request(self):
        http_attempts = []

        def transport_handler(request: httpx.Request) -> httpx.Response:
            http_attempts.append(request)
            return make_mock_groq_response(
                content="Hello there! How can I help you?",
                prompt_tokens=25,
                completion_tokens=10
            )

        provider = GroqLLMProvider(temperature=0.3)
        # Attach MockTransport directly to underlying HTTPX client
        provider._client.client._client._client._transport = httpx.MockTransport(transport_handler)

        response = provider.invoke("Hello")

        # Verify transport-level interception
        self.assertEqual(len(http_attempts), 1, "Exactly one HTTP request must be sent across the wire")
        self.assertEqual(http_attempts[0].method, "POST")
        self.assertEqual(str(http_attempts[0].url), "https://api.groq.com/openai/v1/chat/completions")

        # Verify request metrics
        metrics = usage_tracker.get_query_metrics()
        self.assertEqual(metrics["api_requests"], 1)
        self.assertEqual(metrics["logical_invocations"], 1)
        self.assertEqual(metrics["successful_responses"], 1)
        self.assertEqual(metrics["retries"], 0)
        self.assertEqual(metrics["prompt_tokens"], 25)
        self.assertEqual(metrics["completion_tokens"], 10)
        self.assertEqual(metrics["total_tokens"], 35)

    # 2. Verify transient 429 RPM rate limit triggers retry at HTTP transport level
    def test_02_transient_rpm_rate_limit_retry_sequence(self):
        http_attempts = []

        def transport_handler(request: httpx.Request) -> httpx.Response:
            http_attempts.append(request)
            if len(http_attempts) == 1:
                # First attempt: 429 RPM with retry-after header
                return make_mock_groq_error_response(
                    status_code=429,
                    message="Rate limit reached for model qwen/qwen3.8-27b on requests per minute (RPM). Please try again in 0.2s.",
                    error_type="requests",
                    error_code="rate_limit_exceeded",
                    headers={"retry-after": "0.1", "x-ratelimit-remaining-requests": "0"}
                )
            else:
                # Second attempt: succeeds
                return make_mock_groq_response(
                    content="Recovered after rate limit",
                    prompt_tokens=30,
                    completion_tokens=8
                )

        provider = GroqLLMProvider(temperature=0.3, max_retries=2)
        provider._client.client._client._client._transport = httpx.MockTransport(transport_handler)

        response = provider.invoke("Test query")
        self.assertIn("Recovered", response.content)

        # Verify exact HTTP attempts at transport layer
        self.assertEqual(len(http_attempts), 2, "Exactly 2 physical HTTP requests must occur (1 failed + 1 retry)")
        
        metrics = usage_tracker.get_query_metrics()
        self.assertEqual(metrics["api_requests"], 2)
        self.assertEqual(metrics["logical_invocations"], 1)
        self.assertEqual(metrics["successful_responses"], 1)
        self.assertEqual(metrics["retries"], 1)
        self.assertEqual(metrics["prompt_tokens"], 30)
        self.assertEqual(metrics["completion_tokens"], 8)
        self.assertEqual(metrics["total_tokens"], 38)

    # 3. Verify permanent 429 TPD daily quota exhaustion halts immediately
    def test_03_permanent_tpd_quota_exhaustion_no_retry(self):
        http_attempts = []

        def transport_handler(request: httpx.Request) -> httpx.Response:
            http_attempts.append(request)
            return make_mock_groq_error_response(
                status_code=429,
                message="Rate limit reached for model qwen/qwen3.8-27b on tokens per day (TPD). Exceeded your current quota.",
                error_type="tokens",
                error_code="rate_limit_exceeded",
                headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "18h30m"}
            )

        provider = GroqLLMProvider(temperature=0.3, max_retries=2)
        provider._client.client._client._client._transport = httpx.MockTransport(transport_handler)

        with self.assertRaises(GroqProviderError) as ctx:
            provider.invoke("Test query")

        # Verify permanent failure halts immediately without further retries
        self.assertEqual(len(http_attempts), 1, "Must NOT retry after daily TPD quota exhaustion")
        self.assertEqual(ctx.exception.category, "quota_exceeded")
        self.assertEqual(ctx.exception.error_code, "QUOTA_EXCEEDED")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertFalse(ctx.exception.is_retryable)

        # Verify metrics: 1 attempt, 0 successful responses, 0 retries, 0 tokens
        metrics = usage_tracker.get_query_metrics()
        self.assertEqual(metrics["api_requests"], 1)
        self.assertEqual(metrics["successful_responses"], 0)
        self.assertEqual(metrics["retries"], 0)
        self.assertEqual(metrics["total_tokens"], 0)

    # 4. Verify permanent 401 Authentication error halts immediately
    def test_04_permanent_auth_error_no_retry(self):
        http_attempts = []

        def transport_handler(request: httpx.Request) -> httpx.Response:
            http_attempts.append(request)
            return make_mock_groq_error_response(
                status_code=401,
                message="Invalid API Key provided: gsk_secret_1234567890abcdef",
                error_type="invalid_request_error",
                error_code="invalid_api_key"
            )

        provider = GroqLLMProvider(temperature=0.3, max_retries=2)
        provider._client.client._client._client._transport = httpx.MockTransport(transport_handler)

        with self.assertRaises(GroqProviderError) as ctx:
            provider.invoke("Test query")

        # Verify exactly 1 HTTP attempt and no retries
        self.assertEqual(len(http_attempts), 1)
        self.assertEqual(ctx.exception.category, "authentication")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertFalse(ctx.exception.is_retryable)
        self.assertNotIn("secret", str(ctx.exception.original_error))
        self.assertIn("REDACTED", str(ctx.exception.original_error))

    # 5. Verify no hidden SDK retries bypass the tracker
    def test_05_no_hidden_sdk_retries(self):
        provider = GroqLLMProvider(temperature=0.3)
        # Check that underlying groq.Groq client has max_retries=0
        groq_client = provider._client.client._client
        self.assertEqual(groq_client.max_retries, 0, "ChatGroq must set underlying Groq client max_retries to 0")

    # 6. Verify rate limit headers parsing
    def test_06_rate_limit_headers_parsing(self):
        limiter = GroqRateLimiter()

        class MockResp:
            headers = {
                "x-ratelimit-reset-requests": "4.5s",
                "x-ratelimit-remaining-requests": "0"
            }
        err = Exception("Rate limit reached")
        err.response = MockResp()
        limiter.handle_rate_limit_error(err)
        self.assertIsNotNone(limiter._retry_after_until)

        # Test minutes parsing e.g. "2m30s"
        class MockRespMin:
            headers = {
                "x-ratelimit-reset-tokens": "2m30s",
                "x-ratelimit-remaining-tokens": "0"
            }
        err_min = Exception("Rate limit reached")
        err_min.response = MockRespMin()
        limiter.handle_rate_limit_error(err_min)
        self.assertIsNotNone(limiter._retry_after_until)


if __name__ == "__main__":
    unittest.main(verbosity=2)
