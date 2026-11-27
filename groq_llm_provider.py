"""
Groq-Only LLM Provider with Rate Limiting, Request Tracking, and Quota Protection

Focused, reliable integration with Groq API via ChatGroq.
No fallback. No secondary providers. Only Groq.

Configuration:
- Provider: Groq
- Model: GROQ_MODEL environment variable (default: qwen/qwen3.8-27b)
- API Key: GROQ_API_KEY environment variable

Features:
- Minimal in-process rate coordination via GroqRateLimiter
- Bounded retry for transient failures (RPM/TPM, 5xx, timeout, connection)
- Fast-fail for permanent failures (TPD/RPD daily limits, 401 auth, 400 invalid model/request)
- Never leaks API keys or sensitive headers in error messages
- Per-query actual HTTP request, retry, and token usage tracking for quota visibility
- Single-instance caching by (model, temperature) to prevent duplicate provider initialization
- Implements LangChain BaseChatModel interface for drop-in compatibility
"""

import inspect
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_groq import ChatGroq
from pydantic import Field

from groq_rate_limiter import get_groq_limiter, GroqRateLimiter

logger = logging.getLogger(__name__)


def sanitize_error_text(text: str, api_key: Optional[str] = None) -> str:
    """Strip API keys, tokens, and sensitive headers from error messages."""
    if not text:
        return ""
    sanitized = str(text)
    if api_key and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    # Redact Authorization headers or raw tokens
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9_\-]+', 'Bearer [REDACTED]', sanitized)
    sanitized = re.sub(r'gsk_[A-Za-z0-9_\-]{8,}', 'gsk_[REDACTED]', sanitized)
    sanitized = re.sub(r'sk-[A-Za-z0-9_\-]{8,}', 'sk-[REDACTED]', sanitized)
    return sanitized


class GroqProviderError(RuntimeError):
    """
    Structured exception for Groq API errors.
    Carries classification metadata without exposing sensitive credentials.
    """
    def __init__(
        self,
        message: str,
        category: str = "unexpected_response",
        status_code: int = 503,
        error_code: str = "GROQ_ERROR",
        is_retryable: bool = False,
        original_error: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.status_code = status_code
        self.error_code = error_code
        self.is_retryable = is_retryable
        self.original_error = original_error

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "category": self.category,
            "status_code": self.status_code,
            "is_retryable": self.is_retryable,
            "message": self.message
        }


class GroqUsageTracker:
    """
    Thread-safe tracker for Groq API usage.
    Distinguishes:
    - logical model invocations called by application agents
    - actual physical HTTP/API request attempts across the wire
    - successful HTTP responses received
    - retries triggered
    - tokens consumed (prompt, completion, total)
    - breakdown by workflow stage
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.query_logical_invocations: int = 0
        self.query_api_requests: int = 0
        self.query_successful_responses: int = 0
        self.query_invocations: int = 0  # for backward compatibility
        self.query_retries: int = 0
        self.query_prompt_tokens: int = 0
        self.query_completion_tokens: int = 0
        self.query_total_tokens: int = 0
        self.query_stage_calls: Dict[str, Dict[str, int]] = {}

        self.total_requests: int = 0
        self.total_tokens: int = 0

    def record_logical_invocation(self, stage: Optional[str] = None) -> None:
        """Record a logical invocation from an agent or graph node."""
        stg = stage or "unspecified"
        with self._lock:
            self.query_logical_invocations += 1
            self.query_invocations += 1
            if stg not in self.query_stage_calls:
                self.query_stage_calls[stg] = {
                    "invocations": 0, "requests": 0, "successful_responses": 0,
                    "retries": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0
                }
            self.query_stage_calls[stg]["invocations"] = self.query_stage_calls[stg].get("invocations", 0) + 1

    def record_request_attempt(self, stage: Optional[str] = None) -> None:
        """Record an actual physical HTTP API request attempt across the transport."""
        stg = stage or "unspecified"
        with self._lock:
            self.query_api_requests += 1
            self.total_requests += 1
            if stg not in self.query_stage_calls:
                self.query_stage_calls[stg] = {
                    "invocations": 0, "requests": 0, "successful_responses": 0,
                    "retries": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0
                }
            self.query_stage_calls[stg]["requests"] += 1

    def record_retry(self, stage: Optional[str] = None) -> None:
        """Record a retry attempt after a transient failure."""
        stg = stage or "unspecified"
        with self._lock:
            self.query_retries += 1
            if stg in self.query_stage_calls:
                self.query_stage_calls[stg]["retries"] += 1

    def record_response(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        stage: Optional[str] = None
    ) -> None:
        """Record tokens and successful HTTP response from Groq."""
        stg = stage or "unspecified"
        tot = total_tokens if total_tokens > 0 else (prompt_tokens + completion_tokens)
        with self._lock:
            self.query_successful_responses += 1
            if self.query_logical_invocations < self.query_successful_responses:
                self.query_logical_invocations = self.query_successful_responses
                self.query_invocations = self.query_successful_responses
            self.query_prompt_tokens += prompt_tokens
            self.query_completion_tokens += completion_tokens
            self.query_total_tokens += tot
            self.total_tokens += tot

            if stg not in self.query_stage_calls:
                self.query_stage_calls[stg] = {
                    "invocations": 0, "requests": 0, "successful_responses": 0,
                    "retries": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0
                }
            self.query_stage_calls[stg]["successful_responses"] = self.query_stage_calls[stg].get("successful_responses", 0) + 1
            self.query_stage_calls[stg]["prompt_tokens"] += prompt_tokens
            self.query_stage_calls[stg]["completion_tokens"] += completion_tokens
            self.query_stage_calls[stg]["total_tokens"] += tot

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0) -> None:
        """Backward compatibility method."""
        self.record_logical_invocation()
        self.record_request_attempt()
        self.record_response(prompt_tokens, completion_tokens, total_tokens)

    def reset_query_metrics(self) -> None:
        with self._lock:
            self.query_logical_invocations = 0
            self.query_api_requests = 0
            self.query_successful_responses = 0
            self.query_invocations = 0
            self.query_retries = 0
            self.query_prompt_tokens = 0
            self.query_completion_tokens = 0
            self.query_total_tokens = 0
            self.query_stage_calls = {}

    def get_query_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests": self.query_api_requests,  # for backward compatibility
                "api_requests": self.query_api_requests,
                "logical_invocations": self.query_logical_invocations,
                "model_invocations": self.query_invocations,
                "successful_responses": self.query_successful_responses,
                "retries": self.query_retries,
                "prompt_tokens": self.query_prompt_tokens,
                "completion_tokens": self.query_completion_tokens,
                "total_tokens": self.query_total_tokens,
                "stage_breakdown": {k: dict(v) for k, v in self.query_stage_calls.items()},
            }

    def get_total_metrics(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
            }


# Global usage tracker instance
usage_tracker = GroqUsageTracker()


def _detect_caller_stage() -> str:
    """Detect which workflow component or agent is calling the LLM."""
    try:
        stack = inspect.stack()
        for frame_info in stack[2:12]:
            module_name = frame_info.frame.f_globals.get("__name__", "")
            func_name = frame_info.function
            
            if "local_guardrails" in module_name:
                if "check_input" in func_name:
                    return "guardrail_input"
                if "check_output" in func_name:
                    return "guardrail_output"
                return "guardrails"
            elif "agent_decision" in module_name:
                if "route_to_agent" in func_name:
                    return "agent_decision_routing"
                if "run_conversation_agent" in func_name:
                    return "conversation_agent"
                if "run_clinical" in func_name:
                    return "clinical_agent"
            elif "query_expander" in module_name:
                return "rag_query_expander"
            elif "response_generator" in module_name:
                return "rag_response_generator"
            elif "content_processor" in module_name:
                return "rag_chunker"
            elif "web_search" in module_name:
                if "web_search_query" in func_name:
                    return "web_search_query_gen"
                return "web_search_processor"
    except Exception:
        pass
    return "general_llm_call"


class GroqLLMProvider(BaseChatModel):
    """
    Groq-only LLM provider using ChatGroq.
    Implements LangChain BaseChatModel interface for drop-in compatibility with agents.
    Strictly zero fallback: no Gemini, no DeepSeek.
    """

    model_id: str = Field(default="qwen/qwen3.8-27b", description="Groq model ID")
    temperature: float = Field(default=0.3, description="Temperature for generation")
    groq_api_key: str = Field(..., description="Groq API key")
    max_retries: int = Field(default=2, description="Max retries for transient errors")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for completion")

    # Internal state
    _client: Optional[ChatGroq] = None
    _rate_limiter: Optional[GroqRateLimiter] = None

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.3,
        groq_api_key: Optional[str] = None,
        max_retries: int = 4,
        timeout: int = 60,
        max_tokens: Optional[int] = None,
    ):
        actual_model = model_id or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        actual_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not actual_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass groq_api_key parameter."
            )

        super().__init__(
            model_id=actual_model,
            temperature=temperature,
            groq_api_key=actual_key,
            max_retries=max_retries,
            timeout=timeout,
            max_tokens=max_tokens,
        )

        client_kwargs = {
            "model": self.model_id,
            "temperature": self.temperature,
            "api_key": self.groq_api_key,
            "max_retries": 0,
            "timeout": self.timeout,
        }
        if self.max_tokens:
            client_kwargs["max_tokens"] = self.max_tokens

        # Set max_retries=0 on ChatGroq so underlying client does not perform hidden unmetered retries.
        # All retries are explicit, coordinated by GroqRateLimiter, and accurately measured by GroqUsageTracker.
        self._client = ChatGroq(**client_kwargs)
        self._rate_limiter = get_groq_limiter()

        logger.info(
            f"GroqLLMProvider initialized: model={self.model_id}, "
            f"temperature={self.temperature}, max_retries={self.max_retries}"
        )

    def _classify_error(self, error: Exception) -> dict:
        """
        Classify error into permanent vs transient with appropriate metadata.
        Distinguishes RPD, TPD, RPM, TPM, auth, invalid model, and server/connection errors.
        """
        error_str = sanitize_error_text(str(error), self.groq_api_key)
        error_lower = error_str.lower()

        # 1. Daily Token Limit (TPD) - Permanent for today / 24h rolling window, DO NOT RETRY
        if any(p in error_lower for p in ["tokens per day", "tokens_per_day", "daily token limit", "daily quota", "(tpd)"]):
            # Extract wait time if present for clarity
            wait_info = ""
            m_wait = re.search(r'try again in\s+([^\.]+)', error_str, re.IGNORECASE)
            if m_wait:
                wait_info = f" (Resets in {m_wait.group(1).strip()})"
            return {
                "category": "quota_exceeded",
                "status_code": 429,
                "error_code": "QUOTA_EXCEEDED",
                "is_retryable": False,
                "message": f"Groq API daily token quota (200,000 TPD) reached{wait_info}. Please check your Groq API plan or quota limits.",
                "original_error": error_str
            }

        # 2. Daily Request Limit (RPD) - Permanent for today, NEVER RETRY
        if any(p in error_lower for p in ["rpd", "requests per day", "requests_per_day", "daily request limit"]):
            return {
                "category": "daily_request_limit",
                "status_code": 429,
                "error_code": "DAILY_REQUEST_LIMIT_EXCEEDED",
                "is_retryable": False,
                "message": "Groq API daily request limit (RPD) exceeded.",
                "original_error": error_str
            }

        # 3. Per-Minute Token Limit (TPM) - Transient, retryable with backoff
        if "try again in" in error_lower or any(p in error_lower for p in ["tpm", "tokens per minute", "tokens_per_minute"]):
            return {
                "category": "rate_limit_tpm",
                "status_code": 503,
                "error_code": "TPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "Groq API per-minute token limit (TPM) reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 4. Per-Minute Request Limit (RPM) - Transient, retryable with backoff
        if any(p in error_lower for p in ["rpm", "requests per minute", "requests_per_minute"]):
            return {
                "category": "rate_limit_rpm",
                "status_code": 503,
                "error_code": "RPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "Groq API per-minute request limit (RPM) reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 3. HTTP 401 / 403 - Authentication or Permission - Permanent, NEVER RETRY
        if any(p in error_lower for p in ["401", "403", "invalid_api_key", "invalid api key", "authentication", "unauthorized", "forbidden"]):
            return {
                "category": "authentication",
                "status_code": 401 if "403" not in error_lower else 403,
                "error_code": "AUTHENTICATION_ERROR",
                "is_retryable": False,
                "message": "Groq API authentication failed. Please verify your GROQ_API_KEY.",
                "original_error": error_str
            }

        # 4. Invalid Model Configuration (400/404) - Permanent, NEVER RETRY
        if any(p in error_lower for p in ["model_not_found", "model not found", "does not exist", "invalid model"]):
            return {
                "category": "invalid_model",
                "status_code": 400,
                "error_code": "INVALID_MODEL",
                "is_retryable": False,
                "message": f"Configured Groq model '{self.model_id}' is not found or unsupported.",
                "original_error": error_str
            }

        # 5. HTTP 400 / 422 - Invalid Request - Permanent, NEVER RETRY
        if any(p in error_lower for p in ["400", "422", "invalid_request", "invalid request"]):
            return {
                "category": "invalid_request",
                "status_code": 400,
                "error_code": "INVALID_REQUEST",
                "is_retryable": False,
                "message": "Invalid request or payload configuration for Groq API.",
                "original_error": error_str
            }

        # 6. Per-Minute Request Limit (RPM) - Transient, retryable with backoff
        if any(p in error_lower for p in ["rpm", "requests per minute", "requests_per_minute"]):
            return {
                "category": "rate_limit_rpm",
                "status_code": 503,
                "error_code": "RPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "Groq API per-minute request limit (RPM) reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 7. Per-Minute Token Limit (TPM) - Transient, retryable with backoff
        if any(p in error_lower for p in ["tpm", "tokens per minute", "tokens_per_minute"]):
            return {
                "category": "rate_limit_tpm",
                "status_code": 503,
                "error_code": "TPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "Groq API per-minute token limit (TPM) reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 8. Unspecified HTTP 429 - Rate limit - Transient, retryable
        if any(p in error_lower for p in ["429", "rate limit", "rate_limit", "too many requests", "throttle"]):
            return {
                "category": "rate_limit",
                "status_code": 503,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "Groq API rate limit reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 9. HTTP 5xx - Provider server error - Transient, retryable
        if any(p in error_lower for p in ["500", "502", "503", "504", "server error", "internal server error", "bad gateway", "service unavailable"]):
            return {
                "category": "server_error",
                "status_code": 503,
                "error_code": "PROVIDER_SERVER_ERROR",
                "is_retryable": True,
                "message": "Groq server error. Service temporarily unavailable.",
                "original_error": error_str
            }

        # 10. Connection Errors - Transient, retryable
        if any(p in error_lower for p in ["connection error", "connection refused", "failed to resolve", "getaddrinfo failed", "connecterror", "connectionpool", "apiconnectionerror"]):
            return {
                "category": "connection_error",
                "status_code": 503,
                "error_code": "CONNECTION_ERROR",
                "is_retryable": True,
                "message": "Failed to connect to Groq API. Network issue or provider endpoint unreachable.",
                "original_error": error_str
            }

        # 11. Request Timeout - Transient, retryable
        if any(p in error_lower for p in ["timeout", "timed out", "readtimeout", "connecttimeout"]):
            return {
                "category": "timeout",
                "status_code": 503,
                "error_code": "TIMEOUT",
                "is_retryable": True,
                "message": "Groq API request timed out.",
                "original_error": error_str
            }

        # 12. Unknown / Unexpected Provider Response
        return {
            "category": "unexpected_response",
            "status_code": 503,
            "error_code": "UNEXPECTED_RESPONSE",
            "is_retryable": False,
            "message": "Unexpected error received from Groq provider.",
            "original_error": error_str
        }

    def _track_tokens_from_response(self, response: Any, stage: Optional[str] = None) -> None:
        """Extract and record token usage metrics from an AIMessage or ChatResult."""
        try:
            if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    prompt_toks = usage.get("prompt_tokens", 0)
                    compl_toks = usage.get("completion_tokens", 0)
                    tot_toks = usage.get("total_tokens", 0)
                    usage_tracker.record_response(prompt_toks, compl_toks, tot_toks, stage=stage)
                    return
            if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                usage = response.llm_output.get("token_usage", {})
                if usage:
                    usage_tracker.record_response(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                        stage=stage
                    )
                    return
            usage_tracker.record_response(0, 0, 0, stage=stage)
        except Exception as e:
            logger.debug(f"Could not track tokens: {e}")

    def invoke(self, input: Union[str, list, dict, Any], config=None, **kwargs) -> AIMessage:
        """
        Invoke Groq LLM with rate limiting, attempt tracking, and bounded retries.
        Permanent errors (daily quota, auth, invalid model) fail immediately.
        """
        if config is None:
            config = {}

        if self._client is None:
            raise RuntimeError("Groq LLM client is not initialized.")

        # Determine workflow stage for instrumentation
        stage = kwargs.pop("stage", None) or config.get("metadata", {}).get("stage") or _detect_caller_stage()

        # Record high-level logical model invocation
        usage_tracker.record_logical_invocation(stage=stage)

        attempt = 0
        last_classified = None

        while attempt <= self.max_retries:
            # Record each physical API request attempt
            usage_tracker.record_request_attempt(stage=stage)

            if attempt > 0:
                usage_tracker.record_retry(stage=stage)

            try:
                # Rate limiter coordination
                if self._rate_limiter:
                    self._rate_limiter.wait_if_needed()

                logger.debug(f"Groq invoke [{stage}] attempt {attempt + 1}/{self.max_retries + 1}")
                response = self._client.invoke(input, config=config, **kwargs)
                if self._rate_limiter:
                    self._rate_limiter.record_completion()

                # Record token usage metrics and successful invocation
                self._track_tokens_from_response(response, stage=stage)
                return response

            except Exception as e:
                classified = self._classify_error(e)
                last_classified = classified

                # Notify rate limiter of rate-limit error to set backoff/Retry-After cooldown
                if self._rate_limiter and classified.get("is_retryable"):
                    self._rate_limiter.handle_rate_limit_error(e)

                # Check if error is permanent - fail immediately without retrying
                if not classified["is_retryable"]:
                    logger.error(
                        f"Groq permanent failure ({classified['category']}): {classified['error_code']}"
                    )
                    raise GroqProviderError(
                        message=classified["message"],
                        category=classified["category"],
                        status_code=classified["status_code"],
                        error_code=classified["error_code"],
                        is_retryable=False,
                        original_error=classified["original_error"]
                    ) from e

                # Check if retry limit reached
                if attempt >= self.max_retries:
                    logger.error(
                        f"Groq API transient failure persisted after {self.max_retries + 1} attempts. "
                        f"Category: {classified['category']}, Code: {classified['error_code']}"
                    )
                    raise GroqProviderError(
                        message=classified["message"],
                        category=classified["category"],
                        status_code=classified["status_code"],
                        error_code=classified["error_code"],
                        is_retryable=True,
                        original_error=classified["original_error"]
                    ) from e

                # Bounded exponential backoff: 1s, 2s (capped at 3s)
                backoff_seconds = min(1.0 * (2 ** attempt), 3.0)
                logger.warning(
                    f"Groq transient failure on attempt {attempt + 1}/{self.max_retries + 1}. "
                    f"Retrying in {backoff_seconds:.1f}s. Category: {classified['category']}"
                )
                time.sleep(backoff_seconds)
                attempt += 1

        if last_classified:
            raise GroqProviderError(
                message=last_classified["message"],
                category=last_classified["category"],
                status_code=last_classified["status_code"],
                error_code=last_classified["error_code"],
                is_retryable=last_classified["is_retryable"],
                original_error=last_classified["original_error"]
            )
        raise GroqProviderError("Groq API invocation failed.")

    def stream(self, input: Union[str, list, dict, Any], config=None, **kwargs):
        """
        Stream Groq LLM response chunks with rate limiting, attempt tracking, and bounded retries.
        Yields AIMessageChunk instances as tokens arrive.
        """
        if config is None:
            config = {}

        if self._client is None:
            raise RuntimeError("Groq LLM client is not initialized.")

        stage = kwargs.pop("stage", None) or config.get("metadata", {}).get("stage") or _detect_caller_stage()

        usage_tracker.record_logical_invocation(stage=stage)

        attempt = 0
        last_classified = None

        while attempt <= self.max_retries:
            usage_tracker.record_request_attempt(stage=stage)
            if attempt > 0:
                usage_tracker.record_retry(stage=stage)

            if self._rate_limiter:
                self._rate_limiter.wait_if_needed()

            logger.debug(f"Groq stream [{stage}] attempt {attempt + 1}/{self.max_retries + 1}")
            tokens_yielded = 0
            try:
                # We obtain the iterator and fetch the first chunk to ensure connection succeeded
                stream_iter = iter(self._client.stream(input, config=config, **kwargs))
                try:
                    first_chunk = next(stream_iter)
                except StopIteration:
                    return

                if first_chunk and hasattr(first_chunk, "content") and first_chunk.content:
                    tokens_yielded += 1
                yield first_chunk

                for chunk in stream_iter:
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        tokens_yielded += 1
                    yield chunk

                if self._rate_limiter:
                    self._rate_limiter.record_completion()

                usage_tracker.record_response(
                    prompt_tokens=0,
                    completion_tokens=tokens_yielded,
                    total_tokens=tokens_yielded,
                    stage=stage
                )
                return

            except Exception as e:
                classified = self._classify_error(e)
                last_classified = classified

                if self._rate_limiter and classified.get("is_retryable"):
                    self._rate_limiter.handle_rate_limit_error(e)

                # If we already yielded tokens, we cannot cleanly retry the stream
                if tokens_yielded > 0 or not classified.get("is_retryable") or attempt >= self.max_retries:
                    logger.error(f"Groq stream failure ({classified['category']}): {classified['error_code']}")
                    raise GroqProviderError(
                        message=classified["message"],
                        category=classified["category"],
                        status_code=classified["status_code"],
                        error_code=classified["error_code"],
                        is_retryable=classified["is_retryable"],
                        original_error=classified["original_error"]
                    ) from e

                backoff_seconds = min(1.0 * (2 ** attempt), 3.0)
                logger.warning(
                    f"Groq stream transient failure on attempt {attempt + 1}/{self.max_retries + 1}. "
                    f"Retrying in {backoff_seconds:.1f}s. Category: {classified['category']}"
                )
                time.sleep(backoff_seconds)
                attempt += 1

        if last_classified:
            raise GroqProviderError(
                message=last_classified["message"],
                category=last_classified["category"],
                status_code=last_classified["status_code"],
                error_code=last_classified["error_code"],
                is_retryable=last_classified["is_retryable"],
                original_error=last_classified["original_error"]
            )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ):
        """LangChain internal method for streaming."""
        for chunk in self.stream(messages, stop=stop, **kwargs):
            generation_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=generation_chunk)
            yield generation_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> ChatResult:
        """LangChain internal method for generation."""
        response = self.invoke(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier."""
        return "groq"


# Provider cache to prevent duplicate provider initialization
_provider_cache: Dict[Tuple[str, float], GroqLLMProvider] = {}
_cache_lock = threading.Lock()


def create_groq_provider(
    model_id: Optional[str] = None,
    temperature: float = 0.3,
    groq_api_key: Optional[str] = None,
    max_retries: int = 2,
    timeout: int = 60,
    max_tokens: Optional[int] = None,
) -> GroqLLMProvider:
    """
    Factory function to create or retrieve cached Groq provider.
    Reuses existing instances with matching (model, temperature, max_tokens) to optimize memory and connections.
    """
    actual_model = model_id or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    cache_key = (actual_model, round(temperature, 2), max_tokens)

    with _cache_lock:
        if cache_key in _provider_cache:
            return _provider_cache[cache_key]

        provider = GroqLLMProvider(
            model_id=actual_model,
            temperature=temperature,
            groq_api_key=groq_api_key,
            max_retries=max_retries,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        _provider_cache[cache_key] = provider
        return provider
