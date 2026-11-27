"""
DeepSeek-Only LLM Provider

Focused, reliable integration with DeepSeek API via OpenAI-compatible client.
No fallback. No provider switching. Only DeepSeek.

Configuration:
- Provider: DeepSeek  
- Base URL: https://api.deepseek.com
- Model: deepseek-flash
- API Key: DEEPSEEK_API_KEY environment variable

Error Handling:
- Transient errors (429, 5xx, timeout, connection): bounded retry with capped backoff
- Permanent errors (402 balance, 401/403 auth, 400 bad request): fail immediately without retrying
- Never leaks API keys or raw stack traces in error messages
- No silent fallback to other providers

Implements LangChain BaseChatModel interface for drop-in compatibility.
"""

import logging
import os
import re
import time
from typing import Any, Optional, Union
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import Field

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
    sanitized = re.sub(r'sk-[A-Za-z0-9_\-]{8,}', 'sk-[REDACTED]', sanitized)
    return sanitized


class DeepSeekProviderError(RuntimeError):
    """
    Structured exception for DeepSeek API errors.
    Carries classification metadata without exposing sensitive credentials.
    """
    def __init__(
        self,
        message: str,
        category: str = "unexpected_response",
        status_code: int = 503,
        error_code: str = "DEEPSEEK_ERROR",
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


class DeepSeekLLMProvider(BaseChatModel):
    """
    DeepSeek-only LLM provider using OpenAI-compatible API.
    Implements LangChain BaseChatModel interface for compatibility with agents.
    """
    
    # Pydantic fields
    model_id: str = Field(default="deepseek-flash", description="DeepSeek model ID")
    temperature: float = Field(..., description="Temperature for generation")
    deepseek_api_key: str = Field(..., description="DeepSeek API key")
    base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek base URL")
    max_retries: int = Field(default=2, description="Max retries for transient errors")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
    # Internal state
    _client: ChatOpenAI = None
    
    def __init__(
        self,
        model_id: str = "deepseek-flash",
        temperature: float = 0.3,
        deepseek_api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 2,
        timeout: int = 60,
    ):
        """
        Initialize DeepSeek provider.
        
        Args:
            model_id: DeepSeek model ID (default: deepseek-flash)
            temperature: Temperature for generation
            deepseek_api_key: DeepSeek API key (uses env var DEEPSEEK_API_KEY if not provided)
            base_url: DeepSeek base URL
            max_retries: Max retries for transient errors
            timeout: Request timeout in seconds
        """
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            deepseek_api_key=deepseek_api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        
        if not self.deepseek_api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY environment variable or pass deepseek_api_key parameter."
            )
        
        # Initialize OpenAI-compatible client (configured for DeepSeek)
        # max_retries is set to 0 on ChatOpenAI to avoid nested duplicate retries
        self._client = ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=self.deepseek_api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout,
        )
        
        logger.info(
            f"DeepSeekLLMProvider initialized: "
            f"model={model_id}, "
            f"base_url={base_url}, "
            f"temperature={temperature}, "
            f"max_retries={max_retries}"
        )

    def _classify_error(self, error: Exception) -> dict:
        """
        Classify error into permanent vs transient with appropriate metadata.
        """
        error_str = sanitize_error_text(str(error), self.deepseek_api_key)
        error_lower = error_str.lower()

        # 1. HTTP 402 - Insufficient Balance (Permanent billing failure - NEVER RETRY)
        if "402" in error_lower or "insufficient balance" in error_lower or "insufficient_balance" in error_lower:
            return {
                "category": "billing",
                "status_code": 402,
                "error_code": "INSUFFICIENT_BALANCE",
                "is_retryable": False,
                "message": "DeepSeek API balance insufficient. Please check your account balance and billing details.",
                "original_error": error_str
            }

        # 2. HTTP 401 / 403 - Authentication or Permission (Permanent - NEVER RETRY)
        if any(p in error_lower for p in ["401", "403", "invalid_api_key", "authentication", "unauthorized", "forbidden"]):
            return {
                "category": "authentication",
                "status_code": 401 if "403" not in error_lower else 403,
                "error_code": "AUTHENTICATION_ERROR",
                "is_retryable": False,
                "message": "DeepSeek API authentication failed. Please verify your DEEPSEEK_API_KEY.",
                "original_error": error_str
            }

        # 3. HTTP 400 / 422 - Invalid Request or Model Configuration (Permanent - NEVER RETRY)
        if any(p in error_lower for p in ["400", "422", "invalid_request", "invalid request", "model_not_found"]):
            return {
                "category": "invalid_request",
                "status_code": 400,
                "error_code": "INVALID_REQUEST",
                "is_retryable": False,
                "message": "Invalid request or model configuration for DeepSeek API.",
                "original_error": error_str
            }

        # 4. HTTP 429 - Rate limit (Transient - retryable with bounded backoff)
        if any(p in error_lower for p in ["429", "rate limit", "rate_limit", "too many requests", "throttle"]):
            return {
                "category": "rate_limit",
                "status_code": 503,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "is_retryable": True,
                "message": "DeepSeek API rate limit reached. Service temporarily busy.",
                "original_error": error_str
            }

        # 5. HTTP 5xx - Provider server error (Transient - retryable)
        if any(p in error_lower for p in ["500", "502", "503", "504", "server error", "internal server error", "bad gateway", "service unavailable"]):
            return {
                "category": "server_error",
                "status_code": 503,
                "error_code": "PROVIDER_SERVER_ERROR",
                "is_retryable": True,
                "message": "DeepSeek server error. Service temporarily unavailable.",
                "original_error": error_str
            }

        # 6. Connection Errors (Transient - retryable)
        if any(p in error_lower for p in ["connection error", "connection refused", "failed to resolve", "getaddrinfo failed", "connecterror", "connectionpool"]):
            return {
                "category": "connection_error",
                "status_code": 503,
                "error_code": "CONNECTION_ERROR",
                "is_retryable": True,
                "message": "Failed to connect to DeepSeek API. Network issue or provider endpoint unreachable.",
                "original_error": error_str
            }

        # 7. Request Timeout (Transient - retryable)
        if any(p in error_lower for p in ["timeout", "timed out", "readtimeout", "connecttimeout"]):
            return {
                "category": "timeout",
                "status_code": 503,
                "error_code": "TIMEOUT",
                "is_retryable": True,
                "message": "DeepSeek API request timed out.",
                "original_error": error_str
            }

        # 8. Unknown / Unexpected Provider Response (Treat as permanent to avoid infinite retries)
        return {
            "category": "unexpected_response",
            "status_code": 503,
            "error_code": "UNEXPECTED_RESPONSE",
            "is_retryable": False,
            "message": "Unexpected error received from DeepSeek provider.",
            "original_error": error_str
        }

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if error is transient (retryable)."""
        classification = self._classify_error(error)
        return classification.get("is_retryable", False)
    
    def invoke(self, input: Union[str, list, dict], config=None, **kwargs):
        """
        Invoke DeepSeek LLM with bounded retry logic for transient errors.
        Permanent errors fail immediately without retries.
        
        Args:
            input: Prompt text or message list
            config: LangChain config dict
            **kwargs: Additional arguments
        
        Returns:
            AIMessage response
        
        Raises:
            DeepSeekProviderError: If API call fails
            ValueError: If input is invalid
        """
        if config is None:
            config = {}
        
        # Convert input to string representation if needed
        if isinstance(input, list):
            text = "\n".join(str(m) for m in input)
        elif isinstance(input, dict):
            text = str(input)
        else:
            text = str(input)
        
        if not text or not text.strip():
            raise ValueError("Cannot invoke with empty input")
        
        last_classified = None
        
        # Retry loop with bounded backoff
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"DeepSeek invoke attempt {attempt + 1}/{self.max_retries + 1}")
                response = self._client.invoke(text)
                return response
            
            except Exception as e:
                classified = self._classify_error(e)
                last_classified = classified
                
                # Check if error is permanent - fail immediately
                if not classified["is_retryable"]:
                    logger.error(
                        f"DeepSeek permanent failure ({classified['category']}): {classified['error_code']}"
                    )
                    raise DeepSeekProviderError(
                        message=classified["message"],
                        category=classified["category"],
                        status_code=classified["status_code"],
                        error_code=classified["error_code"],
                        is_retryable=False,
                        original_error=classified["original_error"]
                    ) from e
                
                # Transient error check if last attempt reached
                if attempt >= self.max_retries:
                    logger.error(
                        f"DeepSeek API transient failure persisted after {self.max_retries + 1} attempts. "
                        f"Category: {classified['category']}, Code: {classified['error_code']}"
                    )
                    raise DeepSeekProviderError(
                        message=classified["message"],
                        category=classified["category"],
                        status_code=classified["status_code"],
                        error_code=classified["error_code"],
                        is_retryable=True,
                        original_error=classified["original_error"]
                    ) from e
                
                # Bounded backoff: 1s, 2s (capped at 3s)
                backoff_seconds = min(1.0 * (2 ** attempt), 3.0)
                logger.warning(
                    f"DeepSeek transient failure on attempt {attempt + 1}/{self.max_retries + 1}. "
                    f"Retrying in {backoff_seconds:.1f}s. "
                    f"Category: {classified['category']}"
                )
                time.sleep(backoff_seconds)
        
        if last_classified:
            raise DeepSeekProviderError(
                message=last_classified["message"],
                category=last_classified["category"],
                status_code=last_classified["status_code"],
                error_code=last_classified["error_code"],
                is_retryable=last_classified["is_retryable"],
                original_error=last_classified["original_error"]
            )
        raise DeepSeekProviderError("DeepSeek API invocation failed.")
    
    def _generate(self, messages: list, stop: Optional[list] = None, **kwargs):
        """LangChain internal method for generation."""
        return self.invoke(messages)
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier."""
        return "deepseek"


def create_deepseek_provider(
    model_id: str = "deepseek-flash",
    temperature: float = 0.3,
    deepseek_api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    max_retries: int = 2,
    timeout: int = 60,
) -> DeepSeekLLMProvider:
    """Factory function to create DeepSeek provider."""
    return DeepSeekLLMProvider(
        model_id=model_id,
        temperature=temperature,
        deepseek_api_key=deepseek_api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
