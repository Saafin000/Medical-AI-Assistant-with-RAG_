"""
Groq Rate Limiter — Minimal in-process request coordination
Task 1.8.0K: 429 Stabilization

Coordinates all Groq requests to avoid concurrent 429 errors.

Does NOT:
- Change models, prompts, or token counts
- Introduce external dependencies
- Modify any agents or workflows
- Replace LLM Decision with deterministic logic

Does:
- Serialize Groq requests with bounded exponential backoff
- Respect Retry-After headers
- Return clear errors when quota is exhausted
"""

import time
import threading
from typing import Callable, TypeVar, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class GroqRateLimiter:
    """
    Shared rate limiter for all Groq LLM invocations.
    
    Prevents concurrent Groq API calls that trigger 429 errors.
    Uses bounded exponential backoff with jitter.
    """
    
    def __init__(self, min_interval_seconds: float = 0.1, max_retries: int = 3):
        """
        Initialize rate limiter.
        
        Args:
            min_interval_seconds: Minimum time between requests (default: 100ms)
            max_retries: Maximum retry attempts on 429/503 (default: 3)
        """
        self._lock = threading.RLock()
        self._last_request_time = None
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._retry_after_until = None
        
        logger.info(
            f"GroqRateLimiter initialized: "
            f"min_interval={min_interval_seconds}s, max_retries={max_retries}"
        )
    
    def wait_if_needed(self) -> None:
        """
        Wait until the minimum interval has passed since last request.
        
        Respects Retry-After cooldown if previously encountered.
        """
        with self._lock:
            now = datetime.now()
            
            # Check if we're in a Retry-After cooldown period
            if self._retry_after_until and now < self._retry_after_until:
                wait_secs = (self._retry_after_until - now).total_seconds()
                logger.warning(
                    f"Groq rate limit cooldown: waiting {wait_secs:.1f}s "
                    f"until {self._retry_after_until.isoformat()}"
                )
                time.sleep(wait_secs)
                self._retry_after_until = None
            
            # Check minimum interval
            if self._last_request_time:
                elapsed = (now - self._last_request_time).total_seconds()
                if elapsed < self._min_interval:
                    wait_secs = self._min_interval - elapsed
                    time.sleep(wait_secs)
            
            self._last_request_time = datetime.now()

    def record_completion(self) -> None:
        """Record completion of an HTTP request to pace consecutive requests."""
        with self._lock:
            self._last_request_time = datetime.now()
    
    def handle_rate_limit_error(self, error: Exception) -> None:
        """
        Process a rate-limit error and set cooldown if Retry-After provided.
        Inspects error.response headers and error text for explicit wait intervals.
        
        Args:
            error: The exception from Groq API
        """
        with self._lock:
            error_str = str(error)
            retry_secs = None
            
            # 1. First priority: Check exact cooldown in error message (e.g., "Please try again in 1m52.752s" or "try again in 1.6s" or "850ms")
            import re
            m_comp = re.search(r'try again in\s+(?:(\d+)m\s*)?([0-9.]+)\s*(ms|s)?', error_str, re.IGNORECASE)
            if m_comp:
                try:
                    mins = float(m_comp.group(1)) if m_comp.group(1) else 0.0
                    secs_val = float(m_comp.group(2))
                    unit = (m_comp.group(3) or "s").lower()
                    if unit == "ms":
                        secs_val = secs_val / 1000.0
                    retry_secs = (mins * 60.0) + secs_val
                except ValueError:
                    pass

            # 2. Check token rate-limit reset header (reset-tokens) before requests reset
            response = getattr(error, "response", None)
            if retry_secs is None and response is not None and hasattr(response, "headers"):
                # Prefer x-ratelimit-reset-tokens since TPM is the primary rate limit
                for hdr_name in ["x-ratelimit-reset-tokens", "retry-after", "Retry-After", "x-ratelimit-reset-requests"]:
                    hdr_val = response.headers.get(hdr_name)
                    if hdr_val:
                        # Match milliseconds like "172ms"
                        m_ms = re.search(r'^([0-9.]+)\s*ms$', hdr_val.strip(), re.IGNORECASE)
                        if m_ms:
                            try:
                                retry_secs = float(m_ms.group(1)) / 1000.0
                                break
                            except ValueError:
                                pass
                        # Match pure seconds like "4.375s" or "2s"
                        m = re.search(r'^([0-9.]+)\s*s?$', hdr_val.strip(), re.IGNORECASE)
                        if m:
                            try:
                                retry_secs = float(m.group(1))
                                break
                            except ValueError:
                                pass
                        # Match minutes + seconds like "2m15s"
                        m_min = re.search(r'(\d+)m(?:(\d+)s)?', hdr_val, re.IGNORECASE)
                        if m_min:
                            try:
                                mins = int(m_min.group(1))
                                secs = int(m_min.group(2) or 0)
                                retry_secs = float(mins * 60 + secs)
                                break
                            except ValueError:
                                pass

            if retry_secs is None:
                match = re.search(r'retry-after[:\s]+([0-9.]+)', error_str, re.IGNORECASE)
                if match:
                    try:
                        retry_secs = float(match.group(1))
                    except ValueError:
                        pass

            if retry_secs is not None:
                # Cap maximum single cooldown at 30s to allow normal rolling window recovery
                retry_secs = min(max(retry_secs, 0.5), 30.0)
                self._retry_after_until = datetime.now() + timedelta(seconds=retry_secs)
                logger.warning(
                    f"Groq rate limit cooldown: waiting {retry_secs:.2f}s "
                    f"(until {self._retry_after_until.isoformat()})"
                )
                return

            # Fallback: exponential backoff (1s, 2s, 4s)
            backoff_secs = min(2 ** (self._max_retries - 1), 30)
            self._retry_after_until = datetime.now() + timedelta(seconds=backoff_secs)
            logger.warning(
                f"Groq rate limit detected. Cooldown: {backoff_secs}s"
            )
    
    def invoke_with_retry(
        self,
        llm_invoke_fn: Callable[[], T],
        operation_name: str = "Groq LLM call"
    ) -> T:
        """
        Execute LLM invocation with rate-limit retry logic.
        
        Args:
            llm_invoke_fn: Callable that invokes the LLM (e.g., llm.invoke(prompt))
            operation_name: Human-readable name for logging
        
        Returns:
            LLM response
        
        Raises:
            Exception: If all retries exhausted or non-retryable error
        """
        attempt = 0
        last_error = None
        
        while attempt <= self._max_retries:
            try:
                # Wait for rate limit clearance before attempting
                self.wait_if_needed()
                
                # Execute the LLM call
                logger.debug(f"{operation_name} (attempt {attempt + 1}/{self._max_retries + 1})")
                result = llm_invoke_fn()
                return result
            
            except Exception as e:
                error_str = str(e)
                last_error = e
                attempt += 1
                
                # Check if error is retryable
                is_rate_limit = (
                    "429" in error_str or
                    "Too Many Requests" in error_str or
                    "rate limit" in error_str.lower()
                )
                is_temporary = (
                    "503" in error_str or
                    "Service Unavailable" in error_str or
                    "timeout" in error_str.lower()
                )
                
                if not (is_rate_limit or is_temporary):
                    # Non-retryable error (auth, validation, etc.)
                    logger.error(f"{operation_name} failed (non-retryable): {error_str[:200]}")
                    raise
                
                # Retryable error
                if attempt <= self._max_retries:
                    logger.warning(
                        f"{operation_name} hit {'rate limit' if is_rate_limit else 'temporary failure'} "
                        f"(attempt {attempt}/{self._max_retries})"
                    )
                    
                    # Update cooldown
                    self.handle_rate_limit_error(e)
                else:
                    logger.error(
                        f"{operation_name} exhausted retries after {attempt} attempts: "
                        f"{error_str[:200]}"
                    )
        
        # All retries exhausted
        raise last_error if last_error else RuntimeError(
            f"{operation_name} failed after {self._max_retries} retries"
        )


# Global shared rate limiter instance
_groq_limiter = GroqRateLimiter(
    min_interval_seconds=2.0,  # 2.0s between requests to prevent burst 429
    max_retries=4               # 4 retry attempts
)


def get_groq_limiter() -> GroqRateLimiter:
    """Get the shared Groq rate limiter instance."""
    return _groq_limiter


def invoke_with_rate_limit(
    llm_invoke_fn: Callable[[], T],
    operation_name: str = "Groq LLM call"
) -> T:
    """
    Convenience function to invoke LLM with rate-limit protection.
    
    Example:
        response = invoke_with_rate_limit(
            lambda: llm.invoke(prompt),
            operation_name="Agent Decision"
        )
    """
    return _groq_limiter.invoke_with_retry(llm_invoke_fn, operation_name)
