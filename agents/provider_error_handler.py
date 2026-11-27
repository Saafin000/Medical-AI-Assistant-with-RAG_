"""
Provider Error Handling and Classification - Groq & Provider Agnostic

Handles provider failures with error classification and session preservation.
Does not introduce fallback providers or modify existing medical assistant architecture.

Responsibilities:
1. Classify Groq errors: HTTP 429 (rate limit vs daily quota), 401/403, 400, 5xx, connection, timeout, unexpected.
2. Return controlled HTTP responses with appropriate status codes (503 for temporary, 429 for quota, 401/400 for config).
3. Preserve conversation state and patient information during failures.
4. Never expose raw stack traces, API keys, or sensitive provider details to frontend.
5. Never fabricate medical responses.
"""

import logging
import re
import threading
from typing import Dict, Tuple, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    """Sanitize text to prevent leaking API keys or tokens in logs or messages."""
    if not text:
        return ""
    sanitized = str(text)
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9_\-]+', 'Bearer [REDACTED]', sanitized)
    sanitized = re.sub(r'gsk_[A-Za-z0-9_\-]{8,}', 'gsk_[REDACTED]', sanitized)
    sanitized = re.sub(r'sk-[A-Za-z0-9_\-]{8,}', 'sk-[REDACTED]', sanitized)
    return sanitized


class ProviderErrorClassifier:
    """Classifies provider errors to determine appropriate HTTP status and handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def classify_groq_error(self, error: Union[Exception, str]) -> Dict[str, Any]:
        """
        Classify Groq error into one of the canonical categories.
        """
        # If already structured GroqProviderError
        if hasattr(error, "category") and hasattr(error, "status_code") and hasattr(error, "error_code"):
            return {
                "is_provider_error": True,
                "category": getattr(error, "category", "unexpected_response"),
                "error_type": getattr(error, "category", "unexpected_response"),
                "http_status": getattr(error, "status_code", 503),
                "error_code": getattr(error, "error_code", "GROQ_ERROR"),
                "is_retryable": getattr(error, "is_retryable", False),
                "user_message": sanitize_text(getattr(error, "message", str(error))),
            }

        error_str = sanitize_text(str(error))
        error_lower = error_str.lower()

        # Check if this is an LLM or provider related error
        provider_indicators = [
            "groq", "chatgroq", "deepseek", "openai", "chatopenai", "api", "402", "429", "401", "403", 
            "500", "502", "503", "504", "rate limit", "balance", "insufficient", 
            "quota", "timeout", "connection", "connect", "token", "tpd", "tpm", "rpm"
        ]
        is_provider = any(ind in error_lower for ind in provider_indicators)

        # 1. Daily Token Limit (TPD) - Permanent for today, NEVER RETRY
        if any(p in error_lower for p in ["tpd", "tokens per day", "tokens_per_day", "daily token limit", "quota per day", "exceeded your current quota", "insufficient balance"]):
            return {
                "is_provider_error": True,
                "category": "daily_token_limit",
                "error_type": "quota_exceeded",
                "http_status": 429,
                "error_code": "DAILY_TOKEN_LIMIT_EXCEEDED",
                "is_retryable": False,
                "user_message": "AI services are unavailable due to exceeded Groq API daily token quota (TPD). Please check your API quota or try again later.",
            }

        # 2. Daily Request Limit (RPD) - Permanent for today, NEVER RETRY
        if any(p in error_lower for p in ["rpd", "requests per day", "requests_per_day", "daily request limit"]):
            return {
                "is_provider_error": True,
                "category": "daily_request_limit",
                "error_type": "daily_request_limit",
                "http_status": 429,
                "error_code": "DAILY_REQUEST_LIMIT_EXCEEDED",
                "is_retryable": False,
                "user_message": "AI services are unavailable due to exceeded Groq API daily request quota (RPD). Please try again tomorrow.",
            }

        # 3. HTTP 401 / 403 - Authentication or Permission
        if any(p in error_lower for p in ["401", "403", "invalid_api_key", "invalid api key", "authentication", "unauthorized", "forbidden"]):
            status = 403 if "403" in error_lower else 401
            return {
                "is_provider_error": True,
                "category": "authentication",
                "error_type": "authentication_error",
                "http_status": status,
                "error_code": "AUTHENTICATION_ERROR",
                "is_retryable": False,
                "user_message": "AI services are unavailable due to an authentication error. Please verify your Groq API key configuration.",
            }

        # 4. Invalid Model Configuration (400/404)
        if any(p in error_lower for p in ["model_not_found", "model not found", "does not exist", "invalid model"]) or ("model" in error_lower and "404" in error_lower):
            return {
                "is_provider_error": True,
                "category": "invalid_model",
                "error_type": "invalid_model",
                "http_status": 400,
                "error_code": "INVALID_MODEL",
                "is_retryable": False,
                "user_message": "Configured Groq model is invalid or unavailable. Please verify model configuration.",
            }

        # 5. HTTP 400 / 422 - Invalid Request
        if any(p in error_lower for p in ["400", "422", "invalid_request", "invalid request"]):
            return {
                "is_provider_error": True,
                "category": "invalid_request",
                "error_type": "invalid_request",
                "http_status": 400,
                "error_code": "INVALID_REQUEST",
                "is_retryable": False,
                "user_message": "Invalid request or payload configuration for Groq API. Please verify configuration settings.",
            }

        # 6. Per-Minute Request Limit (RPM)
        if any(p in error_lower for p in ["rpm", "requests per minute", "requests_per_minute"]):
            return {
                "is_provider_error": True,
                "category": "rate_limit_rpm",
                "error_type": "rate_limit_rpm",
                "http_status": 503,
                "error_code": "RPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "user_message": "AI services are temporarily unavailable due to per-minute request rate limits (RPM). Please try again in a few moments.",
            }

        # 7. Per-Minute Token Limit (TPM)
        if any(p in error_lower for p in ["tpm", "tokens per minute", "tokens_per_minute"]):
            return {
                "is_provider_error": True,
                "category": "rate_limit_tpm",
                "error_type": "rate_limit_tpm",
                "http_status": 503,
                "error_code": "TPM_LIMIT_EXCEEDED",
                "is_retryable": True,
                "user_message": "AI services are temporarily unavailable due to per-minute token rate limits (TPM). Please try again in a few moments.",
            }

        # 8. HTTP 429 - General Rate Limit
        if any(p in error_lower for p in ["429", "rate limit", "rate_limit", "too many requests", "throttle"]):
            return {
                "is_provider_error": True,
                "category": "rate_limit",
                "error_type": "rate_limit",
                "http_status": 503,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "is_retryable": True,
                "user_message": "AI services are temporarily unavailable due to rate limits. Please try again in a few moments.",
            }

        # 5. HTTP 5xx - Provider Server Error
        if any(p in error_lower for p in ["500", "502", "503", "504", "server error", "internal server error", "bad gateway", "service unavailable"]):
            return {
                "is_provider_error": True,
                "category": "server_error",
                "error_type": "server_error",
                "http_status": 503,
                "error_code": "PROVIDER_SERVER_ERROR",
                "is_retryable": True,
                "user_message": "Groq AI services are experiencing temporary server difficulties. Please try again shortly.",
            }

        # 6. Connection Errors
        if any(p in error_lower for p in ["connection error", "connection refused", "failed to resolve", "getaddrinfo failed", "connecterror", "connectionpool", "apiconnectionerror"]):
            return {
                "is_provider_error": True,
                "category": "connection_error",
                "error_type": "connection_error",
                "http_status": 503,
                "error_code": "CONNECTION_ERROR",
                "is_retryable": True,
                "user_message": "Failed to connect to Groq API. Please check network connectivity and try again.",
            }

        # 7. Request Timeout
        if any(p in error_lower for p in ["timeout", "timed out", "readtimeout", "connecttimeout"]):
            return {
                "is_provider_error": True,
                "category": "timeout",
                "error_type": "timeout",
                "http_status": 503,
                "error_code": "TIMEOUT",
                "is_retryable": True,
                "user_message": "Groq API request timed out. Please try again.",
            }

        # 8. Unexpected response or other provider error
        if is_provider:
            return {
                "is_provider_error": True,
                "category": "unexpected_response",
                "error_type": "unexpected_response",
                "http_status": 503,
                "error_code": "UNEXPECTED_RESPONSE",
                "is_retryable": False,
                "user_message": "An unexpected error was received from the AI provider. Please try again.",
            }

        # Not a provider error
        return {
            "is_provider_error": False,
            "category": "non_provider_error",
            "error_type": "general_error",
            "http_status": 500,
            "error_code": "INTERNAL_ERROR",
            "is_retryable": False,
            "user_message": "An internal error occurred while processing your request.",
        }

    def classify_deepseek_error(self, error: Union[Exception, str]) -> Dict[str, Any]:
        """Backward compatibility classifier for DeepSeek errors."""
        error_str = sanitize_text(str(error))
        error_lower = error_str.lower()

        if "402" in error_lower or "insufficient balance" in error_lower:
            return {
                "is_provider_error": True,
                "category": "billing",
                "error_type": "insufficient_balance",
                "http_status": 402,
                "error_code": "INSUFFICIENT_BALANCE",
                "is_retryable": False,
                "user_message": "AI services are unavailable due to insufficient DeepSeek API balance. Please check your account balance and billing details.",
            }
        return self.classify_groq_error(error)

    def classify_error(self, error: Union[Exception, str]) -> Dict[str, Any]:
        """General classifier method - defaults to Groq classification."""
        error_str = str(error).lower()
        if "deepseek" in error_str:
            return self.classify_deepseek_error(error)
        return self.classify_groq_error(error)

    # Backward compatibility methods
    def classify_groq_429(self, error_message: str) -> Dict[str, Any]:
        return self.classify_groq_error(error_message)

    def classify_gemini_error(self, error_message: str) -> Dict[str, Any]:
        return self.classify_groq_error(error_message)


class ProviderFailureHandler:
    """Handles provider failures and builds safe, controlled JSON responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = ProviderErrorClassifier()
    
    def handle_provider_failure(
        self,
        error: Exception,
        provider_name: str = "groq",
        patient_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Handle provider failure and return (http_status, response_content).
        
        Args:
            error: The exception from the provider
            provider_name: Provider name (defaults to "groq")
            patient_data: Optional patient data to preserve
            session_id: Session identifier
        
        Returns:
            Tuple of (http_status_code, response_dict)
        """
        classification = self.classifier.classify_error(error)
        http_status = classification["http_status"]
        
        self.logger.error(
            f"Provider failure handled: provider={provider_name}, "
            f"category={classification['category']}, "
            f"code={classification['error_code']}, status={http_status}"
        )
        
        response = {
            "status": "service_unavailable" if http_status == 503 else "error",
            "error_code": classification["error_code"],
            "message": classification["user_message"],
            "detail": classification["user_message"],  # Frontend compatible
            "session_id": session_id,
            "details": {
                "timestamp": datetime.utcnow().isoformat(),
                "provider": provider_name,
                "category": classification["category"],
                "is_retryable": classification["is_retryable"],
                "session_preserved": True,
                "patient_data_preserved": bool(patient_data)
            }
        }
        
        if patient_data:
            response["details"]["collected_info"] = {
                "symptoms": len(patient_data.get("symptoms", {}).get("list", [])),
                "medications": len(patient_data.get("medications", {}).get("list", [])),
                "allergies_status": patient_data.get("allergies", {}).get("status"),
                "medical_history_status": patient_data.get("medical_history", {}).get("status")
            }
            
        return http_status, response
    
    def build_service_unavailable_response(
        self, 
        patient_data: Optional[Dict[str, Any]] = None, 
        session_id: Optional[str] = None,
        message: Optional[str] = None,
        error_code: str = "PROVIDER_UNAVAILABLE"
    ) -> Dict[str, Any]:
        """
        Build HTTP 503 Service Unavailable response for Groq.
        """
        default_message = (
            "Groq AI services are temporarily unavailable. "
            "Your conversation session has been preserved and can resume shortly. "
            "Please try again in a few moments."
        )
        
        response = {
            "status": "service_unavailable",
            "error_code": error_code,
            "message": message or default_message,
            "detail": message or default_message,
            "session_id": session_id,
            "details": {
                "timestamp": datetime.utcnow().isoformat(),
                "provider": "groq",
                "reason": "Groq API temporarily unavailable",
                "session_preserved": True,
                "patient_data_preserved": bool(patient_data)
            }
        }
        
        if patient_data:
            response["details"]["collected_info"] = {
                "symptoms": len(patient_data.get("symptoms", {}).get("list", [])),
                "medications": len(patient_data.get("medications", {}).get("list", [])),
                "allergies_status": patient_data.get("allergies", {}).get("status"),
                "medical_history_status": patient_data.get("medical_history", {}).get("status")
            }
        
        return response


class SessionStatePreserver:
    """Preserves session patient data during provider failures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def save_session_state(self, session_id: str, patient_data: Dict[str, Any]) -> bool:
        """Save patient data for a session thread-safely."""
        if not session_id or not patient_data:
            return False
        try:
            with self._lock:
                self._session_cache[session_id] = {
                    "patient_data": patient_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            self.logger.info(f"Session {session_id[:8]}... state preserved")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")
            return False
    
    def restore_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Restore patient data for a session thread-safely."""
        if not session_id:
            return None
        with self._lock:
            if session_id in self._session_cache:
                data = self._session_cache[session_id]
                self.logger.info(f"Session {session_id[:8]}... state restored")
                return data.get("patient_data")
        return None
    
    def clear_session_state(self, session_id: str) -> bool:
        """Clear session state when completed."""
        if not session_id:
            return False
        with self._lock:
            if session_id in self._session_cache:
                del self._session_cache[session_id]
                self.logger.info(f"Session {session_id[:8]}... state cleared")
                return True
        return False


# Global singletons
error_classifier = ProviderErrorClassifier()
failure_handler = ProviderFailureHandler()
state_preserver = SessionStatePreserver()
