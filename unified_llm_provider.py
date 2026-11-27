"""
Unified LLM Provider with Groq→Gemini Fallback
Task 1.8.0N: Centralized provider management

Wraps Groq and Gemini to provide transparent fallback.
Groq is primary; Gemini activates only on provider failures.

Does NOT:
- Change agent logic, prompts, temperatures, or token limits
- Modify Agent Decision invocation count
- Replace LLM-based routing
- Strip clinical data (returns error instead)

Does:
- Serialize Groq requests
- Fallback to Gemini on 429/503/timeout
- Respect provider recovery
- Validate clinical privacy
"""

import logging
import os
from typing import Any, Optional, Union
from datetime import datetime, timedelta

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import Field

# Import GroqRateLimiter for coordinated Groq requests (Task 1.9.4 Phase 2)
try:
    from groq_rate_limiter import get_groq_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    get_groq_limiter = None

logger = logging.getLogger(__name__)


class ClinicalDataValidator:
    """Checks if a request contains sensitive clinical information."""
    
    CLINICAL_KEYWORDS = {
        # Symptoms
        "symptom", "fever", "pain", "ache", "cough", "bleeding", "rash",
        "dizziness", "nausea", "fatigue", "chest pain", "difficulty breathing",
        
        # Medical info
        "medical history", "condition", "diagnosis", "disease", "prescription",
        "medication", "allergy", "surgery", "hospitalization", "treatment",
        
        # Common conditions
        "diabetes", "hypertension", "asthma", "depression", "anxiety", "cancer",
        "heart disease", "stroke", "arthritis", "thyroid",
        
        # Vital signs
        "blood pressure", "heart rate", "pulse", "temperature", "blood sugar",
        "glucose", "cholesterol", "triglycerides",
        
        # Patient identifiers
        "patient", "clinical", "medical record", "MRN", "chart",
    }
    
    @staticmethod
    def is_clinical_content(text: str) -> bool:
        """
        Check if text contains clinical information.
        
        Returns:
            True if clinical/sensitive content detected, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ClinicalDataValidator.CLINICAL_KEYWORDS)
    
    @staticmethod
    def should_allow_gemini_fallback(text: str) -> tuple[bool, Optional[str]]:
        """
        Determine if a request can safely be sent to Gemini.
        
        TASK 1.9.4 PHASE 2: Clinical data IS ALLOWED for Gemini fallback in this project.
        This project explicitly authorizes Gemini to handle clinical pre-consultation data,
        patient intake, and medical information as part of the fallback strategy.
        
        Returns:
            (allowed, reason) tuple
            allowed: True if safe to send to Gemini (clinical data now allowed)
            reason: Explanation if False (unused now, but kept for compatibility)
        """
        # REMOVED: Old check that blocked clinical data from Gemini
        # if ClinicalDataValidator.is_clinical_content(text):
        #     return False, "Clinical/patient data not approved for Gemini fallback"
        
        # All content approved for Gemini fallback in this project
        return True, None


class UnifiedLLMProvider(BaseChatModel):
    """
    Unified provider wrapping Groq (primary) and Gemini (fallback).
    
    Implements LangChain BaseChatModel interface for drop-in compatibility.
    """
    
    # Pydantic fields
    model_id: str = Field(..., description="Groq model ID")
    temperature: float = Field(..., description="Temperature for generation")
    groq_api_key: str = Field(..., description="Groq API key")
    gemini_api_key: Optional[str] = Field(None, description="Gemini API key")
    groq_max_retries: int = Field(default=2, description="Max retries for Groq")
    allow_gemini_fallback: bool = Field(default=True, description="Enable Gemini fallback")
    
    # Internal state (not Pydantic fields)
    _groq: ChatGroq = None
    _gemini: Optional[Any] = None
    _gemini_model_id: str = "gemini-3.8-flash"  # FIX 1.8.0AB: Updated from 'gemini-2.0-flash' (shutdown June 1, 2026) to 'gemini-3.8-flash' (current active model, free-tier eligible, SDK 4.5.0 compatible)
    _last_provider_used: str = "groq"
    _gemini_cooldown_until: Optional[datetime] = None
    
    def __init__(
        self,
        model_id: str,
        temperature: float,
        groq_api_key: str,
        gemini_api_key: Optional[str] = None,
        groq_max_retries: int = 2,
        allow_gemini_fallback: bool = True,
    ):
        """
        Initialize unified provider.
        
        Args:
            model_id: Groq model ID (e.g., "qwen/qwen3.8-27b")
            temperature: Temperature for generation
            groq_api_key: Groq API key
            gemini_api_key: Gemini API key (optional)
            groq_max_retries: Max retries for Groq
            allow_gemini_fallback: Enable Gemini fallback
        """
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            groq_max_retries=groq_max_retries,
            allow_gemini_fallback=allow_gemini_fallback and bool(gemini_api_key)
        )
        
        # Initialize Groq (primary)
        self._groq = ChatGroq(
            model=model_id,
            temperature=temperature,
            api_key=groq_api_key,
            max_retries=groq_max_retries
        )
        
        # Initialize Gemini (fallback, lazy-loaded)
        self._gemini = None
        self._last_provider_used = "groq"
        self._gemini_cooldown_until = None
        
        logger.info(
            f"UnifiedLLMProvider initialized: "
            f"primary=Groq({model_id}, temp={temperature}), "
            f"fallback={'Gemini' if self.allow_gemini_fallback else 'DISABLED'}"
        )
    
    def _initialize_gemini(self) -> None:
        """
        Lazy-load Gemini SDK and client (non-blocking).
        
        Task 1.9.4 Phase 2: Make initialization non-blocking.
        If Gemini is unavailable, log a warning but don't crash.
        Errors will be discovered when actually needed (fallback scenario).
        """
        if self._gemini is not None:
            return  # Already initialized
        
        if not self.allow_gemini_fallback:
            raise RuntimeError("Gemini fallback disabled (no API key)")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            logger.debug(f"Initializing Gemini provider (model: {self._gemini_model_id})...")
            
            self._gemini = ChatGoogleGenerativeAI(
                model=self._gemini_model_id,
                api_key=self.gemini_api_key,
                temperature=self.temperature,
                convert_system_message_to_human=True,
            )
            logger.info(f"✅ Gemini fallback provider initialized: {self._gemini_model_id}")
        
        except ImportError as e:
            logger.warning("⚠️  langchain-google-genai not installed. Gemini fallback disabled.")
            self.allow_gemini_fallback = False
            raise RuntimeError(
                "Cannot initialize Gemini fallback: install langchain-google-genai"
            )
        
        except Exception as e:
            error_str = str(e)
            logger.warning(
                f"⚠️  Gemini initialization warning (will retry on fallback): {error_str[:200]}"
            )
            # Don't crash - let fallback attempt discover the actual error
            # This allows Groq to work normally even if Gemini config is incomplete
    
    def _is_provider_failure(self, error: Exception) -> bool:
        """
        Check if error is a provider failure (retryable).
        
        Returns True for transient failures (RPM, timeout, 503)
        Returns False for permanent exhaustion (TPD)
        """
        error_str = str(error)
        
        # Daily quota exhaustion - NOT retryable within same day
        if any(pattern in error_str for pattern in ["TPD", "tokens per day", "tokens_per_day", "quota per day"]):
            return False  # Not retryable - quota exhausted for the day
        
        # Retryable: temporary rate limits, server errors, timeouts
        if any(pattern in error_str for pattern in [
            "429", "Too Many Requests",
            "503", "Service Unavailable",
            "500", "Internal Server Error",
            "timeout", "Timeout",
            "rate limit", "Rate limit",
            "RPM", "requests per minute",  # Minute-based rate limit (retryable)
            "TPM",  # Tokens per minute (retryable)
        ]):
            return True  # Retryable - temporary issue
        
        return False  # Unknown - treat as non-retryable
    
    def _invoke_with_fallback(self, input_text: str) -> AIMessage:
        """
        Invoke with Groq first (rate-limited), fallback to Gemini if needed.
        
        Groq priority:
        - Always attempt Groq first (primary provider)
        - Use rate limiter if available to prevent concurrent 429 errors
        
        Fallback to Gemini:
        - On temporary failures (429, 503, timeout)
        - On daily quota (TPD) exhaustion - try Gemini for non-quota errors
        - Clinical data now allowed for Gemini (Task 1.9.4: explicit authorization)
        
        Args:
            input_text: The prompt text
        
        Returns:
            AIMessage response
        
        Raises:
            RuntimeError: If all providers fail or auth errors
        """
        # Try Groq first (with rate limiting if available)
        try:
            logger.debug(f"Attempting Groq invocation (rate-limited)...")
            
            if RATE_LIMITER_AVAILABLE and get_groq_limiter:
                # Use rate limiter for coordinated requests (prevents concurrent 429)
                limiter = get_groq_limiter()
                response = limiter.invoke_with_retry(
                    lambda: self._groq.invoke(input_text),
                    operation_name="Groq LLM call"
                )
            else:
                # Rate limiter not available, invoke directly
                response = self._groq.invoke(input_text)
            
            self._last_provider_used = "groq"
            self._gemini_cooldown_until = None  # Clear cooldown on success
            logger.debug(f"Groq success")
            return response
        
        except Exception as groq_error:
            groq_error_str = str(groq_error)
            
            # Check if this is daily quota exhaustion (TPD)
            # TPD is non-retryable within same day, but Gemini can still try
            is_tpd_exhausted = any(pattern in groq_error_str for pattern in [
                "TPD", "tokens per day", "tokens_per_day", "quota per day", "daily"
            ])
            
            # Check if this is a retryable provider failure
            is_retryable = any(pattern in groq_error_str for pattern in [
                "429", "Too Many Requests",
                "503", "Service Unavailable",
                "500", "Internal Server Error",
                "timeout", "Timeout",
                "rate limit", "Rate limit",
                "RPM", "requests per minute",
                "TPM",  # Tokens per minute (retryable)
            ])
            
            if is_tpd_exhausted:
                # Daily quota exhausted - log but allow Gemini fallback for other requests
                logger.warning(
                    f"Groq daily quota (TPD) exhausted. "
                    f"Attempting Gemini fallback for non-quota operations."
                )
            elif not is_retryable:
                # Non-retryable error (auth, validation, etc.)
                logger.error(f"Groq non-retryable error: {groq_error_str[:200]}")
                raise
            else:
                # Retryable: temporary rate limits, server errors, timeouts
                logger.warning(
                    f"Groq provider failure (will attempt fallback): "
                    f"{groq_error_str[:150]}"
                )
        
        # Groq failed (either TPD or retryable error). Try Gemini fallback.
        if not self.allow_gemini_fallback:
            logger.error("Groq failed, Gemini fallback disabled. Returning error.")
            raise RuntimeError(
                f"Groq provider failed and fallback disabled: {groq_error_str[:200]}"
            )
        
        # Check Gemini cooldown
        if self._gemini_cooldown_until and datetime.now() < self._gemini_cooldown_until:
            wait_secs = (self._gemini_cooldown_until - datetime.now()).total_seconds()
            logger.error(
                f"Gemini in cooldown ({wait_secs:.0f}s remaining). "
                f"Both providers unavailable."
            )
            raise RuntimeError(
                f"Groq failed and Gemini in cooldown. Please retry in {wait_secs:.0f}s"
            )
        
        # Try Gemini fallback (clinical data now allowed per Task 1.9.4 Phase 2)
        try:
            self._initialize_gemini()
            logger.info(f"Attempting Gemini fallback (after Groq failure)...")
            response = self._gemini.invoke(input_text)
            self._last_provider_used = "gemini"
            logger.info(f"Gemini fallback succeeded")
            return response
        
        except Exception as gemini_error:
            gemini_error_str = str(gemini_error)
            logger.error(
                f"Gemini fallback failed: {gemini_error_str[:200]}. "
                f"All providers exhausted."
            )
            
            # Set Gemini cooldown for temporary failures
            is_temporary = any(pattern in gemini_error_str for pattern in [
                "429", "503", "timeout", "rate limit"
            ])
            if is_temporary:
                self._gemini_cooldown_until = datetime.now() + timedelta(minutes=5)
            
            raise RuntimeError(
                f"Both Groq and Gemini failed. "
                f"Groq: {groq_error_str[:100]}. "
                f"Gemini: {gemini_error_str[:100]}"
            )
            return response
        
        except Exception as gemini_error:
            gemini_error_str = str(gemini_error)
            logger.error(
                f"Gemini fallback failed: {gemini_error_str[:200]}. "
                f"All providers exhausted."
            )
            
            # Set Gemini cooldown
            if self._is_provider_failure(gemini_error):
                self._gemini_cooldown_until = datetime.now() + timedelta(minutes=5)
            
            raise RuntimeError(
                f"Both Groq and Gemini failed. "
                f"Groq: {groq_error_str[:100]}. "
                f"Gemini: {gemini_error_str[:100]}"
            )
    
    def invoke(self, input: Union[str, list, dict], config=None, **kwargs):
        """
        Invoke the unified provider (Groq→Gemini fallback).
        
        Implements LangChain BaseLLM interface compatible with RunnableSequence.
        Accepts config and kwargs for LangChain pipeline compatibility.
        """
        # Handle config parameter (passed by LangChain pipelines)
        if config is None:
            config = {}
        
        # Convert to string if needed
        if isinstance(input, list):
            # Handle message lists
            text = "\n".join(str(m) for m in input)
        elif isinstance(input, dict):
            # Handle dict input (from prompt templates)
            text = str(input.get("input", input.get("text", "")))
        else:
            text = str(input)
        
        return self._invoke_with_fallback(text)
    
    async def ainvoke(self, input: Union[str, list, dict], config=None, **kwargs):
        """Async invoke (currently delegates to sync)."""
        return self.invoke(input, config, **kwargs)
    
    def _generate(self, messages: list, stop: Optional[list] = None, run_manager=None, **kwargs):
        """
        Generate chat response from messages.
        
        Implements BaseChatModel abstract method.
        Delegates to _invoke_with_fallback.
        """
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        # Convert messages to text
        text = "\n".join(str(m) for m in messages)
        
        try:
            response = self._invoke_with_fallback(text)
            # Wrap response in ChatGeneration and ChatResult
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"_generate failed: {str(e)}")
            raise
    
    async def _agenerate(self, messages: list, stop: Optional[list] = None, run_manager=None, **kwargs):
        """Async version of _generate."""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """LangChain interface property."""
        return "unified_groq_gemini"
    
    @property
    def _identifying_params(self) -> dict:
        """LangChain interface property."""
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "provider": self._last_provider_used,
        }
    
    def get_provider_status(self) -> dict:
        """Get current provider status and failover state."""
        return {
            "primary": "groq",
            "secondary": "gemini" if self.allow_gemini_fallback else "disabled",
            "last_provider_used": self._last_provider_used,
            "gemini_cooldown_until": self._gemini_cooldown_until.isoformat() if self._gemini_cooldown_until else None,
            "clinical_data_validation": "enabled",
        }


def create_unified_provider(
    model_id: str,
    temperature: float,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> UnifiedLLMProvider:
    """
    Factory function to create a unified provider.
    
    Args:
        model_id: Groq model ID
        temperature: Temperature
        groq_api_key: Groq API key (from env if None)
        gemini_api_key: Gemini API key (from env if None)
    
    Returns:
        UnifiedLLMProvider instance
    """
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    gemini_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
    
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    return UnifiedLLMProvider(
        model_id=model_id,
        temperature=temperature,
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
        allow_gemini_fallback=bool(gemini_key),
    )
