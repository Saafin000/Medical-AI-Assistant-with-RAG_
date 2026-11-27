"""
Provider Configuration Adapter - Groq as Exclusive LLM Provider

Configures Groq as the primary and exclusive LLM provider across all application
components, including the Clinical Pre-Consultation Agent.

Zero fallback: no automatic fallback to Gemini, DeepSeek, or other secondary providers.
"""

import os
import logging
from typing import Optional, Union, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Import Groq provider
try:
    from groq_llm_provider import GroqLLMProvider, create_groq_provider, usage_tracker
    GROQ_PROVIDER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Groq provider not available: {e}")
    GROQ_PROVIDER_AVAILABLE = False


def get_groq_provider(
    model_id: Optional[str] = None,
    temperature: float = 0.3,
    max_retries: int = 4,
    timeout: int = 60,
    max_tokens: Optional[int] = None,
) -> GroqLLMProvider:
    """
    Get Groq LLM provider for all project components.
    
    Args:
        model_id: Groq model identifier (defaults to GROQ_MODEL env var or qwen/qwen3.8-27b)
        temperature: Temperature setting
        max_retries: Maximum retries for transient errors
        timeout: Request timeout in seconds
        max_tokens: Maximum tokens for completion (limits rate limit reservation)
        
    Returns:
        GroqLLMProvider instance
    """
    if not GROQ_PROVIDER_AVAILABLE:
        raise ImportError(
            "Groq provider not available. Install langchain-groq: pip install langchain-groq"
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required. Please set it before running the application."
        )

    model = model_id or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return create_groq_provider(
        model_id=model,
        temperature=temperature,
        groq_api_key=groq_api_key,
        max_retries=max_retries,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def get_llm_provider(
    model_id: str = "qwen/qwen3.8-27b",
    temperature: float = 0.3,
    use_fallback: bool = False,
) -> GroqLLMProvider:
    """
    Get LLM provider.
    
    All application components use GroqLLMProvider exclusively.
    
    Args:
        model_id: Model ID (defaults to project Groq model)
        temperature: Temperature setting
        use_fallback: Kept for backward compatibility, strictly False
        
    Returns:
        GroqLLMProvider instance
    """
    effective_model = None if model_id == "deepseek-flash" else model_id
    return get_groq_provider(
        model_id=effective_model,
        temperature=temperature,
    )


# Fallback is strictly disabled. No secondary provider switching.
LLM_PROVIDER_FALLBACK_ENABLED = False
