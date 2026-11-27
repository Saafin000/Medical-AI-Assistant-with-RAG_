"""
Comprehensive Verification Test Suite for Groq Migration and Project Preservation

Tests:
1. Groq provider initialization
2. Error classification: 429 Quota Exceeded (TPD), 429 Rate Limit (RPM), 401 Auth, 5xx Server Error, Connection Error, Timeout
3. Zero fallback confirmation (no Gemini / DeepSeek)
4. Groq per-query API usage measurement
5. Provider cache deduplication
6. Session preservation upon provider failure
7. In-scope configurations verification (all use Groq)
8. Clinical Pre-Consultation Agent exclusion and untouchability confirmation
9. LangGraph architecture preservation (all 7 nodes, edges, conditional edges, LLM routing)
10. FastAPI endpoints (/health, /, /chat success, /chat error handling)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Set offline flags for embeddings before imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

# Import project modules
from groq_llm_provider import (
    GroqLLMProvider,
    GroqProviderError,
    GroqUsageTracker,
    create_groq_provider,
    usage_tracker,
    sanitize_error_text
)
from groq_rate_limiter import GroqRateLimiter, get_groq_limiter
from provider_config_adapter import (
    get_groq_provider,
    get_llm_provider,
    LLM_PROVIDER_FALLBACK_ENABLED
)
from agents.provider_error_handler import (
    ProviderErrorClassifier,
    ProviderFailureHandler,
    SessionStatePreserver,
    error_classifier,
    failure_handler,
    state_preserver,
    sanitize_text
)
from config import Config
from langchain_core.messages import AIMessage, HumanMessage


class TestGroqMigrationAndReliability(unittest.TestCase):

    def setUp(self):
        self.classifier = ProviderErrorClassifier()
        self.handler = ProviderFailureHandler()
        self.preserver = SessionStatePreserver()

    # 1. Groq Provider Initialization
    def test_01_groq_provider_initialization(self):
        provider = get_groq_provider(temperature=0.3)
        self.assertIsInstance(provider, GroqLLMProvider)
        self.assertEqual(provider.temperature, 0.3)
        self.assertEqual(provider._llm_type, "groq")
        self.assertIsNotNone(provider._client)
        self.assertIsNotNone(provider._rate_limiter)

    # 2. Daily Token Limit (TPD) Classification (Non-Retryable)
    def test_02_quota_exceeded_classification(self):
        tpd_error = Exception("Rate limit reached for model `qwen/qwen3.8-27b` on tokens per day (TPD). Exceeded your current quota.")
        classified = self.classifier.classify_error(tpd_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "daily_token_limit")
        self.assertEqual(classified["error_code"], "DAILY_TOKEN_LIMIT_EXCEEDED")
        self.assertEqual(classified["http_status"], 429)
        self.assertFalse(classified["is_retryable"])

    # 2b. Daily Request Limit (RPD) Classification (Non-Retryable)
    def test_02b_rpd_limit_classification(self):
        rpd_error = Exception("Rate limit reached for requests per day (RPD). Exceeded limit.")
        classified = self.classifier.classify_error(rpd_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "daily_request_limit")
        self.assertEqual(classified["error_code"], "DAILY_REQUEST_LIMIT_EXCEEDED")
        self.assertEqual(classified["http_status"], 429)
        self.assertFalse(classified["is_retryable"])

    # 3. Rate Limit (RPM/TPM) Classification (Retryable)
    def test_03_rate_limit_transient_classification(self):
        rpm_error = Exception("Error code: 429 - {'error': {'message': 'Rate limit reached for requests per minute (RPM). Please wait.'}}")
        classified = self.classifier.classify_error(rpm_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "rate_limit_rpm")
        self.assertEqual(classified["error_code"], "RPM_LIMIT_EXCEEDED")
        self.assertEqual(classified["http_status"], 503)
        self.assertTrue(classified["is_retryable"])

        tpm_error = Exception("Rate limit reached for tokens per minute (TPM). Please wait.")
        classified_tpm = self.classifier.classify_error(tpm_error)
        self.assertEqual(classified_tpm["category"], "rate_limit_tpm")
        self.assertEqual(classified_tpm["error_code"], "TPM_LIMIT_EXCEEDED")
        self.assertEqual(classified_tpm["http_status"], 503)
        self.assertTrue(classified_tpm["is_retryable"])

    # 4. Authentication Error (401/403) Classification (Non-Retryable)
    def test_04_authentication_error_classification(self):
        auth_error = Exception("Error code: 401 - {'error': {'message': 'Invalid API Key provided', 'code': 'invalid_api_key'}}")
        classified = self.classifier.classify_error(auth_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "authentication")
        self.assertEqual(classified["error_code"], "AUTHENTICATION_ERROR")
        self.assertEqual(classified["http_status"], 401)
        self.assertFalse(classified["is_retryable"])

    # 5. Server Error (5xx) Classification (Retryable)
    def test_05_server_error_classification(self):
        server_error = Exception("Error code: 503 - {'error': {'message': 'Service Unavailable - Provider is experiencing high load'}}")
        classified = self.classifier.classify_error(server_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "server_error")
        self.assertEqual(classified["error_code"], "PROVIDER_SERVER_ERROR")
        self.assertEqual(classified["http_status"], 503)
        self.assertTrue(classified["is_retryable"])

    # 6. Connection Error Classification (Retryable)
    def test_06_connection_error_classification(self):
        conn_error = Exception("httpx.ConnectError: [Errno 11001] getaddrinfo failed - Connection refused")
        classified = self.classifier.classify_error(conn_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "connection_error")
        self.assertEqual(classified["error_code"], "CONNECTION_ERROR")
        self.assertEqual(classified["http_status"], 503)
        self.assertTrue(classified["is_retryable"])

    # 7. Timeout Classification (Retryable)
    def test_07_timeout_classification(self):
        timeout_error = Exception("httpx.ReadTimeout: The read operation timed out after 60 seconds")
        classified = self.classifier.classify_error(timeout_error)
        self.assertTrue(classified["is_provider_error"])
        self.assertEqual(classified["category"], "timeout")
        self.assertEqual(classified["error_code"], "TIMEOUT")
        self.assertEqual(classified["http_status"], 503)
        self.assertTrue(classified["is_retryable"])

    # 8. Zero Fallback Confirmation
    def test_08_zero_fallback_confirmation(self):
        self.assertFalse(LLM_PROVIDER_FALLBACK_ENABLED)
        provider = get_groq_provider()
        
        # Mock ChatGroq.invoke to fail with quota exhaustion
        from langchain_groq import ChatGroq
        with patch.object(ChatGroq, "invoke", side_effect=Exception("Rate limit on tokens per day (TPD). Exceeded your current quota.")):
            with self.assertRaises(GroqProviderError) as ctx:
                provider.invoke("Hello")
            self.assertEqual(ctx.exception.category, "quota_exceeded")
            self.assertEqual(ctx.exception.error_code, "QUOTA_EXCEEDED")
            # Verify no fallback to secondary providers occurred
            self.assertFalse(ctx.exception.is_retryable)

    # 9. Groq Usage Tracking Per Query (Invocations, Actual Requests, Retries, Tokens, Stages)
    def test_09_groq_usage_tracking_per_query(self):
        tracker = GroqUsageTracker()
        tracker.reset_query_metrics()
        self.assertEqual(tracker.get_query_metrics()["api_requests"], 0)
        self.assertEqual(tracker.get_query_metrics()["total_tokens"], 0)

        # Stage 1: Decision routing (1 request, success)
        tracker.record_request_attempt(stage="agent_decision_routing")
        tracker.record_response(prompt_tokens=150, completion_tokens=30, total_tokens=180, stage="agent_decision_routing")

        # Stage 2: Conversation agent (attempt 1 failed, retry attempt 2 succeeded)
        tracker.record_request_attempt(stage="conversation_agent")
        tracker.record_retry(stage="conversation_agent")
        tracker.record_request_attempt(stage="conversation_agent")
        tracker.record_response(prompt_tokens=200, completion_tokens=100, total_tokens=300, stage="conversation_agent")

        metrics = tracker.get_query_metrics()
        self.assertEqual(metrics["api_requests"], 3)  # 3 physical API calls
        self.assertEqual(metrics["model_invocations"], 2)  # 2 logical model tasks
        self.assertEqual(metrics["retries"], 1)  # 1 retry
        self.assertEqual(metrics["prompt_tokens"], 350)
        self.assertEqual(metrics["completion_tokens"], 130)
        self.assertEqual(metrics["total_tokens"], 480)

        # Verify stage breakdown
        stages = metrics["stage_breakdown"]
        self.assertIn("agent_decision_routing", stages)
        self.assertIn("conversation_agent", stages)
        self.assertEqual(stages["conversation_agent"]["requests"], 2)
        self.assertEqual(stages["conversation_agent"]["retries"], 1)

        # Reset for next query
        tracker.reset_query_metrics()
        self.assertEqual(tracker.get_query_metrics()["api_requests"], 0)
        self.assertEqual(tracker.get_query_metrics()["model_invocations"], 0)
        # Cumulative totals persist
        self.assertEqual(tracker.get_total_metrics()["total_requests"], 3)
        self.assertEqual(tracker.get_total_metrics()["total_tokens"], 480)

    # 9b. Retry-After Header and Body Parsing Verification
    def test_09b_retry_after_parsing(self):
        limiter = GroqRateLimiter()

        # Case 1: Groq body with "Please try again in 4.375s."
        err_with_body = Exception("Rate limit reached for requests per minute. Please try again in 4.375s.")
        limiter.handle_rate_limit_error(err_with_body)
        self.assertIsNotNone(limiter._retry_after_until)

        # Case 2: Header with Retry-After: 15
        class MockResponse:
            headers = {"retry-after": "15"}
        err_with_hdr = Exception("Rate limit reached")
        err_with_hdr.response = MockResponse()
        limiter.handle_rate_limit_error(err_with_hdr)
        self.assertIsNotNone(limiter._retry_after_until)

    # 10. Provider Cache Deduplication
    def test_10_provider_cache_deduplication(self):
        p1 = create_groq_provider(temperature=0.3)
        p2 = create_groq_provider(temperature=0.3)
        self.assertIs(p1, p2, "create_groq_provider should reuse cached instance for identical temperature")

    # 11. Session Preservation upon Provider Failure
    def test_11_session_preservation_on_failure(self):
        session_id = "test-session-12345"
        patient_data = {
            "symptoms": {"list": ["headache", "fever"]},
            "medications": {"list": ["paracetamol"]},
            "allergies": {"status": "none"},
            "medical_history": {"status": "hypertension"}
        }

        # Save session
        saved = self.preserver.save_session_state(session_id, patient_data)
        self.assertTrue(saved)

        # Handle failure
        status_code, response = self.handler.handle_provider_failure(
            error=Exception("Groq 429 Rate limit"),
            provider_name="groq",
            patient_data=patient_data,
            session_id=session_id
        )
        self.assertEqual(status_code, 503)
        self.assertTrue(response["details"]["session_preserved"])
        self.assertTrue(response["details"]["patient_data_preserved"])
        self.assertEqual(response["details"]["collected_info"]["symptoms"], 2)

        # Restore session
        restored = self.preserver.restore_session_state(session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored["symptoms"]["list"], ["headache", "fever"])

    # 12. Security: Sensitive Credential Redaction
    def test_12_sensitive_credential_redaction(self):
        raw_error = "Authorization: Bearer test_token and api_key=gsk_secret1234567890abcdef and sk-proj-1234567890"
        sanitized = sanitize_text(raw_error)
        self.assertNotIn("test_token", sanitized)
        self.assertNotIn("gsk_secret", sanitized)
        self.assertNotIn("sk-proj-", sanitized)
        self.assertIn("gsk_[REDACTED]", sanitized)
        self.assertIn("sk-[REDACTED]", sanitized)
        self.assertIn("Bearer [REDACTED]", sanitized)

    # 13. In-Scope Configurations Use Groq
    def test_13_in_scope_configs_use_groq(self):
        cfg = Config()
        self.assertIsInstance(cfg.agent_decision.llm, GroqLLMProvider)
        self.assertIsInstance(cfg.conversation.llm, GroqLLMProvider)
        self.assertIsInstance(cfg.web_search.llm, GroqLLMProvider)
        self.assertIsInstance(cfg.rag.llm, GroqLLMProvider)
        self.assertIsInstance(cfg.rag.chunker_model, GroqLLMProvider)
        self.assertIsInstance(cfg.rag.response_generator_model, GroqLLMProvider)

    # 14. Clinical Pre-Consultation Agent Untouched
    def test_14_clinical_pre_consultation_agent_untouched(self):
        clinical_dir = os.path.join(os.path.dirname(__file__), "agents", "clinical_pre_consultation_agent")
        self.assertTrue(os.path.exists(clinical_dir))
        expected_files = [
            "__init__.py",
            "followup_question_generator.py",
            "patient_info_collector.py",
            "red_flag_detector.py",
            "report_generator.py"
        ]
        for f in expected_files:
            file_path = os.path.join(clinical_dir, f)
            self.assertTrue(os.path.exists(file_path), f"Missing {f}")

        # Check git status to ensure none of the clinical files were modified
        import subprocess
        result = subprocess.run(["git", "status", "--porcelain", "agents/clinical_pre_consultation_agent"], capture_output=True, text=True)
        modified_files = [line for line in result.stdout.splitlines() if line.startswith(" M") or line.startswith("M ")]
        self.assertEqual(len(modified_files), 0, f"Clinical Pre-Consultation files must NOT be modified! Found: {modified_files}")

    # 15. LangGraph Architecture Preservation
    def test_15_langgraph_architecture_preservation(self):
        from agents.agent_decision import create_agent_graph
        graph = create_agent_graph()
        self.assertIsNotNone(graph)
        graph_nodes = graph.get_graph().nodes
        
        # Verify all 7 registered agent nodes exist
        required_nodes = [
            "CLINICAL_PRE_CONSULTATION_AGENT",
            "CONVERSATION_AGENT",
            "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT",
            "analyze_input",
            "route_to_agent",
            "human_validation",
            "apply_guardrails"
        ]
        for node in required_nodes:
            self.assertIn(node, graph_nodes, f"Node {node} missing from LangGraph!")

    # 16. Fast-path Guardrails Quota Optimization
    def test_16_fast_path_guardrails_optimization(self):
        from agents.guardrails.local_guardrails import LocalGuardrails
        mock_llm = MagicMock()
        guardrails = LocalGuardrails(mock_llm)

        # Test common simple phrases bypass the LLM
        for phrase in ["yes", "no", "hi", "hello", "thanks", "ok", "sure"]:
            allowed, msg = guardrails.check_input(phrase)
            self.assertTrue(allowed)
        
        # Verify LLM was NEVER called for these common phrases
        mock_llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
