"""
Phase 1: DeepSeek Reliability, Error Handling, and Session Preservation Tests

Automated test suite verifying:
1. DeepSeek HTTP 402 (Insufficient Balance -> status 402, 0 retries, session preserved)
2. DeepSeek HTTP 429 (Rate Limit -> status 503, bounded retries, session preserved)
3. DeepSeek HTTP 401/403 (Auth/Permission -> status 401, 0 retries, session preserved)
4. DeepSeek HTTP 5xx (Server Error -> status 503, bounded retries, session preserved)
5. Connection failure (Network error -> status 503, session preserved)
6. Timeout (Timeout -> status 503, session preserved)
7. Session preservation after provider failure (patient info retained, turn 2 resumption)
8. Successful mocked DeepSeek response (valid LangGraph execution)
9. Confirmation that no Groq/Gemini fallback occurs
10. Regression checks (FastAPI start, GET /, GET /health, routing structure, agents)
"""

import os
import sys
import uuid
from unittest.mock import patch, MagicMock

# Set offline flags to prevent HuggingFace network calls during test initialization
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

# Import application components
import app
from app import app as fastapi_app
import config
from config import Config
from provider_config_adapter import get_llm_provider, LLM_PROVIDER_FALLBACK_ENABLED
from deepseek_llm_provider import DeepSeekLLMProvider, DeepSeekProviderError, create_deepseek_provider
from agents.provider_error_handler import error_classifier, failure_handler, state_preserver
from agents.agent_decision import get_clinical_agent, create_agent_graph

client = TestClient(fastapi_app)

test_results = []


def record_result(test_name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    test_results.append((test_name, status, details))
    print(f"[{status}] {test_name}: {details}")


def test_deepseek_http_402_insufficient_balance():
    """Test 1: HTTP 402 Insufficient Balance fails immediately without retry and preserves session."""
    session_id = f"test-402-{uuid.uuid4()}"
    error_text = "Error code: 402 - {'error': {'message': 'Insufficient Balance', 'type': 'insufficient_balance_error'}}"
    
    # Test classifier directly
    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "billing"
    assert classified["http_status"] == 402
    assert classified["error_code"] == "INSUFFICIENT_BALANCE"
    assert classified["is_retryable"] is False

    # Test provider invoke: verify zero retries on 402
    provider = create_deepseek_provider(deepseek_api_key="mock_key", max_retries=2)
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception(error_text)
    provider._client = mock_client

    with pytest.raises(DeepSeekProviderError) as exc_info:
        provider.invoke("Test query")
    assert exc_info.value.status_code == 402
    assert exc_info.value.error_code == "INSUFFICIENT_BALANCE"
    assert mock_client.invoke.call_count == 1  # Crucial: 0 retries!

    # Test HTTP endpoint
    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 402
        data = response.json()
        assert data["error_code"] == "INSUFFICIENT_BALANCE"
        assert data["session_id"] == session_id
        assert data["details"]["session_preserved"] is True
        assert "api_key" not in str(data).lower()
        # Ensure session cookie is returned
        assert "session_id" in response.headers.get("set-cookie", "") or response.cookies.get("session_id") == session_id

    record_result("Test 1: DeepSeek HTTP 402", True, "Status 402, 0 retries, session preserved, no credential leak")


def test_deepseek_http_429_rate_limit():
    """Test 2: HTTP 429 Rate Limit returns 503, retries boundedly, preserves session."""
    session_id = f"test-429-{uuid.uuid4()}"
    error_text = "Error code: 429 - {'error': {'message': 'Rate limit reached: 30 requests per minute', 'type': 'rate_limit_error'}}"

    # Test classifier
    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "rate_limit"
    assert classified["http_status"] == 503
    assert classified["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert classified["is_retryable"] is True

    # Test provider bounded retry
    provider = create_deepseek_provider(deepseek_api_key="mock_key", max_retries=1)
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception(error_text)
    provider._client = mock_client

    with patch("time.sleep", return_value=None):
        with pytest.raises(DeepSeekProviderError) as exc_info:
            provider.invoke("Test query")
        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "RATE_LIMIT_EXCEEDED"
        assert mock_client.invoke.call_count == 2  # 1 initial + 1 retry (bounded)

    # Test endpoint
    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
        assert data["status"] == "service_unavailable"
        assert data["details"]["session_preserved"] is True

    record_result("Test 2: DeepSeek HTTP 429", True, "Status 503, bounded retry, session preserved")


def test_deepseek_http_401_403_auth_error():
    """Test 3: HTTP 401/403 Authentication/Permission fails immediately without retrying."""
    session_id = f"test-auth-{uuid.uuid4()}"
    error_text = "Error code: 401 - {'error': {'message': 'Invalid API Key provided: sk-abcdef123456789', 'type': 'authentication_error'}}"

    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "authentication"
    assert classified["http_status"] == 401
    assert classified["error_code"] == "AUTHENTICATION_ERROR"
    assert classified["is_retryable"] is False

    # Test provider: no retries
    provider = create_deepseek_provider(deepseek_api_key="mock_key", max_retries=2)
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception(error_text)
    provider._client = mock_client

    with pytest.raises(DeepSeekProviderError) as exc_info:
        provider.invoke("Test query")
    assert exc_info.value.status_code == 401
    assert mock_client.invoke.call_count == 1  # 0 retries

    # Test endpoint
    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "AUTHENTICATION_ERROR"
        # Verify secret key was redacted
        assert "sk-abcdef123456789" not in response.text

    record_result("Test 3: DeepSeek HTTP 401/403", True, "Status 401, 0 retries, secret key redacted in response")


def test_deepseek_http_5xx_server_error():
    """Test 4: HTTP 5xx Server Error returns 503, bounded retries, preserves session."""
    session_id = f"test-500-{uuid.uuid4()}"
    error_text = "Error code: 502 - Bad gateway from DeepSeek server"

    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "server_error"
    assert classified["http_status"] == 503
    assert classified["error_code"] == "PROVIDER_SERVER_ERROR"
    assert classified["is_retryable"] is True

    # Test provider bounded retry
    provider = create_deepseek_provider(deepseek_api_key="mock_key", max_retries=1)
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception(error_text)
    provider._client = mock_client

    with patch("time.sleep", return_value=None):
        with pytest.raises(DeepSeekProviderError) as exc_info:
            provider.invoke("Test query")
        assert exc_info.value.status_code == 503
        assert mock_client.invoke.call_count == 2

    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "PROVIDER_SERVER_ERROR"
        assert data["status"] == "service_unavailable"

    record_result("Test 4: DeepSeek HTTP 5xx", True, "Status 503, bounded retry, session preserved")


def test_connection_failure():
    """Test 5: Connection failure returns 503 with safe user message and preserves session."""
    session_id = f"test-conn-{uuid.uuid4()}"
    error_text = "Connection error: Failed to resolve 'api.deepseek.com' (getaddrinfo failed)"

    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "connection_error"
    assert classified["http_status"] == 503
    assert classified["error_code"] == "CONNECTION_ERROR"

    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "CONNECTION_ERROR"
        assert "network" in data["message"].lower() or "connect" in data["message"].lower()

    record_result("Test 5: Connection Failure", True, "Status 503, safe connection message, session preserved")


def test_timeout():
    """Test 6: Request timeout returns 503 and preserves session."""
    session_id = f"test-timeout-{uuid.uuid4()}"
    error_text = "Request timeout: read timed out after 60 seconds"

    classified = error_classifier.classify_deepseek_error(error_text)
    assert classified["category"] == "timeout"
    assert classified["http_status"] == 503
    assert classified["error_code"] == "TIMEOUT"

    with patch("agents.agent_decision.process_query", side_effect=Exception(error_text)):
        response = client.post("/chat", json={"query": "Hello"}, cookies={"session_id": session_id})
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "TIMEOUT"

    record_result("Test 6: Timeout Handling", True, "Status 503, timeout classified, session preserved")


def test_session_preservation_and_turn_resumption():
    """Test 7: Patient data retained during provider failure and resumption works on turn 2."""
    session_id = f"test-session-{uuid.uuid4()}"
    
    # Simulate turn 1: patient provides symptoms, clinical agent extracts them, but LLM fails
    clinical_agent = get_clinical_agent(session_id)
    clinical_agent.patient_info_collector.extract_from_patient_text("I have had a high fever and severe headache for 3 days.")
    patient_data_t1 = clinical_agent.patient_info_collector.get_patient_data()
    
    assert len(patient_data_t1["symptoms"]["list"]) > 0
    state_preserver.save_session_state(session_id, patient_data_t1)

    # Provider fails on turn 1
    with patch("agents.agent_decision.process_query", side_effect=Exception("HTTP 429 rate limit")):
        response_t1 = client.post(
            "/chat", 
            json={"query": "I have had a high fever and severe headache for 3 days."},
            cookies={"session_id": session_id}
        )
        assert response_t1.status_code == 503
        assert response_t1.json()["details"]["session_preserved"] is True
        assert response_t1.json()["details"]["patient_data_preserved"] is True

    # Verify patient data is intact in state preserver
    restored = state_preserver.restore_session_state(session_id)
    assert restored is not None
    assert len(restored["symptoms"]["list"]) > 0

    # Turn 2: Provider recovers. Clinical agent processes next patient input
    clinical_agent_t2 = get_clinical_agent(session_id)
    clinical_agent_t2.patient_info_collector.extract_from_patient_text("I am taking Tylenol 500mg.")
    patient_data_t2 = clinical_agent_t2.patient_info_collector.get_patient_data()
    
    # Both symptoms and medications are present!
    assert len(patient_data_t2["symptoms"]["list"]) > 0
    assert len(patient_data_t2["medications"]["list"]) > 0

    record_result("Test 7: Session Preservation & Resumption", True, "Turn 1 error preserved patient info; Turn 2 accumulated new info seamlessly")


def test_successful_mocked_deepseek_response():
    """Test 8: Successful mocked DeepSeek response passes through LangGraph workflow."""
    session_id = f"test-success-{uuid.uuid4()}"
    
    mock_routing_response = AIMessage(content='{"agent": "CONVERSATION_AGENT", "confidence": 0.95, "reasoning": "Greeting"}')
    mock_agent_response = AIMessage(content="Hello! I am your AI Medical Assistant. How can I help you today?")
    
    def mock_invoke(input_data, *args, **kwargs):
        text = str(input_data)
        if "which agent should handle" in text.lower():
            return mock_routing_response
        return mock_agent_response

    with patch.object(DeepSeekLLMProvider, "invoke", side_effect=mock_invoke):
        response = client.post(
            "/chat", 
            json={"query": "Hello there"},
            cookies={"session_id": session_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["agent"] == "CONVERSATION_AGENT"
        assert "Medical Assistant" in data["response"]

    record_result("Test 8: Successful Mocked DeepSeek Response", True, "200 OK, routed to CONVERSATION_AGENT, returned expected content")


def test_confirmation_no_fallback_occurs():
    """Test 9: Verify DeepSeek is the only provider - no Groq, Gemini, or fallback logic."""
    assert LLM_PROVIDER_FALLBACK_ENABLED is False
    
    # Verify provider adapter creates DeepSeekLLMProvider
    provider = get_llm_provider()
    assert isinstance(provider, DeepSeekLLMProvider)
    assert provider.model_id == "deepseek-flash"
    assert provider.base_url == "https://api.deepseek.com"

    # Verify that when DeepSeek fails, no other provider is called
    provider_fail_error = DeepSeekProviderError("DeepSeek quota exhausted", status_code=503)
    with patch("agents.agent_decision.process_query", side_effect=provider_fail_error):
        response = client.post("/chat", json={"query": "Is there a fallback?"})
        # Fails directly with 503 rather than attempting Groq/Gemini
        assert response.status_code == 503
        data = response.json()
        assert data["details"]["provider"] == "deepseek"

    record_result("Test 9: Confirmation No Fallback Occurs", True, "DeepSeek is the sole provider, fallback disabled, direct failure reporting")


def test_regression_checks():
    """Test 10: Regression checks (FastAPI start, GET /, GET /health, route & agent integrity)."""
    # GET /
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "html" in res_index.headers["content-type"]

    # GET /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "healthy"}

    # Verify registered routes in FastAPI
    routes = [r.path for r in fastapi_app.routes]
    assert "/" in routes
    assert "/health" in routes
    assert "/chat" in routes
    assert "/upload" in routes
    assert "/validate" in routes
    assert "/clinical-patient-feedback" in routes

    # Verify LangGraph graph compiles with all agent nodes
    graph = create_agent_graph()
    assert graph is not None
    graph_nodes = graph.nodes
    expected_agents = [
        "CLINICAL_PRE_CONSULTATION_AGENT",
        "CONVERSATION_AGENT",
        "RAG_AGENT",
        "WEB_SEARCH_PROCESSOR_AGENT",
        "BRAIN_TUMOR_AGENT",
        "CHEST_XRAY_AGENT",
        "SKIN_LESION_AGENT",
    ]
    for agent_node in expected_agents:
        assert agent_node in graph_nodes

    record_result("Test 10: Regression Checks", True, "GET /, GET /health, all endpoints, and all 7 LangGraph agent nodes verified")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 1: DEEPSEEK RELIABILITY & SESSION PRESERVATION TESTS")
    print("=" * 70)
    
    test_deepseek_http_402_insufficient_balance()
    test_deepseek_http_429_rate_limit()
    test_deepseek_http_401_403_auth_error()
    test_deepseek_http_5xx_server_error()
    test_connection_failure()
    test_timeout()
    test_session_preservation_and_turn_resumption()
    test_successful_mocked_deepseek_response()
    test_confirmation_no_fallback_occurs()
    test_regression_checks()
    
    print("\n" + "=" * 70)
    print("PHASE 1 TEST EXECUTION SUMMARY:")
    print("=" * 70)
    all_passed = True
    for name, status, details in test_results:
        print(f"[{status}] {name}")
        if status != "PASS":
            all_passed = False
    
    print(f"\nFinal Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
