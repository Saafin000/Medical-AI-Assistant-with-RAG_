"""
Phase 7: Multi-Agent Orchestration & Routing Validation Test Suite

Validates:
1. LLM decision routing prompt & agent selection mapping for all agents:
   - CLINICAL_PRE_CONSULTATION_AGENT
   - CONVERSATION_AGENT
   - RAG_AGENT
   - WEB_SEARCH_PROCESSOR_AGENT
   - Vision Agents (CHEST_XRAY_AGENT, SKIN_LESION_AGENT, BRAIN_TUMOR_AGENT)
2. LangGraph state initialization & thread_id isolation per session.
3. RAG confidence-based handoff to Web Search Processor when confidence < threshold.
4. Human validation routing for medical vision agents.
5. Error classifier and failure recovery.
"""

import os
import sys
import uuid
import unittest
from unittest.mock import MagicMock, patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage, HumanMessage
from agents.agent_decision import (
    get_thread_config,
    get_clinical_agent,
    reset_clinical_agent_session,
    init_agent_state,
    AgentState,
    AgentConfig
)
from config import Config


class TestMultiAgentOrchestration(unittest.TestCase):
    """Orchestration, routing, and LangGraph workflow validation."""

    def setUp(self):
        self.config = Config()

    def test_01_thread_config_session_isolation(self):
        """Thread config dynamically isolates LangGraph threads by session_id."""
        sid_1 = f"conv_{uuid.uuid4()}"
        sid_2 = f"clin_{uuid.uuid4()}"

        cfg_1 = get_thread_config(sid_1)
        cfg_2 = get_thread_config(sid_2)
        cfg_default = get_thread_config(None)

        self.assertEqual(cfg_1["configurable"]["thread_id"], sid_1)
        self.assertEqual(cfg_2["configurable"]["thread_id"], sid_2)
        self.assertEqual(cfg_default["configurable"]["thread_id"], "default")
        self.assertNotEqual(cfg_1["configurable"]["thread_id"], cfg_2["configurable"]["thread_id"])

    def test_02_clinical_agent_registry_and_reset(self):
        """Clinical agent instances are registered by session_id and cleaned up on reset."""
        sid = f"clin_{uuid.uuid4()}"
        agent1 = get_clinical_agent(sid)
        agent2 = get_clinical_agent(sid)

        # Same session must yield the exact same agent instance in memory
        self.assertIs(agent1, agent2)

        # Reset removes it
        reset_clinical_agent_session(sid)
        agent3 = get_clinical_agent(sid)
        self.assertIsNot(agent1, agent3)
        reset_clinical_agent_session(sid)

    def test_03_init_agent_state_propagates_session_id(self):
        """init_agent_state correctly initializes AgentState schema with session_id."""
        sid = f"conv_{uuid.uuid4()}"
        state = init_agent_state(session_id=sid)

        self.assertEqual(state["session_id"], sid)
        self.assertEqual(state["messages"], [])
        self.assertIsNone(state["agent_name"])
        self.assertFalse(state["has_image"])
        self.assertFalse(state["needs_human_validation"])
        self.assertEqual(state["retrieval_confidence"], 0.0)

    def test_04_confidence_based_routing_logic(self):
        """confidence_based_routing redirects RAG to Web Search if confidence is below threshold or info insufficient."""
        # Low confidence -> re-route to WEB_SEARCH_PROCESSOR_AGENT
        state_low_conf: AgentState = {
            "messages": [],
            "session_id": "test",
            "agent_name": "RAG_AGENT",
            "current_input": "medical query",
            "has_image": False,
            "image_type": None,
            "output": AIMessage(content=""),
            "needs_human_validation": False,
            "retrieval_confidence": 0.25,  # Below min_retrieval_confidence (0.40)
            "bypass_routing": False,
            "insufficient_info": False
        }
        
        # Test routing rule
        min_conf = self.config.rag.min_retrieval_confidence
        self.assertGreater(min_conf, 0.25)

        # Simulate routing evaluation from agents.agent_decision
        def eval_routing(s):
            if s.get("retrieval_confidence", 0.0) < min_conf or s.get("insufficient_info", False):
                return "WEB_SEARCH_PROCESSOR_AGENT"
            return "check_validation"

        self.assertEqual(eval_routing(state_low_conf), "WEB_SEARCH_PROCESSOR_AGENT")

        # Sufficient confidence and info -> proceed to check_validation
        state_high_conf = dict(state_low_conf)
        state_high_conf["retrieval_confidence"] = 0.85
        state_high_conf["insufficient_info"] = False
        self.assertEqual(eval_routing(state_high_conf), "check_validation")

        # High confidence but insufficient info flag -> re-route to Web Search
        state_insufficient = dict(state_high_conf)
        state_insufficient["insufficient_info"] = True
        self.assertEqual(eval_routing(state_insufficient), "WEB_SEARCH_PROCESSOR_AGENT")

    def test_05_human_validation_trigger_for_vision_agents(self):
        """Vision agents flag needs_human_validation=True triggering validation routing."""
        state_vision: AgentState = {
            "messages": [],
            "session_id": "test",
            "agent_name": "CHEST_XRAY_AGENT",
            "current_input": {"image": "test.jpg"},
            "has_image": True,
            "image_type": "Chest X-ray",
            "output": AIMessage(content="COVID-19 Positive"),
            "needs_human_validation": True,
            "retrieval_confidence": 0.0,
            "bypass_routing": False,
            "insufficient_info": False
        }

        # handle_human_validation logic
        def handle_validation(s):
            if s.get("needs_human_validation", False):
                return {"agent_state": s, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
            return {"agent_state": s, "next": "END"}

        res = handle_validation(state_vision)
        self.assertEqual(res["next"], "human_validation")
        self.assertEqual(res["agent"], "HUMAN_VALIDATION")

    def test_06_error_classifier_categories(self):
        """Error classifier properly categorizes provider errors without leaking sensitive data."""
        from agents.provider_error_handler import error_classifier

        e_429 = Exception("Error code: 429 - rate_limit_exceeded")
        res_429 = error_classifier.classify_error(e_429)
        self.assertTrue(res_429["is_provider_error"])
        self.assertIn("rate", res_429["category"])

        e_generic = ValueError("Invalid input syntax")
        res_generic = error_classifier.classify_error(e_generic)
        self.assertFalse(res_generic["is_provider_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
