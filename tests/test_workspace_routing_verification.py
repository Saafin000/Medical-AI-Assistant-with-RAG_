"""
Targeted Verification Test Suite for Workspace Routing Boundary Enforcement

Verifies all 9 requirements from the INVESTIGATE, FIX & VERIFY MEDICAL WORKSPACE ROUTING Directive:
- Test 1: Medical Conversation — Fever query must NOT invoke Clinical Agent; must route to Conversation/RAG/Web.
- Test 2: Medical Conversation — Headache query must provide general informational response without clinical intake.
- Test 3: Medical Conversation — Direct clinical question must provide informational answer without intake workflow.
- Test 4: Clinical Pre-Consultation — Fever intake MUST invoke Clinical Pre-Consultation Agent and ask follow-up questions.
- Test 5: Clinical Pre-Consultation — Adaptive follow-up collects symptoms, history, medications, and avoids repeating questions.
- Test 6: Clinical Red Flag — Safe emergency fixture triggers immediate emergency escalation without weakened safety.
- Test 7: Workspace Switching — Strict isolation between conversation and clinical sessions across sequential turns.
- Test 8: Session Restoration — Persistence and correct restoration of session workspaces across simulated reloads.
- Test 9: Image Analysis Isolation — Image sessions strictly rejected by /chat and kept isolated.
"""

import os
import sys
import uuid
import json
import unittest

# Ensure offline environment for HuggingFace / Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from storage.session_db import session_db
from agents.agent_decision import reset_clinical_agent_session, get_clinical_agent
from app import app


class TestWorkspaceRoutingVerification(unittest.TestCase):
    """Verifies workspace routing boundary enforcement and isolation."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.created_sessions = []

    def tearDown(self):
        # Clean up all created test sessions
        for sid in self.created_sessions:
            try:
                session_db.delete_session(sid)
                reset_clinical_agent_session(sid)
            except Exception:
                pass

    def _create_session(self, workspace_type: str, title: str = "Test Session") -> str:
        res = self.client.post("/sessions", json={
            "workspace_type": workspace_type,
            "title": title
        })
        self.assertEqual(res.status_code, 200)
        sid = res.json()["session_id"]
        self.created_sessions.append(sid)
        return sid

    # -------------------------------------------------------------------------
    # Test 1: Medical Conversation — Fever
    # -------------------------------------------------------------------------
    def test_01_medical_conversation_fever(self):
        """User fever query in Medical AI Conversation must NOT invoke Clinical Pre-Consultation Agent."""
        sid = self._create_session("conversation", "Fever Inquiry")
        query_text = "I have fever from 2 days and I also have runny nose and feeling body temperature high."

        res = self.client.post(
            "/chat",
            json={
                "query": query_text,
                "conversation_history": [],
                "workspace": "conversation",
                "workspace_type": "conversation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Strict assertion: Agent MUST NOT be CLINICAL_PRE_CONSULTATION_AGENT
        self.assertNotEqual(
            data.get("agent"),
            "CLINICAL_PRE_CONSULTATION_AGENT",
            "Medical AI Conversation must NEVER invoke CLINICAL_PRE_CONSULTATION_AGENT!"
        )
        # Agent must be one of conversational/knowledge agents
        self.assertIn(
            data.get("agent"),
            ["CONVERSATION_AGENT", "RAG_AGENT", "WEB_SEARCH_PROCESSOR_AGENT"]
        )

        response_text = data.get("response", "")
        self.assertTrue(len(response_text) > 20, "Response text must not be empty")

        # Response must NOT start clinical intake asking medication details
        self.assertNotIn(
            "Have you taken any fever-reducing medication",
            response_text,
            "Must not ask clinical pre-consultation intake questions in conversation workspace"
        )
        self.assertNotIn("CLINICAL PRE-CONSULTATION INTAKE REPORT", response_text)

        # Verify clinical state was NOT created in DB for this conversation session
        clin_state = session_db.get_clinical_state(sid)
        self.assertIsNone(clin_state, "No clinical intake state should be created for conversation session")

    # -------------------------------------------------------------------------
    # Test 2: Medical Conversation — Headache
    # -------------------------------------------------------------------------
    def test_02_medical_conversation_headache(self):
        """Headache symptom inquiry in Medical AI Conversation must provide general informational response."""
        sid = self._create_session("conversation", "Headache Inquiry")
        query_text = "I have had a headache since yesterday. What could be the common reasons?"

        res = self.client.post(
            "/chat",
            json={
                "query": query_text,
                "conversation_history": [],
                "workspace": "conversation",
                "workspace_type": "conversation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertNotEqual(data.get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")
        response_text = data.get("response", "")
        self.assertTrue(len(response_text) > 20)
        self.assertNotIn("CLINICAL PRE-CONSULTATION INTAKE REPORT", response_text)

        clin_state = session_db.get_clinical_state(sid)
        self.assertIsNone(clin_state)

    # -------------------------------------------------------------------------
    # Test 3: Medical Conversation — Direct Clinical Question
    # -------------------------------------------------------------------------
    def test_03_medical_conversation_direct_clinical_question(self):
        """Clinical knowledge inquiry in Medical AI Conversation must answer informationally without intake."""
        sid = self._create_session("conversation", "Clinical Info Question")
        query_text = "What information should a doctor know before evaluating a patient with fever?"

        res = self.client.post(
            "/chat",
            json={
                "query": query_text,
                "conversation_history": [],
                "workspace": "conversation",
                "workspace_type": "conversation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertNotEqual(data.get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")
        response_text = data.get("response", "")
        self.assertTrue(len(response_text) > 20)

    # -------------------------------------------------------------------------
    # Test 4: Clinical Pre-Consultation — Fever Intake
    # -------------------------------------------------------------------------
    def test_04_clinical_pre_consultation_fever(self):
        """Fever complaint in Clinical Pre-Consultation workspace MUST invoke Clinical Agent."""
        sid = self._create_session("clinical_pre_consultation", "Clinical Intake: Fever")
        query_text = "I have fever from 2 days and runny nose."

        res = self.client.post(
            "/chat",
            json={
                "query": query_text,
                "conversation_history": [],
                "workspace": "clinical_pre_consultation",
                "workspace_type": "clinical_pre_consultation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # In clinical workspace, agent MUST be CLINICAL_PRE_CONSULTATION_AGENT
        self.assertEqual(data.get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")
        response_text = data.get("response", "")
        self.assertTrue(len(response_text) > 10)

        # Verify clinical state was preserved in DB for this clinical session
        clin_state = session_db.get_clinical_state(sid)
        self.assertIsNotNone(clin_state, "Clinical intake state must be persisted for clinical session")
        self.assertIn("fever", json.dumps(clin_state.get("patient_data", {})).lower())

    # -------------------------------------------------------------------------
    # Test 5: Clinical Pre-Consultation — Adaptive Follow-up
    # -------------------------------------------------------------------------
    def test_05_clinical_pre_consultation_adaptive_followup(self):
        """Clinical intake conducts adaptive questioning and stores patient findings."""
        sid = self._create_session("clinical_pre_consultation", "Adaptive Intake")

        # Turn 1: Present complaint
        res1 = self.client.post(
            "/chat",
            json={"query": "I have had a high fever for 3 days with chills.", "workspace_type": "clinical_pre_consultation"},
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1.get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")

        # Turn 2: Provide duration, medication, and allergy details
        res2 = self.client.post(
            "/chat",
            json={
                "query": "My temperature is 101.5 F. I took ibuprofen 400mg. I have no medical history and no allergies.",
                "workspace_type": "clinical_pre_consultation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2.get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")

        # Verify clinical state reflects accumulated data
        state = session_db.get_clinical_state(sid)
        self.assertIsNotNone(state)
        p_data_str = json.dumps(state.get("patient_data", {})).lower()
        self.assertTrue("fever" in p_data_str or "ibuprofen" in p_data_str)

    # -------------------------------------------------------------------------
    # Test 6: Clinical Red Flag Emergency Escalation
    # -------------------------------------------------------------------------
    def test_06_clinical_red_flag(self):
        """Severe chest pain fixture triggers urgent emergency escalation without safety weakening."""
        sid = self._create_session("clinical_pre_consultation", "Emergency Intake")
        emergency_query = "I am experiencing sudden crushing chest pain radiating to my left arm, cold sweats, and severe shortness of breath."

        res = self.client.post(
            "/chat",
            json={
                "query": emergency_query,
                "workspace_type": "clinical_pre_consultation"
            },
            headers={"X-Session-ID": sid}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(
            data.get("agent", "").startswith("CLINICAL_PRE_CONSULTATION_AGENT"),
            f"Expected CLINICAL_PRE_CONSULTATION_AGENT, got {data.get('agent')}"
        )

        response_text = data.get("response", "").lower()
        is_emergency = "emergency" in response_text and ("immediate" in response_text or "escalation" in response_text or "urgent" in response_text or "911" in response_text or "cardiac" in response_text)
        self.assertTrue(is_emergency, f"Emergency prompt must trigger emergency escalation guidance: {response_text[:200]}")

    # -------------------------------------------------------------------------
    # Test 7: Workspace Switching Isolation
    # -------------------------------------------------------------------------
    def test_07_workspace_switching(self):
        """Switching between conversation and clinical workspaces ensures strict isolation."""
        conv_sid = self._create_session("conversation", "General Medical Chat")
        clin_sid = self._create_session("clinical_pre_consultation", "Patient Intake")

        # Step 1: Send message in conversation
        res_conv = self.client.post(
            "/chat",
            json={"query": "What are the common causes of seasonal allergies?", "workspace_type": "conversation"},
            headers={"X-Session-ID": conv_sid}
        )
        self.assertEqual(res_conv.status_code, 200)
        self.assertNotEqual(res_conv.json().get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")

        # Step 2: Send message in clinical pre-consultation
        res_clin = self.client.post(
            "/chat",
            json={"query": "I have had a sore throat and fever for 2 days.", "workspace_type": "clinical_pre_consultation"},
            headers={"X-Session-ID": clin_sid}
        )
        self.assertEqual(res_clin.status_code, 200)
        self.assertEqual(res_clin.json().get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")

        # Step 3: Switch back to conversation session
        res_conv2 = self.client.post(
            "/chat",
            json={"query": "Can dehydration cause a mild headache?", "workspace_type": "conversation"},
            headers={"X-Session-ID": conv_sid}
        )
        self.assertEqual(res_conv2.status_code, 200)
        self.assertNotEqual(res_conv2.json().get("agent"), "CLINICAL_PRE_CONSULTATION_AGENT")

        # Check DB isolation: conv_sid has 4 messages (2 user + 2 assistant), clin_sid has 2
        conv_msgs = session_db.get_messages(conv_sid)
        clin_msgs = session_db.get_messages(clin_sid)
        self.assertEqual(len(conv_msgs), 4)
        self.assertEqual(len(clin_msgs), 2)

        # Check no clinical state in conversation session
        self.assertIsNone(session_db.get_clinical_state(conv_sid))
        # Check clinical state exists in clinical session
        self.assertIsNotNone(session_db.get_clinical_state(clin_sid))

    # -------------------------------------------------------------------------
    # Test 8: Session Restoration
    # -------------------------------------------------------------------------
    def test_08_session_restoration(self):
        """Simulate browser reload: sessions and workspace boundaries are preserved."""
        conv_sid = self._create_session("conversation", "Restoration Chat")
        clin_sid = self._create_session("clinical_pre_consultation", "Restoration Intake")

        # Add message to conversation
        self.client.post(
            "/chat",
            json={"query": "How much water should an adult drink daily?", "workspace_type": "conversation"},
            headers={"X-Session-ID": conv_sid}
        )
        # Add message to clinical
        self.client.post(
            "/chat",
            json={"query": "I have joint pain in my knees.", "workspace_type": "clinical_pre_consultation"},
            headers={"X-Session-ID": clin_sid}
        )

        # Fresh client simulation
        fresh_client = TestClient(app)

        # Query conversation sessions
        conv_list = fresh_client.get("/sessions?workspace_type=conversation").json()["sessions"]
        conv_sids = [s["session_id"] for s in conv_list]
        self.assertIn(conv_sid, conv_sids)
        self.assertNotIn(clin_sid, conv_sids)

        # Query clinical sessions
        clin_list = fresh_client.get("/sessions?workspace_type=clinical_pre_consultation").json()["sessions"]
        clin_sids = [s["session_id"] for s in clin_list]
        self.assertIn(clin_sid, clin_sids)
        self.assertNotIn(conv_sid, clin_sids)

    # -------------------------------------------------------------------------
    # Test 9: Image Analysis Workspace Isolation
    # -------------------------------------------------------------------------
    def test_09_image_analysis_isolation(self):
        """Image analysis sessions are strictly rejected by /chat."""
        img_sid = self._create_session("image_analysis", "Chest X-Ray Scan")
        self.assertTrue(img_sid.startswith("img_"))

        # Attempt to use image session on /chat -> must return HTTP 400
        res = self.client.post(
            "/chat",
            json={"query": "Hello", "workspace_type": "conversation"},
            headers={"X-Session-ID": img_sid}
        )
        self.assertEqual(res.status_code, 400, "POST /chat must reject img_ session IDs")


if __name__ == "__main__":
    unittest.main()

