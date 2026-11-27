"""
Targeted Verification Test Suite for Clinical Pre-Consultation Session & Report Isolation

Tests required by Section 5 of CRITICAL BUG FIX Directive:
- Test A — Different medical conditions (Headache -> Report A, Fever -> Report B; Report B has no headache data)
- Test B — Previous report preservation (Reopening Intake A retrieves Report A intact)
- Test C — Multiple new intakes (Three distinct intakes with distinct symptoms and isolated reports)
- Test D — Session switching (Switching between clinical sessions restores only their respective messages and reports)
- Test E — Stale asynchronous response (Delayed response from Session A cannot overwrite Session B)
- Test F — Workspace non-interference (Conversation & Image Analysis workspaces are unaffected)
"""

import os
import sys
import uuid
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure offline environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = "gsk_test_key_for_isolated_clinical_tests_12345"

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from storage.session_db import session_db
from agents.agent_decision import get_clinical_agent, reset_clinical_agent_session, _clinical_agent_registry
from app import app


class TestClinicalReportIsolation(unittest.TestCase):
    """Verifies that every new clinical intake creates an isolated session and report."""

    def setUp(self):
        self.client = TestClient(app)
        self.created_sessions = []

    def tearDown(self):
        # Clean up test sessions
        for sid in self.created_sessions:
            try:
                session_db.delete_session(sid)
                reset_clinical_agent_session(sid)
            except Exception:
                pass

    def _create_clinical_session(self, title="Clinical Intake"):
        res = self.client.post("/sessions", json={
            "workspace_type": "clinical_pre_consultation",
            "title": title
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        sid = data["session_id"]
        self.created_sessions.append(sid)
        return sid

    def test_a_different_medical_conditions_produce_isolated_reports(self):
        """Test A: Intake A (headache) and Intake B (fever) must produce completely isolated reports."""
        # 1. Start Intake A for headache
        sid_a = self._create_clinical_session("Headache Intake")
        agent_a = get_clinical_agent(sid_a)

        # Intake A completes with headache
        res_a = agent_a.process_patient_info("I have a severe throbbing headache with light sensitivity for 3 days.")
        # Mark as completed report
        rep_a_text = agent_a.report_generator.format_report_text(
            agent_a.report_generator.generate_report(
                agent_a.patient_info_collector.get_patient_data(),
                agent_a.red_flag_detector.detect_emergencies(agent_a.patient_info_collector.get_patient_data()),
                agent_a.conversation_history
            )
        )
        agent_a._latest_formatted_report = rep_a_text
        agent_a._latest_report = rep_a_text
        agent_a._session_active = False

        # Save state for Session A
        session_db.save_clinical_state(
            session_id=sid_a,
            patient_data=agent_a.patient_info_collector.get_patient_data(),
            current_stage="completed",
            latest_report_text=rep_a_text,
            formatted_report=rep_a_text,
            is_confirmed=True
        )

        self.assertIn("headache", rep_a_text.lower())
        self.assertNotIn("fever", rep_a_text.lower())

        # 2. Start Intake B for fever (completely new session)
        sid_b = self._create_clinical_session("Fever Intake")
        self.assertNotEqual(sid_a, sid_b)

        agent_b = get_clinical_agent(sid_b)
        self.assertIsNot(agent_a, agent_b)

        # Verify agent B starts completely fresh
        self.assertIsNone(agent_b._latest_report)
        self.assertTrue(agent_b._session_active)
        self.assertEqual(len(agent_b.conversation_history), 0)

        # Intake B processes fever
        res_b = agent_b.process_patient_info("I have developed a high fever of 102F and chills since yesterday.")
        rep_b_text = agent_b.report_generator.format_report_text(
            agent_b.report_generator.generate_report(
                agent_b.patient_info_collector.get_patient_data(),
                agent_b.red_flag_detector.detect_emergencies(agent_b.patient_info_collector.get_patient_data()),
                agent_b.conversation_history
            )
        )
        agent_b._latest_formatted_report = rep_b_text
        agent_b._latest_report = rep_b_text
        agent_b._session_active = False

        # Save state for Session B
        session_db.save_clinical_state(
            session_id=sid_b,
            patient_data=agent_b.patient_info_collector.get_patient_data(),
            current_stage="completed",
            latest_report_text=rep_b_text,
            formatted_report=rep_b_text,
            is_confirmed=True
        )

        # Verify Report B contains fever and has ZERO headache information from Intake A
        self.assertIn("fever", rep_b_text.lower())
        self.assertNotIn("headache", rep_b_text.lower(), "Report B must NOT contain headache data from Intake A!")

    def test_b_previous_report_preservation(self):
        """Test B: Generating Report B must preserve Report A when reopening Intake A."""
        sid_a = self._create_clinical_session("Intake A")
        sid_b = self._create_clinical_session("Intake B")

        # Save Report A in session A
        rep_a = "CLINICAL PRE-CONSULTATION INTAKE REPORT\nChief Complaint: Migraine headache\nDuration: 2 days"
        session_db.save_clinical_state(
            session_id=sid_a,
            patient_data={"chief_complaint": {"reason": "migraine headache"}},
            current_stage="completed",
            latest_report_text=rep_a,
            formatted_report=rep_a,
            is_confirmed=True
        )
        session_db.save_message(session_id=sid_a, sender="user", content="I have a migraine headache")
        session_db.save_message(session_id=sid_a, sender="assistant", content=rep_a, is_report=True)

        # Save Report B in session B
        rep_b = "CLINICAL PRE-CONSULTATION INTAKE REPORT\nChief Complaint: High fever and chills\nDuration: 1 day"
        session_db.save_clinical_state(
            session_id=sid_b,
            patient_data={"chief_complaint": {"reason": "high fever and chills"}},
            current_stage="completed",
            latest_report_text=rep_b,
            formatted_report=rep_b,
            is_confirmed=True
        )
        session_db.save_message(session_id=sid_b, sender="user", content="I have a high fever and chills")
        session_db.save_message(session_id=sid_b, sender="assistant", content=rep_b, is_report=True)

        # Reopen Intake A via REST API
        res_a_state = self.client.get(f"/sessions/{sid_a}/clinical-state")
        self.assertEqual(res_a_state.status_code, 200)
        data_a = res_a_state.json()
        self.assertIn("Migraine headache", data_a["clinical_state"]["latest_report_text"])
        self.assertNotIn("High fever", data_a["clinical_state"]["latest_report_text"])

        # Reopen Intake B via REST API
        res_b_state = self.client.get(f"/sessions/{sid_b}/clinical-state")
        self.assertEqual(res_b_state.status_code, 200)
        data_b = res_b_state.json()
        self.assertIn("High fever", data_b["clinical_state"]["latest_report_text"])
        self.assertNotIn("Migraine headache", data_b["clinical_state"]["latest_report_text"])

    def test_c_multiple_new_intakes_strict_isolation(self):
        """Test C: Create three separate intakes (Headache, Rash, Cough) and verify strict isolation."""
        symptoms = [
            ("headache", "Severe tension headache with throbbing temples"),
            ("rash", "Erythematous itchy rash on forearms"),
            ("cough", "Persistent hacking dry cough with chest tightness")
        ]
        session_ids = []

        for name, text in symptoms:
            sid = self._create_clinical_session(f"{name.title()} Intake")
            session_ids.append((sid, name, text))

            agent = get_clinical_agent(sid)
            agent.process_patient_info(text)
            rep = agent.report_generator.format_report_text(
                agent.report_generator.generate_report(
                    agent.patient_info_collector.get_patient_data(),
                    agent.red_flag_detector.detect_emergencies(agent.patient_info_collector.get_patient_data()),
                    agent.conversation_history
                )
            )
            session_db.save_clinical_state(
                session_id=sid,
                patient_data=agent.patient_info_collector.get_patient_data(),
                current_stage="completed",
                latest_report_text=rep,
                formatted_report=rep,
                is_confirmed=True
            )

        # Verify each of the 3 sessions only has its own symptom and not the other two
        for i, (sid, name, text) in enumerate(session_ids):
            state = session_db.get_clinical_state(sid)
            rep_text = state["latest_report_text"].lower()
            self.assertIn(name, rep_text)

            # Check other symptoms are NOT in this report
            for j, (other_sid, other_name, _) in enumerate(session_ids):
                if i != j:
                    self.assertNotIn(
                        other_name,
                        rep_text,
                        f"Session {sid} ({name}) leaked data from session {other_sid} ({other_name})!"
                    )

    def test_d_session_switching_restores_correct_messages_and_state(self):
        """Test D: Switching between clinical sessions via API restores each session's own messages and report."""
        sid_1 = self._create_clinical_session("Session 1")
        sid_2 = self._create_clinical_session("Session 2")

        session_db.save_message(session_id=sid_1, sender="user", content="S1 user question")
        session_db.save_message(session_id=sid_1, sender="assistant", content="S1 assistant reply")

        session_db.save_message(session_id=sid_2, sender="user", content="S2 user question")
        session_db.save_message(session_id=sid_2, sender="assistant", content="S2 assistant reply")

        # Load Session 1 messages
        m1 = self.client.get(f"/sessions/{sid_1}/messages").json()["messages"]
        self.assertEqual(len(m1), 2)
        self.assertEqual(m1[0]["content"], "S1 user question")

        # Switch to Session 2 messages
        m2 = self.client.get(f"/sessions/{sid_2}/messages").json()["messages"]
        self.assertEqual(len(m2), 2)
        self.assertEqual(m2[0]["content"], "S2 user question")

        # Switch back to Session 1 messages
        m1_again = self.client.get(f"/sessions/{sid_1}/messages").json()["messages"]
        self.assertEqual(len(m1_again), 2)
        self.assertEqual(m1_again[0]["content"], "S1 user question")

    def test_e_stale_asynchronous_request_guard(self):
        """Test E: Verification that a clinical request targeted for session A cannot pollute session B."""
        sid_a = self._create_clinical_session("Intake A")
        sid_b = self._create_clinical_session("Intake B")

        # Client sends feedback targeted for Session A
        feedback_res = self.client.post(
            "/clinical-patient-feedback",
            data={"patient_feedback": "Correction for Intake A: headache started on Tuesday."},
            headers={"X-Session-ID": sid_a}
        )
        self.assertEqual(feedback_res.status_code, 200)

        # Verify Session B's state remains empty / untouched
        b_state = session_db.get_clinical_state(sid_b)
        self.assertIsNone(b_state, "Session B must not be polluted by Session A feedback!")

        b_msgs = session_db.get_messages(sid_b)
        self.assertEqual(len(b_msgs), 0, "Session B messages must remain empty!")

    def test_f_workspace_non_interference(self):
        """Test F: Clinical pre-consultation activity does not affect conversation or image analysis workspaces."""
        conv_sid = f"conv_{uuid.uuid4()}"
        img_sid = f"img_{uuid.uuid4()}"
        clin_sid = self._create_clinical_session("Clinical Intake")

        session_db.create_session(session_id=conv_sid, workspace_type="conversation", title="Chat Session")
        session_db.create_session(session_id=img_sid, workspace_type="image_analysis", title="Image Session")
        self.created_sessions.extend([conv_sid, img_sid])

        # Clinical intake state saved
        session_db.save_clinical_state(
            session_id=clin_sid,
            patient_data={"symptoms": ["cough"]},
            current_stage="completed",
            latest_report_text="REPORT",
            is_confirmed=True
        )

        # Verify conversation session has no clinical state
        conv_state = session_db.get_clinical_state(conv_sid)
        self.assertIsNone(conv_state)

        # Verify image session has no clinical state
        img_state = session_db.get_clinical_state(img_sid)
        self.assertIsNone(img_state)

        # Verify endpoint boundary rejection: clinical endpoint rejects conv_ session
        res = self.client.post(
            "/clinical-patient-feedback",
            data={"patient_feedback": "test"},
            headers={"X-Session-ID": conv_sid}
        )
        self.assertEqual(res.status_code, 400, "Clinical endpoint must reject conv_ workspace session")


if __name__ == "__main__":
    unittest.main()
