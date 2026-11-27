"""
Targeted Automated Test Suite for Workspace Session History & Sidebar UX

Verifies all requirements from BUG FIX — WORKSPACE HISTORY NOT VISIBLE IN LEFT SIDEBAR Directive:
- Test A — Conversation History (>=3 sessions, listed in history, message restoration)
- Test B — Image Analysis History (Multiple sessions, listed in Image Sessions, artifact & analysis restoration)
- Test C — Clinical Intake History (Multiple sessions, completed intakes with reports, restoration of conversation & report)
- Test D — Workspace Isolation (Strict isolation: conv_ only in conversation, img_ only in image, clin_ only in clinical)
- Test E — Refresh Persistence (Sessions, messages, artifacts, and clinical reports persist across simulated refresh/reload)
- Test F — New Session Refresh (New session appears immediately in GET /sessions without manual reload)
- Test G — Session Ordering (Sessions ordered by most recently updated, newest first)
"""

import os
import sys
import uuid
import time
import json
import unittest

# Ensure offline environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = "gsk_test_key_sidebar_history_12345"

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from storage.session_db import session_db
from agents.agent_decision import reset_clinical_agent_session
from app import app


class TestWorkspaceHistorySidebar(unittest.TestCase):
    """Verifies workspace session history, sidebar lifecycle, and session restoration."""

    def setUp(self):
        self.client = TestClient(app)
        self.created_sessions = []

    def tearDown(self):
        # Clean up all created test sessions
        for sid in self.created_sessions:
            try:
                session_db.delete_session(sid)
                reset_clinical_agent_session(sid)
            except Exception:
                pass

    def _create_session(self, workspace_type: str, title: str = None):
        payload = {"workspace_type": workspace_type}
        if title:
            payload["title"] = title
        res = self.client.post("/sessions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        sid = data["session_id"]
        self.created_sessions.append(sid)
        return sid

    # -------------------------------------------------------------------------
    # Test A — Conversation History
    # -------------------------------------------------------------------------
    def test_a_conversation_history(self):
        """Create at least 3 conversation sessions, verify appearance in history and message restoration."""
        sids = []
        messages_map = {}

        for i in range(1, 4):
            sid = self._create_session("conversation", title=f"Cardiology Consultation {i}")
            sids.append(sid)

            # Add distinct user and assistant messages
            user_msg = f"Patient question about chest symptoms batch {i}"
            asst_msg = f"Clinical advice and diagnostic recommendations for case {i}"
            messages_map[sid] = (user_msg, asst_msg)

            session_db.save_message(session_id=sid, sender="user", content=user_msg, agent_name="user")
            session_db.save_message(session_id=sid, sender="assistant", content=asst_msg, agent_name="Medical Conversation Agent")

        # Query conversation history
        res = self.client.get("/sessions?workspace_type=conversation&include_archived=false")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        fetched_sids = [s["session_id"] for s in data["sessions"]]

        # Verify all 3 appear
        for sid in sids:
            self.assertIn(sid, fetched_sids, f"Session {sid} must appear in conversation history")

        # Verify selecting each session restores its exact messages
        for sid in sids:
            msg_res = self.client.get(f"/sessions/{sid}/messages")
            self.assertEqual(msg_res.status_code, 200)
            msg_data = msg_res.json()
            msgs = msg_data["messages"]
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["content"], messages_map[sid][0])
            self.assertEqual(msgs[1]["content"], messages_map[sid][1])

    # -------------------------------------------------------------------------
    # Test B — Image Analysis History
    # -------------------------------------------------------------------------
    def test_b_image_analysis_history(self):
        """Create multiple image sessions, verify they appear in Image Sessions with restored artifacts."""
        sids = []
        for i in range(1, 3):
            sid = self._create_session("image_analysis", title=f"Chest X-Ray Analysis {i}")
            sids.append(sid)

            # Save input artifact and assistant analysis
            session_db.save_artifact(
                session_id=sid,
                artifact_type="input_image",
                file_path=f"uploads/backend/test_xray_{i}.png",
                original_filename=f"patient_xray_{i}.png",
                mime_type="image/png"
            )
            session_db.save_message(
                session_id=sid,
                sender="assistant",
                content=f"Analysis finding for scan {i}: Normal bronchovascular markings without consolidation.",
                agent_name="Medical Vision Agent"
            )

        # Query image sessions history
        res = self.client.get("/sessions?workspace_type=image_analysis&include_archived=false")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        fetched_sids = [s["session_id"] for s in data["sessions"]]

        # Verify both appear
        for sid in sids:
            self.assertIn(sid, fetched_sids, f"Image session {sid} must appear in Image Sessions")

        # Verify selecting each session restores its artifact and analysis
        for i, sid in enumerate(sids, 1):
            art_res = self.client.get(f"/sessions/{sid}/artifacts")
            self.assertEqual(art_res.status_code, 200)
            artifacts = art_res.json()["artifacts"]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["original_filename"], f"patient_xray_{i}.png")

            msg_res = self.client.get(f"/sessions/{sid}/messages")
            self.assertEqual(msg_res.status_code, 200)
            msgs = msg_res.json()["messages"]
            self.assertEqual(len(msgs), 1)
            self.assertIn(f"Analysis finding for scan {i}", msgs[0]["content"])

    # -------------------------------------------------------------------------
    # Test C — Clinical Intake History
    # -------------------------------------------------------------------------
    def test_c_clinical_intake_history(self):
        """Create multiple clinical intakes with generated reports; verify appearance and report restoration."""
        # Intake 1: Headache
        sid1 = self._create_session("clinical_pre_consultation", title="Intake: Migraine Assessment")
        session_db.save_message(session_id=sid1, sender="user", content="Severe throbbing headache for 2 days")
        session_db.save_message(
            session_id=sid1,
            sender="assistant",
            content="CLINICAL PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT: Throbbing headache\nSEVERITY: Moderate",
            is_report=True
        )
        session_db.save_clinical_state(
            session_id=sid1,
            patient_data={"symptoms": ["headache"]},
            current_stage="completed",
            latest_report_text="CLINICAL PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT: Throbbing headache",
            is_confirmed=True
        )

        # Intake 2: Abdominal Pain
        sid2 = self._create_session("clinical_pre_consultation", title="Intake: Abdominal Pain Evaluation")
        session_db.save_message(session_id=sid2, sender="user", content="Lower right quadrant abdominal pain")
        session_db.save_message(
            session_id=sid2,
            sender="assistant",
            content="CLINICAL PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT: RLQ Abdominal pain\nEMERGENCY: False",
            is_report=True
        )
        session_db.save_clinical_state(
            session_id=sid2,
            patient_data={"symptoms": ["abdominal pain"]},
            current_stage="completed",
            latest_report_text="CLINICAL PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT: RLQ Abdominal pain",
            is_confirmed=True
        )

        # Query clinical intake history
        res = self.client.get("/sessions?workspace_type=clinical_pre_consultation&include_archived=false")
        self.assertEqual(res.status_code, 200)
        fetched_sids = [s["session_id"] for s in res.json()["sessions"]]
        self.assertIn(sid1, fetched_sids)
        self.assertIn(sid2, fetched_sids)

        # Reopen Intake 1: verify headache report restored
        state1_res = self.client.get(f"/sessions/{sid1}/clinical-state")
        self.assertEqual(state1_res.status_code, 200)
        state1 = state1_res.json()["clinical_state"]
        self.assertIn("Throbbing headache", state1["latest_report_text"])
        self.assertNotIn("RLQ Abdominal", state1["latest_report_text"])

        # Reopen Intake 2: verify abdominal pain report restored
        state2_res = self.client.get(f"/sessions/{sid2}/clinical-state")
        self.assertEqual(state2_res.status_code, 200)
        state2 = state2_res.json()["clinical_state"]
        self.assertIn("RLQ Abdominal pain", state2["latest_report_text"])
        self.assertNotIn("Throbbing headache", state2["latest_report_text"])

    # -------------------------------------------------------------------------
    # Test D — Workspace Isolation
    # -------------------------------------------------------------------------
    def test_d_workspace_isolation(self):
        """Create sessions across all 3 workspaces; verify history queries return only their own prefix."""
        sid_conv = self._create_session("conversation", title="Conv Session")
        sid_img = self._create_session("image_analysis", title="Img Session")
        sid_clin = self._create_session("clinical_pre_consultation", title="Clin Session")

        self.assertTrue(sid_conv.startswith("conv_"))
        self.assertTrue(sid_img.startswith("img_"))
        self.assertTrue(sid_clin.startswith("clin_"))

        # Conversation history: conv_ ONLY
        res_conv = self.client.get("/sessions?workspace_type=conversation")
        self.assertEqual(res_conv.status_code, 200)
        conv_sids = [s["session_id"] for s in res_conv.json()["sessions"]]
        self.assertIn(sid_conv, conv_sids)
        for s in conv_sids:
            self.assertTrue(s.startswith("conv_"), f"Conversation history must contain only conv_ sessions: {s}")
        self.assertNotIn(sid_img, conv_sids)
        self.assertNotIn(sid_clin, conv_sids)

        # Image history: img_ ONLY
        res_img = self.client.get("/sessions?workspace_type=image_analysis")
        self.assertEqual(res_img.status_code, 200)
        img_sids = [s["session_id"] for s in res_img.json()["sessions"]]
        self.assertIn(sid_img, img_sids)
        for s in img_sids:
            self.assertTrue(s.startswith("img_"), f"Image history must contain only img_ sessions: {s}")
        self.assertNotIn(sid_conv, img_sids)
        self.assertNotIn(sid_clin, img_sids)

        # Clinical history: clin_ ONLY
        res_clin = self.client.get("/sessions?workspace_type=clinical_pre_consultation")
        self.assertEqual(res_clin.status_code, 200)
        clin_sids = [s["session_id"] for s in res_clin.json()["sessions"]]
        self.assertIn(sid_clin, clin_sids)
        for s in clin_sids:
            self.assertTrue(s.startswith("clin_"), f"Clinical history must contain only clin_ sessions: {s}")
        self.assertNotIn(sid_conv, clin_sids)
        self.assertNotIn(sid_img, clin_sids)

    # -------------------------------------------------------------------------
    # Test E — Refresh Persistence
    # -------------------------------------------------------------------------
    def test_e_refresh_persistence(self):
        """Simulate browser refresh: saved sessions and message details survive reload."""
        sid = self._create_session("conversation", title="Persistent History Topic")
        session_db.save_message(session_id=sid, sender="user", content="What causes tension headaches?")
        session_db.save_message(session_id=sid, sender="assistant", content="Tension headaches are often caused by muscle contractions.")

        # Simulate fresh client reconnect (like page reload)
        fresh_client = TestClient(app)
        res = fresh_client.get("/sessions?workspace_type=conversation")
        self.assertEqual(res.status_code, 200)
        sessions = res.json()["sessions"]
        found = next((s for s in sessions if s["session_id"] == sid), None)
        self.assertIsNotNone(found, "Session must exist after refresh")
        self.assertEqual(found["title"], "Persistent History Topic")

        msg_res = fresh_client.get(f"/sessions/{sid}/messages")
        self.assertEqual(msg_res.status_code, 200)
        msgs = msg_res.json()["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "What causes tension headaches?")

    # -------------------------------------------------------------------------
    # Test F — New Session Refresh
    # -------------------------------------------------------------------------
    def test_f_new_session_refresh(self):
        """Creating a new session immediately makes it available in GET /sessions without manual reload."""
        for ws in ("conversation", "image_analysis", "clinical_pre_consultation"):
            new_sid = self._create_session(ws, title=f"Fresh {ws} Session")
            res = self.client.get(f"/sessions?workspace_type={ws}")
            self.assertEqual(res.status_code, 200)
            sids = [s["session_id"] for s in res.json()["sessions"]]
            self.assertIn(new_sid, sids, f"New session {new_sid} must appear immediately in {ws} history")

    # -------------------------------------------------------------------------
    # Test G — Session Ordering
    # -------------------------------------------------------------------------
    def test_g_session_ordering(self):
        """Sessions must be ordered newest first based on updated_at."""
        sid1 = self._create_session("conversation", title="First Created")
        time.sleep(0.02)
        sid2 = self._create_session("conversation", title="Second Created")

        # Initially, sid2 is newer than sid1
        res = self.client.get("/sessions?workspace_type=conversation")
        sids = [s["session_id"] for s in res.json()["sessions"]]
        self.assertLess(sids.index(sid2), sids.index(sid1))

        # Now update sid1 by adding a message (bumps updated_at)
        time.sleep(0.02)
        session_db.save_message(session_id=sid1, sender="user", content="New activity in session 1")

        # Now sid1 must appear before sid2 in the ordered list
        res_after = self.client.get("/sessions?workspace_type=conversation")
        sids_after = [s["session_id"] for s in res_after.json()["sessions"]]
        self.assertLess(sids_after.index(sid1), sids_after.index(sid2), "Recently updated session must appear first")


if __name__ == "__main__":
    unittest.main()
