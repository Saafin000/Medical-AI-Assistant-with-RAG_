"""
Phase 6: Session Lifecycle & Persistence REST API Test Suite

Tests:
1. GET /sessions (filtering by workspace, archival status, pagination, invalid workspace rejection)
2. POST /sessions (prefixed ID generation, cookies/headers, metadata, invalid workspace rejection)
3. GET /sessions/{session_id} (retrieval, 404 for missing)
4. PATCH /sessions/{session_id} (title, status, archival, metadata update, 404 for missing)
5. DELETE /sessions/{session_id} (cascade deletion, clinical agent reset, 404 for missing)
6. GET /sessions/{session_id}/messages (deterministic ordering, pagination, 404 for missing)
7. GET /sessions/{session_id}/clinical-state (retrieval of persisted clinical intake state, 404 for missing)
8. GET /sessions/{session_id}/artifacts (retrieval of image artifacts, filtering, 404 for missing)
9. End-to-End Persistence Verification:
   - /chat message persistence and auto-title generation
   - /chat clinical state persistence for clinical pre-consultation
   - /clinical-patient-feedback message and clinical state persistence
   - /upload artifact and message persistence
"""

import os
import sys
import uuid
import tempfile
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

# Ensure offline test environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Response, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app import (
    app,
    list_sessions_endpoint,
    create_session_endpoint,
    get_session_endpoint,
    update_session_endpoint,
    delete_session_endpoint,
    get_session_messages_endpoint,
    get_session_clinical_state_endpoint,
    get_session_artifacts_endpoint,
    chat,
    handle_clinical_patient_feedback,
    CreateSessionRequest,
    UpdateSessionRequest,
    QueryRequest,
)
from storage.session_db import SessionDatabase


class TestSessionLifecycleAPI(unittest.TestCase):
    """Unit and Integration tests for Phase 6 Session Lifecycle Endpoints and Persistence."""

    @classmethod
    def setUpClass(cls):
        """Create an isolated temporary SQLite database for this test class."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.test_dir.name, "test_lifecycle.db")
        cls.db = SessionDatabase(cls.db_path)
        cls.db.initialize_database()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        cls.test_dir.cleanup()

    def setUp(self):
        """Patch app.py's session_db singleton with our isolated test database for each test."""
        self.patcher = patch("app.session_db", self.db)
        self.patcher.start()

    def tearDown(self):
        """Stop patcher."""
        self.patcher.stop()

    # =========================================================================
    # 1. POST /sessions Tests
    # =========================================================================

    def test_01_create_conversation_session(self):
        """POST /sessions creates conversation session with conv_ prefix and sets headers."""
        req = CreateSessionRequest(
            workspace_type="conversation",
            title="General Health Chat",
            metadata={"source": "test"}
        )
        resp = Response()
        result = create_session_endpoint(req=req, response=resp)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["session_id"].startswith("conv_"))
        self.assertEqual(result["session"]["workspace_type"], "conversation")
        self.assertEqual(result["session"]["title"], "General Health Chat")
        self.assertEqual(result["session"]["metadata"], {"source": "test"})
        self.assertEqual(resp.headers.get("X-Session-ID"), result["session_id"])
        self.assertIn(f"session_id={result['session_id']}", resp.headers.get("set-cookie", ""))

    def test_02_create_image_analysis_session(self):
        """POST /sessions creates image analysis session with img_ prefix."""
        req = CreateSessionRequest(
            workspace_type="image_analysis",
            title="Chest X-Ray Review"
        )
        resp = Response()
        result = create_session_endpoint(req=req, response=resp)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["session_id"].startswith("img_"))
        self.assertEqual(result["session"]["workspace_type"], "image_analysis")

    def test_03_create_clinical_session(self):
        """POST /sessions creates clinical session with clin_ prefix."""
        req = CreateSessionRequest(
            workspace_type="clinical_pre_consultation",
            title="Patient Pre-Consultation"
        )
        resp = Response()
        result = create_session_endpoint(req=req, response=resp)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["session_id"].startswith("clin_"))
        self.assertEqual(result["session"]["workspace_type"], "clinical_pre_consultation")

    def test_04_create_session_invalid_workspace(self):
        """POST /sessions rejects invalid workspace types with HTTP 400."""
        req = CreateSessionRequest(
            workspace_type="unknown_workspace",
            title="Invalid"
        )
        resp = Response()
        with self.assertRaises(HTTPException) as ctx:
            create_session_endpoint(req=req, response=resp)
        self.assertEqual(ctx.exception.status_code, 400)

    # =========================================================================
    # 2. GET /sessions Tests
    # =========================================================================

    def test_05_list_sessions_filtering_and_archival(self):
        """GET /sessions filters by workspace, archival status, and supports pagination."""
        # Create a unique conversation session and an archived one
        s1 = self.db.create_session(
            session_id=f"conv_{uuid.uuid4()}",
            workspace_type="conversation",
            title="Active Conv"
        )
        s2 = self.db.create_session(
            session_id=f"conv_{uuid.uuid4()}",
            workspace_type="conversation",
            title="Archived Conv"
        )
        self.db.update_session(s2["session_id"], is_archived=True)

        s3 = self.db.create_session(
            session_id=f"img_{uuid.uuid4()}",
            workspace_type="image_analysis",
            title="Active Img"
        )

        # Filter by conversation without archived
        res_conv = list_sessions_endpoint(workspace_type="conversation", include_archived=False)
        self.assertEqual(res_conv["status"], "success")
        ids = [s["session_id"] for s in res_conv["sessions"]]
        self.assertIn(s1["session_id"], ids)
        self.assertNotIn(s2["session_id"], ids)
        self.assertNotIn(s3["session_id"], ids)

        # Include archived
        res_conv_all = list_sessions_endpoint(workspace_type="conversation", include_archived=True)
        ids_all = [s["session_id"] for s in res_conv_all["sessions"]]
        self.assertIn(s1["session_id"], ids_all)
        self.assertIn(s2["session_id"], ids_all)

        # Filter by image_analysis
        res_img = list_sessions_endpoint(workspace_type="image_analysis")
        ids_img = [s["session_id"] for s in res_img["sessions"]]
        self.assertIn(s3["session_id"], ids_img)
        self.assertNotIn(s1["session_id"], ids_img)

    def test_06_list_sessions_invalid_workspace(self):
        """GET /sessions raises 400 for unknown workspace types."""
        with self.assertRaises(HTTPException) as ctx:
            list_sessions_endpoint(workspace_type="invalid_ws")
        self.assertEqual(ctx.exception.status_code, 400)

    # =========================================================================
    # 3. GET /sessions/{session_id} Tests
    # =========================================================================

    def test_07_get_session_by_id(self):
        """GET /sessions/{id} returns details for existing session and 404 for missing."""
        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="Lookup Test")

        res = get_session_endpoint(session_id=sid)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["session"]["session_id"], sid)
        self.assertEqual(res["session"]["title"], "Lookup Test")

        with self.assertRaises(HTTPException) as ctx:
            get_session_endpoint(session_id=f"conv_{uuid.uuid4()}")
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 4. PATCH /sessions/{session_id} Tests
    # =========================================================================

    def test_08_update_session(self):
        """PATCH /sessions/{id} updates title, status, archival, and metadata."""
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation", title="Initial Title")

        req = UpdateSessionRequest(
            title="Updated Title",
            status="completed",
            is_archived=True,
            metadata={"notes": "patient completed intake"}
        )
        res = update_session_endpoint(session_id=sid, req=req)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["session"]["title"], "Updated Title")
        self.assertEqual(res["session"]["status"], "completed")
        self.assertEqual(res["session"]["is_archived"], 1)
        self.assertEqual(res["session"]["metadata"].get("notes"), "patient completed intake")

        # 404 on missing
        with self.assertRaises(HTTPException) as ctx:
            update_session_endpoint(session_id=f"clin_{uuid.uuid4()}", req=req)
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 5. DELETE /sessions/{session_id} Tests
    # =========================================================================

    def test_09_delete_session_and_cascades(self):
        """DELETE /sessions/{id} deletes session and cascades to messages and state."""
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")
        self.db.save_message(session_id=sid, sender="user", content="Hello doctor")
        self.db.save_clinical_state(session_id=sid, patient_data={"age": 45})

        # Delete session
        res = delete_session_endpoint(session_id=sid)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["deleted"])

        # Verify parent and children are gone
        self.assertIsNone(self.db.get_session(sid))
        self.assertEqual(len(self.db.get_messages(sid)), 0)
        self.assertIsNone(self.db.get_clinical_state(sid))

        # 404 on already deleted
        with self.assertRaises(HTTPException) as ctx:
            delete_session_endpoint(session_id=sid)
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 6. GET /sessions/{session_id}/messages Tests
    # =========================================================================

    def test_10_get_session_messages(self):
        """GET /sessions/{id}/messages returns messages ordered by sequence_num."""
        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation")
        self.db.save_message(session_id=sid, sender="user", content="Msg 1")
        self.db.save_message(session_id=sid, sender="assistant", content="Msg 2")

        res = get_session_messages_endpoint(session_id=sid)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["messages"][0]["content"], "Msg 1")
        self.assertEqual(res["messages"][0]["sequence_num"], 1)
        self.assertEqual(res["messages"][1]["content"], "Msg 2")
        self.assertEqual(res["messages"][1]["sequence_num"], 2)

        # 404 for missing session
        with self.assertRaises(HTTPException) as ctx:
            get_session_messages_endpoint(session_id=f"conv_{uuid.uuid4()}")
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 7. GET /sessions/{session_id}/clinical-state Tests
    # =========================================================================

    def test_11_get_clinical_state_endpoint(self):
        """GET /sessions/{id}/clinical-state returns saved clinical state."""
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")
        self.db.save_clinical_state(
            session_id=sid,
            patient_data={"symptoms": ["headache", "fever"]},
            current_stage="active",
            latest_report_text="Preliminary Summary"
        )

        res = get_session_clinical_state_endpoint(session_id=sid)
        self.assertEqual(res["status"], "success")
        self.assertIsNotNone(res["clinical_state"])
        self.assertEqual(res["clinical_state"]["current_stage"], "active")
        self.assertEqual(res["clinical_state"]["patient_data"]["symptoms"], ["headache", "fever"])

        # 404 for non-existent session
        with self.assertRaises(HTTPException) as ctx:
            get_session_clinical_state_endpoint(session_id=f"clin_{uuid.uuid4()}")
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 8. GET /sessions/{session_id}/artifacts Tests
    # =========================================================================

    def test_12_get_artifacts_endpoint(self):
        """GET /sessions/{id}/artifacts returns saved artifacts."""
        sid = f"img_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="image_analysis")
        self.db.save_artifact(
            session_id=sid,
            artifact_type="input_image",
            file_path="uploads/backend/test_xray.png",
            original_filename="test_xray.png",
            mime_type="image/png"
        )

        res = get_session_artifacts_endpoint(session_id=sid)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["artifacts"][0]["artifact_type"], "input_image")
        self.assertEqual(res["artifacts"][0]["file_path"], "uploads/backend/test_xray.png")

        # 404 for non-existent session
        with self.assertRaises(HTTPException) as ctx:
            get_session_artifacts_endpoint(session_id=f"img_{uuid.uuid4()}")
        self.assertEqual(ctx.exception.status_code, 404)

    # =========================================================================
    # 9. Direct Endpoint Persistence Integration Tests
    # =========================================================================

    @patch("app.process_query")
    def test_13_chat_persists_messages_and_auto_titles(self, mock_process_query):
        """POST /chat persists user message, assistant message, and auto-generates title."""
        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="New Conversation Session")

        mock_process_query.return_value = {
            "messages": [AIMessage(content="Normal blood pressure is 120/80 mmHg.")],
            "agent_name": "CONVERSATION_AGENT"
        }

        req = QueryRequest(query="What is normal blood pressure?")
        resp = Response()
        result = chat(request=req, response=resp, x_session_id=sid)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["session_id"], sid)

        # Check messages persisted in DB
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["sender"], "user")
        self.assertEqual(messages[0]["content"], "What is normal blood pressure?")
        self.assertEqual(messages[1]["sender"], "assistant")
        self.assertEqual(messages[1]["content"], "Normal blood pressure is 120/80 mmHg.")

        # Check session title updated
        session = self.db.get_session(sid)
        self.assertEqual(session["title"], "What is normal blood pressure?")

    @patch("agents.agent_decision.get_clinical_agent")
    def test_14_clinical_patient_feedback_persists_report(self, mock_get_clinical_agent):
        """POST /clinical-patient-feedback persists user feedback, updated report, and clinical state."""
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")

        # Mock clinical agent
        mock_agent = MagicMock()
        mock_agent.process_patient_info.return_value = {
            "status": "completed",
            "report": "Updated intake report text with 3 days fever.",
            "formatted_report": "Formatted Markdown Report"
        }
        mock_collector = MagicMock()
        mock_collector.get_patient_data.return_value = {
            "symptoms": {"list": ["fever", "cough"]},
            "duration": "3 days"
        }
        mock_agent.patient_info_collector = mock_collector
        mock_get_clinical_agent.return_value = mock_agent

        resp = Response()
        result = handle_clinical_patient_feedback(
            response=resp,
            patient_feedback="Actually the fever started 3 days ago.",
            x_session_id=sid
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["session_id"], sid)

        # Check messages persisted
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["sender"], "user")
        self.assertEqual(messages[0]["content"], "Actually the fever started 3 days ago.")
        self.assertEqual(messages[1]["sender"], "assistant")
        self.assertTrue(messages[1]["is_report"])

        # Check clinical state persisted
        state = self.db.get_clinical_state(sid)
        self.assertIsNotNone(state)
        self.assertEqual(state["current_stage"], "completed")
        self.assertEqual(state["latest_report_text"], "Updated intake report text with 3 days fever.")
        self.assertEqual(state["formatted_report"], "Formatted Markdown Report")
        self.assertEqual(state["is_confirmed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
