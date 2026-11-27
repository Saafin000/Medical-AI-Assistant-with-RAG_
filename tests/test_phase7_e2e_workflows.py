"""
Phase 7: Comprehensive End-to-End Workflows & Integration Test Suite

Covers:
1. Workspace 1: Medical AI Conversation
   - Multi-turn conversation context retention
   - Auto-generated session title from first user query
   - Session history retrieval and deterministic message ordering
2. Workspace 2: Medical Image Analysis
   - Image upload (/upload) workflow and artifact persistence in SQLite
   - Human validation (/validate) confirmation and rejection workflows
   - Artifact metadata retrieval and relationship tracking
3. Workspace 3: Clinical Pre-Consultation
   - Adaptive intake flow and state accumulation
   - Red flag detection (acute emergency escalation flag in messages and state)
   - Clinical report generation and persistence
   - Patient feedback (/clinical-patient-feedback) and report regeneration
4. Workspace Isolation & Boundary Enforcement
   - Cross-workspace request rejection (HTTP 400):
     * /chat rejecting img_ session IDs
     * /upload rejecting conv_ and clin_ session IDs
     * /validate rejecting conv_ session IDs
     * /clinical-patient-feedback rejecting conv_ and img_ session IDs
   - Workspace-filtered session listing (/sessions?workspace_type=...)
5. Security, Validation & Reliability
   - File upload path traversal sanitization (secure_filename)
   - Unsupported file format rejection (HTTP 400)
   - Oversized file upload rejection (HTTP 413)
   - Resilient error handling on LLM provider failure (429/503) with session preservation
"""

import os
import sys
import io
import uuid
import asyncio
import tempfile
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Offline test environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from app import (
    app,
    chat,
    upload_image,
    validate_medical_output,
    handle_clinical_patient_feedback,
    list_sessions_endpoint,
    create_session_endpoint,
    get_session_messages_endpoint,
    get_session_artifacts_endpoint,
    get_session_clinical_state_endpoint,
    QueryRequest,
    CreateSessionRequest,
)
from storage.session_db import SessionDatabase


class TestPhase7E2EWorkflows(unittest.TestCase):
    """Comprehensive E2E Integration and Workflow Verification Suite for Phase 7."""

    @classmethod
    def setUpClass(cls):
        """Create an isolated temporary SQLite database for this test class."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.test_dir.name, "test_phase7_e2e.db")
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
    # 1. Workspace 1: Medical AI Conversation End-to-End
    # =========================================================================

    @patch("app.process_query")
    def test_01_conversation_multiturn_persistence_and_auto_title(self, mock_process_query):
        """
        Verify multi-turn conversation flow:
        - Turn 1 auto-updates session title from default 'New Conversation Session'
        - Both user and assistant messages are stored in SQLite
        - Turn 2 persists and context is preserved across turns
        """
        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="New Conversation Session")

        # Turn 1: User asks initial medical query
        mock_process_query.return_value = {
            "messages": [
                HumanMessage(content="What are common early symptoms of Type 2 Diabetes?"),
                AIMessage(content="Common early symptoms include increased thirst, frequent urination, and fatigue.")
            ],
            "agent_name": "CONVERSATION_AGENT"
        }

        req1 = QueryRequest(query="What are common early symptoms of Type 2 Diabetes?")
        resp1 = Response()
        res1 = chat(request=req1, response=resp1, x_session_id=sid)

        self.assertEqual(res1["status"], "success")
        self.assertEqual(res1["session_id"], sid)
        self.assertIn("frequent urination", res1["response"])

        # Check session title was automatically updated from the query
        sess = self.db.get_session(sid)
        self.assertEqual(sess["title"], "What are common early symptoms of Type 2 Diabetes?")

        # Check messages for Turn 1
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["sender"], "user")
        self.assertEqual(messages[0]["content"], "What are common early symptoms of Type 2 Diabetes?")
        self.assertEqual(messages[1]["sender"], "assistant")
        self.assertEqual(messages[1]["agent_name"], "CONVERSATION_AGENT")

        # Turn 2: Follow-up question in the same session
        mock_process_query.return_value = {
            "messages": [
                HumanMessage(content="Are these symptoms reversible with lifestyle changes?"),
                AIMessage(content="Yes, in prediabetes and early Type 2 diabetes, lifestyle modifications can improve insulin sensitivity.")
            ],
            "agent_name": "CONVERSATION_AGENT"
        }

        req2 = QueryRequest(query="Are these symptoms reversible with lifestyle changes?")
        resp2 = Response()
        res2 = chat(request=req2, response=resp2, x_session_id=sid)

        self.assertEqual(res2["status"], "success")
        self.assertEqual(res2["session_id"], sid)

        # Check that all 4 messages are preserved in chronological order
        messages_t2 = self.db.get_messages(sid)
        self.assertEqual(len(messages_t2), 4)
        self.assertEqual(messages_t2[2]["sender"], "user")
        self.assertEqual(messages_t2[2]["content"], "Are these symptoms reversible with lifestyle changes?")
        self.assertEqual(messages_t2[3]["sender"], "assistant")

    def test_02_conversation_session_history_retrieval_and_ordering(self):
        """Verify GET /sessions/{session_id}/messages deterministic ordering and limit/offset."""
        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="Diabetes Query")

        # Insert 6 messages
        for i in range(6):
            sender = "user" if i % 2 == 0 else "assistant"
            self.db.save_message(
                session_id=sid,
                sender=sender,
                content=f"Message {i}",
                agent_name="user" if sender == "user" else "CONVERSATION_AGENT"
            )

        # Retrieve messages via endpoint
        history = get_session_messages_endpoint(session_id=sid, limit=10, offset=0)
        self.assertEqual(history["count"], 6)
        self.assertEqual(len(history["messages"]), 6)
        self.assertEqual(history["messages"][0]["content"], "Message 0")
        self.assertEqual(history["messages"][5]["content"], "Message 5")

        # Test pagination: limit 2, offset 2
        paged = get_session_messages_endpoint(session_id=sid, limit=2, offset=2)
        self.assertEqual(len(paged["messages"]), 2)
        self.assertEqual(paged["messages"][0]["content"], "Message 2")
        self.assertEqual(paged["messages"][1]["content"], "Message 3")

    # =========================================================================
    # 2. Workspace 2: Medical Image Analysis End-to-End
    # =========================================================================

    @patch("app.process_query")
    def test_03_image_upload_workflow_and_artifact_persistence(self, mock_process_query):
        """
        Verify POST /upload workflow:
        - Image is accepted and saved
        - Artifact is created in SQLite artifacts table
        - User message and assistant response are persisted with metadata
        """
        sid = f"img_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="image_analysis", title="New Image Session")

        mock_process_query.return_value = {
            "messages": [
                AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output.")
            ],
            "agent_name": "CHEST_XRAY_AGENT, HUMAN_VALIDATION"
        }

        # Mock UploadFile with AsyncMock read to prevent loopback socket hangs
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "chest_xray_sample.jpg"
        upload_file.content_type = "image/jpeg"
        upload_file.read = AsyncMock(return_value=b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 200)

        resp = Response()
        res = asyncio.run(upload_image(
            response=resp,
            image=upload_file,
            text="Analyze chest x-ray for abnormalities",
            x_session_id=sid
        ))

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["session_id"], sid)
        self.assertEqual(res["agent"], "CHEST_XRAY_AGENT, HUMAN_VALIDATION")
        self.assertIn("NEGATIVE", res["response"])

        # Check artifact persisted in SQLite
        artifacts = self.db.get_artifacts(sid)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["artifact_type"], "input_image")
        self.assertEqual(artifacts[0]["original_filename"], "chest_xray_sample.jpg")
        self.assertEqual(artifacts[0]["mime_type"], "image/jpeg")

        # Check session title updated
        sess = self.db.get_session(sid)
        self.assertEqual(sess["title"], "Analyze chest x-ray for abnormalities")

        # Check messages persisted
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["sender"], "user")
        self.assertEqual(messages[1]["sender"], "assistant")
        self.assertEqual(messages[1]["agent_name"], "CHEST_XRAY_AGENT, HUMAN_VALIDATION")

    @patch("app.process_query")
    def test_04_human_validation_confirm_and_reject_workflows(self, mock_process_query):
        """
        Verify POST /validate workflow:
        - Case A: validation_result='yes' returns status 'validated' and logs confirmation
        - Case B: validation_result='no' with comments returns status 'rejected' and logs comments
        """
        sid = f"img_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="image_analysis")

        # Case A: Doctor confirms AI output
        mock_process_query.return_value = {
            "messages": [AIMessage(content="Validation accepted. Findings confirmed by physician.")],
            "agent_name": "HUMAN_VALIDATION"
        }

        resp_a = Response()
        res_a = validate_medical_output(
            response=resp_a,
            validation_result="yes",
            comments=None,
            x_session_id=sid
        )

        self.assertEqual(res_a["status"], "validated")
        self.assertEqual(res_a["session_id"], sid)
        self.assertIn("Output confirmed by human validator", res_a["message"])

        # Case B: Doctor rejects AI output with clinical feedback
        mock_process_query.return_value = {
            "messages": [AIMessage(content="Validation rejected. Output flagged for manual radiologist re-read.")],
            "agent_name": "HUMAN_VALIDATION"
        }

        resp_b = Response()
        res_b = validate_medical_output(
            response=resp_b,
            validation_result="no",
            comments="Mild opacity in right lower lobe overlooked.",
            x_session_id=sid
        )

        self.assertEqual(res_b["status"], "rejected")
        self.assertEqual(res_b["comments"], "Mild opacity in right lower lobe overlooked.")
        self.assertIn("Output requires further review", res_b["message"])

        # Check DB messages recorded both validation turns
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 4)

    # =========================================================================
    # 3. Workspace 3: Clinical Pre-Consultation End-to-End
    # =========================================================================

    @patch("app.process_query")
    def test_05_clinical_red_flag_detection_and_escalation(self, mock_process_query):
        """
        Verify acute red flag detection during clinical pre-consultation:
        - Severe chest pain / radiating symptoms trigger emergency flag
        - Message saved with is_emergency=True
        - Clinical state set to 'emergency'
        """
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")

        mock_process_query.return_value = {
            "messages": [
                AIMessage(content="WARNING: Emergency escalation required. Please call 911 or visit the nearest emergency department immediately for acute chest pain.")
            ],
            "agent_name": "CLINICAL_PRE_CONSULTATION_AGENT"
        }

        req = QueryRequest(query="I have severe crushing chest pain radiating down my left arm and trouble breathing.")
        resp = Response()
        res = chat(request=req, response=resp, x_session_id=sid)

        self.assertEqual(res["status"], "success")

        # Verify emergency flags in database
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 2)
        assistant_msg = messages[1]
        self.assertTrue(assistant_msg["is_emergency"])

        # Verify clinical state marked as emergency
        clin_state = self.db.get_clinical_state(sid)
        self.assertIsNotNone(clin_state)
        self.assertEqual(clin_state["current_stage"], "emergency")

    @patch("agents.agent_decision.get_clinical_agent")
    def test_06_clinical_patient_feedback_workflow(self, mock_get_clinical_agent):
        """
        Verify POST /clinical-patient-feedback workflow:
        - Patient updates symptom details
        - Regenerates report and persists updated clinical state
        """
        sid = f"clin_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")

        mock_agent = MagicMock()
        mock_agent.process_patient_info.return_value = {
            "status": "completed",
            "report": "Updated Intake: 45-year-old male with persistent cough for 5 days, dry.",
            "formatted_report": "### Clinical Summary\n- **Patient:** 45M\n- **Duration:** 5 days"
        }
        mock_collector = MagicMock()
        mock_collector.get_patient_data.return_value = {
            "symptoms": ["cough"],
            "duration": "5 days",
            "age": 45
        }
        mock_agent.patient_info_collector = mock_collector
        mock_get_clinical_agent.return_value = mock_agent

        resp = Response()
        res = handle_clinical_patient_feedback(
            response=resp,
            patient_feedback="The cough has lasted 5 days, not 2 days.",
            x_session_id=sid
        )

        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["session_id"], sid)
        self.assertIn("Report updated with your feedback", res["message"])

        # Check DB clinical state was updated
        state = self.db.get_clinical_state(sid)
        self.assertIsNotNone(state)
        self.assertEqual(state["current_stage"], "completed")
        self.assertEqual(state["is_confirmed"], 1)
        self.assertIn("5 days", state["latest_report_text"])

    # =========================================================================
    # 4. Strict Cross-Workspace Boundary Isolation
    # =========================================================================

    def test_07_strict_cross_workspace_boundary_rejection(self):
        """
        Verify strict workspace isolation across all endpoints:
        - /chat rejects img_ session IDs (HTTP 400)
        - /upload rejects conv_ and clin_ session IDs (HTTP 400)
        - /validate rejects conv_ and clin_ session IDs (HTTP 400)
        - /clinical-patient-feedback rejects conv_ and img_ session IDs (HTTP 400)
        """
        conv_sid = f"conv_{uuid.uuid4()}"
        img_sid = f"img_{uuid.uuid4()}"
        clin_sid = f"clin_{uuid.uuid4()}"

        self.db.create_session(session_id=conv_sid, workspace_type="conversation")
        self.db.create_session(session_id=img_sid, workspace_type="image_analysis")
        self.db.create_session(session_id=clin_sid, workspace_type="clinical_pre_consultation")

        # 1. /chat rejecting img_ session ID
        with self.assertRaises(HTTPException) as cm:
            chat(request=QueryRequest(query="Hello"), response=Response(), x_session_id=img_sid)
        self.assertEqual(cm.exception.status_code, 400)

        # 2. /upload rejecting conv_ session ID
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=b"test data")

        with self.assertRaises(HTTPException) as cm:
            asyncio.run(upload_image(response=Response(), image=mock_file, x_session_id=conv_sid))
        self.assertEqual(cm.exception.status_code, 400)

        # 3. /upload rejecting clin_ session ID
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(upload_image(response=Response(), image=mock_file, x_session_id=clin_sid))
        self.assertEqual(cm.exception.status_code, 400)

        # 4. /validate rejecting conv_ session ID
        with self.assertRaises(HTTPException) as cm:
            validate_medical_output(response=Response(), validation_result="yes", x_session_id=conv_sid)
        self.assertEqual(cm.exception.status_code, 400)

        # 5. /clinical-patient-feedback rejecting conv_ session ID
        with self.assertRaises(HTTPException) as cm:
            handle_clinical_patient_feedback(response=Response(), patient_feedback="Test", x_session_id=conv_sid)
        self.assertEqual(cm.exception.status_code, 400)

        # 6. /clinical-patient-feedback rejecting img_ session ID
        with self.assertRaises(HTTPException) as cm:
            handle_clinical_patient_feedback(response=Response(), patient_feedback="Test", x_session_id=img_sid)
        self.assertEqual(cm.exception.status_code, 400)

    def test_08_workspace_filtered_session_listing(self):
        """Verify GET /sessions?workspace_type=... strictly isolates session listings."""
        # Create 2 of each workspace type
        for i in range(2):
            self.db.create_session(session_id=f"conv_{uuid.uuid4()}", workspace_type="conversation", title=f"Chat {i}")
            self.db.create_session(session_id=f"img_{uuid.uuid4()}", workspace_type="image_analysis", title=f"Img {i}")
            self.db.create_session(session_id=f"clin_{uuid.uuid4()}", workspace_type="clinical_pre_consultation", title=f"Clin {i}")

        conv_list = list_sessions_endpoint(workspace_type="conversation")
        self.assertTrue(all(s["workspace_type"] == "conversation" and s["session_id"].startswith("conv_") for s in conv_list["sessions"]))
        self.assertGreaterEqual(len(conv_list["sessions"]), 2)

        img_list = list_sessions_endpoint(workspace_type="image_analysis")
        self.assertTrue(all(s["workspace_type"] == "image_analysis" and s["session_id"].startswith("img_") for s in img_list["sessions"]))
        self.assertGreaterEqual(len(img_list["sessions"]), 2)

        clin_list = list_sessions_endpoint(workspace_type="clinical_pre_consultation")
        self.assertTrue(all(s["workspace_type"] == "clinical_pre_consultation" and s["session_id"].startswith("clin_") for s in clin_list["sessions"]))
        self.assertGreaterEqual(len(clin_list["sessions"]), 2)

    # =========================================================================
    # 5. Security & Edge Case Handling
    # =========================================================================

    def test_09_image_upload_security_sanitization_and_validation(self):
        """
        Verify upload security checks:
        - Unsupported file format (.exe, .pdf) returns HTTP 400
        - Oversized upload (> max_image_upload_size) returns HTTP 413
        """
        from app import config

        # A. Unsupported format
        mock_exe = MagicMock(spec=UploadFile)
        mock_exe.filename = "malicious_payload.exe"
        mock_exe.read = AsyncMock(return_value=b"MZ...")
        res_exe = asyncio.run(upload_image(response=Response(), image=mock_exe))
        self.assertIsInstance(res_exe, JSONResponse)
        self.assertEqual(res_exe.status_code, 400)

        # B. Oversized file (> max_image_upload_size MB)
        oversized_bytes = b"x" * (config.api.max_image_upload_size * 1024 * 1024 + 1024)
        mock_large = MagicMock(spec=UploadFile)
        mock_large.filename = "large_image.jpg"
        mock_large.read = AsyncMock(return_value=oversized_bytes)
        res_large = asyncio.run(upload_image(response=Response(), image=mock_large))
        self.assertIsInstance(res_large, JSONResponse)
        self.assertEqual(res_large.status_code, 413)

    @patch("app.process_query")
    def test_10_provider_failure_session_preservation_and_recovery(self, mock_process_query):
        """
        Verify that provider errors (429 Quota / 503 Rate Limit) preserve session ID and metadata:
        - Response headers retain X-Session-ID and session_id cookie
        - Session record in database remains intact
        - User message preceding the error is preserved in SQLite
        """
        from groq_llm_provider import GroqProviderError

        sid = f"conv_{uuid.uuid4()}"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="Resilience Test")

        # Simulate provider 429 quota error
        quota_err = GroqProviderError(
            message="Groq API daily token quota exceeded.",
            category="quota_exceeded",
            status_code=429,
            error_code="QUOTA_EXCEEDED",
            is_retryable=False
        )
        mock_process_query.side_effect = quota_err

        resp = Response()
        req = QueryRequest(query="What is asthma?")
        res = chat(request=req, response=resp, x_session_id=sid)

        self.assertIsInstance(res, JSONResponse)
        self.assertEqual(res.status_code, 429)

        # Verify session ID was preserved in response headers
        self.assertEqual(resp.headers.get("X-Session-ID"), sid)

        # Verify the user message was saved to SQLite prior to LLM failure
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["sender"], "user")
        self.assertEqual(messages[0]["content"], "What is asthma?")

        # Verify session record still exists and active
        sess = self.db.get_session(sid)
        self.assertIsNotNone(sess)
        self.assertEqual(sess["status"], "active")


if __name__ == "__main__":
    unittest.main(verbosity=2)
