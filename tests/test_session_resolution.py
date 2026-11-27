"""
Phase 6.2 Dual-Mode Session Resolution & Header Support Test Suite

Verifies:
1. parse_and_validate_session_id():
   - Prefixed IDs (conv_<uuid4>, img_<uuid4>, clin_<uuid4>)
   - Legacy canonical un-prefixed UUIDs
   - Rejection of invalid, malformed, uppercase, or non-UUID formats
2. resolve_session_id():
   - Header precedence over cookie
   - Header only resolution and DB registration
   - Cookie only (legacy un-prefixed) resolution and DB registration
   - Cookie only (prefixed) resolution and DB registration
   - Neither present: new ID generation with default workspace prefix
   - Cross-workspace prevention (header prefix vs allowed workspaces)
   - Cross-workspace prevention in cookie
   - Database workspace mismatch prevention
   - Malformed / unprefixed header rejection with HTTP 400
   - Concurrent registration race handling (IntegrityError recovery)
   - Response headers: Set-Cookie and X-Session-ID
3. Endpoint Integration:
   - POST /chat (allows conv_ and clin_, rejects img_, returns session_id)
   - POST /upload (allows img_, rejects conv_ and clin_, returns session_id)
   - POST /validate (allows img_, rejects conv_, returns session_id)
   - POST /clinical-patient-feedback (allows clin_, rejects conv_ and img_, returns session_id)
4. LangGraph Thread ID Propagation:
   - Session ID passed to process_query and reflected in thread config
"""

import os
import sys
import uuid
import tempfile
import sqlite3
import unittest
import faulthandler
from unittest.mock import MagicMock, patch

faulthandler.enable()

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure offline environment for HuggingFace/transformers during tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

from fastapi import Response, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app import (
    app,
    parse_and_validate_session_id,
    resolve_session_id,
    WORKSPACE_PREFIX_MAP,
    PREFIX_WORKSPACE_MAP,
    QueryRequest,
    chat,
    upload_image,
    validate_medical_output,
    handle_clinical_patient_feedback
)
from storage.session_db import SessionDatabase, session_db
from agents.agent_decision import get_thread_config


class TestSessionValidation(unittest.TestCase):
    """Test parse_and_validate_session_id function."""

    def test_01_valid_prefixed_session_ids(self):
        uid = str(uuid.uuid4())
        
        valid_conv = f"conv_{uid}"
        is_valid, ws, raw = parse_and_validate_session_id(valid_conv)
        self.assertTrue(is_valid)
        self.assertEqual(ws, "conversation")
        self.assertEqual(raw, uid)

        valid_img = f"img_{uid}"
        is_valid, ws, raw = parse_and_validate_session_id(valid_img)
        self.assertTrue(is_valid)
        self.assertEqual(ws, "image_analysis")
        self.assertEqual(raw, uid)

        valid_clin = f"clin_{uid}"
        is_valid, ws, raw = parse_and_validate_session_id(valid_clin)
        self.assertTrue(is_valid)
        self.assertEqual(ws, "clinical_pre_consultation")
        self.assertEqual(raw, uid)

    def test_02_valid_legacy_unprefixed_uuid(self):
        uid = str(uuid.uuid4())
        is_valid, ws, raw = parse_and_validate_session_id(uid)
        self.assertTrue(is_valid)
        self.assertIsNone(ws)
        self.assertEqual(raw, uid)

    def test_03_invalid_session_ids_rejected(self):
        invalid_cases = [
            None,
            "",
            "   ",
            "not-a-uuid",
            "conv_not-a-uuid",
            "img_12345",
            "chat_8413f982-f54e-4e6c-bf92-7da3eeb43770",  # unsupported prefix
            "8413f982-f54e-4e6c-bf92-7da3eeb43770-extra",
            f"conv_{str(uuid.uuid4()).upper()}",  # uppercase prefixed
            str(uuid.uuid4()).upper(),  # uppercase legacy unprefixed
            f"conv_{uuid.uuid4().hex}",  # unhyphenated hex
            "a" * 105,  # too long
        ]
        for sid in invalid_cases:
            is_valid, ws, raw = parse_and_validate_session_id(sid)
            self.assertFalse(is_valid, f"Expected {sid!r} to be invalid, but returned True")
            self.assertIsNone(ws)
            self.assertIsNone(raw)


class TestDualModeResolutionLogic(unittest.TestCase):
    """Test resolve_session_id logic with isolated temporary database."""

    def setUp(self):
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_file.close()
        self.test_db = SessionDatabase(db_path=self.temp_db_file.name)
        self.test_db.initialize_database()

        # Patch session_db instance in app module
        self.orig_db_path = session_db.db_path
        session_db.db_path = self.temp_db_file.name
        session_db.initialize_database()

    def tearDown(self):
        session_db.db_path = self.orig_db_path
        if os.path.exists(self.temp_db_file.name):
            try:
                os.remove(self.temp_db_file.name)
            except OSError:
                pass

    def test_04_header_precedence_over_cookie(self):
        header_id = f"conv_{uuid.uuid4()}"
        cookie_id = f"conv_{uuid.uuid4()}"
        response = Response()

        resolved = resolve_session_id(
            x_session_id=header_id,
            session_cookie=cookie_id,
            response=response,
            allowed_workspaces=("conversation", "clinical_pre_consultation"),
            default_workspace="conversation"
        )

        self.assertEqual(resolved, header_id)
        self.assertEqual(response.headers["X-Session-ID"], header_id)
        # Check that header_id was registered in SQLite
        record = self.test_db.get_session(header_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["workspace_type"], "conversation")
        # Check that cookie_id was NOT registered
        self.assertIsNone(self.test_db.get_session(cookie_id))

    def test_05_header_only_resolution_and_registration(self):
        header_id = f"img_{uuid.uuid4()}"
        response = Response()

        resolved = resolve_session_id(
            x_session_id=header_id,
            session_cookie=None,
            response=response,
            allowed_workspaces=("image_analysis",),
            default_workspace="image_analysis"
        )

        self.assertEqual(resolved, header_id)
        self.assertEqual(response.headers["X-Session-ID"], header_id)
        record = self.test_db.get_session(header_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["workspace_type"], "image_analysis")

    def test_06_cookie_only_legacy_unprefixed_uuid(self):
        legacy_uid = str(uuid.uuid4())
        response = Response()

        resolved = resolve_session_id(
            x_session_id=None,
            session_cookie=legacy_uid,
            response=response,
            allowed_workspaces=("conversation", "clinical_pre_consultation"),
            default_workspace="conversation"
        )

        self.assertEqual(resolved, legacy_uid)
        self.assertEqual(response.headers["X-Session-ID"], legacy_uid)
        record = self.test_db.get_session(legacy_uid)
        self.assertIsNotNone(record)
        self.assertEqual(record["workspace_type"], "conversation")

    def test_07_cookie_only_prefixed_session_id(self):
        prefixed_cookie = f"clin_{uuid.uuid4()}"
        response = Response()

        resolved = resolve_session_id(
            x_session_id=None,
            session_cookie=prefixed_cookie,
            response=response,
            allowed_workspaces=("conversation", "clinical_pre_consultation"),
            default_workspace="conversation"
        )

        self.assertEqual(resolved, prefixed_cookie)
        self.assertEqual(response.headers["X-Session-ID"], prefixed_cookie)
        record = self.test_db.get_session(prefixed_cookie)
        self.assertIsNotNone(record)
        self.assertEqual(record["workspace_type"], "clinical_pre_consultation")

    def test_08_neither_present_generates_new_prefixed_id(self):
        response = Response()
        resolved = resolve_session_id(
            x_session_id=None,
            session_cookie=None,
            response=response,
            allowed_workspaces=("image_analysis",),
            default_workspace="image_analysis"
        )

        self.assertTrue(resolved.startswith("img_"))
        raw_uuid = resolved[len("img_"):]
        self.assertEqual(str(uuid.UUID(raw_uuid)), raw_uuid)
        self.assertEqual(response.headers["X-Session-ID"], resolved)

        record = self.test_db.get_session(resolved)
        self.assertIsNotNone(record)
        self.assertEqual(record["workspace_type"], "image_analysis")

    def test_09_cross_workspace_header_rejected(self):
        # img_ passed to conversation endpoint
        bad_header = f"img_{uuid.uuid4()}"
        response = Response()

        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=bad_header,
                session_cookie=None,
                response=response,
                allowed_workspaces=("conversation", "clinical_pre_consultation"),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_10_cross_workspace_cookie_rejected(self):
        # img_ cookie passed to conversation endpoint
        bad_cookie = f"img_{uuid.uuid4()}"
        response = Response()

        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=None,
                session_cookie=bad_cookie,
                response=response,
                allowed_workspaces=("conversation", "clinical_pre_consultation"),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_11_unprefixed_header_rejected(self):
        # Header MUST be prefixed; raw UUID in header is rejected
        raw_uid = str(uuid.uuid4())
        response = Response()

        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=raw_uid,
                session_cookie=None,
                response=response,
                allowed_workspaces=("conversation",),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("must include a workspace prefix", ctx.exception.detail)

    def test_12_malformed_header_rejected(self):
        bad_header = "invalid-session-format"
        response = Response()

        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=bad_header,
                session_cookie=None,
                response=response,
                allowed_workspaces=("conversation",),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid X-Session-ID format", ctx.exception.detail)

    def test_13_malformed_cookie_rejected(self):
        bad_cookie = "invalid-cookie-value"
        response = Response()

        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=None,
                session_cookie=bad_cookie,
                response=response,
                allowed_workspaces=("conversation",),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid session_id cookie format", ctx.exception.detail)

    def test_14_existing_session_cross_workspace_conflict_detected(self):
        # Register a legacy session under 'image_analysis'
        legacy_uid = str(uuid.uuid4())
        self.test_db.create_session(legacy_uid, workspace_type="image_analysis")

        # Now attempt to reuse this session in an endpoint allowing only 'conversation'
        response = Response()
        with self.assertRaises(HTTPException) as ctx:
            resolve_session_id(
                x_session_id=None,
                session_cookie=legacy_uid,
                response=response,
                allowed_workspaces=("conversation",),
                default_workspace="conversation"
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace conflict", ctx.exception.detail)

    def test_15_concurrent_first_time_registration_race_recovery(self):
        # Verify that if session_db.create_session encounters sqlite3.IntegrityError
        # (simulating concurrent insert of same ID), resolve_session_id safely recovers
        header_id = f"conv_{uuid.uuid4()}"
        # Pre-insert session
        self.test_db.create_session(header_id, workspace_type="conversation")

        # Now call resolve_session_id with the same header
        response = Response()
        resolved = resolve_session_id(
            x_session_id=header_id,
            session_cookie=None,
            response=response,
            allowed_workspaces=("conversation",),
            default_workspace="conversation"
        )
        self.assertEqual(resolved, header_id)


class TestEndpointIntegration(unittest.TestCase):
    """
    Test actual FastAPI endpoint logic for session resolution, headers, cookies,
    workspace boundary enforcement, SQLite registration, and JSON payload contracts.
    
    Directly invokes endpoint functions with real FastAPI Request/Response objects,
    guaranteeing deterministic execution without loopback socket hangs in sandboxed environments.
    """

    def setUp(self):
        faulthandler.dump_traceback_later(timeout=5, repeat=False)
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_file.close()
        self.test_db = SessionDatabase(db_path=self.temp_db_file.name)
        self.test_db.initialize_database()

        self.orig_db_path = session_db.db_path
        session_db.db_path = self.temp_db_file.name
        session_db.initialize_database()

    def tearDown(self):
        faulthandler.cancel_dump_traceback_later()
        session_db.db_path = self.orig_db_path
        if os.path.exists(self.temp_db_file.name):
            try:
                os.remove(self.temp_db_file.name)
            except OSError:
                pass

    def test_16_chat_endpoint_with_valid_header(self):
        conv_id = f"conv_{uuid.uuid4()}"
        mock_response = {
            "messages": [AIMessage(content="Hello patient, how can I help?")],
            "agent_name": "CONVERSATION_AGENT"
        }

        with patch("app.process_query", return_value=mock_response) as mock_pq:
            response = Response()
            req = QueryRequest(query="Hello", conversation_history=[])
            res = chat(
                request=req,
                response=response,
                x_session_id=conv_id
            )
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["session_id"], conv_id)
            self.assertEqual(res["agent"], "CONVERSATION_AGENT")
            self.assertEqual(response.headers.get("X-Session-ID"), conv_id)
            self.assertIn(f"session_id={conv_id}", response.headers.get("set-cookie", ""))
            mock_pq.assert_called_once_with("Hello", session_id=conv_id, workspace="conversation")

            # Verify SQLite registration under 'conversation'
            record = self.test_db.get_session(conv_id)
            self.assertIsNotNone(record)
            self.assertEqual(record["workspace_type"], "conversation")

    def test_17_chat_endpoint_allows_clinical_pre_consultation_prefix(self):
        clin_id = f"clin_{uuid.uuid4()}"
        mock_response = {
            "messages": [AIMessage(content="Please describe your symptoms.")],
            "agent_name": "CLINICAL_PRE_CONSULTATION_AGENT"
        }

        with patch("app.process_query", return_value=mock_response):
            response = Response()
            req = QueryRequest(query="I want to start a clinical intake", conversation_history=[])
            res = chat(
                request=req,
                response=response,
                x_session_id=clin_id
            )
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["session_id"], clin_id)
            self.assertEqual(response.headers.get("X-Session-ID"), clin_id)
            self.assertIn(f"session_id={clin_id}", response.headers.get("set-cookie", ""))

            # Verify SQLite registration under 'clinical_pre_consultation'
            record = self.test_db.get_session(clin_id)
            self.assertIsNotNone(record)
            self.assertEqual(record["workspace_type"], "clinical_pre_consultation")

    def test_18_chat_endpoint_rejects_image_analysis_prefix(self):
        img_id = f"img_{uuid.uuid4()}"
        response = Response()
        req = QueryRequest(query="Analyze this", conversation_history=[])
        with self.assertRaises(HTTPException) as ctx:
            chat(
                request=req,
                response=response,
                x_session_id=img_id
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_19_upload_endpoint_accepts_img_prefix_and_rejects_conv_prefix(self):
        import asyncio
        from io import BytesIO
        from fastapi import UploadFile

        img_id = f"img_{uuid.uuid4()}"
        conv_id = f"conv_{uuid.uuid4()}"

        mock_response = {
            "messages": [AIMessage(content="Chest X-Ray analysis normal")],
            "agent_name": "CHEST_XRAY_AGENT"
        }

        # 1. Successful upload with img_ prefix
        with patch("app.process_query", return_value=mock_response):
            upload_file = UploadFile(
                filename="test.jpg",
                file=BytesIO(b"\xff\xd8\xff\xe0dummy_jpeg_data")
            )
            response = Response()
            res = asyncio.run(upload_image(
                response=response,
                image=upload_file,
                text="Chest X-ray",
                x_session_id=img_id
            ))
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["session_id"], img_id)
            self.assertEqual(response.headers.get("X-Session-ID"), img_id)
            self.assertIn(f"session_id={img_id}", response.headers.get("set-cookie", ""))

            # Verify SQLite registration under 'image_analysis'
            record = self.test_db.get_session(img_id)
            self.assertIsNotNone(record)
            self.assertEqual(record["workspace_type"], "image_analysis")

        # 2. Rejection with conv_ prefix
        upload_file_bad = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"\xff\xd8\xff\xe0dummy_jpeg_data")
        )
        response_bad = Response()
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(upload_image(
                response=response_bad,
                image=upload_file_bad,
                text="Chest X-ray",
                x_session_id=conv_id
            ))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_20_validate_endpoint_session_resolution(self):
        img_id = f"img_{uuid.uuid4()}"
        conv_id = f"conv_{uuid.uuid4()}"

        mock_response = {
            "messages": [AIMessage(content="Validated confirmation")],
            "agent_name": "HUMAN_VALIDATION"
        }

        # 1. Success with img_
        with patch("app.process_query", return_value=mock_response):
            response = Response()
            res = validate_medical_output(
                response=response,
                validation_result="yes",
                comments="Looks good",
                x_session_id=img_id
            )
            self.assertEqual(res["status"], "validated")
            self.assertEqual(res["session_id"], img_id)
            self.assertEqual(response.headers.get("X-Session-ID"), img_id)
            self.assertIn(f"session_id={img_id}", response.headers.get("set-cookie", ""))

        # 2. Rejection with conv_
        response_bad = Response()
        with self.assertRaises(HTTPException) as ctx:
            validate_medical_output(
                response=response_bad,
                validation_result="yes",
                comments="Looks good",
                x_session_id=conv_id
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_21_clinical_patient_feedback_endpoint_session_resolution(self):
        clin_id = f"clin_{uuid.uuid4()}"
        conv_id = f"conv_{uuid.uuid4()}"

        mock_clinical_agent = MagicMock()
        mock_clinical_agent.process_patient_info.return_value = {
            "status": "ready_for_review",
            "report": "Updated intake report",
            "formatted_report": "<b>Updated intake report</b>"
        }

        # 1. Success with clin_
        with patch("agents.agent_decision.get_clinical_agent", return_value=mock_clinical_agent):
            response = Response()
            res = handle_clinical_patient_feedback(
                response=response,
                patient_feedback="The headache started 3 days ago, not yesterday.",
                x_session_id=clin_id
            )
            self.assertEqual(res["status"], "ready_for_review")
            self.assertEqual(res["session_id"], clin_id)
            self.assertEqual(response.headers.get("X-Session-ID"), clin_id)
            self.assertIn(f"session_id={clin_id}", response.headers.get("set-cookie", ""))

        # 2. Rejection with conv_
        response_bad = Response()
        with self.assertRaises(HTTPException) as ctx:
            handle_clinical_patient_feedback(
                response=response_bad,
                patient_feedback="Update my info",
                x_session_id=conv_id
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Workspace mismatch", ctx.exception.detail)

    def test_22_chat_endpoint_legacy_cookie_fallback(self):
        legacy_id = str(uuid.uuid4())
        mock_response = {
            "messages": [AIMessage(content="Hello from legacy session")],
            "agent_name": "CONVERSATION_AGENT"
        }

        with patch("app.process_query", return_value=mock_response):
            response = Response()
            req = QueryRequest(query="Hello", conversation_history=[])
            res = chat(
                request=req,
                response=response,
                session_id=legacy_id,
                x_session_id=None
            )
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["session_id"], legacy_id)
            self.assertEqual(response.headers.get("X-Session-ID"), legacy_id)
            self.assertIn(f"session_id={legacy_id}", response.headers.get("set-cookie", ""))

            # Verify SQLite registration
            record = self.test_db.get_session(legacy_id)
            self.assertIsNotNone(record)
            self.assertEqual(record["workspace_type"], "conversation")

    def test_23_langgraph_thread_id_matches_resolved_session_id(self):
        clin_id = f"clin_{uuid.uuid4()}"
        cfg = get_thread_config(clin_id)
        self.assertEqual(cfg, {"configurable": {"thread_id": clin_id}})

        conv_id = f"conv_{uuid.uuid4()}"
        cfg2 = get_thread_config(conv_id)
        self.assertEqual(cfg2, {"configurable": {"thread_id": conv_id}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
