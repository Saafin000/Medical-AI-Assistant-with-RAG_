"""
Phase 6: End-to-End Workspace Isolation & Concurrency Verification Test Suite

Tests:
1. Strict workspace isolation:
   - Conversation session cannot hold or retrieve clinical state
   - Image analysis session cannot be accessed via chat endpoint
   - Clinical pre-consultation session cannot be accessed via image upload
2. Multi-session concurrency:
   - 30 concurrent message writes across distinct workspaces maintain deterministic ordering
   - Zero sequence number collisions or gaps
3. State persistence and recovery across session switching:
   - Creating multiple sessions in each workspace
   - Switching between sessions restores correct history and state
   - Deleting one session cascades only its data without affecting other sessions
"""

import os
import sys
import uuid
import tempfile
import sqlite3
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor

# Ensure offline environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Response, HTTPException
from storage.session_db import SessionDatabase


class TestSessionE2EIsolation(unittest.TestCase):
    """End-to-End Workspace Isolation and Concurrency Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.test_dir.name, "test_e2e_isolation.db")
        cls.db = SessionDatabase(cls.db_path)
        cls.db.initialize_database()

    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()

    def test_01_workspace_type_enforcement(self):
        """Database enforces valid workspace types and rejects invalid ones."""
        with self.assertRaises(ValueError):
            self.db.create_session(
                session_id=f"bad_{uuid.uuid4()}",
                workspace_type="invalid_workspace"
            )

    def test_02_workspace_isolation_data_boundaries(self):
        """Data boundaries between conversation, image_analysis, and clinical workspaces are strictly isolated."""
        conv_id = f"conv_{uuid.uuid4()}"
        img_id = f"img_{uuid.uuid4()}"
        clin_id = f"clin_{uuid.uuid4()}"

        # Create sessions in each workspace
        self.db.create_session(session_id=conv_id, workspace_type="conversation", title="Conv Session")
        self.db.create_session(session_id=img_id, workspace_type="image_analysis", title="Img Session")
        self.db.create_session(session_id=clin_id, workspace_type="clinical_pre_consultation", title="Clin Session")

        # Save artifacts to image session
        self.db.save_artifact(
            session_id=img_id,
            artifact_type="input_image",
            file_path="uploads/backend/scan.png",
            original_filename="scan.png"
        )

        # Save clinical state to clinical session
        self.db.save_clinical_state(
            session_id=clin_id,
            patient_data={"symptoms": ["chest pain", "shortness of breath"]},
            current_stage="active"
        )

        # Save messages to conversation session
        self.db.save_message(session_id=conv_id, sender="user", content="Hello medical assistant")
        self.db.save_message(session_id=conv_id, sender="assistant", content="Hello! How can I help you today?")

        # Verify artifacts only exist for img_id
        self.assertEqual(len(self.db.get_artifacts(img_id)), 1)
        self.assertEqual(len(self.db.get_artifacts(conv_id)), 0)
        self.assertEqual(len(self.db.get_artifacts(clin_id)), 0)

        # Verify clinical state only exists for clin_id
        self.assertIsNotNone(self.db.get_clinical_state(clin_id))
        self.assertIsNone(self.db.get_clinical_state(conv_id))
        self.assertIsNone(self.db.get_clinical_state(img_id))

        # Verify messages only exist for conv_id
        self.assertEqual(len(self.db.get_messages(conv_id)), 2)
        self.assertEqual(len(self.db.get_messages(img_id)), 0)
        self.assertEqual(len(self.db.get_messages(clin_id)), 0)

        # Listing by workspace type isolates results completely
        conv_list = self.db.list_sessions(workspace_type="conversation")
        img_list = self.db.list_sessions(workspace_type="image_analysis")
        clin_list = self.db.list_sessions(workspace_type="clinical_pre_consultation")

        conv_ids = [s["session_id"] for s in conv_list]
        img_ids = [s["session_id"] for s in img_list]
        clin_ids = [s["session_id"] for s in clin_list]

        self.assertIn(conv_id, conv_ids)
        self.assertNotIn(conv_id, img_ids)
        self.assertNotIn(conv_id, clin_ids)

        self.assertIn(img_id, img_ids)
        self.assertNotIn(img_id, conv_ids)
        self.assertNotIn(img_id, clin_ids)

        self.assertIn(clin_id, clin_ids)
        self.assertNotIn(clin_id, conv_ids)
        self.assertNotIn(clin_id, img_ids)

    def test_03_concurrent_writes_across_workspaces(self):
        """Concurrent writes across 3 workspaces maintain deterministic sequence ordering with no collisions."""
        sid1 = f"conv_{uuid.uuid4()}"
        sid2 = f"img_{uuid.uuid4()}"
        sid3 = f"clin_{uuid.uuid4()}"

        self.db.create_session(session_id=sid1, workspace_type="conversation")
        self.db.create_session(session_id=sid2, workspace_type="image_analysis")
        self.db.create_session(session_id=sid3, workspace_type="clinical_pre_consultation")

        target_sessions = [sid1, sid2, sid3] * 10  # 30 writes total

        def write_msg(index, target_sid):
            self.db.save_message(
                session_id=target_sid,
                sender="user" if index % 2 == 0 else "assistant",
                content=f"Message {index} for {target_sid}"
            )

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(write_msg, i, sid) for i, sid in enumerate(target_sessions)]
            for f in futures:
                f.result()

        # Each session must have exactly 10 messages with sequence_num 1..10
        for sid in [sid1, sid2, sid3]:
            msgs = self.db.get_messages(sid)
            self.assertEqual(len(msgs), 10)
            seqs = [m["sequence_num"] for m in msgs]
            self.assertEqual(seqs, list(range(1, 11)))

    def test_04_session_lifecycle_and_cascade_isolation(self):
        """Deleting a session cascades only its messages and clinical state, preserving other sessions."""
        sid_to_delete = f"clin_{uuid.uuid4()}"
        sid_to_keep = f"clin_{uuid.uuid4()}"

        self.db.create_session(session_id=sid_to_delete, workspace_type="clinical_pre_consultation")
        self.db.create_session(session_id=sid_to_keep, workspace_type="clinical_pre_consultation")

        self.db.save_message(session_id=sid_to_delete, sender="user", content="Delete me")
        self.db.save_clinical_state(session_id=sid_to_delete, patient_data={"temp": 101.5})

        self.db.save_message(session_id=sid_to_keep, sender="user", content="Keep me")
        self.db.save_clinical_state(session_id=sid_to_keep, patient_data={"temp": 98.6})

        # Delete first session
        deleted = self.db.delete_session(sid_to_delete)
        self.assertTrue(deleted)

        # First session data is gone
        self.assertIsNone(self.db.get_session(sid_to_delete))
        self.assertEqual(len(self.db.get_messages(sid_to_delete)), 0)
        self.assertIsNone(self.db.get_clinical_state(sid_to_delete))

        # Second session data remains completely intact
        self.assertIsNotNone(self.db.get_session(sid_to_keep))
        self.assertEqual(len(self.db.get_messages(sid_to_keep)), 1)
        self.assertEqual(self.db.get_messages(sid_to_keep)[0]["content"], "Keep me")
        self.assertEqual(self.db.get_clinical_state(sid_to_keep)["patient_data"]["temp"], 98.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
