"""
Comprehensive Test Suite for SQLite Storage Layer (Phase 6.1).
Tests all 16 required capabilities including database initialization,
foreign keys, cascading deletion, WAL concurrency, JSON handling, and reinitialization.
"""

import os
import sys
import json
import sqlite3
import tempfile
import unittest
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.session_db import SessionDatabase, SUPPORTED_WORKSPACES


class TestSessionDatabase(unittest.TestCase):
    """Test suite verifying all capabilities of SessionDatabase."""

    def setUp(self):
        """Create a fresh temporary directory and database for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_sessions.db")
        self.db = SessionDatabase(self.db_path)
        self.db.initialize_database()

    def tearDown(self):
        """Clean up temporary directory and database."""
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Test 1: Database initialization
    # -------------------------------------------------------------------------
    def test_01_database_initialization(self):
        """Verify database initialization creates the file, tables, and sets WAL mode."""
        self.assertTrue(os.path.exists(self.db_path), "Database file must exist")
        self.assertTrue(self.db.verify_database(), "verify_database() must return True")

        # Verify WAL mode
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0].lower()
            self.assertEqual(journal_mode, "wal", "Journal mode must be WAL")

            # Verify foreign keys are enabled
            cursor.execute("PRAGMA foreign_keys;")
            fk = cursor.fetchone()[0]
            self.assertEqual(fk, 1, "Foreign keys must be ON")

    # -------------------------------------------------------------------------
    # Test 2: Repeated initialization without data loss
    # -------------------------------------------------------------------------
    def test_02_repeated_initialization_without_data_loss(self):
        """Verify calling initialize_database multiple times does not corrupt or wipe data."""
        # Create a session
        sess = self.db.create_session(
            session_id="session_repeat_init",
            workspace_type="conversation",
            title="Initial Session"
        )
        self.assertIsNotNone(sess)

        # Re-run initialization 3 times
        self.assertTrue(self.db.initialize_database())
        self.assertTrue(self.db.initialize_database())
        self.assertTrue(self.db.initialize_database())

        # Verify the session still exists intact
        retrieved = self.db.get_session("session_repeat_init")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["title"], "Initial Session")

        # Verify schema version table has exactly 1 row for v1 and 1 for v2
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = 1;")
            count_v1 = cursor.fetchone()[0]
            self.assertEqual(count_v1, 1, "Migration v1 must only be recorded once")

            cursor.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = 2;")
            count_v2 = cursor.fetchone()[0]
            self.assertEqual(count_v2, 1, "Migration v2 must only be recorded once")

            cursor.execute("SELECT MAX(version) FROM schema_migrations;")
            max_ver = cursor.fetchone()[0]
            self.assertEqual(max_ver, 2, "Current schema version must be 2")

    # -------------------------------------------------------------------------
    # Test 3: Correct creation of all four tables and indexes
    # -------------------------------------------------------------------------
    def test_03_correct_creation_of_all_four_tables(self):
        """Verify sessions, session_messages, session_artifacts, clinical_intake_state exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "schema_migrations",
                "sessions",
                "session_messages",
                "session_artifacts",
                "clinical_intake_state"
            }
            self.assertTrue(expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}")

            # Verify indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = {row[0] for row in cursor.fetchall()}
            expected_indexes = {
                "idx_sessions_workspace",
                "idx_sessions_updated_at",
                "idx_sessions_workspace_archive_updated",
                "idx_messages_session_id",
                "idx_messages_session_order",
                "uq_session_messages_seq",
                "idx_artifacts_session_id"
            }
            self.assertTrue(expected_indexes.issubset(indexes), f"Missing indexes: {expected_indexes - indexes}")

    # -------------------------------------------------------------------------
    # Test 4: Session creation and retrieval
    # -------------------------------------------------------------------------
    def test_04_session_creation_and_retrieval(self):
        """Verify creating and retrieving sessions for all supported workspaces."""
        for ws in SUPPORTED_WORKSPACES:
            sid = f"sess_{ws}_1"
            created = self.db.create_session(
                session_id=sid,
                workspace_type=ws,
                title=f"Test {ws.title()}",
                metadata={"test_key": f"val_{ws}", "nested": {"count": 42}}
            )
            self.assertEqual(created["session_id"], sid)
            self.assertEqual(created["workspace_type"], ws)
            self.assertEqual(created["status"], "active")
            self.assertEqual(created["is_archived"], 0)
            self.assertEqual(created["metadata"]["nested"]["count"], 42)

            # Retrieve
            retrieved = self.db.get_session(sid)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["session_id"], sid)
            self.assertEqual(retrieved["workspace_type"], ws)
            self.assertEqual(retrieved["metadata"]["test_key"], f"val_{ws}")
            self.assertEqual(retrieved["metadata"]["nested"]["count"], 42)

    # -------------------------------------------------------------------------
    # Test 5: Workspace validation
    # -------------------------------------------------------------------------
    def test_05_workspace_validation(self):
        """Verify invalid workspace types are rejected with ValueError."""
        invalid_types = ["general", "admin", "unknown", "", None]
        for inv in invalid_types:
            with self.assertRaises(ValueError, msg=f"Workspace '{inv}' should be rejected"):
                self.db.create_session(workspace_type=inv)

        # Verify listing with invalid workspace raises ValueError
        with self.assertRaises(ValueError):
            self.db.list_sessions(workspace_type="invalid_type")

    # -------------------------------------------------------------------------
    # Test 6: Message insertion and deterministic ordering
    # -------------------------------------------------------------------------
    def test_06_message_insertion_and_ordering(self):
        """Verify messages are stored and retrieved with deterministic sequence numbering."""
        sid = "sess_ordering_test"
        self.db.create_session(session_id=sid, workspace_type="conversation")

        # Insert 5 messages
        m1 = self.db.save_message(sid, sender="user", content="Hello doctor")
        m2 = self.db.save_message(sid, sender="assistant", content="Hello, how can I help you?", agent_name="CONVERSATION_AGENT")
        m3 = self.db.save_message(sid, sender="user", content="I have a headache")
        m4 = self.db.save_message(sid, sender="assistant", content="How long has it lasted?", agent_name="CONVERSATION_AGENT")
        m5 = self.db.save_message(sid, sender="user", content="Since this morning")

        self.assertEqual(m1["sequence_num"], 1)
        self.assertEqual(m2["sequence_num"], 2)
        self.assertEqual(m3["sequence_num"], 3)
        self.assertEqual(m4["sequence_num"], 4)
        self.assertEqual(m5["sequence_num"], 5)

        # Retrieve and verify exact sequence order
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 5)
        for i, msg in enumerate(messages, start=1):
            self.assertEqual(msg["sequence_num"], i)

        self.assertEqual(messages[0]["content"], "Hello doctor")
        self.assertEqual(messages[4]["content"], "Since this morning")

    # -------------------------------------------------------------------------
    # Test 7: Clinical state persistence and restoration
    # -------------------------------------------------------------------------
    def test_07_clinical_state_persistence_and_restoration(self):
        """Verify clinical state (patient data, symptoms, stage, report) is fully preserved."""
        sid = "sess_clinical_test"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")

        patient_info = {
            "symptoms": {
                "list": [
                    {"term": "headache", "severity": "moderate", "duration": "3 days"},
                    {"term": "nausea", "severity": "mild", "duration": "1 day"}
                ],
                "onset": "sudden"
            },
            "allergies": {"status": "none"},
            "medications": {"list": ["ibuprofen 200mg"]},
            "medical_history": {"conditions": ["hypertension"]}
        }

        saved = self.db.save_clinical_state(
            session_id=sid,
            patient_data=patient_info,
            current_stage="symptom_deep_dive",
            latest_report_text="Patient presents with acute headache...",
            formatted_report="### SBAR REPORT\nPatient presents with acute headache...",
            is_confirmed=False
        )

        self.assertEqual(saved["session_id"], sid)
        self.assertEqual(saved["current_stage"], "symptom_deep_dive")
        self.assertEqual(saved["is_confirmed"], 0)

        # Retrieve and verify nested dictionary equality
        retrieved = self.db.get_clinical_state(sid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["current_stage"], "symptom_deep_dive")
        self.assertEqual(len(retrieved["patient_data"]["symptoms"]["list"]), 2)
        self.assertEqual(retrieved["patient_data"]["symptoms"]["list"][0]["term"], "headache")
        self.assertEqual(retrieved["patient_data"]["medications"]["list"], ["ibuprofen 200mg"])
        self.assertEqual(retrieved["latest_report_text"], "Patient presents with acute headache...")

        # Update confirmation without destroying existing data
        updated = self.db.save_clinical_state(
            session_id=sid,
            patient_data=patient_info,
            is_confirmed=True
        )
        self.assertEqual(updated["is_confirmed"], 1)
        self.assertEqual(updated["current_stage"], "symptom_deep_dive")  # Preserved

    # -------------------------------------------------------------------------
    # Test 8: Artifact metadata persistence
    # -------------------------------------------------------------------------
    def test_08_artifact_metadata_persistence(self):
        """Verify image artifact records and rejection of binary byte data."""
        sid = "sess_artifact_test"
        self.db.create_session(session_id=sid, workspace_type="image_analysis")

        # Save an image artifact
        art = self.db.save_artifact(
            session_id=sid,
            artifact_type="chest_xray",
            file_path="uploads/artifacts/sess_artifact_test/xray_01.png",
            original_filename="patient_chest.png",
            mime_type="image/png",
            metadata={"model": "covid_chest_xray_model.pth", "confidence": 0.98}
        )
        self.assertIsNotNone(art)
        self.assertEqual(art["artifact_type"], "chest_xray")
        self.assertEqual(art["metadata"]["confidence"], 0.98)

        # Retrieve
        artifacts = self.db.get_artifacts(sid)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["original_filename"], "patient_chest.png")

        # Rejection of raw binary bytes
        with self.assertRaises(TypeError):
            self.db.save_artifact(
                session_id=sid,
                artifact_type="raw_bytes",
                file_path=b"\x89PNG\r\n\x1a\n"  # Invalid binary payload
            )

    # -------------------------------------------------------------------------
    # Test 9: Session updates without unrelated field loss
    # -------------------------------------------------------------------------
    def test_09_session_updates_without_unrelated_field_loss(self):
        """Verify updating selected fields retains existing fields and merges metadata."""
        sid = "sess_update_test"
        initial = self.db.create_session(
            session_id=sid,
            workspace_type="conversation",
            title="Old Title",
            metadata={"user_pref": "dark", "lang": "en"}
        )

        # Update title only
        up1 = self.db.update_session(sid, title="New Title")
        self.assertEqual(up1["title"], "New Title")
        self.assertEqual(up1["metadata"]["user_pref"], "dark")
        self.assertEqual(up1["created_at"], initial["created_at"])

        # Update metadata partially (should merge)
        up2 = self.db.update_session(sid, metadata={"voice": "alloy"})
        self.assertEqual(up2["title"], "New Title")
        self.assertEqual(up2["metadata"]["user_pref"], "dark")
        self.assertEqual(up2["metadata"]["lang"], "en")
        self.assertEqual(up2["metadata"]["voice"], "alloy")

    # -------------------------------------------------------------------------
    # Test 10: Foreign-key enforcement
    # -------------------------------------------------------------------------
    def test_10_foreign_key_enforcement(self):
        """Verify message, artifact, or clinical state cannot be saved to non-existent session."""
        fake_sid = "non_existent_session_123"

        with self.assertRaises((ValueError, sqlite3.IntegrityError)):
            self.db.save_message(fake_sid, sender="user", content="Ghost message")

        with self.assertRaises((ValueError, sqlite3.IntegrityError)):
            self.db.save_artifact(fake_sid, artifact_type="scan", file_path="path/to/scan.png")

        with self.assertRaises((ValueError, sqlite3.IntegrityError)):
            self.db.save_clinical_state(fake_sid, patient_data={"test": True})

    # -------------------------------------------------------------------------
    # Test 11: Cascading deletion
    # -------------------------------------------------------------------------
    def test_11_cascading_deletion(self):
        """Verify deleting a session removes all related messages, artifacts, and clinical state."""
        sid = "sess_cascade_test"
        self.db.create_session(session_id=sid, workspace_type="clinical_pre_consultation")
        self.db.save_message(sid, sender="user", content="My head hurts")
        self.db.save_artifact(sid, artifact_type="notes", file_path="uploads/notes.pdf")
        self.db.save_clinical_state(sid, patient_data={"symptom": "headache"})

        # Verify child records exist
        self.assertEqual(len(self.db.get_messages(sid)), 1)
        self.assertEqual(len(self.db.get_artifacts(sid)), 1)
        self.assertIsNotNone(self.db.get_clinical_state(sid))

        # Delete session
        deleted = self.db.delete_session(sid)
        self.assertTrue(deleted)

        # Verify parent session is gone
        self.assertIsNone(self.db.get_session(sid))

        # Verify child records cascaded away
        self.assertEqual(len(self.db.get_messages(sid)), 0)
        self.assertEqual(len(self.db.get_artifacts(sid)), 0)
        self.assertIsNone(self.db.get_clinical_state(sid))

    # -------------------------------------------------------------------------
    # Test 12: Transaction rollback on failure
    # -------------------------------------------------------------------------
    def test_12_transaction_rollback_on_failure(self):
        """Verify transactions roll back cleanly when a multi-statement operation fails."""
        sid = "sess_tx_test"
        self.db.create_session(session_id=sid, workspace_type="conversation")

        # Simulate a transaction with intentional error
        try:
            with self.db.get_connection() as conn:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO session_messages (
                            message_id, session_id, sequence_num, sender, content, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        ("msg_tx_1", sid, 1, "user", "Valid message", "2026-01-01T00:00:00Z")
                    )
                    # Trigger integrity violation (duplicate primary key)
                    conn.execute(
                        """
                        INSERT INTO session_messages (
                            message_id, session_id, sequence_num, sender, content, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        ("msg_tx_1", sid, 2, "user", "Duplicate ID message", "2026-01-01T00:00:01Z")
                    )
        except sqlite3.IntegrityError:
            pass

        # Verify the entire transaction rolled back; msg_tx_1 should NOT exist
        messages = self.db.get_messages(sid)
        self.assertEqual(len(messages), 0, "Failed transaction must roll back completely")

    # -------------------------------------------------------------------------
    # Test 13: Multiple concurrent readers
    # -------------------------------------------------------------------------
    def test_13_multiple_concurrent_readers(self):
        """Verify multiple threads can read sessions concurrently without lock contention."""
        sid = "sess_concurrent_read"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="Concurrent Test")
        for i in range(10):
            self.db.save_message(sid, sender="user", content=f"Message {i}")

        results = []
        errors = []

        def reader_task(reader_id: int):
            try:
                session = self.db.get_session(sid)
                messages = self.db.get_messages(sid)
                return len(messages)
            except Exception as e:
                errors.append(f"Reader {reader_id} error: {e}")
                return 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(reader_task, i) for i in range(20)]
            for fut in as_completed(futures):
                results.append(fut.result())

        self.assertEqual(len(errors), 0, f"Concurrent reads had errors: {errors}")
        self.assertTrue(all(count == 10 for count in results), "All readers must read 10 messages")

    # -------------------------------------------------------------------------
    # Test 14: Concurrent write behavior under WAL
    # -------------------------------------------------------------------------
    def test_14_concurrent_write_behavior_under_wal(self):
        """Verify multiple threads can write sessions and messages concurrently in WAL mode."""
        num_threads = 10
        errors = []

        def writer_task(worker_id: int):
            try:
                ws_sid = f"sess_wal_worker_{worker_id}"
                self.db.create_session(session_id=ws_sid, workspace_type="conversation", title=f"Worker {worker_id}")
                for m in range(5):
                    self.db.save_message(ws_sid, sender="user", content=f"Worker {worker_id} - msg {m}")
                return True
            except Exception as e:
                errors.append(f"Worker {worker_id} failed: {e}")
                return False

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(writer_task, i) for i in range(num_threads)]
            for fut in as_completed(futures):
                fut.result()

        self.assertEqual(len(errors), 0, f"Concurrent writes had errors: {errors}")

        # Verify all sessions and messages were persisted
        for i in range(num_threads):
            sess = self.db.get_session(f"sess_wal_worker_{i}")
            self.assertIsNotNone(sess)
            msgs = self.db.get_messages(f"sess_wal_worker_{i}")
            self.assertEqual(len(msgs), 5)

    # -------------------------------------------------------------------------
    # Test 15: JSON serialization and invalid JSON handling
    # -------------------------------------------------------------------------
    def test_15_json_serialization_and_invalid_json_handling(self):
        """Verify safe handling of nested dictionaries, booleans, and corrupted JSON."""
        sid = "sess_json_test"
        complex_meta = {
            "unicode_test": "🩺 Doctor & 患者",
            "numbers": [1, 2.5, 300],
            "boolean": True,
            "null_val": None,
            "deep": {"a": {"b": {"c": "found"}}}
        }

        self.db.create_session(session_id=sid, workspace_type="conversation", metadata=complex_meta)
        retrieved = self.db.get_session(sid)
        self.assertEqual(retrieved["metadata"]["unicode_test"], "🩺 Doctor & 患者")
        self.assertEqual(retrieved["metadata"]["deep"]["a"]["b"]["c"], "found")

        # Manually corrupt JSON in database to test graceful parsing
        with self.db.get_connection() as conn:
            with conn:
                conn.execute(
                    "UPDATE sessions SET metadata_json = 'INVALID_JSON{corrupted' WHERE session_id = ?;",
                    (sid,)
                )

        corrupt_retrieved = self.db.get_session(sid)
        self.assertIsNotNone(corrupt_retrieved)
        # Should gracefully return raw wrapped value instead of raising exception
        self.assertIn("raw", corrupt_retrieved["metadata"])

    # -------------------------------------------------------------------------
    # Test 16: Persistence across connection/process-style reinitialization
    # -------------------------------------------------------------------------
    def test_16_persistence_across_reopening(self):
        """Verify database state persists when a brand new SessionDatabase instance is opened."""
        sid = "sess_reopen_test"
        self.db.create_session(session_id=sid, workspace_type="image_analysis", title="Reopen Test")
        self.db.save_message(sid, sender="user", content="Check this MRI")
        self.db.save_artifact(sid, artifact_type="brain_mri", file_path="uploads/mri.png")

        # Close/discard the original db instance
        self.db.close()
        del self.db

        # Open brand new SessionDatabase instance pointing to the exact same file
        reopened_db = SessionDatabase(self.db_path)
        self.assertTrue(reopened_db.verify_database())

        sess = reopened_db.get_session(sid)
        self.assertIsNotNone(sess)
        self.assertEqual(sess["title"], "Reopen Test")

        msgs = reopened_db.get_messages(sid)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "Check this MRI")

        artifacts = reopened_db.get_artifacts(sid)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["file_path"], "uploads/mri.png")

    # -------------------------------------------------------------------------
    # Test 17: FIX A - 20 simultaneous writes to the same session
    # -------------------------------------------------------------------------
    def test_17_concurrent_writes_same_session_20_unique_sequence_numbers(self):
        """Verify 20 concurrent threads writing to the SAME session all receive unique sequence numbers 1..20."""
        sid = "sess_concurrent_same_session_20"
        self.db.create_session(session_id=sid, workspace_type="conversation", title="20 Writers Same Session")

        errors = []
        results = []

        def writer_task(msg_idx: int):
            try:
                msg = self.db.save_message(
                    session_id=sid,
                    sender="user" if msg_idx % 2 == 0 else "assistant",
                    content=f"Concurrent message payload {msg_idx}"
                )
                return msg
            except Exception as e:
                errors.append(f"Writer {msg_idx} failed: {e}")
                return None

        # Execute 20 concurrent writes simultaneously
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(writer_task, i) for i in range(20)]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)

        self.assertEqual(len(errors), 0, f"Concurrent writes had errors: {errors}")
        self.assertEqual(len(results), 20, "All 20 writes must succeed")

        # Verify all sequence numbers are strictly unique and span 1..20
        seq_nums = [r["sequence_num"] for r in results]
        self.assertEqual(len(set(seq_nums)), 20, f"Duplicate sequence numbers found: {seq_nums}")
        self.assertEqual(sorted(seq_nums), list(range(1, 21)), "Sequence numbers must be 1 through 20")

        # Verify deterministic retrieval order
        retrieved = self.db.get_messages(sid)
        self.assertEqual(len(retrieved), 20)
        retrieved_seqs = [m["sequence_num"] for m in retrieved]
        self.assertEqual(retrieved_seqs, list(range(1, 21)), "Retrieved messages must be ordered 1..20")

    # -------------------------------------------------------------------------
    # Test 18: FIX A - UNIQUE(session_id, sequence_num) enforced by SQLite
    # -------------------------------------------------------------------------
    def test_18_unique_sequence_constraint_enforced_by_sqlite(self):
        """Verify that attempting to insert a duplicate sequence_num fails with IntegrityError."""
        sid = "sess_unique_constraint_test"
        self.db.create_session(session_id=sid, workspace_type="conversation")
        self.db.save_message(sid, sender="user", content="First message")

        with self.db.get_connection() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                with conn:
                    # Attempt raw SQL insert with duplicate sequence_num = 1
                    conn.execute(
                        """
                        INSERT INTO session_messages (
                            message_id, session_id, sequence_num, sender, content, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        ("msg_dup_test", sid, 1, "user", "Duplicate sequence message", "2026-01-01T00:00:00Z")
                    )

    # -------------------------------------------------------------------------
    # Test 19: FIX A & B - Migration v2 applies cleanly and preserves records
    # -------------------------------------------------------------------------
    def test_19_migration_v2_clean_and_preserves_records(self):
        """Verify migration v2 applies to existing v1 database and preserves all existing records."""
        with tempfile.TemporaryDirectory() as mig_dir:
            v1_db_path = os.path.join(mig_dir, "v1_test.db")
            conn = sqlite3.connect(v1_db_path)
            conn.row_factory = sqlite3.Row

            # Apply only v1 schema
            v1_db = SessionDatabase(v1_db_path)
            v1_db._apply_migration_v1(conn)
            conn.execute(
                "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT, description TEXT);"
            )
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, description) VALUES (1, '2026-01-01', 'V1');"
            )
            # Insert existing session and message
            conn.execute(
                "INSERT INTO sessions (session_id, workspace_type, title, created_at, updated_at, status, is_archived) VALUES ('s_mig', 'conversation', 'Preserve Me', '2026-01-01', '2026-01-01', 'active', 0);"
            )
            conn.execute(
                "INSERT INTO session_messages (message_id, session_id, sequence_num, sender, content, created_at) VALUES ('m_mig_1', 's_mig', 1, 'user', 'Existing Message', '2026-01-01');"
            )
            conn.commit()
            conn.close()

            # Now initialize with SessionDatabase which should run migration v2
            upgraded_db = SessionDatabase(v1_db_path)
            upgraded_db.initialize_database()

            # Verify schema version is now 2
            with upgraded_db.get_connection() as c:
                cur = c.cursor()
                cur.execute("SELECT MAX(version) FROM schema_migrations;")
                self.assertEqual(cur.fetchone()[0], 2)

                # Verify indexes exist
                cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
                idx_names = {r[0] for r in cur.fetchall()}
                self.assertIn("uq_session_messages_seq", idx_names)
                self.assertIn("idx_sessions_workspace_archive_updated", idx_names)

            # Verify existing records were preserved
            sess = upgraded_db.get_session("s_mig")
            self.assertIsNotNone(sess)
            self.assertEqual(sess["title"], "Preserve Me")

            msgs = upgraded_db.get_messages("s_mig")
            self.assertEqual(len(msgs), 1)
            self.assertEqual(msgs[0]["content"], "Existing Message")
            self.assertEqual(msgs[0]["sequence_num"], 1)

    # -------------------------------------------------------------------------
    # Test 20: FIX A - Migration v2 safely rejects pre-existing duplicates
    # -------------------------------------------------------------------------
    def test_20_migration_v2_safely_rejects_pre_existing_duplicates(self):
        """Verify migration v2 aborts safely when pre-existing duplicate sequence numbers exist without modifying data."""
        with tempfile.TemporaryDirectory() as mig_dir:
            dup_db_path = os.path.join(mig_dir, "dup_v1.db")
            conn = sqlite3.connect(dup_db_path)
            conn.row_factory = sqlite3.Row

            v1_db = SessionDatabase(dup_db_path)
            v1_db._apply_migration_v1(conn)
            conn.execute(
                "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT, description TEXT);"
            )
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, description) VALUES (1, '2026-01-01', 'V1');"
            )
            conn.execute(
                "INSERT INTO sessions (session_id, workspace_type, title, created_at, updated_at, status, is_archived) VALUES ('s_dup', 'conversation', 'Dup Test', '2026-01-01', '2026-01-01', 'active', 0);"
            )
            # Force insert DUPLICATE sequence numbers (seq=1 for two different messages)
            conn.execute(
                "INSERT INTO session_messages (message_id, session_id, sequence_num, sender, content, created_at) VALUES ('m_1', 's_dup', 1, 'user', 'Msg 1', '2026-01-01');"
            )
            conn.execute(
                "INSERT INTO session_messages (message_id, session_id, sequence_num, sender, content, created_at) VALUES ('m_2', 's_dup', 1, 'user', 'Msg 2 duplicate seq', '2026-01-01');"
            )
            conn.commit()
            conn.close()

            # Attempting to initialize should abort on migration v2
            db_with_dups = SessionDatabase(dup_db_path)
            with self.assertRaises(RuntimeError) as ctx:
                db_with_dups.initialize_database()

            err_msg = str(ctx.exception)
            self.assertIn("duplicate (session_id, sequence_num)", err_msg)
            self.assertIn("s_dup", err_msg)

            # Verify that database records remain completely untouched and schema version remains 1
            raw_conn = sqlite3.connect(dup_db_path)
            raw_cur = raw_conn.cursor()
            raw_cur.execute("SELECT MAX(version) FROM schema_migrations;")
            self.assertEqual(raw_cur.fetchone()[0], 1, "Schema version must remain 1 when migration aborts")

            raw_cur.execute("SELECT COUNT(*) FROM session_messages WHERE session_id = 's_dup';")
            self.assertEqual(raw_cur.fetchone()[0], 2, "Existing messages must remain completely intact")
            raw_conn.close()

    # -------------------------------------------------------------------------
    # Test 21: FIX B - Workspace history index optimization & query plan
    # -------------------------------------------------------------------------
    def test_21_workspace_history_index_and_query_plan(self):
        """Verify composite index idx_sessions_workspace_archive_updated eliminates USE TEMP B-TREE FOR ORDER BY."""
        with self.db.get_connection() as conn:
            # 1. Verify index columns via PRAGMA index_info
            cursor = conn.cursor()
            cursor.execute("PRAGMA index_info(idx_sessions_workspace_archive_updated);")
            cols = [r["name"] for r in cursor.fetchall()]
            self.assertEqual(cols, ["workspace_type", "is_archived", "updated_at"], "Index columns must match query pattern")

            # 2. Verify EXPLAIN QUERY PLAN on list_sessions query
            query = """
                EXPLAIN QUERY PLAN
                SELECT session_id, workspace_type, title, created_at, updated_at, last_activity, status, is_archived, metadata_json
                FROM sessions
                WHERE workspace_type = ? AND is_archived = 0
                ORDER BY updated_at DESC
                LIMIT 50 OFFSET 0;
            """
            cursor.execute(query, ("conversation",))
            plan_rows = cursor.fetchall()
            plan_details = [r["detail"] for r in plan_rows]

            # Verify the composite index is used
            index_used = any("idx_sessions_workspace_archive_updated" in detail for detail in plan_details)
            self.assertTrue(index_used, f"Expected index not used in plan: {plan_details}")

            # Verify USE TEMP B-TREE is NOT used
            temp_btree_used = any("USE TEMP B-TREE" in detail for detail in plan_details)
            self.assertFalse(temp_btree_used, f"Query plan should not require TEMP B-TREE sorting: {plan_details}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
