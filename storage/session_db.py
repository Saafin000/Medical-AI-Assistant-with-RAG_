"""
Session Database Storage Layer for Multi-Agent Medical AI Assistant.
Provides SQLite-based transactional persistence for workspaces, sessions,
messages, clinical intake state, and image artifacts.

Supports:
- SQLite WAL mode for high-concurrency read/write operations.
- PRAGMA foreign_keys = ON for cascading integrity.
- Thread-safe connection-per-operation semantics.
- Idempotent schema migrations.
- Strict workspace type validation.
- UTC ISO 8601 timestamps throughout.
"""

import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

# Supported workspace types for feature isolation
SUPPORTED_WORKSPACES = ("conversation", "image_analysis", "clinical_pre_consultation")

# Default database location under the project's data directory
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.db"
)

# Current schema version
CURRENT_SCHEMA_VERSION = 2


def _utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dumps(data: Any) -> Optional[str]:
    """Serialize data to JSON string safely; return None if data is empty or None."""
    if data is None:
        return None
    try:
        return json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize metadata to JSON: {e}")
        return json.dumps(str(data))


def _safe_json_loads(data: Optional[str]) -> Optional[Dict[str, Any]]:
    """Deserialize JSON string safely; return empty dict or None on failure."""
    if not data:
        return None
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Corrupt JSON in storage, returning raw wrapped value.")
        return {"raw": data}


class SessionDatabase:
    """
    SQLite persistence manager for multi-agent medical assistant sessions.
    
    Provides thread-safe, transactional operations with WAL mode, foreign keys,
    and schema versioning.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. Defaults to DEFAULT_DB_PATH.
        """
        self.db_path = os.path.abspath(db_path or os.environ.get("SESSIONS_DB_PATH", DEFAULT_DB_PATH))
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Create parent directory for database if it does not exist."""
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for obtaining a thread-safe SQLite connection.
        Configures row_factory, foreign keys, and busy timeout on every connection.
        """
        self._ensure_parent_dir()
        conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            # Enforce foreign keys and configure busy timeout
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # Database Lifecycle & Migrations
    # =========================================================================

    def initialize_database(self) -> bool:
        """
        Idempotently initialize SQLite database, apply WAL mode, and execute pending migrations.
        Safe to call multiple times without data loss.
        """
        self._ensure_parent_dir()
        with self.get_connection() as conn:
            # Enable WAL mode for high concurrency
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")

            # Ensure schema_migrations table exists
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL,
                        description TEXT
                    );
                    """
                )

                # Check current schema version
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(version) FROM schema_migrations;")
                row = cursor.fetchone()
                current_ver = row[0] if (row and row[0] is not None) else 0

                # Apply migration v1 if not applied
                if current_ver < 1:
                    self._apply_migration_v1(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?);",
                        (1, _utc_now_iso(), "V1: Initial schema for sessions, messages, artifacts, clinical state")
                    )
                    logger.info("Applied schema migration v1 successfully.")

                # Apply migration v2 if not applied
                if current_ver < 2:
                    self._apply_migration_v2(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?);",
                        (2, _utc_now_iso(), "V2: Unique sequence index and workspace history query composite index")
                    )
                    logger.info("Applied schema migration v2 successfully.")

        return True

    def _apply_migration_v1(self, conn: sqlite3.Connection) -> None:
        """Apply Version 1 schema creating core tables and indexes."""
        # 1. sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                workspace_type TEXT NOT NULL CHECK(workspace_type IN ('conversation', 'image_analysis', 'clinical_pre_consultation')),
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_activity TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                is_archived INTEGER NOT NULL DEFAULT 0 CHECK(is_archived IN (0, 1)),
                metadata_json TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);")

        # 2. session_messages table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_name TEXT,
                is_report INTEGER NOT NULL DEFAULT 0 CHECK(is_report IN (0, 1)),
                is_emergency INTEGER NOT NULL DEFAULT 0 CHECK(is_emergency IN (0, 1)),
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON session_messages(session_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_order ON session_messages(session_id, sequence_num ASC, created_at ASC);")

        # 3. session_artifacts table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_artifacts (
                artifact_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_filename TEXT,
                mime_type TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON session_artifacts(session_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_type ON session_artifacts(artifact_type);")

        # 4. clinical_intake_state table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinical_intake_state (
                session_id TEXT PRIMARY KEY,
                current_stage TEXT,
                patient_data_json TEXT NOT NULL,
                latest_report_text TEXT,
                formatted_report TEXT,
                is_confirmed INTEGER NOT NULL DEFAULT 0 CHECK(is_confirmed IN (0, 1)),
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """
        )

    def _apply_migration_v2(self, conn: sqlite3.Connection) -> None:
        """
        Apply Version 2 migration:
        1. Inspect existing duplicate (session_id, sequence_num) groups.
           If duplicates exist, abort safely without modifying or renumbering records.
        2. Apply UNIQUE index uq_session_messages_seq on session_messages(session_id, sequence_num).
        3. Apply composite index idx_sessions_workspace_archive_updated on sessions(workspace_type, is_archived, updated_at DESC).
        """
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_id, sequence_num, COUNT(*) as count
            FROM session_messages
            GROUP BY session_id, sequence_num
            HAVING COUNT(*) > 1;
            """
        )
        duplicates = cursor.fetchall()
        if duplicates:
            dup_details = [
                f"session_id='{r['session_id']}', sequence_num={r['sequence_num']} (count={r['count']})"
                for r in duplicates
            ]
            raise RuntimeError(
                f"Cannot apply migration v2: Found {len(duplicates)} duplicate (session_id, sequence_num) group(s) "
                f"in session_messages: {'; '.join(dup_details)}. "
                "Existing data preserved without modification. Resolution required before unique constraint can be applied."
            )

        # Apply unique index on session_messages(session_id, sequence_num)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_session_messages_seq
            ON session_messages(session_id, sequence_num);
            """
        )

        # Apply composite index matching list_sessions query:
        # WHERE workspace_type = ? AND is_archived = 0 ORDER BY updated_at DESC
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_workspace_archive_updated
            ON sessions(workspace_type, is_archived, updated_at DESC);
            """
        )

    def verify_database(self) -> bool:
        """
        Verify database integrity and table accessibility.
        Returns True if database is healthy.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                result = cursor.fetchone()
                if not result or result[0] != "ok":
                    return False

                # Verify all expected tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = {row[0] for row in cursor.fetchall()}
                required = {"schema_migrations", "sessions", "session_messages", "session_artifacts", "clinical_intake_state"}
                return required.issubset(tables)
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False

    def close(self) -> None:
        """Resource cleanup hook."""
        pass

    # =========================================================================
    # Session Management API
    # =========================================================================

    def create_session(
        self,
        session_id: Optional[str] = None,
        workspace_type: str = "conversation",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "active"
    ) -> Dict[str, Any]:
        """
        Create a new isolated workspace session.
        
        Args:
            session_id: Optional explicit session ID; if omitted, generates UUID.
            workspace_type: One of ('conversation', 'image_analysis', 'clinical_pre_consultation').
            title: Human-readable session title.
            metadata: Optional dictionary of arbitrary metadata.
            status: Initial session status (default 'active').
            
        Returns:
            Dict representing the created session.
        """
        if workspace_type not in SUPPORTED_WORKSPACES:
            raise ValueError(
                f"Invalid workspace_type '{workspace_type}'. Must be one of: {SUPPORTED_WORKSPACES}"
            )

        sid = session_id or str(uuid.uuid4())
        now = _utc_now_iso()
        default_title = title or f"New {workspace_type.replace('_', ' ').title()} Session"
        meta_json = _safe_json_dumps(metadata)

        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, workspace_type, title, created_at, updated_at,
                        last_activity, status, is_archived, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?);
                    """,
                    (sid, workspace_type, default_title, now, now, now, status, meta_json)
                )

        return {
            "session_id": sid,
            "workspace_type": workspace_type,
            "title": default_title,
            "created_at": now,
            "updated_at": now,
            "last_activity": now,
            "status": status,
            "is_archived": 0,
            "metadata": metadata or {}
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by its session_id."""
        if not session_id:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, workspace_type, title, created_at, updated_at,
                       last_activity, status, is_archived, metadata_json
                FROM sessions
                WHERE session_id = ?;
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "session_id": row["session_id"],
                "workspace_type": row["workspace_type"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_activity": row["last_activity"],
                "status": row["status"],
                "is_archived": row["is_archived"],
                "metadata": _safe_json_loads(row["metadata_json"]) or {}
            }

    def list_sessions(
        self,
        workspace_type: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List sessions filtered by workspace type and archival status.
        Ordered by updated_at DESC.
        """
        if workspace_type and workspace_type not in SUPPORTED_WORKSPACES:
            raise ValueError(
                f"Invalid workspace_type '{workspace_type}'. Must be one of: {SUPPORTED_WORKSPACES}"
            )

        query = "SELECT session_id, workspace_type, title, created_at, updated_at, last_activity, status, is_archived, metadata_json FROM sessions WHERE 1=1"
        params: List[Any] = []

        if workspace_type:
            query += " AND workspace_type = ?"
            params.append(workspace_type)

        if not include_archived:
            query += " AND is_archived = 0"

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?;"
        params.extend([max(1, limit), max(0, offset)])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "session_id": r["session_id"],
                    "workspace_type": r["workspace_type"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "last_activity": r["last_activity"],
                    "status": r["status"],
                    "is_archived": r["is_archived"],
                    "metadata": _safe_json_loads(r["metadata_json"]) or {}
                }
                for r in rows
            ]

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
        is_archived: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        last_activity: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update selected fields of a session without losing other fields.
        Automatically updates updated_at.
        """
        existing = self.get_session(session_id)
        if not existing:
            return None

        now = _utc_now_iso()
        new_title = title if title is not None else existing["title"]
        new_status = status if status is not None else existing["status"]
        new_archived = int(is_archived) if is_archived is not None else existing["is_archived"]
        new_activity = last_activity or now

        # Merge metadata rather than silently overwriting unrelated keys
        if metadata is not None:
            merged_meta = dict(existing.get("metadata") or {})
            merged_meta.update(metadata)
        else:
            merged_meta = existing.get("metadata") or {}

        meta_json = _safe_json_dumps(merged_meta)

        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET title = ?, status = ?, is_archived = ?, metadata_json = ?,
                        updated_at = ?, last_activity = ?
                    WHERE session_id = ?;
                    """,
                    (new_title, new_status, new_archived, meta_json, now, new_activity, session_id)
                )

        return {
            "session_id": session_id,
            "workspace_type": existing["workspace_type"],
            "title": new_title,
            "created_at": existing["created_at"],
            "updated_at": now,
            "last_activity": new_activity,
            "status": new_status,
            "is_archived": new_archived,
            "metadata": merged_meta
        }

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its cascading messages, artifacts, and clinical state.
        Returns True if deleted, False if session did not exist.
        """
        if not session_id:
            return False

        with self.get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
                return cursor.rowcount > 0

    # =========================================================================
    # Message Management API
    # =========================================================================

    def save_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        message_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        is_report: bool = False,
        is_emergency: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a conversational or clinical message in a session.
        Guarantees deterministic message sequence ordering.
        Updates parent session's updated_at and last_activity timestamps.
        """
        if not session_id or not sender or content is None:
            raise ValueError("session_id, sender, and content are required to save a message.")

        mid = message_id or f"msg_{uuid.uuid4()}"
        now = created_at or _utc_now_iso()
        meta_json = _safe_json_dumps(metadata)
        int_report = 1 if is_report else 0
        int_emergency = 1 if is_emergency else 0

        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                # Validate session exists to enforce foreign key immediately
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?;", (session_id,))
                if not cursor.fetchone():
                    raise ValueError(f"Cannot save message: Session '{session_id}' does not exist.")

                # Calculate next deterministic sequence number for this session
                cursor.execute(
                    "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM session_messages WHERE session_id = ?;",
                    (session_id,)
                )
                seq_num = cursor.fetchone()[0]

                # Insert message
                conn.execute(
                    """
                    INSERT INTO session_messages (
                        message_id, session_id, sequence_num, sender, content,
                        agent_name, is_report, is_emergency, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (mid, session_id, seq_num, sender, content, agent_name, int_report, int_emergency, now, meta_json)
                )

                # Touch parent session timestamps
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, last_activity = ? WHERE session_id = ?;",
                    (now, now, session_id)
                )

                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

        return {
            "message_id": mid,
            "session_id": session_id,
            "sequence_num": seq_num,
            "sender": sender,
            "content": content,
            "agent_name": agent_name,
            "is_report": int_report,
            "is_emergency": int_emergency,
            "created_at": now,
            "metadata": metadata or {}
        }

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages for a session in deterministic sequence order.
        """
        if not session_id:
            return []

        query = """
            SELECT message_id, session_id, sequence_num, sender, content,
                   agent_name, is_report, is_emergency, created_at, metadata_json
            FROM session_messages
            WHERE session_id = ?
            ORDER BY sequence_num ASC, created_at ASC
        """
        params: List[Any] = [session_id]

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([max(1, limit), max(0, offset)])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "message_id": r["message_id"],
                    "session_id": r["session_id"],
                    "sequence_num": r["sequence_num"],
                    "sender": r["sender"],
                    "content": r["content"],
                    "agent_name": r["agent_name"],
                    "is_report": r["is_report"],
                    "is_emergency": r["is_emergency"],
                    "created_at": r["created_at"],
                    "metadata": _safe_json_loads(r["metadata_json"]) or {}
                }
                for r in rows
            ]

    def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session; returns count of deleted messages."""
        if not session_id:
            return 0

        with self.get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_messages WHERE session_id = ?;", (session_id,))
                return cursor.rowcount

    # =========================================================================
    # Clinical Intake State API
    # =========================================================================

    def save_clinical_state(
        self,
        session_id: str,
        patient_data: Union[Dict[str, Any], str],
        current_stage: Optional[str] = None,
        latest_report_text: Optional[str] = None,
        formatted_report: Optional[str] = None,
        is_confirmed: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Upsert clinical intake state for a clinical pre-consultation session.
        Preserves existing fields when only partial updates are supplied.
        """
        if not session_id:
            raise ValueError("session_id is required to save clinical state.")

        # Serialize patient data
        if isinstance(patient_data, str):
            # Validate JSON string
            try:
                parsed_data = json.loads(patient_data)
                patient_data_json = patient_data
            except Exception:
                parsed_data = {"raw": patient_data}
                patient_data_json = json.dumps(parsed_data)
        elif isinstance(patient_data, dict):
            parsed_data = patient_data
            patient_data_json = json.dumps(patient_data, ensure_ascii=False)
        else:
            raise ValueError("patient_data must be a dict or valid JSON string.")

        now = _utc_now_iso()

        with self.get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?;", (session_id,))
                if not cursor.fetchone():
                    raise ValueError(f"Cannot save clinical state: Session '{session_id}' does not exist.")

                # Fetch existing record if present to avoid overwriting unsupplied fields
                cursor.execute(
                    """
                    SELECT current_stage, latest_report_text, formatted_report, is_confirmed
                    FROM clinical_intake_state
                    WHERE session_id = ?;
                    """,
                    (session_id,)
                )
                existing = cursor.fetchone()

                new_stage = current_stage if current_stage is not None else (existing["current_stage"] if existing else None)
                new_report_text = latest_report_text if latest_report_text is not None else (existing["latest_report_text"] if existing else None)
                new_formatted = formatted_report if formatted_report is not None else (existing["formatted_report"] if existing else None)
                new_confirmed = int(is_confirmed) if is_confirmed is not None else (existing["is_confirmed"] if existing else 0)

                conn.execute(
                    """
                    INSERT INTO clinical_intake_state (
                        session_id, current_stage, patient_data_json, latest_report_text,
                        formatted_report, is_confirmed, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        current_stage = excluded.current_stage,
                        patient_data_json = excluded.patient_data_json,
                        latest_report_text = excluded.latest_report_text,
                        formatted_report = excluded.formatted_report,
                        is_confirmed = excluded.is_confirmed,
                        updated_at = excluded.updated_at;
                    """,
                    (session_id, new_stage, patient_data_json, new_report_text, new_formatted, new_confirmed, now)
                )

                # Touch parent session timestamps
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, last_activity = ? WHERE session_id = ?;",
                    (now, now, session_id)
                )

        return {
            "session_id": session_id,
            "current_stage": new_stage,
            "patient_data": parsed_data,
            "latest_report_text": new_report_text,
            "formatted_report": new_formatted,
            "is_confirmed": new_confirmed,
            "updated_at": now
        }

    def get_clinical_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve clinical intake state for a session."""
        if not session_id:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, current_stage, patient_data_json, latest_report_text,
                       formatted_report, is_confirmed, updated_at
                FROM clinical_intake_state
                WHERE session_id = ?;
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "session_id": row["session_id"],
                "current_stage": row["current_stage"],
                "patient_data": _safe_json_loads(row["patient_data_json"]) or {},
                "latest_report_text": row["latest_report_text"],
                "formatted_report": row["formatted_report"],
                "is_confirmed": row["is_confirmed"],
                "updated_at": row["updated_at"]
            }

    # =========================================================================
    # Artifact Management API
    # =========================================================================

    def save_artifact(
        self,
        session_id: str,
        artifact_type: str,
        file_path: str,
        artifact_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record artifact file reference and metadata in database.
        Stores file path references only, NEVER raw file bytes.
        """
        if not session_id or not artifact_type or not file_path:
            raise ValueError("session_id, artifact_type, and file_path are required.")

        if isinstance(file_path, (bytes, bytearray)):
            raise TypeError("Raw binary data cannot be stored as file_path. Provide filesystem path string.")

        aid = artifact_id or f"art_{uuid.uuid4()}"
        now = _utc_now_iso()
        meta_json = _safe_json_dumps(metadata)

        with self.get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?;", (session_id,))
                if not cursor.fetchone():
                    raise ValueError(f"Cannot save artifact: Session '{session_id}' does not exist.")

                conn.execute(
                    """
                    INSERT INTO session_artifacts (
                        artifact_id, session_id, artifact_type, file_path,
                        original_filename, mime_type, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (aid, session_id, artifact_type, str(file_path), original_filename, mime_type, now, meta_json)
                )

                # Touch parent session timestamps
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, last_activity = ? WHERE session_id = ?;",
                    (now, now, session_id)
                )

        return {
            "artifact_id": aid,
            "session_id": session_id,
            "artifact_type": artifact_type,
            "file_path": str(file_path),
            "original_filename": original_filename,
            "mime_type": mime_type,
            "created_at": now,
            "metadata": metadata or {}
        }

    def get_artifacts(
        self,
        session_id: str,
        artifact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve artifacts for a session, optionally filtered by artifact_type."""
        if not session_id:
            return []

        query = """
            SELECT artifact_id, session_id, artifact_type, file_path,
                   original_filename, mime_type, created_at, metadata_json
            FROM session_artifacts
            WHERE session_id = ?
        """
        params: List[Any] = [session_id]

        if artifact_type:
            query += " AND artifact_type = ?"
            params.append(artifact_type)

        query += " ORDER BY created_at DESC;"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "artifact_id": r["artifact_id"],
                    "session_id": r["session_id"],
                    "artifact_type": r["artifact_type"],
                    "file_path": r["file_path"],
                    "original_filename": r["original_filename"],
                    "mime_type": r["mime_type"],
                    "created_at": r["created_at"],
                    "metadata": _safe_json_loads(r["metadata_json"]) or {}
                }
                for r in rows
            ]

    def delete_session_artifacts(self, session_id: str) -> int:
        """
        Delete artifact database records for a session.
        Note: Physical disk file deletion is handled in later storage maintenance phases.
        """
        if not session_id:
            return 0

        with self.get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_artifacts WHERE session_id = ?;", (session_id,))
                return cursor.rowcount


# Default singleton instance for convenience
session_db = SessionDatabase()

# Convenience module-level functions delegated to the singleton
def initialize_database(db_path: Optional[str] = None) -> bool:
    db = SessionDatabase(db_path) if db_path else session_db
    return db.initialize_database()

def verify_database(db_path: Optional[str] = None) -> bool:
    db = SessionDatabase(db_path) if db_path else session_db
    return db.verify_database()

def create_session(*args, **kwargs):
    return session_db.create_session(*args, **kwargs)

def get_session(*args, **kwargs):
    return session_db.get_session(*args, **kwargs)

def list_sessions(*args, **kwargs):
    return session_db.list_sessions(*args, **kwargs)

def update_session(*args, **kwargs):
    return session_db.update_session(*args, **kwargs)

def delete_session(*args, **kwargs):
    return session_db.delete_session(*args, **kwargs)

def save_message(*args, **kwargs):
    return session_db.save_message(*args, **kwargs)

def get_messages(*args, **kwargs):
    return session_db.get_messages(*args, **kwargs)

def delete_session_messages(*args, **kwargs):
    return session_db.delete_session_messages(*args, **kwargs)

def save_clinical_state(*args, **kwargs):
    return session_db.save_clinical_state(*args, **kwargs)

def get_clinical_state(*args, **kwargs):
    return session_db.get_clinical_state(*args, **kwargs)

def save_artifact(*args, **kwargs):
    return session_db.save_artifact(*args, **kwargs)

def get_artifacts(*args, **kwargs):
    return session_db.get_artifacts(*args, **kwargs)

def delete_session_artifacts(*args, **kwargs):
    return session_db.delete_session_artifacts(*args, **kwargs)
