"""
Storage package for Multi-Agent Medical AI Assistant.
Provides SQLite-based persistence for sessions, messages, artifacts, and clinical state.
"""

from storage.session_db import (
    SessionDatabase,
    session_db,
    initialize_database,
    verify_database,
    create_session,
    get_session,
    list_sessions,
    update_session,
    delete_session,
    save_message,
    get_messages,
    delete_session_messages,
    save_clinical_state,
    get_clinical_state,
    save_artifact,
    get_artifacts,
    delete_session_artifacts,
    SUPPORTED_WORKSPACES,
    DEFAULT_DB_PATH
)

__all__ = [
    "SessionDatabase",
    "session_db",
    "initialize_database",
    "verify_database",
    "create_session",
    "get_session",
    "list_sessions",
    "update_session",
    "delete_session",
    "save_message",
    "get_messages",
    "delete_session_messages",
    "save_clinical_state",
    "get_clinical_state",
    "save_artifact",
    "get_artifacts",
    "delete_session_artifacts",
    "SUPPORTED_WORKSPACES",
    "DEFAULT_DB_PATH"
]
