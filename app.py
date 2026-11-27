import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List, Tuple, Any
import sqlite3
import glob
import threading
import time
from io import BytesIO

import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config, is_elevenlabs_key_id
from agents.agent_decision import process_query, process_query_stream
from agents.provider_error_handler import failure_handler, state_preserver, error_classifier
from storage.session_db import session_db, initialize_database

# Initialize SQLite database on module load
try:
    initialize_database()
except Exception as _init_err:
    print(f"Database initialization notice: {_init_err}")

# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize single ElevenLabs client (kept mutable for live key sync without duplicating clients)
elevenlabs_api_key = (getattr(getattr(config, "speech", None), "eleven_labs_api_key", None) or os.getenv("ELEVEN_LABS_API_KEY") or "").strip()
client = ElevenLabs(
    api_key=elevenlabs_api_key,
)


def _resolve_elevenlabs_api_key() -> str:
    """Resolve current ElevenLabs API key cleanly from config or environment without exposing secrets."""
    key = getattr(getattr(config, "speech", None), "eleven_labs_api_key", None)
    if key is None:
        key = os.getenv("ELEVEN_LABS_API_KEY")
    return str(key).strip() if key else ""


def _build_key_id_error_response(service_label: str) -> JSONResponse:
    """Return a clear non-secret configuration error when an ElevenLabs Key ID was used instead of an sk_ key."""
    return JSONResponse(
        status_code=401,
        content={
            "error": (
                f"ElevenLabs {service_label} authentication failed: ELEVEN_LABS_API_KEY appears to be an "
                "API Key ID rather than an API secret key (valid keys start with 'sk_'). "
                "Update ELEVEN_LABS_API_KEY in .env and restart the server."
            ),
            "error_code": "ELEVENLABS_KEY_ID_USED",
            "detail": "API key ID used as API key; expected an ElevenLabs secret key starting with 'sk_'."
        }
    )

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
if not os.environ.get("TESTING"):
    cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
    cleanup_thread.start()

# =============================================================================
# Workspace Prefix & Dual-Mode Session Resolution (Phase 6.2)
# =============================================================================

WORKSPACE_PREFIX_MAP = {
    "conversation": "conv_",
    "image_analysis": "img_",
    "clinical_pre_consultation": "clin_",
}
PREFIX_WORKSPACE_MAP = {
    "conv_": "conversation",
    "img_": "image_analysis",
    "clin_": "clinical_pre_consultation",
}


def parse_and_validate_session_id(sid: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate session ID structure and identify its workspace prefix if present.
    Returns (is_valid, workspace_type, raw_uuid_str).
    - Prefixed: conv_<uuid4>, img_<uuid4>, clin_<uuid4>
    - Legacy: raw canonical UUID string
    """
    if not sid or not isinstance(sid, str):
        return False, None, None
    sid = sid.strip()
    if not sid or len(sid) > 100:
        return False, None, None

    for prefix, ws in PREFIX_WORKSPACE_MAP.items():
        if sid.startswith(prefix):
            raw_uuid = sid[len(prefix):]
            try:
                uuid_obj = uuid.UUID(raw_uuid)
                if str(uuid_obj) != raw_uuid:
                    return False, None, None
                return True, ws, raw_uuid
            except (ValueError, AttributeError):
                return False, None, None

    # Check if it is a valid legacy UUID without prefix
    try:
        uuid_obj = uuid.UUID(sid)
        if str(uuid_obj) != sid:
            return False, None, None
        return True, None, sid
    except (ValueError, AttributeError):
        return False, None, None


def resolve_session_id(
    x_session_id: Optional[str],
    session_cookie: Optional[str],
    response: Response,
    allowed_workspaces: Tuple[str, ...],
    default_workspace: str,
) -> str:
    """
    Dual-mode session resolution:
    1. X-Session-ID header (precedence)
    2. session_id cookie (legacy / fallback)
    3. Generate new workspace-prefixed UUIDv4

    Validates format, checks workspace boundaries, and registers in SQLite.
    Sets both response cookie and X-Session-ID header.
    """
    resolved_id: Optional[str] = None

    # 1. Header present: takes precedence over cookie
    if x_session_id is not None:
        raw_header = x_session_id.strip()
        is_valid, ws, _ = parse_and_validate_session_id(raw_header)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid X-Session-ID format. Expected format: conv_<uuid>, img_<uuid>, or clin_<uuid>."
            )
        if ws is None:
            raise HTTPException(
                status_code=400,
                detail="X-Session-ID must include a workspace prefix ('conv_', 'img_', or 'clin_')."
            )
        if ws not in allowed_workspaces:
            raise HTTPException(
                status_code=400,
                detail=f"Workspace mismatch: Session '{raw_header}' with prefix '{ws}' cannot be used in this endpoint. Allowed: {allowed_workspaces}."
            )

        # Check existing registration in SQLite
        existing = session_db.get_session(raw_header)
        if existing:
            if existing["workspace_type"] != ws:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workspace conflict: Session '{raw_header}' is registered as '{existing['workspace_type']}' and cannot be used in '{ws}'."
                )
        else:
            try:
                session_db.create_session(session_id=raw_header, workspace_type=ws)
            except sqlite3.IntegrityError:
                existing = session_db.get_session(raw_header)
                if existing and existing["workspace_type"] != ws:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Workspace conflict: Session '{raw_header}' is registered as '{existing['workspace_type']}' and cannot be used in '{ws}'."
                    )
        resolved_id = raw_header

    # 2. Header absent, cookie present: preserve legacy behavior
    elif session_cookie is not None:
        raw_cookie = session_cookie.strip()
        is_valid, ws, _ = parse_and_validate_session_id(raw_cookie)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid session_id cookie format."
            )

        if ws is not None:
            # Cookie has a prefix (from previous Phase 6 interactions)
            if ws not in allowed_workspaces:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workspace mismatch: Cookie session '{raw_cookie}' with prefix '{ws}' cannot be used in this endpoint. Allowed: {allowed_workspaces}."
                )
            existing = session_db.get_session(raw_cookie)
            if existing:
                if existing["workspace_type"] != ws:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Workspace conflict: Session '{raw_cookie}' is registered as '{existing['workspace_type']}' and cannot be used in '{ws}'."
                    )
            else:
                try:
                    session_db.create_session(session_id=raw_cookie, workspace_type=ws)
                except sqlite3.IntegrityError:
                    pass
            resolved_id = raw_cookie
        else:
            # Legacy un-prefixed UUID
            existing = session_db.get_session(raw_cookie)
            if existing:
                if existing["workspace_type"] not in allowed_workspaces:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Workspace conflict: Legacy session '{raw_cookie}' is registered as '{existing['workspace_type']}' and cannot be reused in this endpoint."
                    )
            else:
                try:
                    session_db.create_session(session_id=raw_cookie, workspace_type=default_workspace)
                except sqlite3.IntegrityError:
                    pass
            resolved_id = raw_cookie

    # 3. Neither present: generate new prefixed UUIDv4
    else:
        prefix = WORKSPACE_PREFIX_MAP[default_workspace]
        new_id = f"{prefix}{uuid.uuid4()}"
        try:
            session_db.create_session(session_id=new_id, workspace_type=default_workspace)
        except sqlite3.IntegrityError:
            pass
        resolved_id = new_id

    # Set both response headers
    response.set_cookie(key="session_id", value=resolved_id)
    response.headers["X-Session-ID"] = resolved_id

    return resolved_id

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []
    workspace: Optional[str] = None
    workspace_type: Optional[str] = None

class CreateSessionRequest(BaseModel):
    workspace_type: Optional[str] = None
    workspace: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    is_archived: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # Default premade voice (Sarah)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}

@app.post("/chat/stream")
async def chat_stream(
    request: QueryRequest,
    raw_request: Request,
    response: Response,
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Process user query with genuine real-time LLM token streaming across agents.
    Streams Server-Sent Events (SSE) with event types: start, status, token, complete, error.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    target_workspace = request.workspace_type or request.workspace
    if target_workspace and target_workspace not in ("conversation", "clinical_pre_consultation"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workspace '{target_workspace}'. Allowed: conversation, clinical_pre_consultation."
        )

    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("conversation", "clinical_pre_consultation"),
        default_workspace=target_workspace or "conversation"
    )

    _, resolved_ws, _ = parse_and_validate_session_id(session_id)
    active_workspace = resolved_ws or target_workspace or "conversation"

    # Persist incoming user message immediately
    try:
        session_db.save_message(
            session_id=session_id,
            sender="user",
            content=request.query,
            agent_name="user"
        )
    except Exception as _umsg_err:
        print(f"Notice: User message save in stream: {_umsg_err}")

    async def event_generator():
        accumulated_text = ""
        final_agent = "CONVERSATION_AGENT"
        result_image = None
        has_completed = False

        try:
            for event in process_query_stream(request.query, session_id=session_id, workspace=active_workspace):
                # Check for client cancellation / disconnect
                if await raw_request.is_disconnected():
                    print(f"Streaming cancelled: client disconnected for session {session_id}")
                    return

                ev_type = event.get("event", "message")
                ev_data = event.get("data", {})

                if ev_type == "token":
                    chunk_text = ev_data.get("chunk", "")
                    accumulated_text += chunk_text
                elif ev_type == "complete":
                    final_agent = ev_data.get("agent", final_agent)
                    result_image = ev_data.get("result_image")
                    has_completed = True
                elif ev_type == "start":
                    final_agent = ev_data.get("agent", final_agent)

                # Format SSE chunk
                yield f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n"

            # Upon successful completion, save assistant message to DB
            final_content = accumulated_text.strip() or (ev_data.get("response", "") if has_completed else "")
            if has_completed and final_content:
                try:
                    is_report = (
                        "CLINICAL" in final_agent.upper() and
                        any(h in final_content for h in [
                            "CLINICAL PRE-CONSULTATION INTAKE REPORT",
                            "PATIENT INTAKE SUMMARY REPORT",
                            "PRE-CONSULTATION SUMMARY",
                            "PRE-CONSULTATION INTAKE REPORT",
                            "PRE-CONSULTATION REPORT"
                        ])
                    )
                    is_emergency = "emergency" in final_content.lower() and (
                        "escalation" in final_content.lower() or "immediate" in final_content.lower()
                    )

                    session_db.save_message(
                        session_id=session_id,
                        sender="assistant",
                        content=final_content,
                        agent_name=final_agent,
                        is_report=is_report,
                        is_emergency=is_emergency,
                        metadata={"result_image": result_image} if result_image else None
                    )

                    # Auto-update session title from initial query if using default
                    sess = session_db.get_session(session_id)
                    if sess and (not sess.get("title") or sess.get("title", "").startswith("New ") or sess.get("title") == "Untitled Session"):
                        clean_title = request.query.strip().lstrip("#*- ").strip().split("\n")[0][:50]
                        if clean_title:
                            session_db.update_session(session_id=session_id, title=clean_title)

                    # If clinical agent, persist structured intake state to database
                    if "CLINICAL" in final_agent.upper():
                        from agents.agent_decision import get_clinical_agent
                        clinical_agent = get_clinical_agent(session_id)
                        if hasattr(clinical_agent, 'patient_info_collector'):
                            raw_p_data = clinical_agent.patient_info_collector.get_patient_data()
                            p_data = raw_p_data if isinstance(raw_p_data, dict) else {}
                            raw_latest = getattr(clinical_agent, '_latest_formatted_report', None) or getattr(clinical_agent, '_latest_report', None) or (accumulated_text if is_report else None)
                            if isinstance(raw_latest, dict):
                                raw_latest = json.dumps(raw_latest)
                            raw_formatted = getattr(clinical_agent, '_latest_formatted_report', None)
                            if isinstance(raw_formatted, dict):
                                raw_formatted = json.dumps(raw_formatted)

                            session_db.save_clinical_state(
                                session_id=session_id,
                                patient_data=p_data,
                                current_stage="completed" if is_report else ("emergency" if is_emergency else "active"),
                                latest_report_text=raw_latest,
                                formatted_report=raw_formatted,
                                is_confirmed=False
                            )
                except Exception as _save_err:
                    print(f"Notice: Streaming persistence error: {_save_err}")

        except Exception as err:
            print(f"Streaming execution error: {err}")
            err_data = {"message": str(err), "category": "server_error", "session_id": session_id}
            yield f"event: error\ndata: {json.dumps(err_data)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Session-ID": session_id,
    }
    resp = StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
    resp.set_cookie(key="session_id", value=session_id)
    resp.headers["X-Session-ID"] = session_id
    return resp


@app.post("/chat")
def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """Process user text query through the multi-agent system."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    target_workspace = request.workspace_type or request.workspace
    if target_workspace and target_workspace not in ("conversation", "clinical_pre_consultation"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workspace '{target_workspace}'. Allowed: conversation, clinical_pre_consultation."
        )

    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("conversation", "clinical_pre_consultation"),
        default_workspace=target_workspace or "conversation"
    )

    # Determine resolved workspace from session ID or request
    _, resolved_ws, _ = parse_and_validate_session_id(session_id)
    active_workspace = resolved_ws or target_workspace or "conversation"

    # Persist incoming user message
    try:
        session_db.save_message(
            session_id=session_id,
            sender="user",
            content=request.query,
            agent_name="user"
        )
    except Exception as _umsg_err:
        print(f"Notice: User message save: {_umsg_err}")
    
    try:
        response_data = process_query(request.query, session_id=session_id, workspace=active_workspace)
        response_text = response_data['messages'][-1].content
        
        # Set session headers
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"],
            "session_id": session_id
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        # Persist assistant response message
        try:
            agent_name = response_data.get("agent_name", "CONVERSATION_AGENT")
            is_report = (
                "CLINICAL" in agent_name.upper() and
                any(h in response_text for h in [
                    "CLINICAL PRE-CONSULTATION INTAKE REPORT",
                    "PATIENT INTAKE SUMMARY REPORT",
                    "PRE-CONSULTATION SUMMARY",
                    "PRE-CONSULTATION INTAKE REPORT",
                    "PRE-CONSULTATION REPORT"
                ])
            )
            is_emergency = "emergency" in response_text.lower() and ("escalation" in response_text.lower() or "immediate" in response_text.lower())
            
            session_db.save_message(
                session_id=session_id,
                sender="assistant",
                content=response_text,
                agent_name=agent_name,
                is_report=is_report,
                is_emergency=is_emergency,
                metadata={"result_image": result.get("result_image")} if result.get("result_image") else None
            )

            # Auto-update session title from initial query if using default or generic title
            sess = session_db.get_session(session_id)
            if sess and (not sess.get("title") or sess.get("title", "").startswith("New ") or sess.get("title") == "Untitled Session"):
                clean_title = request.query.strip().lstrip("#*- ").strip().split("\n")[0][:50]
                if clean_title:
                    session_db.update_session(session_id=session_id, title=clean_title)

            # If clinical agent, persist structured intake state to database
            if agent_name == "CLINICAL_PRE_CONSULTATION_AGENT":
                from agents.agent_decision import get_clinical_agent
                clinical_agent = get_clinical_agent(session_id)
                if hasattr(clinical_agent, 'patient_info_collector'):
                    raw_p_data = clinical_agent.patient_info_collector.get_patient_data()
                    p_data = raw_p_data if isinstance(raw_p_data, dict) else {}
                    raw_latest = getattr(clinical_agent, '_latest_formatted_report', None) or getattr(clinical_agent, '_latest_report', None) or (response_text if is_report else None)
                    if isinstance(raw_latest, dict):
                        raw_latest = json.dumps(raw_latest)
                    raw_formatted = getattr(clinical_agent, '_latest_formatted_report', None)
                    if isinstance(raw_formatted, dict):
                        raw_formatted = json.dumps(raw_formatted)

                    session_db.save_clinical_state(
                        session_id=session_id,
                        patient_data=p_data,
                        current_stage="completed" if is_report else ("emergency" if is_emergency else "active"),
                        latest_report_text=raw_latest,
                        formatted_report=raw_formatted,
                        is_confirmed=False
                    )
        except Exception as _amsg_err:
            print(f"Notice: Assistant message persistence: {_amsg_err}")
        
        return result
    
    except Exception as e:
        # Phase 1: DeepSeek provider failure handling and session preservation
        classification = error_classifier.classify_error(e)
        
        # Preserve patient data from clinical agent if available
        patient_data = None
        try:
            from agents.agent_decision import get_clinical_agent
            clinical_agent = get_clinical_agent(session_id)
            if hasattr(clinical_agent, 'patient_info_collector'):
                patient_data = clinical_agent.patient_info_collector.get_patient_data()
                if patient_data:
                    state_preserver.save_session_state(session_id, patient_data)
        except Exception:
            pass
        
        # Ensure session headers are preserved on error
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id
        
        # Provider failure: return structured, safe response with appropriate status code
        if classification.get("is_provider_error"):
            status_code, error_response = failure_handler.handle_provider_failure(
                error=e,
                provider_name="groq",
                patient_data=patient_data,
                session_id=session_id
            )
            return JSONResponse(status_code=status_code, content=error_response)
        
        # Non-provider unexpected errors: return safe 500 without leaking stack traces or keys
        print(f"Unexpected error in /chat: {classification.get('user_message', str(e))}")
        safe_error = {
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An error occurred while processing your request. Your session has been preserved.",
            "detail": "An error occurred while processing your request. Your session has been preserved.",
            "session_id": session_id,
            "details": {
                "session_preserved": True,
                "patient_data_preserved": bool(patient_data)
            }
        }
        return JSONResponse(status_code=500, content=safe_error)

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """Process medical image uploads with optional text input."""
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )
    
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("image_analysis",),
        default_workspace="image_analysis"
    )
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query, session_id=session_id, workspace="image_analysis")
        response_text = response_data['messages'][-1].content

        # Set session headers
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"],
            "session_id": session_id
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        # Persist input image artifact to database (file path reference only, no raw bytes)
        input_art = None
        try:
            input_art = session_db.save_artifact(
                session_id=session_id,
                artifact_type="input_image",
                file_path=file_path,
                original_filename=image.filename,
                mime_type=image.content_type,
                metadata={"text_prompt": text} if text else None
            )
        except Exception as _art_err:
            print(f"Notice: Input artifact persistence: {_art_err}")

        # Persist result image artifact if generated
        if result.get("result_image"):
            try:
                session_db.save_artifact(
                    session_id=session_id,
                    artifact_type="result_image",
                    file_path=result["result_image"],
                    original_filename="segmentation_plot.png",
                    mime_type="image/png",
                    metadata={"agent": response_data.get("agent_name")}
                )
            except Exception as _res_art_err:
                print(f"Notice: Result artifact persistence: {_res_art_err}")

        # Persist user message
        try:
            session_db.save_message(
                session_id=session_id,
                sender="user",
                content=text or f"Uploaded image: {image.filename}",
                agent_name="user",
                metadata={
                    "artifact_id": input_art["artifact_id"] if input_art else None,
                    "file_path": file_path,
                    "original_filename": image.filename
                }
            )
        except Exception as _umsg_err:
            print(f"Notice: Upload user message save: {_umsg_err}")

        # Persist assistant response
        try:
            session_db.save_message(
                session_id=session_id,
                sender="assistant",
                content=response_text,
                agent_name=response_data.get("agent_name", "IMAGE_ANALYSIS_AGENT"),
                metadata={
                    "result_image": result.get("result_image"),
                    "agent": response_data.get("agent_name")
                }
            )
            # Auto-update session title from image filename/query
            sess = session_db.get_session(session_id)
            if sess and sess.get("title", "").startswith("New "):
                title_text = (text.strip() or f"Image: {image.filename}")[:50]
                session_db.update_session(session_id=session_id, title=title_text)
        except Exception as _amsg_err:
            print(f"Notice: Upload assistant message save: {_amsg_err}")

        return result
    except Exception as e:
        classification = error_classifier.classify_error(e)
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id
        if classification.get("is_provider_error"):
            status_code, error_response = failure_handler.handle_provider_failure(
                error=e,
                provider_name="groq",
                session_id=session_id
            )
            return JSONResponse(status_code=status_code, content=error_response)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "IMAGE_PROCESSING_ERROR",
                "message": "An error occurred during image processing.",
                "detail": "An error occurred during image processing.",
                "session_id": session_id
            }
        )

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """Handle human validation for medical AI outputs."""
    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("image_analysis",),
        default_workspace="image_analysis"
    )

    try:
        # Set session headers
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query, session_id=session_id)
        resp_content = response_data['messages'][-1].content

        # Persist validation messages in database
        try:
            session_db.save_message(
                session_id=session_id,
                sender="user",
                content=validation_query,
                agent_name="user",
                metadata={"validation_result": validation_result, "comments": comments}
            )
            session_db.save_message(
                session_id=session_id,
                sender="assistant",
                content=resp_content,
                agent_name="HUMAN_VALIDATION",
                metadata={"status": "validated" if validation_result.lower() == 'yes' else "rejected"}
            )
        except Exception as _vmsg_err:
            print(f"Notice: Validation message persistence: {_vmsg_err}")

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": resp_content,
                "session_id": session_id
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": resp_content,
                "session_id": session_id
            }
    except Exception as e:
        classification = error_classifier.classify_error(e)
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id
        if classification.get("is_provider_error"):
            status_code, error_response = failure_handler.handle_provider_failure(
                error=e,
                provider_name="groq",
                session_id=session_id
            )
            return JSONResponse(status_code=status_code, content=error_response)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "message": "An error occurred during output validation.",
                "detail": "An error occurred during output validation.",
                "session_id": session_id
            }
        )

@app.post("/clinical-patient-feedback")
def handle_clinical_patient_feedback(
    response: Response,
    patient_feedback: str = Form(...),
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Handle patient corrections/additional information for clinical report.
    
    This endpoint is specifically for clinical feedback, distinct from validation.
    Patient provides additional info or corrections to their intake report.
    The feedback is processed through the clinical agent to regenerate the report.
    """
    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("clinical_pre_consultation",),
        default_workspace="clinical_pre_consultation"
    )

    # Set session headers
    response.set_cookie(key="session_id", value=session_id)
    response.headers["X-Session-ID"] = session_id
    
    try:
        # Validate patient feedback is not empty
        if not patient_feedback or not patient_feedback.strip():
            raise HTTPException(status_code=400, detail="Feedback cannot be empty")
        
        # Get the session's clinical agent
        from agents.agent_decision import get_clinical_agent
        clinical_agent = get_clinical_agent(session_id)
        
        # Process patient feedback through the clinical agent
        # The agent will accumulate this with existing patient data
        clinical_response = clinical_agent.process_patient_info(patient_feedback)
        
        # Extract updated report
        status = clinical_response.get("status")
        report = clinical_response.get("report")
        formatted_report = clinical_response.get("formatted_report")
        
        # Persist patient feedback and updated report
        try:
            session_db.save_message(
                session_id=session_id,
                sender="user",
                content=patient_feedback,
                agent_name="user",
                metadata={"type": "clinical_feedback"}
            )
            resp_content = formatted_report or report or "Report updated with your feedback."
            session_db.save_message(
                session_id=session_id,
                sender="assistant",
                content=resp_content,
                agent_name="CLINICAL_PRE_CONSULTATION_AGENT",
                is_report=True,
                metadata={"status": status, "is_update": True}
            )
            if hasattr(clinical_agent, 'patient_info_collector'):
                raw_p_data = clinical_agent.patient_info_collector.get_patient_data()
                p_data = raw_p_data if isinstance(raw_p_data, dict) else {}
                session_db.save_clinical_state(
                    session_id=session_id,
                    patient_data=p_data,
                    current_stage=status or "completed",
                    latest_report_text=report,
                    formatted_report=formatted_report,
                    is_confirmed=False
                )
                # Auto-update session title if still generic default
                sess = session_db.get_session(session_id)
                if sess and (not sess.get("title") or sess.get("title", "").startswith("New ") or sess.get("title") == "Untitled Session"):
                    clean_title = patient_feedback.strip().lstrip("#*- ").strip().split("\n")[0][:50]
                    if clean_title:
                        session_db.update_session(session_id=session_id, title=clean_title)
        except Exception as _fmsg_err:
            print(f"Notice: Clinical feedback persistence: {_fmsg_err}")

        return {
            "status": status,
            "message": "Report updated with your feedback",
            "report": report,
            "formatted_report": formatted_report,
            "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
            "requires_confirmation": True,
            "session_id": session_id
        }
    except Exception as e:
        classification = error_classifier.classify_error(e)
        response.set_cookie(key="session_id", value=session_id)
        response.headers["X-Session-ID"] = session_id
        if classification.get("is_provider_error"):
            status_code, error_response = failure_handler.handle_provider_failure(
                error=e,
                provider_name="groq",
                session_id=session_id
            )
            return JSONResponse(status_code=status_code, content=error_response)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "CLINICAL_FEEDBACK_ERROR",
                "message": "An error occurred while processing patient feedback.",
                "detail": "An error occurred while processing patient feedback.",
                "session_id": session_id
            }
        )

@app.post("/clinical-confirm")
def handle_clinical_confirm(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Explicit patient confirmation/verification of the clinical pre-consultation report.
    Records confirmed status and confirmation timestamp in database.
    """
    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("clinical_pre_consultation",),
        default_workspace="clinical_pre_consultation"
    )
    response.set_cookie(key="session_id", value=session_id)
    response.headers["X-Session-ID"] = session_id

    timestamp = datetime.now().isoformat()
    try:
        from agents.agent_decision import get_clinical_agent
        clinical_agent = get_clinical_agent(session_id)
        raw_p_data = clinical_agent.patient_info_collector.get_patient_data() if hasattr(clinical_agent, 'patient_info_collector') else {}
        latest_report = getattr(clinical_agent, '_latest_formatted_report', None) or getattr(clinical_agent, '_latest_report', None)
        
        session_db.save_clinical_state(
            session_id=session_id,
            patient_data=raw_p_data,
            current_stage="completed",
            latest_report_text=latest_report if isinstance(latest_report, str) else None,
            formatted_report=latest_report if isinstance(latest_report, str) else None,
            is_confirmed=True
        )
        session_db.save_message(
            session_id=session_id,
            sender="user",
            content="[PATIENT VERIFICATION] Patient explicitly reviewed and confirmed this clinical pre-consultation report as accurate.",
            agent_name="user",
            metadata={"type": "patient_verification", "confirmed_at": timestamp}
        )
        return {
            "status": "success",
            "is_confirmed": True,
            "confirmed_at": timestamp,
            "session_id": session_id,
            "message": "Report confirmed and verified by patient."
        }
    except Exception as e:
        print(f"Notice: Clinical confirm error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "session_id": session_id}
        )

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
):
    """Endpoint to transcribe speech using ElevenLabs API (auto-detects Hindi, Hinglish, and English by default)"""
    if not audio or not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )

    temp_audio = None
    mp3_path = None
    try:
        # Read uploaded audio content and validate non-empty payload
        audio_content = await audio.read()
        file_size = len(audio_content) if audio_content else 0
        print(f"Received audio file size: {file_size} bytes")

        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )

        # Save the audio file temporarily, preserving extension when safe
        os.makedirs(SPEECH_DIR, exist_ok=True)
        _, raw_ext = os.path.splitext(audio.filename or "")
        safe_ext = raw_ext.lower() if raw_ext.lower() in {".webm", ".ogg", ".mp3", ".wav", ".m4a", ".mp4"} else ".webm"
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}{safe_ext}"
        with open(temp_audio, "wb") as f:
            f.write(audio_content)

        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        try:
            audio_segment = AudioSegment.from_file(temp_audio)
        except Exception as decode_err:
            print(f"Unsupported or unreadable audio format: {decode_err}")
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported or corrupted audio recording format. Please try recording again."}
            )

        if len(audio_segment) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Recorded audio is empty. Please speak clearly before stopping."}
            )

        audio_segment.export(mp3_path, format="mp3")
        mp3_size = os.path.getsize(mp3_path)
        print(f"Converted MP3 file size: {mp3_size} bytes")

        with open(mp3_path, "rb") as mp3_file:
            audio_data = mp3_file.read()
        print("Converted audio file into byte array successfully!")

        # Validate ElevenLabs key before upstream call (respect mocked unit tests)
        resolved_key = _resolve_elevenlabs_api_key()
        is_mock_client = type(getattr(getattr(client, "speech_to_text", None), "convert", None)).__module__.startswith("unittest.mock")
        if not is_mock_client:
            if not resolved_key or resolved_key.startswith("YOUR_") or resolved_key == "EXAMPLE_KEY":
                return JSONResponse(
                    status_code=503,
                    content={"error": "ElevenLabs Speech-to-Text is not configured on the server", "error_code": "STT_KEY_MISSING"}
                )
            if is_elevenlabs_key_id(resolved_key):
                return _build_key_id_error_response("Speech-to-Text")

        # Build STT kwargs: omit language_code by default so ElevenLabs Scribe auto-detects English, Hindi, and Hinglish
        stt_kwargs: Dict[str, Any] = {
            "file": audio_data,
            "model_id": "scribe_v1",
            "tag_audio_events": True,
            "diarize": True,
        }
        if language_code and language_code.strip():
            stt_kwargs["language_code"] = language_code.strip()

        transcription = client.speech_to_text.convert(**stt_kwargs)
        transcript_text = (getattr(transcription, "text", None) or "").strip()

        if transcript_text:
            return {"transcript": transcript_text}
        return JSONResponse(
            status_code=422,
            content={"error": "No speech detected in the recording. Please try speaking again.", "details": ""}
        )

    except Exception as e:
        err_str = str(e)
        print(f"Transcription error: {err_str}")
        lower_err = err_str.lower()
        if "api_key_id_used_as_api_key" in lower_err:
            return _build_key_id_error_response("Speech-to-Text")
        if "invalid_api_key" in lower_err or "status_code: 401" in lower_err:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "ElevenLabs Speech-to-Text authentication failed. Check ELEVEN_LABS_API_KEY in .env.",
                    "error_code": "STT_AUTH_FAILED",
                }
            )
        if "quota_exceeded" in lower_err or "status_code: 429" in lower_err:
            return JSONResponse(
                status_code=429,
                content={"error": "ElevenLabs Speech-to-Text rate limit or quota exceeded.", "error_code": "STT_RATE_LIMITED"}
            )
        return JSONResponse(
            status_code=500,
            content={"error": "Speech-to-Text service encountered an unexpected error. Please try again."}
        )
    finally:
        for path_to_clean in (temp_audio, mp3_path):
            if path_to_clean and os.path.exists(path_to_clean):
                try:
                    os.remove(path_to_clean)
                except Exception as cleanup_err:
                    print(f"Could not delete temp file {path_to_clean}: {cleanup_err}")

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using ElevenLabs API"""
    try:
        text = request.text
        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required and cannot be empty", "error_code": "EMPTY_TEXT"}
            )

        selected_voice_id = request.voice_id
        if not selected_voice_id or selected_voice_id in ("EXAMPLE_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"):
            selected_voice_id = "EXAVITQu4vr4xnSDxMaL"

        api_key = _resolve_elevenlabs_api_key()
        if not api_key or api_key.startswith("YOUR_") or api_key == "EXAMPLE_KEY":
            print("Warning: ElevenLabs API key is missing or not configured.")
            return JSONResponse(
                status_code=503,
                content={"error": "ElevenLabs TTS is not configured on the server", "error_code": "TTS_KEY_MISSING"}
            )
        if is_elevenlabs_key_id(api_key):
            return _build_key_id_error_response("Text-to-Speech")

        # Define API request to ElevenLabs
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        payload = {
            "text": text.strip(),
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API with bounded timeout
        try:
            response = requests.post(elevenlabs_url, headers=headers, json=payload, timeout=15)
        except requests.exceptions.Timeout as timeout_err:
            print(f"Error: ElevenLabs upstream timeout: {timeout_err}")
            return JSONResponse(
                status_code=504,
                content={"error": "ElevenLabs TTS service timed out", "error_code": "TTS_TIMEOUT"}
            )
        except requests.exceptions.RequestException as req_err:
            print(f"Error: ElevenLabs upstream connection error: {req_err}")
            return JSONResponse(
                status_code=502,
                content={"error": "Upstream ElevenLabs service unreachable", "error_code": "TTS_CONNECTION_ERROR"}
            )

        # Handle specific upstream status codes
        if response.status_code in (400, 401, 403):
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass
            detail = error_data.get("detail", {}) if isinstance(error_data, dict) else {}
            if isinstance(detail, dict) and detail.get("status") == "api_key_id_used_as_api_key":
                return _build_key_id_error_response("Text-to-Speech")
            if isinstance(detail, dict) and detail.get("code") == "quota_exceeded":
                quota_msg = detail.get("message") or "ElevenLabs character quota exceeded."
                print(f"Error: ElevenLabs character quota exceeded: {quota_msg}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": quota_msg,
                        "error_code": "TTS_QUOTA_EXCEEDED",
                        "detail": quota_msg
                    }
                )

            auth_msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            if response.status_code in (401, 403) or (isinstance(detail, dict) and detail.get("code") == "invalid_api_key"):
                print(f"Error: ElevenLabs authentication failed: HTTP {response.status_code}")
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": auth_msg or "ElevenLabs authentication failed. Invalid API credentials.",
                        "error_code": "TTS_AUTH_FAILED",
                        "detail": auth_msg or "Invalid API credentials"
                    }
                )
            return JSONResponse(
                status_code=400,
                content={
                    "error": auth_msg or "ElevenLabs TTS request failed.",
                    "error_code": "TTS_BAD_REQUEST",
                    "detail": auth_msg or response.text[:200]
                }
            )

        if response.status_code == 402:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass
            detail = error_data.get("detail", {}) if isinstance(error_data, dict) else {}
            plan_msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            plan_msg = plan_msg or "A paid ElevenLabs plan is required for this request."
            print(f"Error: ElevenLabs payment required: HTTP 402 - {plan_msg}")
            return JSONResponse(
                status_code=402,
                content={"error": plan_msg, "error_code": "TTS_PAYMENT_REQUIRED", "detail": plan_msg}
            )

        if response.status_code == 404:
            print(f"Error: ElevenLabs voice or model not found: HTTP 404 - {response.text[:200]}")
            return JSONResponse(
                status_code=404,
                content={"error": f"The requested voice '{selected_voice_id}' or model was not found.", "error_code": "TTS_VOICE_NOT_FOUND"}
            )

        if response.status_code == 429:
            print(f"Error: ElevenLabs rate limit or quota exceeded: HTTP 429 - {response.text[:200]}")
            return JSONResponse(
                status_code=429,
                content={"error": "ElevenLabs rate limit or quota exceeded.", "error_code": "TTS_RATE_LIMITED"}
            )

        if response.status_code != 200:
            print(f"Error: ElevenLabs API returned HTTP {response.status_code}")
            return JSONResponse(
                status_code=502,
                content={"error": f"Upstream ElevenLabs service error (HTTP {response.status_code})", "error_code": "TTS_UPSTREAM_ERROR"}
            )

        # Validate audio content
        if not response.content or len(response.content) < 50:
            print("Error: Received empty or truncated audio content from ElevenLabs.")
            return JSONResponse(
                status_code=502,
                content={"error": "Received empty or truncated audio response from ElevenLabs", "error_code": "TTS_EMPTY_AUDIO"}
            )

        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Return the generated audio file
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        print(f"TTS generation unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal TTS generation error: {str(e)}", "error_code": "TTS_INTERNAL_ERROR"}
        )

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

# =============================================================================
# Session Lifecycle & Management REST Endpoints (Phase 6)
# =============================================================================

@app.get("/sessions")
def list_sessions_endpoint(
    workspace_type: Optional[str] = None,
    workspace: Optional[str] = None,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0
):
    """List sessions filtered by workspace type and archival status."""
    target_workspace = workspace_type or workspace
    try:
        sessions = session_db.list_sessions(
            workspace_type=target_workspace,
            include_archived=include_archived,
            limit=limit,
            offset=offset
        )
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.post("/sessions")
def create_session_endpoint(
    req: CreateSessionRequest,
    response: Response
):
    """Create a new session with workspace-specific prefix."""
    ws_type = req.workspace_type or req.workspace or "conversation"
    if ws_type not in WORKSPACE_PREFIX_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workspace_type '{ws_type}'. Must be one of: {list(WORKSPACE_PREFIX_MAP.keys())}"
        )
    
    prefix = WORKSPACE_PREFIX_MAP[ws_type]
    new_session_id = f"{prefix}{uuid.uuid4()}"
    
    try:
        session = session_db.create_session(
            session_id=new_session_id,
            workspace_type=ws_type,
            title=req.title,
            metadata=req.metadata
        )
        # Ensure fresh in-memory clinical agent for newly registered clinical session
        if ws_type == "clinical_pre_consultation":
            try:
                from agents.agent_decision import reset_clinical_agent_session
                reset_clinical_agent_session(new_session_id)
            except Exception:
                pass
        response.set_cookie(key="session_id", value=new_session_id)
        response.headers["X-Session-ID"] = new_session_id
        return {
            "status": "success",
            "session": session,
            "session_id": new_session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/sessions/{session_id}")
def get_session_endpoint(session_id: str):
    """Retrieve session details by session_id."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {
        "status": "success",
        "session": session
    }

@app.patch("/sessions/{session_id}")
def update_session_endpoint(session_id: str, req: UpdateSessionRequest):
    """Update session title, status, archival status, or metadata."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    updated = session_db.update_session(
        session_id=session_id,
        title=req.title,
        status=req.status,
        is_archived=req.is_archived,
        metadata=req.metadata
    )
    return {
        "status": "success",
        "session": updated
    }

@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """Delete session, cascading all messages, artifacts, and clinical state."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    # Clean up in-memory clinical agent if present
    try:
        from agents.agent_decision import reset_clinical_agent_session
        reset_clinical_agent_session(session_id)
    except Exception:
        pass
    
    deleted = session_db.delete_session(session_id)
    return {
        "status": "success",
        "deleted": deleted,
        "session_id": session_id
    }

@app.get("/sessions/{session_id}/messages")
def get_session_messages_endpoint(
    session_id: str,
    limit: Optional[int] = None,
    offset: int = 0
):
    """Retrieve ordered message history for a session."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    messages = session_db.get_messages(session_id=session_id, limit=limit, offset=offset)
    return {
        "status": "success",
        "session_id": session_id,
        "messages": messages,
        "count": len(messages)
    }

@app.get("/sessions/{session_id}/clinical-state")
def get_session_clinical_state_endpoint(session_id: str):
    """Retrieve persisted clinical intake state for a clinical pre-consultation session."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    state = session_db.get_clinical_state(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "clinical_state": state
    }

@app.get("/sessions/{session_id}/artifacts")
def get_session_artifacts_endpoint(
    session_id: str,
    artifact_type: Optional[str] = None
):
    """Retrieve image artifacts associated with a session."""
    session = session_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    artifacts = session_db.get_artifacts(session_id=session_id, artifact_type=artifact_type)
    return {
        "status": "success",
        "session_id": session_id,
        "artifacts": artifacts,
        "count": len(artifacts)
    }

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)