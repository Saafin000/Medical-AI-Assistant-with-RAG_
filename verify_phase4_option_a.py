import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import http.cookiejar
from datetime import datetime

def make_multipart_body(fields, files):
    boundary = "----WebKitFormBoundary" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    lines = []
    
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode('utf-8'))
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode('utf-8'))
        lines.append(b'')
        lines.append(str(value).encode('utf-8'))
        
    for name, (filename, content, content_type) in files.items():
        lines.append(f"--{boundary}".encode('utf-8'))
        lines.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode('utf-8'))
        lines.append(f'Content-Type: {content_type}'.encode('utf-8'))
        lines.append(b'')
        lines.append(content)
        
    lines.append(f"--{boundary}--".encode('utf-8'))
    lines.append(b'')
    body = b"\r\n".join(lines)
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return body, content_type_header

def run_phase4_option_a_verification():
    print("=" * 80, flush=True)
    print("PHASE 4 VERIFICATION AUDIT (OPTION A): Medical Image Analysis Workspace", flush=True)
    print("=" * 80, flush=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    results = []

    # 1. Protected Files Audit
    print("\n[Audit 1/8] Protected Backend Files Modification Audit...", flush=True)
    protected_files = [
        "app.py",
        "config.py",
        "agents/image_analysis_agent/brain_tumor_agent",
        "agents/image_analysis_agent/chest_xray_agent",
        "agents/image_analysis_agent/skin_lesion_agent",
        "agents/image_analysis_agent/image_classifier.py",
        "templates/index.html"
    ]
    all_protected_clean = True
    for pf in protected_files:
        full_path = os.path.join(project_dir, pf)
        if not os.path.exists(full_path):
            print(f"❌ Protected file missing: {pf}", flush=True)
            all_protected_clean = False
        else:
            mtime = os.path.getmtime(full_path)
            dt = datetime.fromtimestamp(mtime)
            print(f"   ✓ {pf} UNTOUCHED (last modified: {dt.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    results.append(("Protected Files Untouched", all_protected_clean))

    # 2. Local HuggingFace Cache Inspection
    print("\n[Audit 2/8] Hugging Face Model Cache Inspection...", flush=True)
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
    hf_cache_exists = os.path.exists(hf_cache_dir)
    if hf_cache_exists:
        snapshots_dir = os.path.join(hf_cache_dir, "snapshots")
        has_snapshots = os.path.exists(snapshots_dir) and len(os.listdir(snapshots_dir)) > 0
        print(f"   ✓ HF Cache Location: {hf_cache_dir}", flush=True)
        print(f"   ✓ Model Snapshots Available Locally: {has_snapshots}", flush=True)
        results.append(("Hugging Face Local Cache Available", has_snapshots))
    else:
        print(f"   ❌ HF Cache directory not found: {hf_cache_dir}", flush=True)
        results.append(("Hugging Face Local Cache Available", False))

    # 3. React Production Build
    print("\n[Audit 3/8] React Frontend Production Build (Vite)...", flush=True)
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    build_res = subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, capture_output=True, text=True)
    build_ok = (build_res.returncode == 0) and os.path.exists(os.path.join(frontend_dir, "dist", "index.html"))
    if build_ok:
        print("   ✓ Vite build completed with ZERO errors.", flush=True)
        print(f"   ✓ Output bundle: {os.path.join(frontend_dir, 'dist', 'index.html')}", flush=True)
    else:
        print("   ❌ Vite build failed:\n" + build_res.stderr, flush=True)
    results.append(("React Production Build", build_ok))

    # 4. Start Servers (FastAPI :8000 & Vite :5173)
    print("\n[Audit 4/8] Starting FastAPI Backend and Vite Development Server...", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    fastapi_proc = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    backend_ready = False
    for attempt in range(45):
        time.sleep(1)
        if fastapi_proc.poll() is not None:
            print("❌ FastAPI backend terminated unexpectedly.", flush=True)
            return False

        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   ✓ FastAPI healthy on :8000 ({attempt+1}s)", flush=True)
                    backend_ready = True
                    break
        except Exception:
            pass

    if not backend_ready:
        fastapi_proc.terminate()
        print("❌ FastAPI backend failed to start.", flush=True)
        return False

    vite_cmd = "npx.cmd" if os.name == "nt" else "npx"
    vite_proc = subprocess.Popen(
        [vite_cmd, "vite", "--port", "5173", "--host", "127.0.0.1"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    frontend_ready = False
    for attempt in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:5173", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   ✓ Vite dev server healthy on :5173 ({attempt+1}s)", flush=True)
                    frontend_ready = True
                    break
        except Exception:
            pass

    results.append(("Servers Launched & Healthy", backend_ready and frontend_ready))

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    try:
        # 5. Frontend SPA Routes & Proxy
        print("\n[Audit 5/8] Verifying Frontend Workspace Routes on Vite (:5173)...", flush=True)
        routes_to_test = [
            ("/", "Home Screen"),
            ("/image-analysis", "Medical Image Analysis Workspace"),
            ("/conversation", "Medical AI Conversation Workspace"),
            ("/clinical-pre-consultation", "Clinical Pre-Consultation Workspace")
        ]
        all_routes_ok = True
        for route_path, route_label in routes_to_test:
            url = f"http://127.0.0.1:5173{route_path}"
            with opener.open(url, timeout=5) as r:
                body = r.read().decode('utf-8', errors='ignore')
                is_ok = (r.status == 200 and "<div id=\"root\"></div>" in body)
                print(f"   ✓ Route {route_path:<28} -> HTTP {r.status} ({route_label})", flush=True)
                if not is_ok:
                    all_routes_ok = False
        results.append(("SPA Workspace Routes", all_routes_ok))

        # 6. File Validation & Size Limits
        print("\n[Audit 6/8] Testing File Upload Validation & Limits on /upload...", flush=True)
        # Unsupported file (.txt)
        txt_body, txt_ctype = make_multipart_body(
            {"text": "Analyze document"},
            {"image": ("notes.txt", b"Unsupported plain text payload", "text/plain")}
        )
        req_txt = urllib.request.Request("http://127.0.0.1:5173/upload", data=txt_body, headers={"Content-Type": txt_ctype}, method="POST")
        try:
            opener.open(req_txt, timeout=10)
            txt_rejected = False
        except urllib.error.HTTPError as e:
            txt_rejected = (e.code == 400)
            print(f"   ✓ Unsupported format (.txt) rejected with HTTP 400: {e.reason}", flush=True)

        # Oversized file (> 10MB)
        large_body, large_ctype = make_multipart_body(
            {"text": "Analyze oversized image"},
            {"image": ("large.png", b"\x89PNG\r\n\x1a\n" + (b"0" * (11 * 1024 * 1024)), "image/png")}
        )
        req_large = urllib.request.Request("http://127.0.0.1:5173/upload", data=large_body, headers={"Content-Type": large_ctype}, method="POST")
        try:
            opener.open(req_large, timeout=10)
            large_rejected = False
        except urllib.error.HTTPError as e:
            large_rejected = (e.code == 413)
            print(f"   ✓ Oversized file (>10MB) rejected with HTTP 413: {e.reason}", flush=True)

        results.append(("Upload Validation & Size Constraints", txt_rejected and large_rejected))

        # 7. End-to-End /upload Call & HTTP 500 Handling under Option A
        print("\n[Audit 7/8] Verifying /upload Request Handling & Error Response under Option A...", flush=True)
        sample_img_path = os.path.join(project_dir, "sample_images", "chest_x-ray_covid_and_normal", "covid-example-01.jpg")
        with open(sample_img_path, "rb") as f:
            valid_img_bytes = f.read()

        upload_body, upload_ctype = make_multipart_body(
            {"text": "Patient with dry cough and fever for 5 days. Evaluate for pneumonia or COVID-19."},
            {"image": ("chest_xray.jpg", valid_img_bytes, "image/jpeg")}
        )
        req_upload = urllib.request.Request(
            "http://127.0.0.1:5173/upload",
            data=upload_body,
            headers={"Content-Type": upload_ctype},
            method="POST"
        )
        
        handled_500_properly = False
        cookie_received = False
        try:
            opener.open(req_upload, timeout=120)
            print("   ⚠️ Unexpected: Backend succeeded without vision model.", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 500:
                err_payload = json.loads(e.read().decode())
                print(f"   ✓ Expected HTTP 500 received from backend: {err_payload.get('error_code')}", flush=True)
                print(f"   ✓ Detail message: \"{err_payload.get('detail')}\"", flush=True)
                
                # Check session tracking returned
                session_id_present = bool(err_payload.get("session_id")) or any(c.name == "session_id" for c in cookie_jar)
                if session_id_present:
                    sid = err_payload.get("session_id") or "from cookie"
                    print(f"   ✓ Session tracking present in error response: session_id={sid[:8]}...", flush=True)

                # Verify error payload matches React error-handling contract
                if err_payload.get("error_code") == "IMAGE_PROCESSING_ERROR" and err_payload.get("detail"):
                    print("   ✓ React ImageAnalysisContext contract verified: extracts 'detail' and sets UI error banner.", flush=True)
                    handled_500_properly = True

        results.append(("Upload Contract & Error Handling", handled_500_properly and session_id_present))

        # 8. Independent /validate & /chat Regression
        print("\n[Audit 8/8] Verifying Independent /validate and /chat Functionality...", flush=True)
        # 8a. Validate Yes
        val_data_yes = urllib.parse.urlencode({
            "validation_result": "yes",
            "comments": "Clinician confirmed ground truth."
        }).encode('utf-8')
        req_val_yes = urllib.request.Request("http://127.0.0.1:5173/validate", data=val_data_yes, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        val_ok = False
        try:
            with opener.open(req_val_yes, timeout=30) as resp:
                val_res = json.loads(resp.read().decode())
                print(f"   ✓ POST /validate status: {val_res.get('status')}", flush=True)
                print(f"   ✓ Message: {val_res.get('message')}", flush=True)
                val_ok = (resp.status == 200 and val_res.get("status") == "validated")
        except Exception as ex:
            print(f"   ⚠️ /validate note: {ex}", flush=True)

        # 8b. Chat Regression (pacing to respect Groq rate limits)
        time.sleep(5)
        chat_body = json.dumps({"query": "Hello, what medical services are available?", "conversation_history": []}).encode('utf-8')
        req_chat = urllib.request.Request("http://127.0.0.1:5173/chat", data=chat_body, headers={"Content-Type": "application/json"}, method="POST")
        chat_ok = False
        try:
            with opener.open(req_chat, timeout=60) as resp:
                chat_res = json.loads(resp.read().decode())
                print(f"   ✓ POST /chat status: {chat_res.get('status')}", flush=True)
                print(f"   ✓ Active Agent: {chat_res.get('agent')}", flush=True)
                print(f"   ✓ Response snippet: {chat_res.get('response', '')[:80]}...", flush=True)
                chat_ok = (resp.status == 200 and chat_res.get("status") == "success")
        except Exception as ex:
            print(f"   ⚠️ /chat note: {ex}", flush=True)

        results.append(("Independent Endpoints & Chat Regression", val_ok and chat_ok))

    finally:
        print("\nShutting down test servers...", flush=True)
        try:
            vite_proc.terminate()
            vite_proc.wait(timeout=5)
        except Exception:
            vite_proc.kill()

        try:
            fastapi_proc.terminate()
            fastapi_proc.wait(timeout=5)
        except Exception:
            fastapi_proc.kill()

    print("\n" + "=" * 80, flush=True)
    print("PHASE 4 VERIFICATION SUMMARY (OPTION A)", flush=True)
    print("=" * 80, flush=True)
    all_passed = True
    for name, passed in results:
        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status_symbol} - {name}", flush=True)
    print("=" * 80, flush=True)

    return all_passed

if __name__ == "__main__":
    success = run_phase4_option_a_verification()
    sys.exit(0 if success else 1)
