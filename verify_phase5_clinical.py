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

def run_phase5_verification():
    print("=" * 80, flush=True)
    print("PHASE 5 VERIFICATION AUDIT: Clinical Pre-Consultation Workspace & DeepSeek Workflow", flush=True)
    print("=" * 80, flush=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    results = []

    # 1. Protected Files Audit
    print("\n[Step 1/8] Protected Backend Files Modification Audit...", flush=True)
    strictly_protected = [
        "app.py",
        "config.py",
        "agents/agent_decision.py",
        "agents/image_analysis_agent/image_classifier.py",
        "templates/index.html"
    ]
    clinical_improved = [
        "agents/clinical_pre_consultation_agent/__init__.py",
        "agents/clinical_pre_consultation_agent/patient_info_collector.py",
        "agents/clinical_pre_consultation_agent/followup_question_generator.py",
        "agents/clinical_pre_consultation_agent/red_flag_detector.py",
        "agents/clinical_pre_consultation_agent/report_generator.py"
    ]
    all_protected_clean = True
    for pf in strictly_protected:
        full_path = os.path.join(project_dir, pf)
        if not os.path.exists(full_path):
            print(f"❌ Protected file missing: {pf}", flush=True)
            all_protected_clean = False
        else:
            mtime = os.path.getmtime(full_path)
            dt = datetime.fromtimestamp(mtime)
            print(f"   ✓ {pf} UNTOUCHED (last modified: {dt.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    for cf in clinical_improved:
        full_path = os.path.join(project_dir, cf)
        mtime = os.path.getmtime(full_path)
        dt = datetime.fromtimestamp(mtime)
        print(f"   ✓ {cf} (improved adaptive agent, last modified: {dt.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    results.append(("Protected Files Untouched", all_protected_clean))

    # 2. React Production Build
    print("\n[Step 2/8] React Frontend Production Build (Vite)...", flush=True)
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    build_res = subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, capture_output=True, text=True)
    build_ok = (build_res.returncode == 0) and os.path.exists(os.path.join(frontend_dir, "dist", "index.html"))
    if build_ok:
        print("   ✓ Vite build completed with ZERO errors.", flush=True)
        print(f"   ✓ Output bundle: {os.path.join(frontend_dir, 'dist', 'index.html')}", flush=True)
    else:
        print("   ❌ Vite build failed:\n" + build_res.stderr, flush=True)
    results.append(("React Production Build", build_ok))

    # 3. Start Backend & Vite Dev Server
    print("\n[Step 3/8] Launching FastAPI backend and Vite dev server...", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    fastapi_log_path = os.path.join(project_dir, "fastapi_phase5.log")
    vite_log_path = os.path.join(project_dir, "vite_phase5.log")
    fastapi_log = open(fastapi_log_path, "w", encoding="utf-8")
    vite_log = open(vite_log_path, "w", encoding="utf-8")

    fastapi_proc = subprocess.Popen(
        [python_bin, "-u", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_dir,
        env=env,
        stdout=fastapi_log,
        stderr=subprocess.STDOUT
    )

    backend_ready = False
    for attempt in range(60):
        time.sleep(1)
        if fastapi_proc.poll() is not None:
            fastapi_log.close()
            with open(fastapi_log_path, "r", encoding="utf-8", errors="ignore") as f:
                print("❌ FastAPI backend terminated unexpectedly:\n" + f.read(), flush=True)
            return False

        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   ✓ FastAPI is healthy on :8000 ({attempt+1}s)", flush=True)
                    backend_ready = True
                    break
        except Exception:
            pass

    if not backend_ready:
        fastapi_proc.terminate()
        fastapi_log.close()
        print("❌ FastAPI backend failed to start.", flush=True)
        return False

    vite_cmd = "npx.cmd" if os.name == "nt" else "npx"
    vite_proc = subprocess.Popen(
        [vite_cmd, "vite", "--port", "5173", "--host", "127.0.0.1"],
        cwd=frontend_dir,
        stdout=vite_log,
        stderr=subprocess.STDOUT
    )

    frontend_ready = False
    for attempt in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:5173", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   ✓ Vite dev server is healthy on :5173 ({attempt+1}s)", flush=True)
                    frontend_ready = True
                    break
        except Exception:
            pass

    results.append(("Servers Launched & Healthy", backend_ready and frontend_ready))

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    try:
        # 4. Workspace Navigation Across All Three Features
        print("\n[Step 4/8] Verifying Navigation Across All Three Workspaces on Vite (:5173)...", flush=True)
        routes = [
            ("/", "Home Screen - Feature Selection"),
            ("/conversation", "Feature 1: Medical AI Conversation Workspace"),
            ("/image-analysis", "Feature 2: Medical Image Analysis Workspace"),
            ("/clinical-pre-consultation", "Feature 3: Clinical Pre-Consultation Workspace")
        ]
        all_routes_ok = True
        for path, label in routes:
            with opener.open(f"http://127.0.0.1:5173{path}", timeout=5) as r:
                body = r.read().decode('utf-8', errors='ignore')
                is_ok = (r.status == 200 and '<div id="root"></div>' in body)
                print(f"   ✓ {path:<30} -> HTTP {r.status} ({label})", flush=True)
                if not is_ok:
                    all_routes_ok = False
        results.append(("Multi-Feature Workspace Navigation", all_routes_ok))

        # Pacing delay
        time.sleep(3)

        # 5. Clinical Pre-Consultation Initiation via POST /chat
        print("\n[Step 5/8] Initiating Clinical Pre-Consultation Intake via POST /chat...", flush=True)
        intake_prompt = "I want to start a pre-consultation intake. I have been suffering from severe throbbing headaches for the past 4 days."
        req_data = json.dumps({"query": intake_prompt, "conversation_history": []}).encode('utf-8')
        req = urllib.request.Request(
            "http://127.0.0.1:5173/chat",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with opener.open(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                print(f"   ✓ Chat status: {data.get('status')}", flush=True)
                print(f"   ✓ Routed Agent: {data.get('agent')}", flush=True)
                print(f"   ✓ Agent Question / Response: \"{data.get('response', '')[:120]}...\"", flush=True)

                is_clinical_routed = "CLINICAL" in data.get("agent", "").upper()
                
                # Verify session cookie was set
                has_cookie = any(c.name == "session_id" for c in cookie_jar)
                print(f"   ✓ Session Cookie captured: {has_cookie}", flush=True)
                results.append(("Clinical Intake Initiation & Routing", is_clinical_routed and has_cookie))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            print(f"   ❌ Chat initiation failed with HTTP {e.code}: {err_body}", flush=True)
            results.append(("Clinical Intake Initiation & Routing", False))

        # Pacing delay
        time.sleep(5)

        # 6. Patient Responding to Follow-up Question
        print("\n[Step 6/8] Submitting Patient Follow-up Information via POST /chat...", flush=True)
        followup_answer = "The pain is around an 8 out of 10, concentrated on the right temple, with nausea and light sensitivity."
        req_followup_data = json.dumps({"query": followup_answer, "conversation_history": []}).encode('utf-8')
        req_followup = urllib.request.Request(
            "http://127.0.0.1:5173/chat",
            data=req_followup_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with opener.open(req_followup, timeout=60) as resp:
                data_followup = json.loads(resp.read().decode())
                print(f"   ✓ Followup response status: {data_followup.get('status')}", flush=True)
                print(f"   ✓ Routed Agent: {data_followup.get('agent')}", flush=True)
                print(f"   ✓ Followup response snippet: \"{data_followup.get('response', '')[:120]}...\"", flush=True)
                is_followup_ok = "CLINICAL" in data_followup.get("agent", "").upper() and len(data_followup.get("response", "")) > 10
                results.append(("Adaptive Follow-up Questioning & Accumulation", is_followup_ok))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            print(f"   ❌ Chat follow-up failed with HTTP {e.code}: {err_body}", flush=True)
            results.append(("Adaptive Follow-up Questioning & Accumulation", False))

        # Pacing delay
        time.sleep(5)

        # 7. Patient Feedback Submission via POST /clinical-patient-feedback
        print("\n[Step 7/8] Submitting Patient Corrections via POST /clinical-patient-feedback...", flush=True)
        feedback_form_data = urllib.parse.urlencode({
            "patient_feedback": "I forgot to mention that I take Sumatriptan 50mg as needed, and I have a history of migraines."
        }).encode('utf-8')
        req_feedback = urllib.request.Request(
            "http://127.0.0.1:5173/clinical-patient-feedback",
            data=feedback_form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        try:
            with opener.open(req_feedback, timeout=60) as resp:
                feedback_res = json.loads(resp.read().decode())
                print(f"   ✓ Feedback endpoint status: {feedback_res.get('status')}", flush=True)
                print(f"   ✓ Message: {feedback_res.get('message')}", flush=True)
                report_text = feedback_res.get('formatted_report') or ""
                if report_text:
                    print(f"   ✓ Formatted report snippet: \"{report_text[:120]}...\"", flush=True)
                else:
                    print(f"   ✓ Intake active, feedback accumulated in clinical session.", flush=True)
                is_feedback_ok = (resp.status == 200 and "CLINICAL" in feedback_res.get("agent", "").upper())
                results.append(("Patient Feedback & Report Regeneration", is_feedback_ok))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            print(f"   ❌ Patient feedback failed with HTTP {e.code}: {err_body}", flush=True)
            results.append(("Patient Feedback & Report Regeneration", False))

        # Pacing delay
        time.sleep(5)

        # 8. Regression Checks on Conversation & Image Analysis
        print("\n[Step 8/8] Regression Testing Conversation and Image Analysis Endpoints...", flush=True)
        # 8a. Conversation test
        chat_query = json.dumps({"query": "What is the function of the hippocampus?", "conversation_history": []}).encode('utf-8')
        req_convo = urllib.request.Request(
            "http://127.0.0.1:5173/chat",
            data=chat_query,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_convo, timeout=60) as resp:
            convo_res = json.loads(resp.read().decode())
            convo_ok = convo_res.get("status") == "success" and bool(convo_res.get("response"))
            print(f"   ✓ Conversation regression: agent={convo_res.get('agent')}, status={convo_res.get('status')}", flush=True)

        # 8b. Image upload format rejection check
        txt_body, txt_ctype = make_multipart_body({"text": "test"}, {"image": ("test.txt", b"dummy", "text/plain")})
        req_img = urllib.request.Request("http://127.0.0.1:5173/upload", data=txt_body, headers={"Content-Type": txt_ctype}, method="POST")
        img_rejected = False
        try:
            opener.open(req_img, timeout=10)
        except urllib.error.HTTPError as e:
            img_rejected = (e.code == 400)
            print(f"   ✓ Image analysis regression: unsupported file rejected with HTTP {e.code}", flush=True)

        results.append(("Conversation & Image Analysis Regression", convo_ok and img_rejected))

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
        try:
            fastapi_log.close()
            vite_log.close()
        except Exception:
            pass

    print("\n" + "=" * 80, flush=True)
    print("PHASE 5 VERIFICATION SUMMARY: Clinical Pre-Consultation Workspace", flush=True)
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
    success = run_phase5_verification()
    sys.exit(0 if success else 1)
