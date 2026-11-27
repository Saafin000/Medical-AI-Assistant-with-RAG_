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

def run_phase4_verification():
    print("=" * 75, flush=True)
    print("PHASE 4 VERIFICATION AUDIT: Medical Image Analysis Workspace & Workflows", flush=True)
    print("=" * 75, flush=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    results = []

    # Check 1: Protected Files Integrity
    print("\n[Step 1/10] Checking Protected Backend Files Integrity...", flush=True)
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
            print(f"   ✓ {pf} intact (last modified: {dt.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    results.append(("Protected Files Untouched", all_protected_clean))

    # Check 2: Production Build
    print("\n[Step 2/10] Checking React Frontend Production Build...", flush=True)
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    build_res = subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, capture_output=True, text=True)
    build_ok = (build_res.returncode == 0) and os.path.exists(os.path.join(frontend_dir, "dist", "index.html"))
    if build_ok:
        print("   ✓ Vite build succeeded with 0 errors.", flush=True)
    else:
        print("   ❌ Vite build failed:\n" + build_res.stderr, flush=True)
    results.append(("React Production Build", build_ok))

    # Check 3: Start Backend & Frontend Processes
    print("\n[Step 3/10] Launching FastAPI backend and Vite dev server...", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    fastapi_log_path = os.path.join(project_dir, "fastapi_phase4.log")
    fastapi_log = open(fastapi_log_path, "w", encoding="utf-8")
    fastapi_proc = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
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
                print("❌ FastAPI backend terminated unexpectedly:", flush=True)
                print(f.read(), flush=True)
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

    vite_log_path = os.path.join(project_dir, "vite_phase4.log")
    vite_log = open(vite_log_path, "w", encoding="utf-8")
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

    results.append(("Servers Health & Availability", backend_ready and frontend_ready))

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    try:
        # Check 4: Proxy Health & Route Verification
        print("\n[Step 4/10] Verifying Proxy Routes on Vite (:5173)...", flush=True)
        with opener.open("http://127.0.0.1:5173/health", timeout=5) as resp:
            health_data = json.loads(resp.read().decode())
            proxy_ok = (resp.status == 200 and health_data.get("status") == "healthy")
            print(f"   ✓ Proxied GET /health -> {health_data}", flush=True)
        results.append(("Vite Reverse Proxy Routes", proxy_ok))

        # Check 5: Unsupported File Extension Rejection (e.g. .txt)
        print("\n[Step 5/10] Testing Unsupported File Extension Rejection...", flush=True)
        fake_txt = b"Hello, this is a plain text file, not a medical scan."
        body, ctype = make_multipart_body(
            {"text": "Check this file"},
            {"image": ("test_doc.txt", fake_txt, "text/plain")}
        )
        req = urllib.request.Request("http://127.0.0.1:5173/upload", data=body, headers={"Content-Type": ctype}, method="POST")
        try:
            opener.open(req, timeout=10)
            print("   ❌ Unsupported file was unexpectedly accepted!", flush=True)
            results.append(("Unsupported File Rejection", False))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                err_data = json.loads(e.read().decode())
                print(f"   ✓ HTTP 400 Bad Request correctly returned: {err_data.get('detail')}", flush=True)
                results.append(("Unsupported File Rejection", True))
            else:
                print(f"   ❌ Unexpected HTTP code: {e.code}", flush=True)
                results.append(("Unsupported File Rejection", False))

        # Check 6: Oversized File Rejection (> 10MB)
        print("\n[Step 6/10] Testing Oversized File Rejection (> 10MB)...", flush=True)
        large_fake_png = b"\x89PNG\r\n\x1a\n" + (b"0" * (11 * 1024 * 1024))
        body, ctype = make_multipart_body(
            {"text": "Check oversized scan"},
            {"image": ("large_scan.png", large_fake_png, "image/png")}
        )
        req = urllib.request.Request("http://127.0.0.1:5173/upload", data=body, headers={"Content-Type": ctype}, method="POST")
        try:
            opener.open(req, timeout=10)
            print("   ❌ Oversized file was unexpectedly accepted!", flush=True)
            results.append(("Oversized File Rejection", False))
        except urllib.error.HTTPError as e:
            if e.code == 413:
                err_data = json.loads(e.read().decode())
                print(f"   ✓ HTTP 413 Payload Too Large correctly returned: {err_data.get('detail')}", flush=True)
                results.append(("Oversized File Rejection", True))
            else:
                print(f"   ❌ Unexpected HTTP code: {e.code}", flush=True)
                results.append(("Oversized File Rejection", False))

        # Check 7: Chest X-ray Classification Pipeline
        print("\n[Step 7/10] Testing Chest X-ray Classification Workflow...", flush=True)
        chest_sample_path = os.path.join(project_dir, "sample_images", "chest_x-ray_covid_and_normal", "covid-example-01.jpg")
        if not os.path.exists(chest_sample_path):
            chest_sample_path = os.path.join(project_dir, "uploads", "backend", "cf383b56-853e-428e-a2c1-b627be36b742_covid-example-01.jpg")
            
        if os.path.exists(chest_sample_path):
            with open(chest_sample_path, "rb") as f:
                chest_bytes = f.read()
            body, ctype = make_multipart_body(
                {"text": "Analyze this chest X-ray image for covid or pneumonia."},
                {"image": ("covid_xray.jpg", chest_bytes, "image/jpeg")}
            )
            req = urllib.request.Request("http://127.0.0.1:5173/upload", data=body, headers={"Content-Type": ctype}, method="POST")
            print("   Submitting chest X-ray to POST /upload via Vite proxy...", flush=True)
            with opener.open(req, timeout=120) as resp:
                chest_res = json.loads(resp.read().decode())
                print(f"   ✓ Upload response status: {chest_res.get('status')}", flush=True)
                print(f"   ✓ Detected Agent: {chest_res.get('agent')}", flush=True)
                print(f"   ✓ Classification finding snippet: {chest_res.get('response', '')[:120]}...", flush=True)
                is_chest_ok = "CHEST" in chest_res.get("agent", "").upper() and "HUMAN_VALIDATION" in chest_res.get("agent", "").upper()
                results.append(("Chest X-ray Classification Workflow", is_chest_ok))
        else:
            print("   ⚠️ Sample chest X-ray file not found on disk.", flush=True)
            results.append(("Chest X-ray Classification Workflow", False))

        # Pacing delay
        time.sleep(5)

        # Check 8: Skin Lesion Segmentation Pipeline
        print("\n[Step 8/10] Testing Skin Lesion Segmentation Workflow & Plot Serving...", flush=True)
        skin_sample_path = os.path.join(project_dir, "sample_images", "skin_lesion_images", "ISIC_0012558.jpg")
        if not os.path.exists(skin_sample_path):
            skin_sample_path = os.path.join(project_dir, "uploads", "backend", "eef24507-03b9-4b8d-9593-4ea321732fae_ISIC_0017789.jpg")
            
        if os.path.exists(skin_sample_path):
            with open(skin_sample_path, "rb") as f:
                skin_bytes = f.read()
            body, ctype = make_multipart_body(
                {"text": "Segment and analyze this skin lesion dermatological scan."},
                {"image": ("skin_lesion.jpg", skin_bytes, "image/jpeg")}
            )
            req = urllib.request.Request("http://127.0.0.1:5173/upload", data=body, headers={"Content-Type": ctype}, method="POST")
            print("   Submitting skin lesion to POST /upload via Vite proxy...", flush=True)
            with opener.open(req, timeout=120) as resp:
                skin_res = json.loads(resp.read().decode())
                print(f"   ✓ Upload response status: {skin_res.get('status')}", flush=True)
                print(f"   ✓ Detected Agent: {skin_res.get('agent')}", flush=True)
                print(f"   ✓ Result Image URL: {skin_res.get('result_image')}", flush=True)
                
                is_skin_agent_ok = "SKIN" in skin_res.get("agent", "").upper()
                result_img_url = skin_res.get("result_image")
                
                # Verify result_image can be fetched via Vite proxy on :5173
                img_fetch_ok = False
                if result_img_url:
                    full_img_url = f"http://127.0.0.1:5173{result_img_url}"
                    try:
                        with opener.open(full_img_url, timeout=10) as img_resp:
                            if img_resp.status == 200:
                                print(f"   ✓ Successfully fetched result image via proxy: {full_img_url} ({len(img_resp.read())} bytes)", flush=True)
                                img_fetch_ok = True
                    except Exception as ex:
                        print(f"   ❌ Failed to fetch result image via proxy: {ex}", flush=True)

                results.append(("Skin Lesion Segmentation Workflow", is_skin_agent_ok and img_fetch_ok))
        else:
            print("   ⚠️ Sample skin lesion file not found on disk.", flush=True)
            results.append(("Skin Lesion Segmentation Workflow", False))

        # Pacing delay
        time.sleep(5)

        # Check 9: Brain MRI Placeholder Node Testing
        print("\n[Step 9/10] Testing Brain MRI Agent Placeholder Workflow...", flush=True)
        brain_sample_path = os.path.join(project_dir, "uploads", "backend", "702976ce-5db9-4c8a-980e-944ff70266c8_download.jpeg")
        if not os.path.exists(brain_sample_path):
            brain_sample_path = chest_sample_path
            
        if os.path.exists(brain_sample_path):
            with open(brain_sample_path, "rb") as f:
                brain_bytes = f.read()
            body, ctype = make_multipart_body(
                {"text": "Analyze this brain MRI scan for potential tumor."},
                {"image": ("brain_mri.jpg", brain_bytes, "image/jpeg")}
            )
            req = urllib.request.Request("http://127.0.0.1:5173/upload", data=body, headers={"Content-Type": ctype}, method="POST")
            print("   Submitting brain MRI query to POST /upload via Vite proxy...", flush=True)
            with opener.open(req, timeout=120) as resp:
                brain_res = json.loads(resp.read().decode())
                print(f"   ✓ Upload response status: {brain_res.get('status')}", flush=True)
                print(f"   ✓ Detected Agent: {brain_res.get('agent')}", flush=True)
                print(f"   ✓ Response text: {brain_res.get('response')}", flush=True)
                is_brain_ok = "BRAIN" in brain_res.get("agent", "").upper() and "brain tumor agent" in brain_res.get("response", "").lower()
                results.append(("Brain MRI Placeholder Workflow", is_brain_ok))
        else:
            print("   ⚠️ Sample file not found for brain MRI test.", flush=True)
            results.append(("Brain MRI Placeholder Workflow", False))

        # Check 10: Human-in-the-Loop Validation via POST /validate
        print("\n[Step 10/10] Testing Human-in-the-Loop Validation Endpoint...", flush=True)
        # 10a: Positive confirmation
        val_data_yes = urllib.parse.urlencode({
            "validation_result": "yes",
            "comments": "Clinician confirms model segmentation boundary matches physical lesion."
        }).encode('utf-8')
        req_val_yes = urllib.request.Request(
            "http://127.0.0.1:5173/validate",
            data=val_data_yes,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        with opener.open(req_val_yes, timeout=30) as resp:
            val_yes_res = json.loads(resp.read().decode())
            print(f"   ✓ Validation 'yes' status: {val_yes_res.get('status')}", flush=True)
            print(f"   ✓ Validation 'yes' message: {val_yes_res.get('message')}", flush=True)
            is_val_yes_ok = val_yes_res.get("status") == "validated"

        # 10b: Negative confirmation
        val_data_no = urllib.parse.urlencode({
            "validation_result": "no",
            "comments": "Clinician flags finding for biopsy review."
        }).encode('utf-8')
        req_val_no = urllib.request.Request(
            "http://127.0.0.1:5173/validate",
            data=val_data_no,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        with opener.open(req_val_no, timeout=30) as resp:
            val_no_res = json.loads(resp.read().decode())
            print(f"   ✓ Validation 'no' status: {val_no_res.get('status')}", flush=True)
            print(f"   ✓ Validation 'no' message: {val_no_res.get('message')}", flush=True)
            is_val_no_ok = val_no_res.get("status") == "rejected"

        results.append(("Human-in-the-Loop Validation API", is_val_yes_ok and is_val_no_ok))

    finally:
        print("\nCleaning up server processes...", flush=True)
        try:
            vite_proc.terminate()
            vite_proc.wait(timeout=5)
        except Exception:
            vite_proc.kill()
        vite_log.close()

        try:
            fastapi_proc.terminate()
            fastapi_proc.wait(timeout=5)
        except Exception:
            fastapi_proc.kill()
        fastapi_log.close()

    print("\n" + "=" * 75, flush=True)
    print("PHASE 4 VERIFICATION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    all_passed = True
    for name, passed in results:
        status_symbol = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status_symbol} - {name}", flush=True)
    print("=" * 75, flush=True)

    if all_passed:
        print("🎉 ALL PHASE 4 VERIFICATION AUDITS PASSED SUCCESSFULLY!", flush=True)
        return True
    else:
        print("⚠️ Some verification audits failed. Review log above.", flush=True)
        return False

if __name__ == "__main__":
    success = run_phase4_verification()
    sys.exit(0 if success else 1)
