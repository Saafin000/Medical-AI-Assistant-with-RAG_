import os
import sys
import time
import subprocess
import urllib.request
import json

def test_phase1_proxy():
    print("=================================================================", flush=True)
    print("PHASE 1 VERIFICATION: FastAPI Backend + Vite Proxy Communication", flush=True)
    print("=================================================================", flush=True)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    
    # Locate correct venv python
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable
    print(f"Using Python: {python_bin}", flush=True)
    
    # Set offline HF environment variables to prevent unnecessary network timeouts
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    # Open log file to prevent Windows 4KB pipe buffer deadlock
    log_file_path = os.path.join(project_dir, "fastapi_server.log")
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    # 1. Start FastAPI backend on port 8000
    print("[1/4] Starting FastAPI backend on http://127.0.0.1:8000...", flush=True)
    fastapi_proc = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    # Wait for FastAPI to start
    backend_ready = False
    for attempt in range(60):
        time.sleep(1)
        # Check if process crashed
        if fastapi_proc.poll() is not None:
            log_file.close()
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                print("❌ FastAPI backend terminated unexpectedly:")
                print(f.read())
            return False
            
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    print(f"      Backend is ready in {attempt+1}s! /health returned: {data}", flush=True)
                    backend_ready = True
                    break
        except Exception:
            if (attempt + 1) % 5 == 0:
                print(f"      Waiting for backend models to load ({attempt+1}s)...", flush=True)
            
    if not backend_ready:
        fastapi_proc.terminate()
        log_file.close()
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            print("❌ FastAPI backend failed to start within timeout.")
            print(f.read())
        return False
        
    # 2. Start Vite dev server on port 5173
    print("[2/4] Starting Vite dev server on http://127.0.0.1:5173...", flush=True)
    vite_log_path = os.path.join(project_dir, "vite_server.log")
    vite_log = open(vite_log_path, "w", encoding="utf-8")
    vite_cmd = "npx.cmd" if os.name == "nt" else "npx"
    vite_proc = subprocess.Popen(
        [vite_cmd, "vite", "--port", "5173", "--host", "127.0.0.1"],
        cwd=frontend_dir,
        stdout=vite_log,
        stderr=subprocess.STDOUT
    )
    
    # Wait for Vite to start
    vite_ready = False
    for attempt in range(25):
        time.sleep(1)
        if vite_proc.poll() is not None:
            vite_log.close()
            with open(vite_log_path, "r", encoding="utf-8", errors="ignore") as f:
                print("❌ Vite dev server terminated unexpectedly:")
                print(f.read())
            fastapi_proc.terminate()
            return False
            
        try:
            with urllib.request.urlopen("http://127.0.0.1:5173", timeout=2) as resp:
                if resp.status == 200:
                    print(f"      Vite dev server is ready on http://127.0.0.1:5173 in {attempt+1}s!", flush=True)
                    vite_ready = True
                    break
        except Exception:
            pass
            
    if not vite_ready:
        fastapi_proc.terminate()
        vite_proc.terminate()
        vite_log.close()
        with open(vite_log_path, "r", encoding="utf-8", errors="ignore") as f:
            print("❌ Vite dev server failed to start within timeout.")
            print(f.read())
        return False
        
    # 3. Test Proxy: Query http://127.0.0.1:5173/health (Vite proxy -> FastAPI backend)
    print("[3/4] Testing Vite Proxy: Requesting http://127.0.0.1:5173/health...", flush=True)
    proxy_success = False
    try:
        req = urllib.request.Request("http://127.0.0.1:5173/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status_code = resp.status
            body = resp.read().decode()
            data = json.loads(body)
            print(f"      Proxy Response Status Code: {status_code}", flush=True)
            print(f"      Proxy Response Data: {data}", flush=True)
            if status_code == 200 and data.get("status") == "healthy":
                proxy_success = True
                print("      ✅ PROXY SUCCESS: Vite dev server successfully routed /health to FastAPI!", flush=True)
    except Exception as e:
        print(f"❌ Failed to reach backend via Vite proxy: {e}", flush=True)
        
    # Also test static React HTML serving from Vite
    try:
        req = urllib.request.Request("http://127.0.0.1:5173")
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode()
            if "Multi-Agent Medical Assistant" in html or '<div id="root">' in html:
                print("      ✅ FRONTEND SUCCESS: Vite dev server successfully served React HTML root!", flush=True)
    except Exception as e:
        print(f"⚠️ Warning on root HTML check: {e}", flush=True)
        
    # 4. Clean up processes
    print("[4/4] Shutting down background test servers...", flush=True)
    try:
        fastapi_proc.terminate()
        fastapi_proc.wait(timeout=5)
    except Exception:
        fastapi_proc.kill()
    log_file.close()
        
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(vite_proc.pid)], capture_output=True)
    except Exception:
        vite_proc.kill()
    vite_log.close()
        
    print("=================================================================", flush=True)
    if proxy_success:
        print("🎉 PHASE 1 VERIFICATION PASSED: React + Vite Proxy + Backend Connected!", flush=True)
        print("=================================================================", flush=True)
        return True
    else:
        print("❌ PHASE 1 VERIFICATION FAILED!", flush=True)
        print("=================================================================", flush=True)
        return False

if __name__ == "__main__":
    success = test_phase1_proxy()
    sys.exit(0 if success else 1)
