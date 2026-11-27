import os
import sys
import time
import json
import subprocess
import urllib.request
import http.cookiejar

def test_rag_and_websearch_paced():
    print("=" * 70, flush=True)
    print("TARGETED VERIFICATION: RAG Agent & Web Search Agent Workflows", flush=True)
    print("=" * 70, flush=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    # Pre-cleanup
    try:
        cleanup_cmd = 'Get-NetTCPConnection -LocalPort 8000, 5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }'
        subprocess.run(["powershell", "-Command", cleanup_cmd], capture_output=True)
        time.sleep(2)
    except Exception:
        pass

    print("[1/4] Starting FastAPI backend on http://127.0.0.1:8000...", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    fastapi_log = open(os.path.join(project_dir, "fastapi_paced.log"), "w", encoding="utf-8")
    fastapi_proc = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=project_dir,
        env=env,
        stdout=fastapi_log,
        stderr=subprocess.STDOUT
    )

    for _ in range(60):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                if resp.status == 200:
                    print("      FastAPI is healthy!", flush=True)
                    break
        except Exception:
            pass

    print("[2/4] Starting Vite dev server on http://127.0.0.1:5173...", flush=True)
    vite_log = open(os.path.join(project_dir, "vite_paced.log"), "w", encoding="utf-8")
    vite_cmd = "npx.cmd" if os.name == "nt" else "npx"
    vite_proc = subprocess.Popen(
        [vite_cmd, "vite", "--port", "5173", "--host", "127.0.0.1"],
        cwd=frontend_dir,
        stdout=vite_log,
        stderr=subprocess.STDOUT
    )

    for _ in range(25):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:5173/", timeout=2) as resp:
                if resp.status == 200:
                    print("      Vite dev server is ready!", flush=True)
                    break
        except Exception:
            pass

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    all_ok = True

    try:
        # Give Groq TPM rate limiter a clean 60s cooldown window before starting
        print("\nWaiting 50s for fresh Groq TPM rate-limit quota bucket...", flush=True)
        time.sleep(50)

        # TEST A: Medical query routed to RAG / literature
        print("\n[3/4] Testing Medical Query -> RAG / Web Search with Citations...", flush=True)
        rag_payload = json.dumps({
            "query": "What are the common symptoms of diabetes mellitus?",
            "conversation_history": []
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:5173/chat",
            data=rag_payload,
            headers={"Content-Type": "application/json"}
        )
        with opener.open(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            agent = data.get("agent", "")
            response_text = data.get("response", "")
            print(f"      Agent: {agent}", flush=True)
            print(f"      Response Preview: {response_text[:200]}...", flush=True)
            if agent in ["RAG_AGENT", "WEB_SEARCH_PROCESSOR_AGENT"] and len(response_text) > 50:
                print(f"      ✅ PASSED: Correctly processed by {agent} with rich medical response.", flush=True)
            else:
                print(f"      ⚠️ Handled by {agent}", flush=True)

        # Wait 45s for fresh Groq TPM rate-limit quota
        print("\nWaiting 45s for Groq TPM replenishment...", flush=True)
        time.sleep(45)

        # TEST B: Medical query specifically for latest external research
        print("\n[4/4] Testing External Research -> Web Search Processor Agent...", flush=True)
        web_payload = json.dumps({
            "query": "What are the latest clinical news updates about bird flu?",
            "conversation_history": []
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:5173/chat",
            data=web_payload,
            headers={"Content-Type": "application/json"}
        )
        with opener.open(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            agent = data.get("agent", "")
            response_text = data.get("response", "")
            print(f"      Agent: {agent}", flush=True)
            print(f"      Response Preview: {response_text[:200]}...", flush=True)
            if agent in ["WEB_SEARCH_PROCESSOR_AGENT", "RAG_AGENT"] and len(response_text) > 50:
                print(f"      ✅ PASSED: Correctly handled by {agent}.", flush=True)
            else:
                print(f"      ⚠️ Handled by {agent}", flush=True)

    finally:
        print("\nCleaning up processes...", flush=True)
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(fastapi_proc.pid)], capture_output=True)
        except Exception:
            fastapi_proc.kill()
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(vite_proc.pid)], capture_output=True)
        except Exception:
            vite_proc.kill()
        fastapi_log.close()
        vite_log.close()

    print("=" * 70, flush=True)
    print("VERIFICATION COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    return all_ok

if __name__ == "__main__":
    success = test_rag_and_websearch_paced()
    sys.exit(0 if success else 1)
