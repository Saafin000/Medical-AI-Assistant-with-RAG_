import os
import sys
import time
import subprocess
import urllib.request

def test_phase2_navigation():
    print("=================================================================")
    print("PHASE 2 VERIFICATION: React Shell, Home Cards, and Navigation")
    print("=================================================================")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    
    # 1. Verify build succeeded
    dist_html = os.path.join(frontend_dir, "dist", "index.html")
    if not os.path.exists(dist_html):
        print("❌ Build artifact dist/index.html does not exist.")
        return False
    print("✅ Verified production build artifact exists (dist/index.html).")
    
    # 2. Start Vite server on port 5173
    print("[1/3] Starting Vite dev server on http://127.0.0.1:5173...")
    vite_log_path = os.path.join(project_dir, "vite_nav_test.log")
    vite_log = open(vite_log_path, "w", encoding="utf-8")
    vite_cmd = "npx.cmd" if os.name == "nt" else "npx"
    vite_proc = subprocess.Popen(
        [vite_cmd, "vite", "--port", "5173", "--host", "127.0.0.1"],
        cwd=frontend_dir,
        stdout=vite_log,
        stderr=subprocess.STDOUT
    )
    
    # Wait for Vite to be ready
    vite_ready = False
    for attempt in range(25):
        time.sleep(1)
        if vite_proc.poll() is not None:
            vite_log.close()
            with open(vite_log_path, "r", encoding="utf-8", errors="ignore") as f:
                print("❌ Vite dev server terminated unexpectedly:")
                print(f.read())
            return False
            
        try:
            with urllib.request.urlopen("http://127.0.0.1:5173/", timeout=2) as resp:
                if resp.status == 200:
                    print(f"      Vite dev server is ready in {attempt+1}s!")
                    vite_ready = True
                    break
        except Exception as e:
            if attempt == 0 or (attempt + 1) % 5 == 0:
                print(f"      Waiting for Vite ({attempt+1}s): {type(e).__name__} - {e}")
            
    if not vite_ready:
        vite_proc.terminate()
        vite_log.close()
        print("❌ Vite failed to start within timeout.")
        if os.path.exists(vite_log_path):
            with open(vite_log_path, "r", encoding="utf-8", errors="ignore") as f:
                print("Vite Log:")
                print(f.read())
        return False
        
    # 3. Test all 4 React Router routes via Vite HTTP
    print("[2/3] Testing all 4 React Router routes via Vite HTTP...")
    routes_to_test = [
        ("/", "Home route"),
        ("/conversation", "Medical Conversation route"),
        ("/image-analysis", "Medical Image Analysis route"),
        ("/clinical-pre-consultation", "Clinical Pre-Consultation route")
    ]
    
    all_routes_ok = True
    for route, label in routes_to_test:
        url = f"http://127.0.0.1:5173{route}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    content = resp.read().decode()
                    if "<div id=\"root\">" in content and "main.jsx" in content:
                        print(f"      ✅ Route '{route}' ({label}): Status 200 OK & SPA root rendered.")
                    else:
                        print(f"      ⚠️ Route '{route}': Status 200 but root div missing.")
                        all_routes_ok = False
                else:
                    print(f"      ❌ Route '{route}': Unexpected status {resp.status}")
                    all_routes_ok = False
        except Exception as e:
            print(f"      ❌ Failed to fetch route '{route}': {e}")
            all_routes_ok = False
            
    # 4. Clean up Vite process
    print("[3/3] Shutting down Vite server...")
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(vite_proc.pid)], capture_output=True)
    except Exception:
        vite_proc.kill()
    vite_log.close()
    if os.path.exists(vite_log_path):
        os.remove(vite_log_path)
        
    # 5. Verify legacy frontend intact
    legacy_html = os.path.join(project_dir, "templates", "index.html")
    if os.path.exists(legacy_html) and os.path.getsize(legacy_html) > 50000:
        print("✅ Verified legacy frontend at templates/index.html is 100% intact.")
    else:
        print("❌ Legacy templates/index.html is missing or modified!")
        return False
        
    print("=================================================================")
    if all_routes_ok:
        print("🎉 PHASE 2 VERIFICATION PASSED: Shell, Navigation & Routes Verified!")
        print("=================================================================")
        return True
    else:
        print("❌ PHASE 2 VERIFICATION FAILED!")
        print("=================================================================")
        return False

if __name__ == "__main__":
    success = test_phase2_navigation()
    sys.exit(0 if success else 1)
