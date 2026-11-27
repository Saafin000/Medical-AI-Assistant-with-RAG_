"""
Phase 7 Unified Test Runner and Verification Suite
Executes all automated test suites and outputs structured metrics.
Loads models once to achieve ultra-fast execution of all 83 tests across 7 suites.
"""

import os
import sys
import time
import unittest
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_dummy_test_key_for_testing_1234567890")

if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

TEST_SUITES = [
    ("Session Storage Core (SQLite & Migrations)", "tests/test_session_storage.py"),
    ("Session Resolution & Format Validation", "tests/test_session_resolution.py"),
    ("Session Lifecycle & Persistence REST API", "tests/test_session_lifecycle_api.py"),
    ("Session End-to-End Isolation", "tests/test_session_e2e_isolation.py"),
    ("Multi-Agent Orchestration & Routing Logic", "tests/test_multi_agent_orchestration.py"),
    ("FastAPI Chat Endpoints & Provider Failure Handling", "test_fastapi_chat_endpoints.py"),
    ("Clinical Report & Session Isolation (Tests A-F)", "tests/test_clinical_report_isolation.py"),
    ("Workspace History & Sidebar UX (Tests A-G)", "tests/test_workspace_history_sidebar.py"),
    ("Phase 7 Comprehensive E2E Workflows & Integration", "tests/test_phase7_e2e_workflows.py"),
]

def load_suite_from_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return unittest.TestLoader().loadTestsFromModule(module)

def main():
    print("=" * 88)
    print("PHASE 7: FULL SYSTEM INTEGRATION TEST SUITE EXECUTION")
    print("=" * 88)
    
    overall_start = time.time()
    results = []
    
    for name, rel_path in TEST_SUITES:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        print(f"\n[RUNNING] {name} ({rel_path})...")
        
        if not os.path.exists(full_path):
            print(f"ERROR: Suite file not found: {full_path}")
            results.append((name, rel_path, 0, 0, 1, 0.0, "FILE_NOT_FOUND"))
            continue
            
        suite_start = time.time()
        suite = load_suite_from_file(full_path)
        runner = unittest.TextTestRunner(verbosity=1)
        res = runner.run(suite)
        elapsed = time.time() - suite_start
        
        total = res.testsRun
        failed = len(res.failures) + len(res.errors)
        passed = total - failed
        status = "PASSED" if res.wasSuccessful() else "FAILED"
        
        results.append((name, rel_path, total, passed, failed, elapsed, status))
        print(f"[RESULT] {name}: {total} tests ran, {passed} passed, {failed} failed in {elapsed:.3f}s -> {status}")

    total_time = time.time() - overall_start
    total_run = sum(r[2] for r in results)
    total_passed = sum(r[3] for r in results)
    total_failed = sum(r[4] for r in results)

    print("\n" + "=" * 88)
    print("FINAL TEST EXECUTION SUMMARY")
    print("=" * 88)
    print(f"{'Suite Name':<48} | {'Tests':<6} | {'Passed':<6} | {'Failed':<6} | {'Time (s)':<8} | {'Status':<7}")
    print("-" * 88)
    for name, path, run, p, f, elapsed, status in results:
        print(f"{name:<48} | {run:<6} | {p:<6} | {f:<6} | {elapsed:<8.3f} | {status:<7}")
    print("-" * 88)
    print(f"{'TOTAL':<48} | {total_run:<6} | {total_passed:<6} | {total_failed:<6} | {total_time:<8.3f} | {'PASSED' if total_failed == 0 else 'FAILED'}")
    print("=" * 88)

    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
