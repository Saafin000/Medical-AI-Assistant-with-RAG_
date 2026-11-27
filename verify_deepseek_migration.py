#!/usr/bin/env python
"""Quick verification that DeepSeek migration is complete"""

import os
import sys
from pathlib import Path

print("="*80)
print("DeepSeek Migration Verification")
print("="*80)

project_root = Path(__file__).parent

# 1. Check files exist
print("\n[1] Checking file structure...")
files_to_check = [
    "deepseek_llm_provider.py",
    "provider_config_adapter.py",
    "config.py",
    ".env.example",
]

for f in files_to_check:
    path = project_root / f
    if path.exists():
        print(f"  ✅ {f} exists")
    else:
        print(f"  ❌ {f} missing!")
        sys.exit(1)

# 2. Check config.py doesn't use Groq
print("\n[2] Checking config.py for Groq/Gemini references...")
config_path = project_root / "config.py"
config_content = config_path.read_text()

issues = []
if "from langchain_groq" in config_content:
    issues.append("Still imports langchain_groq")
if "ChatGroq" in config_content:
    issues.append("Still references ChatGroq")
if "from langchain_google_genai" in config_content:
    issues.append("Still imports langchain_google_genai")
if "qwen/qwen3.8-27b" in config_content:
    issues.append("Still uses qwen model ID")

if issues:
    for issue in issues:
        print(f"  ❌ {issue}")
    sys.exit(1)
else:
    print("  ✅ No Groq/Gemini references")

# 3. Check provider_config_adapter uses DeepSeek
print("\n[3] Checking provider_config_adapter.py...")
adapter_path = project_root / "provider_config_adapter.py"
adapter_content = adapter_path.read_text()

if "DeepSeekLLMProvider" not in adapter_content:
    print("  ❌ Doesn't import DeepSeekLLMProvider")
    sys.exit(1)

if "create_deepseek_provider" not in adapter_content:
    print("  ❌ Doesn't call create_deepseek_provider")
    sys.exit(1)

if "unified_llm_provider" in adapter_content.lower() and "import" in adapter_content:
    print("  ❌ Still references unified_llm_provider")
    sys.exit(1)

print("  ✅ Uses DeepSeek provider")

# 4. Check environment documentation
print("\n[4] Checking .env.example...")
env_path = project_root / ".env.example"
env_content = env_path.read_text()

if "DEEPSEEK_API_KEY" not in env_content:
    print("  ❌ DEEPSEEK_API_KEY not documented")
    sys.exit(1)

if "DEEPSEEK_BASE_URL" not in env_content:
    print("  ❌ DEEPSEEK_BASE_URL not documented")
    sys.exit(1)

if "DEEPSEEK_MODEL" not in env_content:
    print("  ❌ DEEPSEEK_MODEL not documented")
    sys.exit(1)

print("  ✅ Environment variables documented")

# 5. Try importing
print("\n[5] Testing imports...")
try:
    from deepseek_llm_provider import DeepSeekLLMProvider, create_deepseek_provider
    print("  ✅ deepseek_llm_provider imports successfully")
except Exception as e:
    print(f"  ❌ Failed to import deepseek_llm_provider: {e}")
    sys.exit(1)

try:
    from provider_config_adapter import get_llm_provider, LLM_PROVIDER_FALLBACK_ENABLED
    print("  ✅ provider_config_adapter imports successfully")
    print(f"  ✅ LLM_PROVIDER_FALLBACK_ENABLED = {LLM_PROVIDER_FALLBACK_ENABLED}")
    if LLM_PROVIDER_FALLBACK_ENABLED:
        print("  ❌ Fallback should be disabled!")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed to import provider_config_adapter: {e}")
    sys.exit(1)

# 6. Check DEEPSEEK_API_KEY
print("\n[6] Checking environment configuration...")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_key:
    print(f"  ✅ DEEPSEEK_API_KEY is set ({len(deepseek_key)} chars)")
else:
    print("  ⚠️  DEEPSEEK_API_KEY not set")
    print("     Set it in .env or as environment variable to test API connectivity")

# Summary
print("\n" + "="*80)
print("✅ MIGRATION VERIFICATION COMPLETE")
print("="*80)
print("\nDeepSeek-only configuration is ready!")
print("\nNext steps:")
print("1. Set DEEPSEEK_API_KEY in your .env file")
print("2. Run: python test_deepseek_integration.py")
print("3. Start the application: python app.py")
print("\nConfiguration variables:")
print("- DEEPSEEK_API_KEY (required)")
print("- DEEPSEEK_BASE_URL (optional, defaults to https://api.deepseek.com)")
print("- DEEPSEEK_MODEL (optional, defaults to deepseek-flash)")
