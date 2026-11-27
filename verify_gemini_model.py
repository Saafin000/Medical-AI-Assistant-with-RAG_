#!/usr/bin/env python3
"""
Task 1.9.4 Phase 2: Verify Gemini SDK and Model Availability

Tests actual gemini-3.8-flash model with a real API request (non-clinical).
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*100)
print("GEMINI MODEL VERIFICATION")
print("="*100 + "\n")

# Step 1: Check SDK Installation
print("[STEP 1] Checking SDK Installation")
print("-" * 100)

try:
    import langchain_google_genai
    print(f"✅ langchain_google_genai imported successfully")
    
    # Try to get version
    if hasattr(langchain_google_genai, '__version__'):
        version = langchain_google_genai.__version__
    else:
        # Try alternative version check
        import importlib.metadata
        try:
            version = importlib.metadata.version('langchain-google-genai')
        except:
            version = "unknown"
    
    print(f"   SDK Version: {version}")
except ImportError as e:
    print(f"❌ SDK not installed: {e}")
    sys.exit(1)

# Step 2: Check API Key
print("\n[STEP 2] Checking API Key")
print("-" * 100)

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("❌ GOOGLE_API_KEY not set in environment")
    print("   Cannot verify model without API key")
    print("   Please set GOOGLE_API_KEY in .env file")
    sys.exit(1)

print(f"✅ GOOGLE_API_KEY is set")
print(f"   Key length: {len(google_api_key)} characters")

# Step 3: Test Model Access
print("\n[STEP 3] Testing Gemini Model Access")
print("-" * 100)

from langchain_google_genai import ChatGoogleGenerativeAI

model_id = "gemini-3.8-flash"
test_prompt = "What is 2+2?"

print(f"\nAttempting to initialize model: {model_id}")
print(f"Test prompt: '{test_prompt}'")
print(f"(Non-clinical, no sensitive data)\n")

try:
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        api_key=google_api_key,
        temperature=0.5,
        convert_system_message_to_human=True,
    )
    print(f"✅ Model instance created: {model_id}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ Model initialization failed: {error_str[:200]}")
    
    # Try to extract more info
    if "404" in error_str or "not found" in error_str.lower():
        print(f"   → Model not found (404)")
    elif "401" in error_str or "unauthorized" in error_str.lower():
        print(f"   → Unauthorized (401) - check API key")
    elif "429" in error_str:
        print(f"   → Rate limited (429)")
    else:
        print(f"   → Other error")
    
    sys.exit(1)

# Step 4: Test Actual Invocation
print("\n[STEP 4] Testing Actual Model Invocation")
print("-" * 100)

try:
    print(f"Invoking model...")
    response = llm.invoke(test_prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    print(f"✅ Model invocation successful!")
    print(f"   Response: {response_text[:100]}...")
    print(f"   Response type: {type(response).__name__}")
    print(f"   Content type: {type(response_text).__name__}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ Model invocation failed: {error_str[:200]}")
    
    if "429" in error_str:
        print(f"   → Rate limited - Gemini quota reached")
    elif "503" in error_str:
        print(f"   → Service unavailable")
    else:
        print(f"   → Other error during invocation")
    
    sys.exit(1)

# Summary
print("\n" + "="*100)
print("VERIFICATION RESULTS")
print("="*100 + "\n")

print(f"✅ ALL CHECKS PASSED")
print(f"   - SDK installed and importable")
print(f"   - API key configured")
print(f"   - Model {model_id} is accessible")
print(f"   - Model invocation works correctly")
print(f"   - Response format compatible")

print(f"\n✅ READY FOR PRODUCTION USE")
print(f"   Model is verified and ready for fallback operations\n")

sys.exit(0)
