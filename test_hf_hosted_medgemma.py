#!/usr/bin/env python3
"""
READ-ONLY VERIFICATION: Hugging Face Hosted MedGemma API Access

Tests whether MedGemma-27B is accessible via Hugging Face's hosted inference API
without downloading model weights locally.
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("HUGGING FACE HOSTED MEDGEMMA API VERIFICATION")
print("="*80)

# Step 1: Check if HF token exists
print("\n[STEP 1: HuggingFace Token Status]")
hf_token_env = os.getenv("HUGGINGFACE_TOKEN")

if hf_token_env:
    print("  Status: FOUND")
    print(f"  Token Length: {len(hf_token_env)} chars")
    print(f"  Token Prefix: {hf_token_env[:10]}..." if len(hf_token_env) > 10 else "  Token: *masked*")
else:
    print("  Status: NOT FOUND")
    print("  Action: Will attempt public model access")

# Step 2: Check for HF Inference client
print("\n[STEP 2: Hugging Face Packages]")

try:
    from huggingface_hub import InferenceClient
    print("  huggingface_hub (InferenceClient): AVAILABLE")
    has_inference_client = True
except ImportError:
    print("  huggingface_hub (InferenceClient): NOT AVAILABLE")
    has_inference_client = False

try:
    from langchain_huggingface import HuggingFaceEndpoint
    print("  langchain_huggingface (HuggingFaceEndpoint): AVAILABLE")
    has_hf_endpoint = True
except ImportError:
    print("  langchain_huggingface (HuggingFaceEndpoint): NOT AVAILABLE")
    has_hf_endpoint = False

try:
    from langchain_community.llms import HuggingFaceHub
    print("  langchain_community (HuggingFaceHub): AVAILABLE")
    has_hf_hub = True
except ImportError:
    print("  langchain_community (HuggingFaceHub): NOT AVAILABLE")
    has_hf_hub = False

# Step 3: Test InferenceClient access
print("\n[STEP 3: Test MedGemma Hosted Inference Access]")

if not has_inference_client:
    print("  Status: CANNOT TEST - InferenceClient not available")
else:
    try:
        from huggingface_hub import InferenceClient
        
        # Create client (token optional for public models)
        client = InferenceClient(
            model="google/medgemma-27b-text-it",
            token=hf_token_env
        )
        print("  InferenceClient created: SUCCESS")
        
        # Test with a simple medical query
        test_prompt = "What is diabetes?"
        print(f"\n  Test Query: '{test_prompt}'")
        print("  Attempting inference...")
        
        try:
            response = client.text_generation(
                prompt=test_prompt,
                max_new_tokens=100,
                temperature=0.3,
            )
            print(f"  Response: SUCCESS")
            print(f"  Output: {response[:150]}..." if len(response) > 150 else f"  Output: {response}")
            print("\n  RESULT: MEDGEMMA HOSTED API IS ACCESSIBLE")
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Response: FAILED")
            print(f"  Error: {error_msg[:200]}")
            
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print("  Reason: Rate limited (model busy or quota exceeded)")
                print("  Status: API ACCESSIBLE but currently overloaded")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                print("  Reason: Authentication failed or token required")
                print("  Status: REQUIRES VALID TOKEN")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print("  Reason: Model not found or not accessible")
                print("  Status: MODEL NOT ACCESSIBLE")
            else:
                print("  Reason: Other error")
                print("  Status: UNKNOWN")
    
    except Exception as e:
        print(f"  Status: FAILED")
        print(f"  Error: {str(e)[:200]}")

# Step 4: Check LangChain HuggingFaceEndpoint integration
print("\n[STEP 4: LangChain HuggingFaceEndpoint Integration]")

if not has_hf_endpoint:
    print("  Status: NOT AVAILABLE")
    print("  Install: pip install langchain-huggingface")
else:
    try:
        from langchain_huggingface import HuggingFaceEndpoint
        
        print("  Status: AVAILABLE")
        print("  Class: HuggingFaceEndpoint")
        print("  Features: Text generation via HF Inference API")
        
        # Check if we can create an instance
        try:
            llm = HuggingFaceEndpoint(
                repo_id="google/medgemma-27b-text-it",
                huggingfacehub_api_token=hf_token_env if hf_token_env else None,
                max_length=100,
                temperature=0.3,
                task="text-generation",
            )
            print("  Instance Creation: SUCCESS")
            print(f"  Repo ID: google/medgemma-27b-text-it")
            print("  Ready for integration: YES")
            
        except Exception as e:
            print(f"  Instance Creation: FAILED")
            print(f"  Error: {str(e)[:150]}")
    
    except Exception as e:
        print(f"  Error checking endpoint: {str(e)}")

# Step 5: Model availability check
print("\n[STEP 5: Model Availability Status]")

try:
    from huggingface_hub import model_info
    
    print("  Checking model card...")
    info = model_info("google/medgemma-27b-text-it", token=hf_token_env)
    
    print(f"  Model: {info.model_id}")
    print(f"  Library: {info.library_name if hasattr(info, 'library_name') else 'N/A'}")
    print(f"  Tags: {', '.join(info.tags) if hasattr(info, 'tags') else 'N/A'}")
    print(f"  Status: MODEL FOUND AND ACCESSIBLE")
    
except Exception as e:
    print(f"  Status: CANNOT VERIFY")
    print(f"  Error: {str(e)[:150]}")

# Step 6: Summary
print("\n" + "="*80)
print("[SUMMARY]")
print("="*80)

print("\nHugging Face Hosted Inference API Status:")

if has_inference_client:
    print("  ✓ InferenceClient available for direct API calls")
else:
    print("  ✗ InferenceClient NOT available (install huggingface-hub)")

if has_hf_endpoint:
    print("  ✓ HuggingFaceEndpoint available for LangChain integration")
else:
    print("  ✗ HuggingFaceEndpoint NOT available (install langchain-huggingface)")

if hf_token_env:
    print("  ✓ HUGGINGFACE_TOKEN configured in .env")
else:
    print("  ⚠ HUGGINGFACE_TOKEN not found (may limit access)")

print("\nMedGemma-27B Access via Hosted Inference:")
print("  Model: google/medgemma-27b-text-it")
print("  Type: Text Generation (instruction-tuned)")
print("  Size: 27B parameters")
print("  Access: Hugging Face Inference API (serverless)")

print("\nNext Step:")
print("  To use in config.py, initialize HuggingFaceEndpoint:")
print("    from langchain_huggingface import HuggingFaceEndpoint")
print("    llm = HuggingFaceEndpoint(repo_id='google/medgemma-27b-text-it')")

print("\n" + "="*80)
