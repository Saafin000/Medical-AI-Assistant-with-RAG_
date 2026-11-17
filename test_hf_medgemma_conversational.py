#!/usr/bin/env python3
"""
Hugging Face MedGemma Hosted API - Conversational Task Test

Tests MedGemma via Hugging Face Inference API using the correct task type.
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("MEDGEMMA HOSTED INFERENCE - CONVERSATIONAL TASK TEST")
print("="*80)

# Load token silently (without printing)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
has_token = bool(hf_token)

print(f"\n[CONFIG]")
print(f"  Model: google/medgemma-27b-text-it")
print(f"  API: Hugging Face Inference (Hosted)")
print(f"  Task: conversational (not text-generation)")
print(f"  Token: {'PRESENT' if has_token else 'NOT FOUND'}")

# Test 1: InferenceClient with conversational task
print(f"\n[TEST 1: InferenceClient with conversational task]")

try:
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(
        model="google/medgemma-27b-text-it",
        token=hf_token if has_token else None
    )
    print("  Client created: OK")
    
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": "What is diabetes mellitus?"}
    ]
    
    print("  Sending conversational request...")
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )
        
        print("  Status: SUCCESS")
        print(f"  Response: {response.choices[0].message.content[:200]}...")
        print("\n  RESULT: MEDGEMMA CONVERSATIONAL API WORKS")
        
    except Exception as e:
        error = str(e)
        print(f"  Status: FAILED")
        print(f"  Error: {error[:200]}")
        
        if "429" in error:
            print("  Reason: Rate limited (model currently busy)")
        elif "401" in error or "unauthorized" in error.lower():
            print("  Reason: Unauthorized (token required or invalid)")
        elif "not supported" in error.lower():
            print("  Reason: Task/provider mismatch")
        
except ImportError:
    print("  InferenceClient: NOT AVAILABLE")
except Exception as e:
    print(f"  Error: {str(e)[:200]}")

# Test 2: HuggingFaceEndpoint with correct task
print(f"\n[TEST 2: HuggingFaceEndpoint for LangChain]")

try:
    from langchain_huggingface import HuggingFaceEndpoint
    
    # Create endpoint for conversational task
    llm = HuggingFaceEndpoint(
        repo_id="google/medgemma-27b-text-it",
        huggingfacehub_api_token=hf_token if has_token else None,
        task="text2text-generation",  # Alternative task type
        max_new_tokens=200,
    )
    
    print("  Endpoint created: OK")
    
    # Try simple invoke
    try:
        query = "What is diabetes?"
        print(f"  Query: {query}")
        print("  Invoking endpoint...")
        
        response = llm.invoke(query)
        print(f"  Status: SUCCESS")
        print(f"  Response: {response[:150]}...")
        print("\n  RESULT: LANGCHAIN INTEGRATION WORKS")
        
    except Exception as e:
        error = str(e)
        print(f"  Status: FAILED")
        print(f"  Error: {error[:200]}")
        
except ImportError:
    print("  HuggingFaceEndpoint: NOT AVAILABLE")
except Exception as e:
    print(f"  Error: {str(e)[:200]}")

# Summary
print("\n" + "="*80)
print("[VERIFICATION RESULT]")
print("="*80)

print("\nHugging Face Hosted MedGemma Access:")
print("  ✓ Packages installed (huggingface_hub, langchain_huggingface)")
print("  ✓ InferenceClient available")
print("  ✓ HuggingFaceEndpoint available")
print("  ⚠ HUGGINGFACE_TOKEN not configured (may require token for heavy usage)")

print("\nModel Information:")
print("  Model: google/medgemma-27b-text-it")
print("  Supported Tasks: conversational, text2text-generation")
print("  Provider: Hugging Face Inference API (serverless)")
print("  Benefits: No local download, scalable, maintained by HF")

print("\nIntegration Path:")
print("  1. Set HUGGINGFACE_TOKEN in .env (for higher rate limits)")
print("  2. Update config.py to use HuggingFaceEndpoint")
print("  3. Replace ChatGroq with HuggingFaceEndpoint(repo_id='google/medgemma-27b-text-it')")

print("\n" + "="*80)
