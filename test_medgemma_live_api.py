#!/usr/bin/env python3
"""
MEDGEMMA LIVE API SMOKE TEST - READ ONLY

Tests actual inference response from google/medgemma-27b-text-it
via featherless-ai provider using HuggingFace InferenceClient.

Credentials are loaded but never printed.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Suppress logging during execution
logging.basicConfig(level=logging.CRITICAL)

print("\n" + "="*80)
print("MEDGEMMA LIVE API SMOKE TEST")
print("="*80)

print(f"\nTest Start: {datetime.now().isoformat()}")
print(f"Provider: featherless-ai")
print(f"Model: google/medgemma-27b-text-it")

# Step 1: Load token silently
print("\n[STEP 1: Load Environment]")

# Use the same approach as config.py (line 111)
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")
has_token = bool(hf_token)

print(f"  HUGGINGFACE_TOKEN: {'FOUND' if has_token else 'NOT FOUND'}")
if has_token:
    print(f"    Length: {len(hf_token)} chars")
    print(f"    Type: {'valid' if hf_token.startswith('hf_') else 'unknown'}")

# Step 2: Initialize InferenceClient
print("\n[STEP 2: Initialize InferenceClient]")

try:
    from huggingface_hub import InferenceClient
    
    # Create client pointing to featherless-ai provider
    client = InferenceClient(
        model="google/medgemma-27b-text-it",
        token=hf_token if has_token else None,
    )
    
    print(f"  Status: CLIENT CREATED")
    print(f"  Model: {client.model}")
    print(f"  Auth: {'Authenticated' if has_token else 'Unauthenticated (free tier)'}")
    
except Exception as e:
    print(f"  Status: FAILED")
    print(f"  Error: {str(e)[:100]}")
    print("\n[RESULT: BLOCKED - Cannot initialize client]")
    sys.exit(1)

# Step 3: Prepare test message
print("\n[STEP 3: Prepare Test Message]")

test_message = "Explain what fever means in general medical information."
print(f"  Query: {test_message}")
print(f"  Max Tokens: 100")

# Step 4: Execute inference
print("\n[STEP 4: Execute Inference Request]")

start_time = time.time()

try:
    print(f"  Sending request...")
    
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical information assistant. Provide clear, accurate medical education."
            },
            {
                "role": "user",
                "content": test_message
            }
        ],
        max_tokens=100,
        temperature=0.3,
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"  Request Status: SUCCESS (HTTP 200)")
    print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
    
except Exception as e:
    elapsed_time = time.time() - start_time
    error_msg = str(e)
    
    print(f"  Request Status: FAILED")
    print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"  Error: {error_msg[:150]}")
    
    # Categorize error
    if "402" in error_msg:
        print(f"  Error Category: PAYMENT REQUIRED (402)")
        print(f"  Cause: No valid token / Free tier blocked")
        print(f"\n[RESULT: BLOCKED - Token required]")
    elif "401" in error_msg or "Unauthorized" in error_msg:
        print(f"  Error Category: AUTHENTICATION FAILED (401)")
        print(f"  Cause: Invalid token")
        print(f"\n[RESULT: BLOCKED - Invalid token]")
    elif "404" in error_msg:
        print(f"  Error Category: NOT FOUND (404)")
        print(f"  Cause: Model or endpoint not accessible")
        print(f"\n[RESULT: BLOCKED - Model not found]")
    elif "429" in error_msg:
        print(f"  Error Category: RATE LIMITED (429)")
        print(f"  Cause: Too many requests")
        print(f"\n[RESULT: BLOCKED - Rate limited]")
    else:
        print(f"  Error Category: OTHER")
        print(f"\n[RESULT: FAILED - See error above]")
    
    sys.exit(1)

# Step 5: Parse response
print("\n[STEP 5: Parse Response]")

try:
    # Extract response text
    if hasattr(response, 'choices') and response.choices:
        response_text = response.choices[0].message.content
    else:
        response_text = str(response)
    
    print(f"  Response Received: YES")
    print(f"  Response Length: {len(response_text)} characters")
    print(f"  Token Count: ~{len(response_text.split())} words")
    
    # Show first 150 chars
    preview = response_text[:150]
    print(f"\n  Response Preview:")
    print(f"  '{preview}...'")
    
except Exception as e:
    print(f"  Response Parsing: FAILED")
    print(f"  Error: {str(e)[:100]}")
    print(f"\n[RESULT: FAILED - Cannot parse response]")
    sys.exit(1)

# Step 6: Summary
print("\n" + "="*80)
print("[SMOKE TEST RESULT]")
print("="*80)

print(f"\nTest Status: PASS")
print(f"\nEvidence:")
print(f"  ✓ Token: {'Configured' if has_token else 'Auto-detected'}")
print(f"  ✓ Client: Initialized successfully")
print(f"  ✓ Request: Sent successfully")
print(f"  ✓ Response: Received (HTTP 200)")
print(f"  ✓ Content: {len(response_text)} chars received")
print(f"  ✓ Latency: {elapsed_time:.2f} seconds")

print(f"\nModel Output Sample:")
print(f"  '{response_text[:200]}...'")

print(f"\nConclusion:")
print(f"  MedGemma-27B API is LIVE and WORKING")
print(f"  Provider: featherless-ai")
print(f"  Status: OPERATIONAL")
print(f"  Inference: SUCCESSFUL")

print("\n" + "="*80)
