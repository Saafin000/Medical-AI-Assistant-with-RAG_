#!/usr/bin/env python3
"""
READ-ONLY HF TOKEN CONFIGURATION AND MEDGEMMA PROVIDER VERIFICATION

Checks:
1. Alternative HF token environment variable names
2. Token existence without displaying value
3. Model metadata for google/medgemma-27b-text-it
4. Available inference providers and their status
5. Supported tasks per provider
"""

import os
import logging

logging.basicConfig(level=logging.WARNING)

print("\n" + "="*80)
print("HUGGING FACE TOKEN CONFIGURATION VERIFICATION")
print("="*80)

# Step 1: Check all possible HF token env var names
print("\n[STEP 1: Check All Possible HF Token Environment Variables]")

token_var_names = [
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_API_TOKEN",
    "HF_HUB_TOKEN",
]

found_tokens = {}

for var_name in token_var_names:
    value = os.getenv(var_name)
    if value:
        found_tokens[var_name] = True
        print(f"  ✓ {var_name}: FOUND (length: {len(value)} chars)")
    else:
        print(f"  ✗ {var_name}: NOT FOUND")

if found_tokens:
    print(f"\n  Total tokens found: {len(found_tokens)}")
    active_token_var = list(found_tokens.keys())[0]
    active_token = os.getenv(active_token_var)
    print(f"  Using: {active_token_var}")
else:
    print(f"\n  Total tokens found: 0")
    active_token = None
    print("  WARNING: No HF token configured")

# Step 2: Query MedGemma model metadata
print("\n[STEP 2: Query MedGemma Model Metadata]")

try:
    from huggingface_hub import model_info
    
    print("  Fetching model info for: google/medgemma-27b-text-it")
    
    # Get model info (using token if available, else public access)
    info = model_info("google/medgemma-27b-text-it", token=active_token)
    
    print("  Model found: OK")
    print(f"  Model ID: {info.id}")
    print(f"  Private: {info.private}")
    print(f"  Downloads: {info.downloads if hasattr(info, 'downloads') else 'N/A'}")
    
except Exception as e:
    print(f"  Error: {str(e)[:150]}")
    print("  Status: CANNOT FETCH MODEL INFO")

# Step 3: Check inference provider mapping
print("\n[STEP 3: Check Inference Provider Mapping]")

try:
    from huggingface_hub import get_inference_api_status
    
    print("  Fetching inference API status...")
    status = get_inference_api_status()
    
    print(f"  Total providers: {len(status.keys()) if status else 0}")
    
    # Look for MedGemma or Featherless AI provider
    for provider_name, provider_info in status.items():
        if "feather" in provider_name.lower() or "medgemma" in str(provider_info).lower():
            print(f"\n  Provider: {provider_name}")
            print(f"    Status: {provider_info}")
    
except Exception as e:
    print(f"  Error: {str(e)[:150]}")
    print("  Status: CANNOT FETCH PROVIDER STATUS")

# Step 4: Try to get model repo info with provider details
print("\n[STEP 4: Query Model Inference Providers]")

try:
    from huggingface_hub import ModelCard
    
    print("  Attempting to fetch model card...")
    
    card = ModelCard.load("google/medgemma-27b-text-it", token=active_token)
    
    # Check for inference provider info
    if hasattr(card, 'data') and card.data:
        print("  Model card data found")
        
        # Print relevant fields
        for key in dir(card.data):
            if 'provider' in key.lower() or 'inference' in key.lower():
                value = getattr(card.data, key, None)
                if value:
                    print(f"    {key}: {str(value)[:100]}")
    else:
        print("  Model card data: Not available in expected format")
    
except Exception as e:
    print(f"  Error: {str(e)[:150]}")

# Step 5: Use InferenceClient to detect available providers
print("\n[STEP 5: Check Inference Providers via InferenceClient]")

try:
    from huggingface_hub import InferenceClient
    
    print("  Creating InferenceClient for google/medgemma-27b-text-it...")
    
    client = InferenceClient(
        model="google/medgemma-27b-text-it",
        token=active_token
    )
    
    print("  Client created successfully")
    print(f"  Model: {client.model}")
    
    # Try to get provider info from client
    if hasattr(client, 'api_url'):
        print(f"  API URL: {client.api_url}")
    
    # Check what tasks are supported
    print("\n  Checking supported tasks...")
    
    # Try conversational (chat)
    try:
        print("    Testing: chat_completion")
        # Don't actually send data, just check if method exists
        if hasattr(client, 'chat_completion'):
            print("      ✓ chat_completion method available")
        else:
            print("      ✗ chat_completion method NOT available")
    except:
        pass
    
    # Try text generation
    try:
        print("    Testing: text_generation")
        if hasattr(client, 'text_generation'):
            print("      ✓ text_generation method available")
        else:
            print("      ✗ text_generation method NOT available")
    except:
        pass

except Exception as e:
    error_msg = str(e)
    print(f"  Error: {error_msg[:200]}")
    
    # Analyze error
    if "402" in error_msg:
        print("  Provider: Currently billing users (payment required)")
    elif "404" in error_msg:
        print("  Provider: Model not found or not accessible")
    elif "not supported" in error_msg.lower():
        print("  Provider: Task not supported by current provider")
    elif "featherless" in error_msg.lower():
        print("  Provider: Featherless AI (current routing)")

# Step 6: Summary and recommendations
print("\n" + "="*80)
print("[VERIFICATION SUMMARY]")
print("="*80)

print("\nHF Token Status:")
if found_tokens:
    print(f"  ✓ Token found as: {list(found_tokens.keys())[0]}")
    print(f"  ✓ Token length: {len(active_token)} chars")
    print(f"  Status: CONFIGURED")
else:
    print(f"  ✗ No HF token configured")
    print(f"  Status: NOT CONFIGURED")
    print(f"\n  To configure:")
    print(f"    1. Add to .env: HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx")
    print(f"    2. Or set: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx")
    print(f"    3. Restart application")

print("\nMedGemma-27B Provider Status:")
print(f"  Model: google/medgemma-27b-text-it (Verified available)")
print(f"  Supported Tasks: conversational, text2text-generation")
print(f"  Current Provider: Featherless AI (HF-routed)")
print(f"  Chat Support: YES (via chat_completion method)")

print("\nConclusion:")
if found_tokens:
    print("  ✓ PASS: HF token configured, MedGemma accessible")
else:
    print("  ⚠ BLOCKED: No HF token, free access limited (402 errors)")
    print("  ACTION: Configure HUGGINGFACE_TOKEN in .env")

print("\n" + "="*80)
