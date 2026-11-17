#!/usr/bin/env python3
"""
Diagnose embedding environment variables
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("EMBEDDING ENVIRONMENT VARIABLES DIAGNOSTIC")
print("=" * 80)

embedding_vars = [
    "embedding_deployment_name",
    "embedding_model_name",
    "embedding_azure_endpoint",
    "embedding_openai_api_key",
    "embedding_openai_api_version",
]

print("\nChecking embedding environment variables:\n")

for var in embedding_vars:
    value = os.getenv(var)
    status = "SET" if value else "NOT SET (None)"
    print(f"  {var:35} -> {status}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check if any are None
missing = [var for var in embedding_vars if os.getenv(var) is None]

if missing:
    print(f"\n⚠ Missing environment variables: {', '.join(missing)}")
    print("\nFIX: Add these to .env file with valid Azure embedding model values:")
    for var in missing:
        print(f"  {var}=<value>")
else:
    print("\n✓ All embedding variables are set")
