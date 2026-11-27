"""
Diagnostic script to identify what's failing during app startup
"""
import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("\n" + "="*80)
print("STARTUP DIAGNOSTICS")
print("="*80)

# Check 1: Environment variables
print("\n[1] Checking Environment Variables...")
print("-" * 80)

required_keys = [
    "GROQ_API_KEY",
    "ELEVEN_LABS_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY"
]

for key in required_keys:
    value = os.getenv(key)
    if value:
        masked = value[:10] + "..." if len(value) > 10 else value
        print(f"  ✓ {key}: {masked}")
    else:
        print(f"  ✗ {key}: NOT SET")

# Check 2: Try importing config
print("\n[2] Loading Config Class...")
print("-" * 80)

try:
    from config import Config
    print("  ✓ Config imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import Config: {e}")
    traceback.print_exc()
    sys.exit(1)

# Check 3: Try initializing config
print("\n[3] Initializing Config...")
print("-" * 80)

try:
    print("  • Initializing Config()...")
    config = Config()
    print("  ✓ Config initialized successfully")
except Exception as e:
    print(f"  ✗ Failed to initialize Config: {e}")
    print(f"  Error type: {type(e).__name__}")
    traceback.print_exc()
    
    # Try to narrow down which component failed
    print("\n[DEBUG] Trying individual config components...")
    
    try:
        from config import APIConfig
        print("    • APIConfig: OK")
    except Exception as e:
        print(f"    • APIConfig: FAILED - {e}")
    
    try:
        from config import RAGConfig
        print("    • RAGConfig: Initializing...")
        rag = RAGConfig()
        print("    • RAGConfig: OK")
    except Exception as e:
        print(f"    • RAGConfig: FAILED - {e}")
        traceback.print_exc()
    
    try:
        from config import MedicalCVConfig
        print("    • MedicalCVConfig: OK")
    except Exception as e:
        print(f"    • MedicalCVConfig: FAILED - {e}")
    
    sys.exit(1)

# Check 4: Try initializing FastAPI app and ElevenLabs client
print("\n[4] Initializing FastAPI and ElevenLabs Client...")
print("-" * 80)

try:
    from fastapi import FastAPI
    from elevenlabs.client import ElevenLabs
    
    print("  • Creating FastAPI app...")
    app = FastAPI(title="Test", version="1.0")
    print("  ✓ FastAPI app created")
    
    print("  • Initializing ElevenLabs client...")
    el_client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)
    print("  ✓ ElevenLabs client initialized")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Check 5: Try importing agent_decision
print("\n[5] Importing Agent Decision Module...")
print("-" * 80)

try:
    print("  • Importing agent_decision...")
    from agents.agent_decision import process_query
    print("  ✓ Agent decision module imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import agent_decision: {e}")
    print(f"  Error type: {type(e).__name__}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL DIAGNOSTICS PASSED")
print("="*80)
print("\nThe server should start successfully. Try running: python app.py")
