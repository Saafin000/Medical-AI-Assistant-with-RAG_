#!/usr/bin/env python3
"""
SAFE TWO-PDF INGESTION SCRIPT WITH ERROR HANDLING

Ingest only:
  1. Diagnosis and Management of Type 2 Diabetes.pdf
  2. Hypertension.pdf

Features:
  - Clear logging of progress
  - Error detection and reporting
  - Chunk count validation
  - Processing time tracking
  - Partial success handling
  
Usage:
  .\venv\Scripts\python.exe ingest_new_pdfs.py
"""

import sys
import logging
from datetime import datetime
from agents.rag_agent import MedicalRAG
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_with_validation(rag, filepath):
    """
    Ingest a single PDF file with validation and error handling.
    
    Args:
        rag: MedicalRAG instance
        filepath: Path to PDF file
        
    Returns:
        dict: Result with success status and details
    """
    print(f"\n{'='*80}")
    print(f"Ingesting: {filepath}")
    print(f"{'='*80}")
    
    try:
        result = rag.ingest_file(filepath)
        
        if result["success"]:
            chunks = result.get("chunks_processed", 0)
            time_taken = result.get("processing_time", 0)
            
            print(f"\n✓ SUCCESS")
            print(f"  Chunks processed: {chunks}")
            print(f"  Processing time: {time_taken:.2f} seconds")
            print(f"  Document ingested: YES")
            
            # Validate chunk count
            if chunks < 10:
                print(f"  ⚠ WARNING: Very few chunks ({chunks}). Check content extraction.")
            elif chunks > 200:
                print(f"  ⚠ WARNING: Many chunks ({chunks}). Consider reviewing chunk size settings.")
            else:
                print(f"  ✓ Chunk count reasonable ({chunks} chunks)")
            
            return {
                "success": True,
                "filepath": filepath,
                "chunks": chunks,
                "time": time_taken
            }
        else:
            error = result.get("error", "Unknown error")
            time_taken = result.get("processing_time", 0)
            
            print(f"\n✗ FAILED")
            print(f"  Error: {error}")
            print(f"  Processing time: {time_taken:.2f} seconds")
            
            return {
                "success": False,
                "filepath": filepath,
                "error": error,
                "time": time_taken
            }
    
    except Exception as e:
        print(f"\n✗ EXCEPTION")
        print(f"  Exception: {str(e)}")
        
        return {
            "success": False,
            "filepath": filepath,
            "error": f"Exception: {str(e)}",
            "exception": True
        }

def main():
    """Main ingestion workflow."""
    
    print("\n" + "="*80)
    print("MEDICAL PDF INGESTION SCRIPT")
    print("="*80)
    print(f"Start Time: {datetime.now().isoformat()}")
    
    # Initialize RAG
    print("\n[STEP 1: Initialize RAG System]")
    try:
        rag = MedicalRAG(Config())
        print("  ✓ MedicalRAG initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize RAG: {e}")
        sys.exit(1)
    
    # Define files to ingest
    print("\n[STEP 2: Define Files to Ingest]")
    files = [
        'data/documents/Diagnosis and Management of Type 2 Diabetes.pdf',
        'data/documents/Hypertension.pdf'
    ]
    
    print(f"  Files to ingest: {len(files)}")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {f}")
    
    # Ingest files
    print("\n[STEP 3: Ingest Files]")
    
    results = []
    total_chunks = 0
    
    for filepath in files:
        result = ingest_with_validation(rag, filepath)
        results.append(result)
        
        if result["success"]:
            total_chunks += result["chunks"]
        else:
            print(f"\nStopping ingestion due to error in: {filepath}")
            print("Note: Previously ingested files remain in the system")
            break
    
    # Summary
    print("\n" + "="*80)
    print("[INGESTION SUMMARY]")
    print("="*80)
    
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    print(f"\nResults: {successful} successful, {failed} failed")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result["success"] else "✗"
        print(f"\n  {i}. {status} {result['filepath']}")
        
        if result["success"]:
            print(f"     Chunks: {result['chunks']}")
            print(f"     Time: {result['time']:.2f}s")
        else:
            print(f"     Error: {result['error'][:80]}")
            if result.get("exception"):
                print(f"     Type: Exception")
            print(f"     Time: {result['time']:.2f}s")
    
    print(f"\nTotal chunks ingested: {total_chunks}")
    print(f"Total processing time: {sum(r['time'] for r in results):.2f}s")
    
    if successful == len(files):
        print("\n✓ SUCCESS: All files ingested successfully")
        
        print("\n[NEXT STEPS]")
        print("  1. Run verification:")
        print("     .\venv\Scripts\python.exe verify_qdrant_sqlite_readonly.py")
        print("  2. Expected collection size: 112-162 points (up from 32)")
        print("  3. Test RAG with queries about diabetes and hypertension")
        
        return 0
    elif successful > 0:
        print(f"\n⚠ PARTIAL SUCCESS: {successful} of {len(files)} files ingested")
        print("  Earlier files are successfully stored")
        print("  You may re-run this script to retry the failed file")
        
        return 1
    else:
        print("\n✗ FAILURE: No files were ingested")
        print("  Check errors above and try again")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        print("Note: Partially ingested data remains in the system")
        exit_code = 130
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        exit_code = 2
    
    print(f"\nEnd Time: {datetime.now().isoformat()}")
    print("="*80 + "\n")
    
    sys.exit(exit_code)
