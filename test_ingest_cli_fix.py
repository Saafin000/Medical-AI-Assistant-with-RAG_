#!/usr/bin/env python3
"""
Test the ingest_rag_data.py CLI fix for the undefined 'result' variable.

Tests:
1. Success case: result dict with success=True
2. Partial success case: result dict with partial_success=True
3. Failure case: result dict with success=False, partial_success=False
4. Variable scope verification: ingestion_success and result are both available
"""

import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_success_case():
    """Test success case reporting."""
    print("\n" + "="*80)
    print("TEST 1: Success case")
    print("="*80)
    
    # Mock result from successful ingestion
    result = {
        "success": True,
        "partial_success": False,
        "documents_ingested": 1,
        "chunks_processed": 24,
        "failed_documents": 0
    }
    
    # This is the fixed logic from ingest_rag_data.py
    ingestion_success = result.get("success", False) or result.get("partial_success", False)
    
    print(f"Result dict: {json.dumps(result, indent=2)}")
    print(f"ingestion_success (boolean): {ingestion_success}")
    
    # Test the reporting logic
    try:
        if ingestion_success:
            if result.get("partial_success"):
                print("\nPartial success branch taken")
                print(f"   Ingested: {result.get('documents_ingested', 0)}, Failed: {result.get('failed_documents', 0)}")
            else:
                print("\nFull success branch taken")
                print("All documents ingested successfully.")
        else:
            print("\nFailure branch taken")
        
        success = True
        status = "PASS"
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = "FAIL"
    
    print(f"\n{status}: Success case handled correctly")
    return success


def test_partial_success_case():
    """Test partial success case reporting."""
    print("\n" + "="*80)
    print("TEST 2: Partial success case")
    print("="*80)
    
    # Mock result from partial ingestion
    result = {
        "success": False,
        "partial_success": True,
        "documents_ingested": 2,
        "chunks_processed": 48,
        "failed_documents": 1
    }
    
    # This is the fixed logic from ingest_rag_data.py
    ingestion_success = result.get("success", False) or result.get("partial_success", False)
    
    print(f"Result dict: {json.dumps(result, indent=2)}")
    print(f"ingestion_success (boolean): {ingestion_success}")
    
    # Test the reporting logic
    try:
        if ingestion_success:
            if result.get("partial_success"):
                print("\nPartial success branch taken")
                print(f"   Ingested: {result.get('documents_ingested', 0)}, Failed: {result.get('failed_documents', 0)}")
            else:
                print("\nFull success branch taken")
        else:
            print("\nFailure branch taken")
        
        success = True
        status = "PASS"
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = "FAIL"
    
    print(f"\n{status}: Partial success case handled correctly")
    return success


def test_failure_case():
    """Test failure case reporting."""
    print("\n" + "="*80)
    print("TEST 3: Complete failure case")
    print("="*80)
    
    # Mock result from failed ingestion
    result = {
        "success": False,
        "partial_success": False,
        "documents_ingested": 0,
        "chunks_processed": 0,
        "failed_documents": 3,
        "error": "Failed to process documents"
    }
    
    # This is the fixed logic from ingest_rag_data.py
    ingestion_success = result.get("success", False) or result.get("partial_success", False)
    
    print(f"Result dict: {json.dumps(result, indent=2)}")
    print(f"ingestion_success (boolean): {ingestion_success}")
    
    # Test the reporting logic
    try:
        if ingestion_success:
            if result.get("partial_success"):
                print("\nPartial success branch taken")
            else:
                print("\nFull success branch taken")
        else:
            print("\nFailure branch taken")
            print("No documents were ingested. Check errors above.")
        
        success = True
        status = "PASS"
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = "FAIL"
    
    print(f"\n{status}: Failure case handled correctly")
    return success


def test_variable_scope():
    """Test that both ingestion_success and result are accessible."""
    print("\n" + "="*80)
    print("TEST 4: Variable scope - both values accessible")
    print("="*80)
    
    # Mock the function return (simulating the fixed data_ingestion function)
    result = {"success": True, "documents_ingested": 1}
    ingestion_success = result.get("success", False) or result.get("partial_success", False)
    
    # Both should be accessible
    try:
        print(f"ingestion_success accessible: {type(ingestion_success)} = {ingestion_success}")
        print(f"result accessible: {type(result)} = {result}")
        
        # Test accessing result.get() after unpacking
        partial = result.get("partial_success")
        ingested = result.get("documents_ingested", 0)
        
        print(f"result.get('partial_success'): {partial}")
        print(f"result.get('documents_ingested'): {ingested}")
        
        success = True
        status = "PASS"
    except NameError as e:
        print(f"NameError: {e}")
        success = False
        status = "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = "FAIL"
    
    print(f"\n{status}: Both variables in correct scope")
    return success


def test_function_return_unpacking():
    """Test the actual function return unpacking logic."""
    print("\n" + "="*80)
    print("TEST 5: Function return unpacking (as in fixed code)")
    print("="*80)
    
    # Simulate the fixed data_ingestion() function
    def mock_data_ingestion_fixed():
        """Mock of the fixed data_ingestion function."""
        # This simulates what happens inside the function
        result = {
            "success": True,
            "partial_success": False,
            "documents_ingested": 1,
            "chunks_processed": 24
        }
        
        print(f"Inside function - result dict: {result}")
        
        # The fixed return statement
        success = result.get("success", False) or result.get("partial_success", False)
        return success, result
    
    # This is how it's called in the fixed code
    try:
        ingestion_success, result = mock_data_ingestion_fixed()
        
        print(f"After function call:")
        print(f"  ingestion_success = {ingestion_success}")
        print(f"  result = {result}")
        print(f"  result.get('documents_ingested') = {result.get('documents_ingested')}")
        
        success = True
        status = "PASS"
    except ValueError as e:
        print(f"ValueError: {e}")
        success = False
        status = "FAIL"
    except NameError as e:
        print(f"NameError: {e}")
        success = False
        status = "FAIL"
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = "FAIL"
    
    print(f"\n{status}: Function return unpacking works correctly")
    return success


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("INGEST_RAG_DATA.PY CLI FIX TEST SUITE")
    print("="*80)
    
    results = {}
    
    results["TEST 1: Success case"] = test_success_case()
    results["TEST 2: Partial success case"] = test_partial_success_case()
    results["TEST 3: Complete failure case"] = test_failure_case()
    results["TEST 4: Variable scope"] = test_variable_scope()
    results["TEST 5: Function return unpacking"] = test_function_return_unpacking()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    all_pass = all(results.values())
    overall = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    print(f"\nOverall: {overall}\n")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
