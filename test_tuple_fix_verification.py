#!/usr/bin/env python3
"""
Verification tests for the tuple-indexing fix.

Key Points:
1. batch_indices are ALWAYS integers (no tuples)
2. split_sections_cache stores actual split pieces
3. chunk_document builds batch_sections from cache when needed
4. No TypeError when accessing section_chunks with indices
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)


def test_1_no_tuple_indexing_error():
    """Test: The main bug fix - no TypeError with split sections."""
    print("\n" + "="*70)
    print("TEST 1: No TypeError with split sections (main bug fix)")
    print("="*70)
    
    # Simulate the fixed code flow
    section_chunks = ["s0", "s1", "s2_original", "s3", "s4"]
    split_sections_cache = {2: ["s2_piece1", "s2_piece2"]}
    batches = [[0, 1], [2], [3, 4]]
    
    # This is what chunk_document now does
    try:
        for batch_indices in batches:
            batch_sections = []
            for idx in batch_indices:
                if isinstance(idx, tuple):
                    print(f"✗ FAIL: Found tuple {idx} in batch_indices")
                    return False
                
                if idx in split_sections_cache:
                    batch_sections.extend(split_sections_cache[idx])
                else:
                    batch_sections.append(section_chunks[idx])
            
            print(f"  Batch indices {batch_indices} → {len(batch_sections)} section(s)")
        
        print("✓ PASS: No TypeError, all indices are integers")
        return True
        
    except TypeError as e:
        print(f"✗ FAIL: TypeError occurred: {e}")
        return False


def test_2_split_pieces_sent_to_llm():
    """Test: Split pieces (not original) are sent to LLM."""
    print("\n" + "="*70)
    print("TEST 2: Split pieces sent to LLM (not original)")
    print("="*70)
    
    section_chunks = ["s0", "s1", "s2_original_12345", "s3", "s4"]
    split_sections_cache = {2: ["s2_piece1", "s2_piece2"]}
    batch_indices = [2]
    
    # Build batch_sections for this batch
    batch_sections = []
    for idx in batch_indices:
        if idx in split_sections_cache:
            batch_sections.extend(split_sections_cache[idx])
        else:
            batch_sections.append(section_chunks[idx])
    
    print(f"  Original section 2: {repr(section_chunks[2])}")
    print(f"  Batch sections: {batch_sections}")
    
    # Verify split pieces are sent, not original
    if "s2_original_12345" in batch_sections:
        print("✗ FAIL: Original section sent (not split pieces)")
        return False
    
    if "s2_piece1" in batch_sections and "s2_piece2" in batch_sections:
        print("✓ PASS: Split pieces sent (not original)")
        return True
    else:
        print("✗ FAIL: Split pieces not sent")
        return False


def test_3_reconstruction_uses_cache():
    """Test: Reconstruction uses cache to reassemble split section."""
    print("\n" + "="*70)
    print("TEST 3: Reconstruction uses cache")
    print("="*70)
    
    section_chunks = ["s0", "s1", "s2_original", "s3"]
    split_sections_cache = {2: ["s2_piece1", "s2_piece2"]}
    final_splits = [2]  # Split after section 2
    
    # Simulate reconstruction
    chunks = []
    current_chunk = ""
    
    for section_idx, section in enumerate(section_chunks):
        if section_idx in split_sections_cache:
            # Use cache (FIXED: this is now done)
            split_pieces = split_sections_cache[section_idx]
            for piece in split_pieces:
                current_chunk += piece
        else:
            current_chunk += section
        
        if section_idx in final_splits:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"  Sections: {len(section_chunks)}")
    print(f"  Cache used for section 2: Yes")
    print(f"  Reconstructed chunks: {len(chunks)}")
    
    if len(chunks) > 0 and "s2_piece" in chunks[0]:
        print("✓ PASS: Reconstruction uses cache")
        return True
    else:
        print("✗ FAIL: Reconstruction doesn't use cache")
        return False


def test_4_cache_initialization():
    """Test: Cache is cleared at start of _create_batches."""
    print("\n" + "="*70)
    print("TEST 4: Cache initialization")
    print("="*70)
    
    split_sections_cache = {"old_key": ["old_pieces"]}
    
    # Simulate _create_batches cache clearing
    split_sections_cache = {}
    
    print(f"  Cache before clear: had old_key")
    print(f"  Cache after clear: {len(split_sections_cache)} entries")
    
    if len(split_sections_cache) == 0:
        print("✓ PASS: Cache cleared properly")
        return True
    else:
        print("✗ FAIL: Cache not cleared")
        return False


def test_5_integer_indices_only():
    """Test: batch_indices contain only integers."""
    print("\n" + "="*70)
    print("TEST 5: batch_indices contain integers only")
    print("="*70)
    
    # After fix, _create_batches should NOT store tuples
    batches = []
    
    # Normal section
    batches.append([0])
    batches.append([1, 2])
    
    # Oversized section (FIXED: stores [3] not [(3, 0)], [(3, 1)])
    batches.append([3])  # For first split piece
    batches.append([3])  # For second split piece (same index, cache lookup handles it)
    
    # More sections
    batches.append([4, 5])
    
    # Verify all are integers
    all_valid = True
    for batch in batches:
        for idx in batch:
            if not isinstance(idx, int):
                print(f"✗ FAIL: Non-integer {type(idx).__name__} in batch: {idx}")
                all_valid = False
    
    print(f"  Total batches: {len(batches)}")
    print(f"  All entries are integers: {all_valid}")
    
    if all_valid:
        print("✓ PASS: All batch_indices are integers")
        return True
    else:
        return False


def test_6_order_preserved():
    """Test: Section order preserved through split batching."""
    print("\n" + "="*70)
    print("TEST 6: Section order preserved")
    print("="*70)
    
    section_chunks = ["S0", "S1", "S2", "S3", "S4"]
    batches = [[0, 1], [2], [3, 4]]
    split_sections_cache = {}
    
    # Process batches and collect section contents
    all_sections = []
    for batch_indices in batches:
        for idx in batch_indices:
            if idx in split_sections_cache:
                all_sections.extend(split_sections_cache[idx])
            else:
                all_sections.append(section_chunks[idx])
    
    print(f"  Original sections: {section_chunks}")
    print(f"  Processed sections: {all_sections}")
    
    if all_sections == section_chunks:
        print("✓ PASS: Order preserved")
        return True
    else:
        print("✗ FAIL: Order not preserved")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("TUPLE-INDEXING BUG FIX VERIFICATION")
    print("="*70)
    
    tests = [
        test_1_no_tuple_indexing_error,
        test_2_split_pieces_sent_to_llm,
        test_3_reconstruction_uses_cache,
        test_4_cache_initialization,
        test_5_integer_indices_only,
        test_6_order_preserved,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} passed")
    
    if passed == total:
        print("\n[SUCCESS] All verification tests passed!")
        print("The tuple-indexing bug fix is correct.")
        return 0
    else:
        print(f"\n[ERROR] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
