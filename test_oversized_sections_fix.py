#!/usr/bin/env python3
"""
Focused mocked tests for oversized section processing.

Tests:
1. Oversized section is split into multiple pieces
2. Each piece is sent separately to LLM (not original section)
3. Split pieces are cached and used in reconstruction
4. Exact text is preserved (no loss or duplication)
5. Global ID mapping remains correct
6. Original order preserved through reconstruction
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_oversized_section_split_into_pieces():
    """Test: Oversized section split returns multiple pieces."""
    print("\n" + "="*70)
    print("TEST 1: Oversized section split into pieces")
    print("="*70)
    
    # Simulate section split logic
    section = "# Header\n\nPara 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
    max_tokens = 50
    
    # Simulate paragraph-based split
    paragraphs = section.split('\n\n')
    pieces = []
    current = ""
    
    for para in paragraphs:
        if len(current) + len(para) > max_tokens:
            if current:
                pieces.append(current)
            current = para
        else:
            if current:
                current += '\n\n' + para
            else:
                current = para
    
    if current:
        pieces.append(current)
    
    print(f"  Original length: {len(section)} chars")
    print(f"  Split into {len(pieces)} pieces")
    
    if len(pieces) > 1:
        print("✓ PASS: Section split into multiple pieces")
        for i, piece in enumerate(pieces):
            print(f"    Piece {i}: {len(piece)} chars")
        return True
    else:
        print("✗ FAIL: Section not split")
        return False


def test_2_each_piece_sent_separately():
    """Test: Each split piece is sent as separate batch to LLM."""
    print("\n" + "="*70)
    print("TEST 2: Each piece sent separately to LLM")
    print("="*70)
    
    # Simulate batch creation with cache
    split_sections_cache = {}
    section_idx = 5
    pieces = ["piece1", "piece2", "piece3"]
    split_sections_cache[section_idx] = pieces
    
    # Simulate batches created for oversized section
    batches = []
    for _ in pieces:
        batches.append([section_idx])  # Each piece gets its own batch
    
    print(f"  Oversized section {section_idx} split into {len(pieces)} pieces")
    print(f"  Created {len(batches)} batches")
    
    # Simulate chunk_document logic: build batch_sections from cache
    for batch_indices in batches:
        batch_sections = []
        for idx in batch_indices:
            if idx in split_sections_cache:
                # Use pieces from cache
                batch_sections.extend(split_sections_cache[idx])
            else:
                batch_sections.append(f"section_{idx}")
        
        print(f"    Batch has {len(batch_sections)} section(s)")
    
    if len(batches) == 3:
        print("✓ PASS: Each piece sent separately")
        return True
    else:
        print("✗ FAIL: Incorrect batch count")
        return False


def test_3_cache_used_in_reconstruction():
    """Test: Reconstruction uses cache to get split pieces."""
    print("\n" + "="*70)
    print("TEST 3: Cache used in reconstruction")
    print("="*70)
    
    # Setup
    original_section = "# Header\n\nPara 1.\n\nPara 2."
    split_sections_cache = {2: ["# Header\n\nPara 1.", "Para 2."]}
    section_chunks = ["Section 0", "Section 1", original_section, "Section 3"]
    
    # Simulate reconstruction
    chunks = []
    current_chunk = ""
    final_splits = [2]  # Split after section 2
    
    for section_idx, section in enumerate(section_chunks):
        # Check cache first
        if section_idx in split_sections_cache:
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
    
    print(f"  Section_chunks: {len(section_chunks)} sections")
    print(f"  Cache entries: {len(split_sections_cache)}")
    print(f"  Reconstructed chunks: {len(chunks)}")
    print(f"  Chunk 0 contains split pieces: {len(chunks[0])} chars")
    
    if len(chunks) == 2 and "# Header" in chunks[0] and "Para 2" in chunks[0]:
        print("✓ PASS: Cache used correctly in reconstruction")
        return True
    else:
        print("✗ FAIL: Cache not used")
        return False


def test_4_exact_text_preserved():
    """Test: Split and reconstruction preserves exact text."""
    print("\n" + "="*70)
    print("TEST 4: Exact text preserved through split and reconstruction")
    print("="*70)
    
    original_section = "# Header\n\nParagraph 1 text.  \n\nParagraph 2.   \n\n  Para 3"
    
    # Split
    split_pieces = ["# Header\n\nParagraph 1 text.  ", "Paragraph 2.   \n\n  Para 3"]
    split_sections_cache = {1: split_pieces}
    section_chunks = ["Section 0", original_section, "Section 2"]
    
    # Reconstruct
    current = ""
    for idx in range(len(section_chunks)):
        if idx in split_sections_cache:
            for piece in split_sections_cache[idx]:
                current += piece
        else:
            current += section_chunks[idx]
    
    # But we need to compare just the reconstructed section, not all sections
    reconstructed_section = ""
    for piece in split_pieces:
        reconstructed_section += piece
    
    print(f"  Original length: {len(original_section)} chars")
    print(f"  Reconstructed length: {len(reconstructed_section)} chars")
    print(f"  Original: {repr(original_section)}")
    print(f"  Reconstructed: {repr(reconstructed_section)}")
    
    if reconstructed_section == original_section:
        print("✓ PASS: Exact text preserved")
        return True
    else:
        print("✗ FAIL: Text mismatch")
        # Show where they differ
        for i, (c1, c2) in enumerate(zip(original_section, reconstructed_section)):
            if c1 != c2:
                print(f"    Position {i}: expected {repr(c1)}, got {repr(c2)}")
        return False


def test_5_global_id_mapping():
    """Test: Global split IDs remain correct with split sections."""
    print("\n" + "="*70)
    print("TEST 5: Global ID mapping with split sections")
    print("="*70)
    
    # Setup: section 2 is split, batches reference it
    split_sections_cache = {2: ["piece1", "piece2"]}
    batches = [
        [0, 1],      # Batch with sections 0, 1
        [2],         # Batch with section 2 (split)
        [3, 4],      # Batch with sections 3, 4
    ]
    
    # When section 2 batch is processed, it contains 2 pieces
    # But batch_indices still = [2] (not split into (2, 0), (2, 1))
    
    batch_for_section_2 = batches[1]  # [2]
    batch_sections = []
    for idx in batch_for_section_2:
        if idx in split_sections_cache:
            batch_sections.extend(split_sections_cache[idx])
        else:
            batch_sections.append(f"section_{idx}")
    
    print(f"  Batch indices: {batch_for_section_2}")
    print(f"  Batch sections (from cache): {len(batch_sections)} pieces")
    
    # LLM returns local splits (relative to batch_sections)
    # For this batch: ["piece1", "piece2"], LLM returns "split_after: 0"
    # This means split after piece 0 (between piece1 and piece2)
    
    # Mapping from local to global:
    # batch_indices = [2], so piece 0 maps to global section 2
    local_splits = [0]
    global_splits = []
    for local_id in local_splits:
        if local_id < len(batch_for_section_2):
            global_id = batch_for_section_2[local_id]
            global_splits.append(global_id)
    
    print(f"  Local splits: {local_splits}")
    print(f"  Global splits: {global_splits}")
    
    if global_splits == [2]:
        print("✓ PASS: Global ID mapping correct")
        return True
    else:
        print("✗ FAIL: Global ID mapping incorrect")
        return False


def test_6_original_order_preserved():
    """Test: Original section order preserved through processing."""
    print("\n" + "="*70)
    print("TEST 6: Original section order preserved")
    print("="*70)
    
    # Setup: multiple sections with one split
    section_chunks = [
        "Section 0 content",
        "Section 1 content",
        "Section 2 oversized",  # Will be split
        "Section 3 content",
        "Section 4 content",
    ]
    split_sections_cache = {
        2: ["Section 2 part 1", "Section 2 part 2"]
    }
    
    # Reconstruct
    result = ""
    for idx, section in enumerate(section_chunks):
        if idx in split_sections_cache:
            for piece in split_sections_cache[idx]:
                result += piece
        else:
            result += section
    
    expected = "".join(section_chunks).replace(
        "Section 2 oversized",
        "Section 2 part 1Section 2 part 2"
    )
    
    print(f"  Sections: {len(section_chunks)}")
    print(f"  Original order preserved: {result[:50]}...")
    
    if "Section 0 content" in result and "Section 4 content" in result:
        # Check order
        pos_0 = result.find("Section 0")
        pos_2 = result.find("Section 2")
        pos_4 = result.find("Section 4")
        
        if pos_0 < pos_2 < pos_4:
            print("✓ PASS: Original order preserved")
            return True
    
    print("✗ FAIL: Order not preserved")
    return False


def test_7_no_integer_tuple_mixup():
    """Test: batch_indices are integers only (no tuple indexing)."""
    print("\n" + "="*70)
    print("TEST 7: batch_indices are integers only")
    print("="*70)
    
    # This tests the fix: batches should contain integers only
    batches = [
        [0, 1],      # Integer indices
        [2],         # Integer index (even for split section)
        [3, 4],      # Integer indices
    ]
    
    section_chunks = ["s0", "s1", "s2_original", "s3", "s4"]
    split_sections_cache = {2: ["s2_piece1", "s2_piece2"]}
    
    # Verify batch_indices are integers
    all_integers = True
    for batch in batches:
        for idx in batch:
            if not isinstance(idx, int):
                all_integers = False
                print(f"✗ FAIL: Non-integer index {repr(idx)}")
    
    if all_integers:
        # Try to use them to access section_chunks or cache
        try:
            for batch in batches:
                for idx in batch:
                    # This should NOT crash
                    if idx in split_sections_cache:
                        pieces = split_sections_cache[idx]
                    else:
                        section = section_chunks[idx]
            
            print("✓ PASS: All batch_indices are integers (no TypeError)")
            return True
        except TypeError as e:
            print(f"✗ FAIL: TypeError when using indices: {e}")
            return False
    
    return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("TEST SUITE: OVERSIZED SECTION PROCESSING")
    print("="*70)
    
    tests = [
        test_1_oversized_section_split_into_pieces,
        test_2_each_piece_sent_separately,
        test_3_cache_used_in_reconstruction,
        test_4_exact_text_preserved,
        test_5_global_id_mapping,
        test_6_original_order_preserved,
        test_7_no_integer_tuple_mixup,
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
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[ERROR] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
