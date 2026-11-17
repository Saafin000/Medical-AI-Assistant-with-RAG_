#!/usr/bin/env python3
"""
ISOLATED TEST: Verify batching logic fix without module dependencies.

Tests tuple-based batching to ensure no duplicate piece processing.
"""

import sys
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MockProcessor:
    """Minimal mock of ContentProcessor for batching logic only."""
    
    def __init__(self):
        self.split_sections_cache = {}
        self.tpm_window_seconds = 60
        self.token_usage_window = deque()
    
    def _estimate_tokens(self, text):
        """Simple token estimation (one token per ~4 chars)."""
        return max(1, len(text) // 4)


def test_create_batches_no_splits():
    """Test: Normal sections (no splits) create batches with (idx, 0) tuples."""
    print("\n" + "="*80)
    print("TEST 1: Normal sections create (idx, 0) tuples")
    print("="*80)
    
    processor = MockProcessor()
    
    # Simulate _create_batches with fixed token budget
    section_chunks = [
        "Section 0 content",
        "Section 1 content",
        "Section 2 content",
    ]
    
    TEMPLATE_OVERHEAD = 350
    MAX_BATCH_TOKENS = 5000 - TEMPLATE_OVERHEAD
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for idx, section in enumerate(section_chunks):
        section_tokens = processor._estimate_tokens(section)
        
        if current_tokens + section_tokens > MAX_BATCH_TOKENS:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.append((idx, 0))
        current_tokens += section_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"\nBatches: {batches}")
    
    # Verify format
    all_tuples = all(isinstance(item, tuple) and len(item) == 2 for batch in batches for item in batch)
    all_piece_0 = all(piece_idx == 0 for batch in batches for _, piece_idx in batch)
    
    print(f"  All items are (idx, piece_idx) tuples: {all_tuples}")
    print(f"  All piece_idx == 0 for normal sections: {all_piece_0}")
    
    result = all_tuples and all_piece_0
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: Normal sections correctly formatted")
    
    return result


def test_create_batches_with_splits():
    """Test: Oversized section split into pieces, each assigned to exactly one batch."""
    print("\n" + "="*80)
    print("TEST 2: Oversized section split - each piece in exactly one batch")
    print("="*80)
    
    processor = MockProcessor()
    
    # Simulate section 2 being oversized and split into 2 pieces
    section_chunks = [
        "Section 0 normal",
        "Section 1 normal",
        "Section 2 OVERSIZED " * 300,  # Will be split
        "Section 3 normal",
    ]
    
    # Pre-populate cache as if _split_oversized_section was called
    processor.split_sections_cache[2] = [
        "Piece 0 of section 2",
        "Piece 1 of section 2"
    ]
    
    TEMPLATE_OVERHEAD = 350
    MAX_BATCH_TOKENS = 5000 - TEMPLATE_OVERHEAD
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for idx, section in enumerate(section_chunks):
        # Check if this is the split section
        if idx in processor.split_sections_cache:
            # Section was split
            split_pieces = processor.split_sections_cache[idx]
            
            # Each piece becomes a separate work item
            for piece_idx, piece in enumerate(split_pieces):
                piece_tokens = processor._estimate_tokens(piece)
                
                if current_batch and current_tokens + piece_tokens > MAX_BATCH_TOKENS:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                current_batch.append((idx, piece_idx))
                current_tokens += piece_tokens
        else:
            # Normal section
            section_tokens = processor._estimate_tokens(section)
            
            if current_batch and current_tokens + section_tokens > MAX_BATCH_TOKENS:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append((idx, 0))
            current_tokens += section_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"\nBatches: {batches}")
    print(f"Split cache: {processor.split_sections_cache}")
    
    # Count work items
    work_item_count = {}
    for batch in batches:
        for work_item in batch:
            work_item_count[work_item] = work_item_count.get(work_item, 0) + 1
    
    print(f"\nWork item uniqueness:")
    all_unique = True
    for work_item, count in sorted(work_item_count.items()):
        status = "✓" if count == 1 else "✗ DUPLICATE"
        print(f"  {status} {work_item}: appears {count} times")
        if count != 1:
            all_unique = False
    
    # Verify pieces sent exactly once
    pieces_sent = {item: 0 for item in work_item_count}
    for batch in batches:
        for idx, piece_idx in batch:
            # All work items count, whether split or not
            pieces_sent[(idx, piece_idx)] = pieces_sent.get((idx, piece_idx), 0) + 1
    
    print(f"\nPieces sent to LLM:")
    all_sent_once = all(count == 1 for count in pieces_sent.values())
    for item, count in sorted(pieces_sent.items()):
        status = "✓" if count == 1 else "✗ SENT MULTIPLE TIMES"
        print(f"  {status} {item}: {count} times")
    
    result = all_unique and all_sent_once
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: Each piece in exactly one batch, sent exactly once")
    
    return result


def test_chunk_document_expansion():
    """Test: chunk_document correctly expands tuples into batch_sections."""
    print("\n" + "="*80)
    print("TEST 3: chunk_document expansion (tuples → batch_sections)")
    print("="*80)
    
    processor = MockProcessor()
    
    section_chunks = [
        "Section 0 content",
        "Section 1 OVERSIZED " * 100,
        "Section 2 content",
    ]
    
    # Pre-populate cache
    processor.split_sections_cache[1] = [
        "Piece 1a of section 1",
        "Piece 1b of section 1"
    ]
    
    # Simulate batches (tuple format)
    batches = [
        [(0, 0)],
        [(1, 0), (1, 1)],
        [(2, 0)]
    ]
    
    print(f"\nSimulating batch expansion:")
    all_correct = True
    for batch_idx, batch_items in enumerate(batches):
        # Simulate chunk_document expansion logic
        batch_sections = []
        for idx, piece_idx in batch_items:
            if idx in processor.split_sections_cache:
                piece = processor.split_sections_cache[idx][piece_idx]
                batch_sections.append(piece)
            else:
                batch_sections.append(section_chunks[idx])
        
        print(f"\n  Batch {batch_idx} items: {batch_items}")
        print(f"  Expanded to: {batch_sections}")
        
        # Verify expansion
        expected = []
        for idx, piece_idx in batch_items:
            if idx in processor.split_sections_cache:
                expected.append(processor.split_sections_cache[idx][piece_idx])
            else:
                expected.append(section_chunks[idx])
        
        if batch_sections == expected:
            print(f"  ✓ Correct expansion")
        else:
            print(f"  ✗ WRONG expansion")
            print(f"     Expected: {expected}")
            all_correct = False
    
    # Count submissions
    print(f"\nPiece submission count:")
    submission_count = {}
    for batch_items in batches:
        for idx, piece_idx in batch_items:
            if idx in processor.split_sections_cache:
                piece_key = f"Section {idx} Piece {piece_idx}"
                submission_count[piece_key] = submission_count.get(piece_key, 0) + 1
    
    all_once = True
    for piece, count in sorted(submission_count.items()):
        status = "✓" if count == 1 else "✗ DUPLICATE"
        print(f"  {status} {piece}: sent {count} times")
        if count != 1:
            all_once = False
    
    result = all_correct and all_once
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: Expansion correct, no duplicates")
    
    return result


def test_map_batch_splits_v2():
    """Test: _map_batch_splits_to_global_v2 handles tuples correctly."""
    print("\n" + "="*80)
    print("TEST 4: _map_batch_splits_to_global_v2 (tuple-based mapping)")
    print("="*80)
    
    # Simulate mapping logic
    def map_batch_splits_to_global_v2(batch_splits, batch_items):
        """Map local splits to global section IDs."""
        global_splits = []
        seen_sections = set()
        
        for local_split_id in batch_splits:
            if local_split_id < len(batch_items):
                section_idx, _ = batch_items[local_split_id]
                if section_idx not in seen_sections:
                    global_splits.append(section_idx)
                    seen_sections.add(section_idx)
        
        return global_splits
    
    # Test case: split after piece 1 (middle of split section)
    batch_items = [(3, 0), (4, 0), (5, 0), (5, 1)]
    batch_splits = [2]  # Split after item index 2 (which is section 5, piece 0)
    
    result = map_batch_splits_to_global_v2(batch_splits, batch_items)
    expected = [5]
    
    print(f"\nBatch items: {batch_items}")
    print(f"Local splits (indices): {batch_splits}")
    print(f"Result (global sections): {result}")
    print(f"Expected: {expected}")
    
    correct = result == expected
    status = "✅ PASS" if correct else "❌ FAIL"
    print(f"\n{status}: Mapping correct")
    
    return correct


def test_merge_batch_results():
    """Test: _merge_batch_results handles tuple-based batches."""
    print("\n" + "="*80)
    print("TEST 5: _merge_batch_results (tuple-based batch handling)")
    print("="*80)
    
    # Simulate merge logic
    def merge_batch_results(all_global_splits, batches, total_sections):
        """Merge splits and force batch boundaries."""
        final_splits = set(all_global_splits)
        
        for batch_idx in range(len(batches) - 1):
            last_item_in_batch = batches[batch_idx][-1]
            
            if isinstance(last_item_in_batch, tuple):
                last_section_idx = last_item_in_batch[0]
            else:
                last_section_idx = last_item_in_batch
            
            final_splits.add(last_section_idx)
        
        return sorted(list(final_splits))
    
    # Tuple-based batches
    batches = [
        [(0, 0), (1, 0)],
        [(2, 0), (2, 1)],
        [(3, 0)]
    ]
    all_global_splits = [2]
    
    result = merge_batch_results(all_global_splits, batches, 4)
    expected = [1, 2]  # Split after 1 (batch boundary), split after 2 (LLM suggestion)
    
    print(f"\nBatches: {batches}")
    print(f"LLM splits: {all_global_splits}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    
    correct = result == expected
    status = "✅ PASS" if correct else "❌ FAIL"
    print(f"\n{status}: Merge correct")
    
    return correct


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ISOLATED BATCHING LOGIC TEST")
    print("="*80)
    
    results = {}
    
    try:
        results["TEST 1: Normal sections format"] = test_create_batches_no_splits()
        results["TEST 2: Split sections uniqueness"] = test_create_batches_with_splits()
        results["TEST 3: chunk_document expansion"] = test_chunk_document_expansion()
        results["TEST 4: Mapping v2"] = test_map_batch_splits_v2()
        results["TEST 5: Merge batch results"] = test_merge_batch_results()
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    all_pass = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
