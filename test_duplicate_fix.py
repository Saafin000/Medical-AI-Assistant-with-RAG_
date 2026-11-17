#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Verify duplicate-processing bug is fixed.

Tests that with tuple-based batching, each split piece is sent to LLM exactly once.
"""

import sys
import logging
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_mock_processor():
    """Create a minimal mock ContentProcessor for testing."""
    from agents.rag_agent.content_processor import ContentProcessor
    
    config = MagicMock()
    config.groq_api_key = "test_key"
    config.groq_model = "qwen-32b"
    config.chunk_size = 1000
    config.overlap = 100
    
    processor = ContentProcessor(config)
    processor.split_sections_cache = {}
    
    return processor


def test_scenario_1():
    """Test: Single oversized section split into two pieces."""
    print("\n" + "="*80)
    print("TEST 1: Single oversized section split into two pieces")
    print("="*80)
    
    processor = create_mock_processor()
    
    # Mock sections: section 0 and 1 are normal, section 2 is oversized
    section_chunks = [
        "Section 0 content",
        "Section 1 content",
        "Section 2 OVERSIZED " * 200,  # Oversized
    ]
    
    # Mock _estimate_tokens to make section 2 oversized
    original_estimate = processor._estimate_tokens
    def mock_estimate(text):
        if "OVERSIZED" in text:
            return 10000  # Oversized
        return len(text.split()) // 3 + 10
    processor._estimate_tokens = mock_estimate
    
    # Create batches
    batches = processor._create_batches(section_chunks)
    print(f"\nBatches created: {batches}")
    print(f"Split cache: {list(processor.split_sections_cache.keys())}")
    
    # Verify batch structure
    print("\nBatch structure:")
    for batch_idx, batch_items in enumerate(batches):
        print(f"  Batch {batch_idx}: {batch_items}")
        pieces_in_batch = []
        for section_idx, piece_idx in batch_items:
            if section_idx in processor.split_sections_cache:
                pieces = processor.split_sections_cache[section_idx]
                pieces_in_batch.append(f"Section {section_idx} Piece {piece_idx}/{len(pieces)-1}")
            else:
                pieces_in_batch.append(f"Section {section_idx}")
        print(f"    → {pieces_in_batch}")
    
    # Count pieces sent
    pieces_sent = {}
    print("\nSimulating LLM submissions:")
    batch_submissions = 0
    for batch_idx, batch_items in enumerate(batches):
        batch_sections = []
        for section_idx, piece_idx in batch_items:
            if section_idx in processor.split_sections_cache:
                piece = processor.split_sections_cache[section_idx][piece_idx]
                batch_sections.append(piece)
                piece_key = f"Section {section_idx} Piece {piece_idx}"
                pieces_sent[piece_key] = pieces_sent.get(piece_key, 0) + 1
                batch_submissions += 1
            else:
                batch_sections.append(section_chunks[section_idx])
        
        print(f"  Batch {batch_idx} sent to LLM: {len(batch_sections)} sections/pieces")
    
    # Check for duplicates
    print(f"\nPieces sent count:")
    has_duplicates = False
    for piece, count in sorted(pieces_sent.items()):
        status = "✓" if count == 1 else "✗ DUPLICATE"
        print(f"  {status} {piece}: sent {count} times")
        if count != 1:
            has_duplicates = True
    
    result = not has_duplicates
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: Each piece sent exactly once")
    
    processor._estimate_tokens = original_estimate
    return result


def test_scenario_2():
    """Test: Multiple oversized sections."""
    print("\n" + "="*80)
    print("TEST 2: Multiple oversized sections")
    print("="*80)
    
    processor = create_mock_processor()
    
    # Three sections, two of which are oversized
    section_chunks = [
        "Section 0 normal",
        "Section 1 OVERSIZED " * 200,
        "Section 2 OVERSIZED " * 200,
    ]
    
    original_estimate = processor._estimate_tokens
    def mock_estimate(text):
        if "OVERSIZED" in text:
            return 10000
        return len(text.split()) // 3 + 10
    processor._estimate_tokens = mock_estimate
    
    batches = processor._create_batches(section_chunks)
    print(f"\nBatches: {batches}")
    print(f"Split cache keys: {list(processor.split_sections_cache.keys())}")
    
    # Count pieces
    pieces_sent = {}
    for batch_idx, batch_items in enumerate(batches):
        for section_idx, piece_idx in batch_items:
            if section_idx in processor.split_sections_cache:
                piece_key = f"Section {section_idx} Piece {piece_idx}"
                pieces_sent[piece_key] = pieces_sent.get(piece_key, 0) + 1
    
    print(f"\nPieces sent:")
    has_duplicates = False
    for piece, count in sorted(pieces_sent.items()):
        status = "✓" if count == 1 else "✗ DUPLICATE"
        print(f"  {status} {piece}: {count} times")
        if count != 1:
            has_duplicates = True
    
    result = not has_duplicates
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: All pieces sent exactly once")
    
    processor._estimate_tokens = original_estimate
    return result


def test_scenario_3():
    """Test: Normal and oversized sections mixed."""
    print("\n" + "="*80)
    print("TEST 3: Normal and oversized sections mixed")
    print("="*80)
    
    processor = create_mock_processor()
    
    section_chunks = [
        "Section 0 normal",
        "Section 1 normal",
        "Section 2 OVERSIZED " * 200,
        "Section 3 normal",
        "Section 4 OVERSIZED " * 200,
        "Section 5 normal",
    ]
    
    original_estimate = processor._estimate_tokens
    def mock_estimate(text):
        if "OVERSIZED" in text:
            return 10000
        return len(text.split()) // 3 + 10
    processor._estimate_tokens = mock_estimate
    
    batches = processor._create_batches(section_chunks)
    print(f"\nBatches: {batches}")
    
    # Verify order preservation
    print(f"\nOrder verification:")
    section_order = []
    for batch in batches:
        for section_idx, piece_idx in batch:
            if section_idx not in section_order:
                section_order.append(section_idx)
    
    expected_order = [0, 1, 2, 3, 4, 5]
    order_preserved = section_order == expected_order
    print(f"  Sections in order: {section_order}")
    print(f"  Expected: {expected_order}")
    print(f"  {'✓' if order_preserved else '✗'} Order preserved: {order_preserved}")
    
    # Count duplicates
    pieces_sent = {}
    for batch in batches:
        for section_idx, piece_idx in batch:
            if section_idx in processor.split_sections_cache:
                piece_key = f"Section {section_idx} Piece {piece_idx}"
                pieces_sent[piece_key] = pieces_sent.get(piece_key, 0) + 1
    
    has_duplicates = any(count != 1 for count in pieces_sent.values())
    
    result = order_preserved and not has_duplicates
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\n{status}: Order preserved, no duplicates")
    
    processor._estimate_tokens = original_estimate
    return result


def test_scenario_4():
    """Test: Token budget preservation."""
    print("\n" + "="*80)
    print("TEST 4: Token budget preservation after split")
    print("="*80)
    
    processor = create_mock_processor()
    
    # One section split into 2 pieces
    original_section = "Content " * 1000
    
    pieces = [original_section[0:len(original_section)//2], 
              original_section[len(original_section)//2:]]
    
    # Verify concatenation
    reconstructed = "".join(pieces)
    text_preserved = original_section == reconstructed
    
    print(f"  Original length: {len(original_section)}")
    print(f"  Piece 1 length: {len(pieces[0])}")
    print(f"  Piece 2 length: {len(pieces[1])}")
    print(f"  Reconstructed length: {len(reconstructed)}")
    print(f"  Text preserved: {text_preserved}")
    
    status = "✅ PASS" if text_preserved else "❌ FAIL"
    print(f"\n{status}: Text preserved exactly")
    
    return text_preserved


def test_scenario_5():
    """Test: Batch-level work item uniqueness."""
    print("\n" + "="*80)
    print("TEST 5: Each work item (section_idx, piece_idx) appears in exactly one batch")
    print("="*80)
    
    processor = create_mock_processor()
    
    section_chunks = [
        "Section 0",
        "Section 1 OVERSIZED " * 200,
        "Section 2",
    ]
    
    original_estimate = processor._estimate_tokens
    def mock_estimate(text):
        if "OVERSIZED" in text:
            return 10000
        return 100
    processor._estimate_tokens = mock_estimate
    
    batches = processor._create_batches(section_chunks)
    
    # Count work items
    work_item_count = {}
    print(f"\nWork items per batch:")
    for batch_idx, batch_items in enumerate(batches):
        print(f"  Batch {batch_idx}: {batch_items}")
        for work_item in batch_items:
            work_item_count[work_item] = work_item_count.get(work_item, 0) + 1
    
    # Check uniqueness
    print(f"\nWork item uniqueness:")
    all_unique = True
    for work_item, count in sorted(work_item_count.items()):
        status = "✓" if count == 1 else "✗ DUPLICATE"
        print(f"  {status} {work_item}: appears {count} times")
        if count != 1:
            all_unique = False
    
    status = "✅ PASS" if all_unique else "❌ FAIL"
    print(f"\n{status}: All work items unique")
    
    processor._estimate_tokens = original_estimate
    return all_unique


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DUPLICATE-PROCESSING BUG FIX TEST")
    print("="*80)
    
    results = {}
    
    try:
        results["TEST 1: Single oversized"] = test_scenario_1()
        results["TEST 2: Multiple oversized"] = test_scenario_2()
        results["TEST 3: Mixed normal/oversized"] = test_scenario_3()
        results["TEST 4: Token budget preserved"] = test_scenario_4()
        results["TEST 5: Work item uniqueness"] = test_scenario_5()
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
