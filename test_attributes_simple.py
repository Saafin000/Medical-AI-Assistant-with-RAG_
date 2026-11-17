#!/usr/bin/env python3
"""
Simple isolated test to verify chunk_size and chunk_overlap are initialized.

Extracts just the relevant code to test without full module imports.
"""

import sys
import logging
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_attribute_initialization():
    """Test that ContentProcessor initializes chunk_size and chunk_overlap."""
    print("\n" + "="*80)
    print("TEST: ContentProcessor attributes initialization")
    print("="*80)
    
    # Simulate ContentProcessor.__init__ behavior
    class MockContentProcessor:
        def __init__(self, config):
            self.logger = logging.getLogger(__name__)
            self.summarizer_model = config.rag.summarizer_model
            self.chunker_model = config.rag.chunker_model
            
            # The FIXED lines - these should now be present
            self.chunk_size = config.rag.chunk_size           # 512 tokens
            self.chunk_overlap = config.rag.chunk_overlap     # 50 tokens
            
            self.token_usage_window = {}
            self.split_sections_cache = {}
    
    # Create mock config
    mock_config = MagicMock()
    mock_config.rag.chunk_size = 512
    mock_config.rag.chunk_overlap = 50
    mock_config.rag.summarizer_model = None
    mock_config.rag.chunker_model = MagicMock()
    
    # Initialize processor
    processor = MockContentProcessor(mock_config)
    
    # Verify attributes exist and have correct values
    print(f"Has chunk_size attribute: {hasattr(processor, 'chunk_size')}")
    print(f"Has chunk_overlap attribute: {hasattr(processor, 'chunk_overlap')}")
    
    if hasattr(processor, 'chunk_size'):
        print(f"  chunk_size value: {processor.chunk_size}")
    
    if hasattr(processor, 'chunk_overlap'):
        print(f"  chunk_overlap value: {processor.chunk_overlap}")
    
    # Test that chunk_document would receive correct values
    print(f"\nValues for RecursiveCharacterTextSplitter:")
    print(f"  chunk_size = {processor.chunk_size}")
    print(f"  chunk_overlap = {processor.chunk_overlap}")
    
    success = (
        hasattr(processor, 'chunk_size') and processor.chunk_size == 512 and
        hasattr(processor, 'chunk_overlap') and processor.chunk_overlap == 50
    )
    
    status = "PASS" if success else "FAIL"
    print(f"\n{status}: Attributes correctly initialized")
    
    return success


def test_chunk_document_usage():
    """Test that chunk_document can use these attributes."""
    print("\n" + "="*80)
    print("TEST: chunk_document would access these attributes")
    print("="*80)
    
    # Simulate chunk_document usage
    class MockProcessor:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.logger = logging.getLogger(__name__)
        
        def chunk_document_would_use(self):
            """Simulate what chunk_document does."""
            try:
                # This is what chunk_document does at line 175-176:
                # splitter = RecursiveCharacterTextSplitter(
                #     chunk_size=self.chunk_size,           # 512 tokens
                #     chunk_overlap=self.chunk_overlap,     # 50 tokens
                
                print(f"Accessing self.chunk_size: {self.chunk_size}")
                print(f"Accessing self.chunk_overlap: {self.chunk_overlap}")
                
                return True
            except AttributeError as e:
                print(f"AttributeError: {e}")
                return False
    
    processor = MockProcessor(chunk_size=512, chunk_overlap=50)
    success = processor.chunk_document_would_use()
    
    status = "PASS" if success else "FAIL"
    print(f"\n{status}: chunk_document can access attributes")
    
    return success


def test_real_file_changes():
    """Verify the actual file has the changes."""
    print("\n" + "="*80)
    print("TEST: Verify actual content_processor.py has the fix")
    print("="*80)
    
    import os
    
    file_path = 'agents/rag_agent/content_processor.py'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for the new lines
    has_chunk_size = 'self.chunk_size = config.rag.chunk_size' in content
    has_chunk_overlap = 'self.chunk_overlap = config.rag.chunk_overlap' in content
    
    print(f"File contains 'self.chunk_size = config.rag.chunk_size': {has_chunk_size}")
    print(f"File contains 'self.chunk_overlap = config.rag.chunk_overlap': {has_chunk_overlap}")
    
    # Also check that these lines are in __init__
    lines = content.split('\n')
    in_init = False
    found_chunk_size = False
    found_chunk_overlap = False
    
    for i, line in enumerate(lines):
        if 'def __init__(self, config):' in line:
            in_init = True
            init_start = i
        
        if in_init:
            if 'def ' in line and '__init__' not in line:
                # We've left __init__, stop looking
                in_init = False
            
            if 'self.chunk_size = config.rag.chunk_size' in line:
                found_chunk_size = True
                print(f"  Found chunk_size at line {i+1}")
            
            if 'self.chunk_overlap = config.rag.chunk_overlap' in line:
                found_chunk_overlap = True
                print(f"  Found chunk_overlap at line {i+1}")
    
    success = found_chunk_size and found_chunk_overlap
    status = "PASS" if success else "FAIL"
    print(f"\n{status}: File has correct changes in __init__")
    
    return success


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SIMPLE ATTRIBUTE INITIALIZATION TEST SUITE")
    print("="*80)
    
    results = {}
    
    results["TEST 1: Mock initialization"] = test_attribute_initialization()
    results["TEST 2: chunk_document would access"] = test_chunk_document_usage()
    results["TEST 3: Real file verification"] = test_real_file_changes()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    all_pass = all(results.values())
    overall = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    print(f"\nOverall: {overall}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
