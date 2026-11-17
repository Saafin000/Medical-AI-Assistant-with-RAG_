#!/usr/bin/env python3
"""
Test to verify ContentProcessor initializes chunk_size and chunk_overlap correctly.

This test verifies:
1. ContentProcessor receives config.rag values
2. chunk_size and chunk_overlap are stored as instance attributes
3. chunk_document() can access these attributes
4. The RecursiveCharacterTextSplitter receives correct values
"""

import sys
import logging
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_content_processor_initialization():
    """Test that ContentProcessor initializes with chunk_size and chunk_overlap."""
    print("\n" + "="*80)
    print("TEST 1: ContentProcessor initialization with config values")
    print("="*80)
    
    # Mock the config with RAG settings
    mock_config = MagicMock()
    mock_config.rag.chunk_size = 512
    mock_config.rag.chunk_overlap = 50
    mock_config.rag.summarizer_model = None
    mock_config.rag.chunker_model = MagicMock()
    
    # Import and initialize ContentProcessor
    sys.path.insert(0, 'Multi-Agent-Medical-Assistant')
    from agents.rag_agent.content_processor import ContentProcessor
    
    processor = ContentProcessor(mock_config)
    
    print(f"chunk_size attribute exists: {hasattr(processor, 'chunk_size')}")
    print(f"chunk_overlap attribute exists: {hasattr(processor, 'chunk_overlap')}")
    
    if hasattr(processor, 'chunk_size'):
        print(f"chunk_size value: {processor.chunk_size}")
    else:
        print("ERROR: chunk_size attribute NOT FOUND")
    
    if hasattr(processor, 'chunk_overlap'):
        print(f"chunk_overlap value: {processor.chunk_overlap}")
    else:
        print("ERROR: chunk_overlap attribute NOT FOUND")
    
    # Verify values
    success = (
        hasattr(processor, 'chunk_size') and processor.chunk_size == 512 and
        hasattr(processor, 'chunk_overlap') and processor.chunk_overlap == 50
    )
    
    status = "PASS" if success else "FAIL"
    print(f"\n{status}: ContentProcessor initialization")
    
    return success, processor


def test_chunk_document_with_real_config():
    """Test chunk_document() with real config."""
    print("\n" + "="*80)
    print("TEST 2: chunk_document() receives correct chunk_size and chunk_overlap")
    print("="*80)
    
    # Import real config
    sys.path.insert(0, 'Multi-Agent-Medical-Assistant')
    from config import RAGConfig
    from agents.rag_agent.content_processor import ContentProcessor
    from unittest.mock import MagicMock
    
    # Create real RAG config
    rag_config = RAGConfig()
    
    # Mock the container config
    mock_full_config = MagicMock()
    mock_full_config.rag = rag_config
    
    print(f"RAG Config chunk_size: {rag_config.chunk_size}")
    print(f"RAG Config chunk_overlap: {rag_config.chunk_overlap}")
    
    # Initialize ContentProcessor
    processor = ContentProcessor(mock_full_config)
    
    print(f"ContentProcessor chunk_size: {processor.chunk_size}")
    print(f"ContentProcessor chunk_overlap: {processor.chunk_overlap}")
    
    # Test chunk_document
    test_doc = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    
    try:
        chunks = processor.chunk_document(test_doc)
        print(f"chunk_document() returned: {len(chunks)} chunks")
        print(f"Type: {type(chunks)}")
        
        success = (
            isinstance(chunks, list) and
            all(isinstance(c, str) for c in chunks) and
            len(chunks) > 0
        )
        status = "PASS" if success else "FAIL"
        print(f"\n{status}: chunk_document() works with config values")
        
        return success
    
    except AttributeError as e:
        print(f"FAIL: AttributeError in chunk_document(): {e}")
        return False
    except Exception as e:
        print(f"FAIL: Exception in chunk_document(): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_splitter_receives_correct_params():
    """Test that RecursiveCharacterTextSplitter receives the correct parameters."""
    print("\n" + "="*80)
    print("TEST 3: RecursiveCharacterTextSplitter receives correct parameters")
    print("="*80)
    
    sys.path.insert(0, 'Multi-Agent-Medical-Assistant')
    from config import RAGConfig
    from agents.rag_agent.content_processor import ContentProcessor
    from unittest.mock import MagicMock, patch
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Create real RAG config
    rag_config = RAGConfig()
    
    # Mock the container config
    mock_full_config = MagicMock()
    mock_full_config.rag = rag_config
    
    # Initialize ContentProcessor
    processor = ContentProcessor(mock_full_config)
    
    # Create a test document
    test_doc = "Word " * 500  # Large document
    
    # Patch RecursiveCharacterTextSplitter to capture constructor args
    with patch('langchain_text_splitters.RecursiveCharacterTextSplitter') as mock_splitter_class:
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter_instance
        
        try:
            chunks = processor.chunk_document(test_doc)
        except:
            pass  # Ignore errors from mocking
        
        # Check if splitter was called with correct parameters
        if mock_splitter_class.called:
            call_kwargs = mock_splitter_class.call_args[1]
            
            print(f"RecursiveCharacterTextSplitter called: YES")
            print(f"Constructor kwargs: {call_kwargs}")
            
            chunk_size = call_kwargs.get('chunk_size')
            chunk_overlap = call_kwargs.get('chunk_overlap')
            separators = call_kwargs.get('separators')
            
            print(f"  chunk_size: {chunk_size}")
            print(f"  chunk_overlap: {chunk_overlap}")
            print(f"  separators: {separators}")
            
            success = (
                chunk_size == 512 and
                chunk_overlap == 50 and
                separators == ["\n\n", "\n", " ", ""]
            )
            
            status = "PASS" if success else "FAIL"
            print(f"\n{status}: RecursiveCharacterTextSplitter parameters correct")
            
            return success
        else:
            print("ERROR: RecursiveCharacterTextSplitter not called")
            return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CONTENTPROCESSOR ATTRIBUTE INITIALIZATION TEST SUITE")
    print("="*80)
    
    results = {}
    
    try:
        success, _ = test_content_processor_initialization()
        results["TEST 1: ContentProcessor initialization"] = success
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        results["TEST 1: ContentProcessor initialization"] = False
    
    try:
        success = test_chunk_document_with_real_config()
        results["TEST 2: chunk_document() with real config"] = success
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        results["TEST 2: chunk_document() with real config"] = False
    
    try:
        success = test_splitter_receives_correct_params()
        results["TEST 3: RecursiveCharacterTextSplitter params"] = success
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        results["TEST 3: RecursiveCharacterTextSplitter params"] = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
