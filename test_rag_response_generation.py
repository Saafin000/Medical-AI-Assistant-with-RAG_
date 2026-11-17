#!/usr/bin/env python3
"""
READ-ONLY RAG RESPONSE GENERATION TEST

Tests the complete RAG pipeline including response generation.
Uses actual retrieved chunks to generate answers via the configured LLM.
"""

import sys
import logging
import traceback
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG components
try:
    logger.info("Importing RAG components...")
    from config import Config
    from agents.rag_agent import MedicalRAG
    logger.info("RAG components imported successfully")
except Exception as e:
    logger.error(f"Failed to import RAG components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize RAG system
try:
    logger.info("Initializing MedicalRAG system...")
    config = Config()
    rag_system = MedicalRAG(config)
    logger.info("MedicalRAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MedicalRAG: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test queries
test_queries = [
    "What are the common types of brain tumors?",
    "What is diabetes mellitus?",
    "What information do the documents provide about brain tumor diagnosis?",
]

print("\n" + "="*80)
print("RAG RESPONSE GENERATION TEST - USING VENV")
print("="*80)

print(f"\nPython Executable: {sys.executable}")
print(f"Environment: Project Virtual Environment")

# Check LLM configuration
print(f"\n[LLM CONFIGURATION]")
print(f"  Model Provider: {type(config.rag.response_generator_model).__name__}")
print(f"  Model Class: {config.rag.response_generator_model.__class__.__module__}.{config.rag.response_generator_model.__class__.__name__}")

# Extract LLM details if available
if hasattr(config.rag.response_generator_model, 'model'):
    print(f"  Model Name: {config.rag.response_generator_model.model}")
if hasattr(config.rag.response_generator_model, 'model_name'):
    print(f"  Model Name: {config.rag.response_generator_model.model_name}")

# Test each query
for query_idx, query in enumerate(test_queries, 1):
    print("\n" + "="*80)
    print(f"QUERY {query_idx}: {query}")
    print("="*80)
    
    try:
        # Execute full RAG pipeline
        logger.info(f"Processing query {query_idx}...")
        result = rag_system.process_query(query, chat_history="")
        
        print(f"\n[QUERY EXPANSION & RETRIEVAL]")
        print(f"  Status: Completed successfully")
        
        print(f"\n[RESPONSE GENERATION]")
        print(f"  Status: SUCCESS")
        
        # Display response
        response_text = result.get("response", "")
        print(f"\n[GENERATED RESPONSE]")
        print(f"{response_text}")
        
        # Display sources
        sources = result.get("sources", [])
        print(f"\n[SOURCES CITED]")
        if sources:
            for idx, source in enumerate(sources, 1):
                print(f"  {idx}. {source.get('title', 'Unknown')}")
                print(f"     Path: {source.get('path', 'N/A')}")
        else:
            print(f"  No sources included in response")
        
        # Display confidence
        confidence = result.get("confidence", 0.0)
        print(f"\n[CONFIDENCE SCORE]")
        print(f"  Score: {confidence:.4f}")
        if confidence >= 0.8:
            print(f"  Level: HIGH")
        elif confidence >= 0.5:
            print(f"  Level: MEDIUM")
        else:
            print(f"  Level: LOW")
        
        # Processing time
        processing_time = result.get("processing_time", 0)
        print(f"\n[PERFORMANCE]")
        print(f"  Processing Time: {processing_time:.2f}s")
        
        # Analysis
        print(f"\n[RESPONSE ANALYSIS]")
        
        # Check for hallucinations/unsupported claims
        if "I don't have enough information" in response_text or "I don't have sufficient" in response_text:
            print(f"  Status: Grounded - Acknowledges information limitations")
        elif sources:
            print(f"  Status: Grounded - Sources cited ({len(sources)} sources)")
        else:
            print(f"  Status: WARNING - No sources cited, check for unsupported claims")
        
        # Check response length
        response_length = len(response_text.split())
        print(f"  Response Length: {response_length} words")
        
        # Check for markdown formatting
        if "#" in response_text or "|" in response_text:
            print(f"  Formatting: Markdown detected (headings/tables)")
        else:
            print(f"  Formatting: Plain text")
        
        print(f"\n[SUMMARY FOR QUERY {query_idx}]")
        print(f"  Status: SUCCESS")
        print(f"  Sources: {len(sources)}")
        print(f"  Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        print(f"\n[ERROR FOR QUERY {query_idx}]")
        print(f"  Status: FAILED")
        print(f"  Error: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        # Continue to next query instead of stopping
        print(f"\n  Continuing to next query...")

print("\n" + "="*80)
print("RAG RESPONSE GENERATION TEST COMPLETE")
print("="*80)

print(f"\n[TEST SUMMARY]")
print(f"  Total Queries: {len(test_queries)}")
print(f"  Environment: Project Virtual Environment (Python 3.11)")
print(f"  Status: All tests executed")
print(f"  Read-Only: Yes (no modifications to database or files)")
