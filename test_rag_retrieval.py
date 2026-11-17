#!/usr/bin/env python3
"""
READ-ONLY RAG Retrieval Test

Tests the existing RAG pipeline with already-ingested documents.
Does NOT ingest, does NOT modify Qdrant or database.
"""

import sys
import logging
import traceback
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG components
try:
    logger.info(f"Importing RAG components...")
    from config import Config
    from agents.rag_agent import MedicalRAG
    logger.info("RAG components imported successfully")
    
    # Initialize config
    logger.info("Initializing configuration...")
    config = Config()
    logger.info("Configuration initialized")
except Exception as e:
    logger.error(f"Failed to import RAG components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize RAG system
try:
    logger.info("Initializing MedicalRAG system...")
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
    "What information does the document provide about brain tumor diagnosis?",
]

print("\n" + "="*80)
print("RAG RETRIEVAL TEST - USING VENV")
print("="*80)

print(f"\nPython Executable: {sys.executable}")
print(f"Environment: Project Virtual Environment")

# Test each query
for query_idx, query in enumerate(test_queries, 1):
    print("\n" + "="*80)
    print(f"QUERY {query_idx}: {query}")
    print("="*80)
    
    try:
        # Step 1: Load vectorstore and docstore
        logger.info(f"Loading vectorstore and docstore for retrieval...")
        vectorstore, docstore = rag_system.vector_store.load_vectorstore()
        logger.info("Vectorstore and docstore loaded")
        
        # Step 2: Expand query
        logger.info(f"Expanding query...")
        expansion_result = rag_system.query_expander.expand_query(query)
        expanded_query = expansion_result["expanded_query"]
        logger.info(f"Original query: '{query}'")
        logger.info(f"Expanded query: '{expanded_query}'")
        print(f"\n[QUERY EXPANSION]")
        print(f"  Original:  {query}")
        print(f"  Expanded:  {expanded_query}")
        
        # Step 3: Dense and Sparse Retrieval
        logger.info(f"Performing dense and sparse retrieval...")
        retrieved_documents = rag_system.vector_store.retrieve_relevant_chunks(
            query=expanded_query,
            vectorstore=vectorstore,
            docstore=docstore
        )
        logger.info(f"Retrieved {len(retrieved_documents)} chunks")
        
        print(f"\n[DENSE/SPARSE RETRIEVAL]")
        print(f"  Chunks retrieved: {len(retrieved_documents)}")
        
        if len(retrieved_documents) == 0:
            print("  NO CHUNKS RETRIEVED")
            continue
        
        # Display retrieval results
        print(f"\n[RETRIEVAL RESULTS - Top {min(3, len(retrieved_documents))} chunks]")
        for idx, doc in enumerate(retrieved_documents[:3], 1):
            print(f"\n  --- Chunk {idx} ---")
            print(f"  Source:     {doc['source']}")
            print(f"  Score:      {doc['score']:.4f}")
            print(f"  Content:    {doc['content'][:200]}..." if len(doc['content']) > 200 else f"  Content:    {doc['content']}")
        
        # Step 4: Reranking
        logger.info(f"Performing reranking...")
        try:
            if rag_system.reranker and len(retrieved_documents) > 1:
                reranked_documents, picture_paths = rag_system.reranker.rerank(
                    expanded_query,
                    retrieved_documents,
                    rag_system.parsed_content_dir
                )
                logger.info(f"✓ Reranked {len(retrieved_documents)} documents")
                print(f"\n[RERANKING]")
                print(f"  Status:      SUCCESS")
                print(f"  Reranked chunks: {len(reranked_documents)}")
                print(f"  Picture paths:   {len(picture_paths)}")
                
                # Show top reranked results
                print(f"\n[RERANKED RESULTS - Top 3]")
                for idx, doc in enumerate(reranked_documents[:3], 1):
                    score = doc.get('score', 'N/A')
                    print(f"\n  --- Reranked Chunk {idx} ---")
                    print(f"  Source:     {doc.get('source', 'N/A')}")
                    print(f"  Score:      {score}")
                    print(f"  Content:    {doc.get('content', '')[:200]}...")
            else:
                logger.info("Skipping reranking (not enough documents)")
                print(f"\n[RERANKING]")
                print(f"  Status:      SKIPPED (insufficient documents or no reranker)")
                reranked_documents = retrieved_documents
        except Exception as rerank_error:
            logger.error(f"Reranking failed: {rerank_error}")
            print(f"\n[RERANKING]")
            print(f"  Status:      FAILED")
            print(f"  Error:       {str(rerank_error)[:100]}")
            reranked_documents = retrieved_documents
        
        # Summary
        print(f"\n[SUMMARY FOR QUERY {query_idx}]")
        print(f"  Retrieved chunks:  {len(retrieved_documents)}")
        if retrieved_documents:
            print(f"  Top source:        {retrieved_documents[0]['source']}")
            print(f"  Top score:         {retrieved_documents[0]['score']:.4f}")
        else:
            print(f"  Top source:        N/A")
            print(f"  Top score:         N/A")
        print(f"  Status:            SUCCESS")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        print(f"\n[ERROR FOR QUERY {query_idx}]")
        print(f"  Status:      FAILED")
        print(f"  Error:       {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()

print("\n" + "="*80)
print("RAG RETRIEVAL TEST COMPLETE")
print("="*80)
