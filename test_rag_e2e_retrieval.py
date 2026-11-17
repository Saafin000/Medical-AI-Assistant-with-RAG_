#!/usr/bin/env python3
"""
READ-ONLY End-to-End Retrieval Test of Medical RAG System

Tests the complete retrieval pipeline:
1. Query expansion
2. Embedding generation (HuggingFace all-MiniLM-L6-v2)
3. Dense/Sparse retrieval from Qdrant
4. Reranking
5. Response generation
6. Source tracking

No modifications, no ingestion, READ-ONLY access only.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def test_rag_e2e_retrieval():
    """Run end-to-end retrieval tests."""
    
    print("\n" + "="*80)
    print("MEDICAL RAG SYSTEM - END-TO-END RETRIEVAL TEST")
    print("="*80)
    
    try:
        # Import RAG components
        print("\n[1/6] Importing RAG components...")
        from config import Config
        from agents.rag_agent import MedicalRAG
        
        # Initialize config and RAG
        print("[2/6] Initializing configuration...")
        config = Config()
        
        print("[3/6] Initializing MedicalRAG system...")
        rag = MedicalRAG(config)
        
        print("✓ RAG system initialized successfully")
        
        # Test queries
        test_queries = [
            "What are the common types of brain tumors?",
            "What is diabetes mellitus?",
            "What information does the document provide about brain tumor diagnosis?"
        ]
        
        results = {}
        
        for query_idx, query in enumerate(test_queries, 1):
            print("\n" + "="*80)
            print(f"QUERY {query_idx}: {query}")
            print("="*80)
            
            try:
                # Process query
                print(f"\n[4/6] Processing query through RAG pipeline...")
                response = rag.process_query(query)
                
                # Extract components
                query_result = {
                    "query": query,
                    "status": "SUCCESS",
                    "processing_time": response.get("processing_time", 0),
                    "response": response.get("response", ""),
                    "sources": response.get("sources", []),
                    "confidence": response.get("confidence", 0),
                }
                
                # Log results
                print(f"\n[5/6] Response Generated:")
                print(f"  - Status: {query_result['status']}")
                print(f"  - Processing time: {query_result['processing_time']:.2f}s")
                print(f"  - Confidence: {query_result['confidence']:.2f}")
                print(f"  - Response length: {len(query_result['response'])} chars")
                
                # Analyze sources
                sources_analysis = {
                    "total_sources": len(query_result["sources"]),
                    "source_files": []
                }
                
                if query_result["sources"]:
                    print(f"\n[6/6] Retrieved Sources:")
                    for i, source in enumerate(query_result["sources"], 1):
                        source_name = source.get("source", "unknown")
                        source_snippet = source.get("snippet", "")[:100]
                        
                        print(f"  Source {i}: {source_name}")
                        print(f"    Preview: {source_snippet}...")
                        
                        if source_name not in sources_analysis["source_files"]:
                            sources_analysis["source_files"].append(source_name)
                
                query_result["sources_analysis"] = sources_analysis
                results[f"Query_{query_idx}"] = query_result
                
            except Exception as e:
                print(f"\n✗ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                
                results[f"Query_{query_idx}"] = {
                    "query": query,
                    "status": "FAILED",
                    "error": str(e),
                    "response": None,
                    "sources": []
                }
        
        return results
        
    except Exception as e:
        print(f"\n✗ Critical error in RAG initialization: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def verify_embedding_model():
    """Verify HuggingFace embedding model."""
    
    print("\n" + "="*80)
    print("EMBEDDING MODEL VERIFICATION")
    print("="*80)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        print("\nInitializing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Test embedding
        test_text = "brain tumor diagnosis"
        embedding = embeddings.embed_query(test_text)
        
        print(f"✓ Model loaded: sentence-transformers/all-MiniLM-L6-v2")
        print(f"✓ Test query: '{test_text}'")
        print(f"✓ Embedding dimension: {len(embedding)}")
        print(f"✓ Expected dimension: 384")
        
        if len(embedding) == 384:
            print("✓ Embedding dimension matches collection (384)")
            return True
        else:
            print(f"✗ Dimension mismatch (expected 384, got {len(embedding)})")
            return False
        
    except Exception as e:
        print(f"✗ Error verifying embedding: {e}")
        return False


def verify_collection_accessibility():
    """Verify Qdrant collection is accessible."""
    
    print("\n" + "="*80)
    print("QDRANT COLLECTION ACCESSIBILITY")
    print("="*80)
    
    try:
        from config import Config
        
        config = Config()
        
        print(f"Collection name: {config.rag.collection_name}")
        print(f"Vector DB type: {config.rag.vector_db_type}")
        print(f"Distance metric: {config.rag.distance_metric}")
        print(f"Embedding dimension: {config.rag.embedding_dim}")
        print(f"Local path: {config.rag.vector_local_path}")
        
        # Verify local path exists
        local_path = Path(config.rag.vector_local_path)
        if local_path.exists():
            print(f"✓ Local Qdrant path exists")
            
            collection_path = local_path / "collection" / config.rag.collection_name
            if collection_path.exists():
                print(f"✓ Collection directory exists")
                return True
            else:
                print(f"✗ Collection directory not found")
                return False
        else:
            print(f"✗ Local Qdrant path does not exist")
            return False
        
    except Exception as e:
        print(f"✗ Error checking collection: {e}")
        return False


def summarize_results(results: Dict[str, Any]):
    """Summarize test results."""
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful_queries = 0
    failed_queries = 0
    total_sources = 0
    all_source_files = set()
    
    for query_key, result in results.items():
        if query_key == "error":
            continue
        
        if result.get("status") == "SUCCESS":
            successful_queries += 1
            total_sources += result.get("sources_analysis", {}).get("total_sources", 0)
            all_source_files.update(result.get("sources_analysis", {}).get("source_files", []))
        else:
            failed_queries += 1
    
    print(f"\nQueries Processed: {successful_queries + failed_queries}")
    print(f"  Successful: {successful_queries}")
    print(f"  Failed: {failed_queries}")
    
    print(f"\nRetrieval Statistics:")
    print(f"  Total sources retrieved: {total_sources}")
    print(f"  Unique source files: {len(all_source_files)}")
    
    if all_source_files:
        print(f"\nSource files accessed:")
        for source_file in sorted(all_source_files):
            print(f"  - {source_file}")
    
    success_rate = (successful_queries / (successful_queries + failed_queries) * 100) if (successful_queries + failed_queries) > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    # Overall status
    if successful_queries == 3 and failed_queries == 0:
        print("\n✓ ALL TESTS PASSED")
        return True
    else:
        print("\n⚠ SOME TESTS FAILED")
        return False


def main():
    """Run all tests."""
    
    print("\n" + "="*80)
    print("MEDICAL RAG SYSTEM E2E RETRIEVAL TEST SUITE")
    print("="*80)
    
    all_pass = True
    
    # Test 1: Verify embedding model
    print("\n" + "="*80)
    print("TEST 1: Embedding Model Verification")
    print("="*80)
    embedding_pass = verify_embedding_model()
    
    # Test 2: Verify collection accessibility
    print("\n" + "="*80)
    print("TEST 2: Collection Accessibility")
    print("="*80)
    collection_pass = verify_collection_accessibility()
    
    # Test 3: End-to-end retrieval
    print("\n" + "=" *80)
    print("TEST 3: End-to-End Retrieval Pipeline")
    print("="*80)
    retrieval_results = test_rag_e2e_retrieval()
    
    # Summarize
    summary_pass = summarize_results(retrieval_results)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    tests = {
        "Embedding Model": embedding_pass,
        "Collection Accessibility": collection_pass,
        "End-to-End Retrieval": summary_pass
    }
    
    for test_name, result in tests.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    all_pass = all(tests.values())
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving detailed results...")
    print("="*80)
    
    detailed_results = {
        "embedding_model": embedding_pass,
        "collection_accessible": collection_pass,
        "retrieval_results": retrieval_results
    }
    
    results_file = Path("test_rag_e2e_results.json")
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
