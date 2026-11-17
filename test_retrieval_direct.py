#!/usr/bin/env python3
"""
Direct Retrieval Pipeline Test - Avoids Unicode Issues and Missing Dependencies

Imports retrieval components directly and tests them.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


def test_dense_retrieval(query: str, config):
    """Test dense vector retrieval directly."""
    
    print("\n" + "="*80)
    print("DENSE RETRIEVAL TEST")
    print("="*80)
    print("Query: " + query)
    
    try:
        # Import directly to avoid __init__ issues
        import sys
        sys.path.insert(0, 'agents/rag_agent')
        from vectorstore_qdrant import VectorStore
        
        print("[1] Initializing VectorStore...")
        vector_store = VectorStore(config)
        
        print("[2] Loading vectorstore and docstore...")
        vectorstore, docstore = vector_store.load_vectorstore()
        
        print("[3] Retrieving relevant chunks (top_k={})...".format(config.rag.top_k))
        
        retrieved_docs = vector_store.retrieve_relevant_chunks(
            query=query,
            vectorstore=vectorstore,
            docstore=docstore
        )
        
        print("\nPASS: Dense retrieval successful")
        print("Retrieved {} documents".format(len(retrieved_docs)))
        
        # Show top 3
        print("\nTop 3 Retrieved Chunks:")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            score = doc.get("score", 0)
            source = doc.get("source", "unknown")
            content = doc.get("content", "")[:150].replace("\n", " ")
            
            print("\n{}. Source: {}".format(i, source))
            print("   Score: {:.4f}".format(score))
            print("   Content: {}...".format(content))
        
        return {
            "status": "SUCCESS",
            "count": len(retrieved_docs),
            "documents": retrieved_docs
        }
        
    except Exception as e:
        print("\nFAIL: Dense retrieval error")
        print("Error: " + str(e))
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e),
            "count": 0,
            "documents": []
        }


def test_reranking(query: str, retrieved_docs: List[Dict], config):
    """Test reranking directly."""
    
    print("\n" + "="*80)
    print("RERANKING TEST")
    print("="*80)
    print("Query: " + query)
    
    if not retrieved_docs:
        print("SKIP: No documents to rerank")
        return {
            "status": "SKIPPED",
            "reason": "No documents"
        }
    
    print("Documents to rerank: {}".format(len(retrieved_docs)))
    
    try:
        import sys
        sys.path.insert(0, 'agents/rag_agent')
        from reranker import Reranker
        
        print("[1] Initializing Reranker...")
        print("    Model: " + config.rag.reranker_model)
        print("    Top-k: {}".format(config.rag.reranker_top_k))
        
        reranker = Reranker(config)
        
        print("[2] Executing rerank()...")
        
        reranked_docs, picture_refs = reranker.rerank(
            query=query,
            documents=retrieved_docs,
            parsed_content_dir=config.rag.parsed_content_dir
        )
        
        print("\nPASS: Reranking successful")
        print("Reranked {} documents".format(len(reranked_docs)))
        print("Picture references: {}".format(len(picture_refs)))
        
        # Show top 3 with scores
        print("\nTop 3 Reranked Documents:")
        for i, doc in enumerate(reranked_docs[:3], 1):
            source = doc.get("source", "unknown")
            original = doc.get("score", 0)
            rerank_score = doc.get("rerank_score", 0)
            combined = doc.get("combined_score", 0)
            content = doc.get("content", "")[:150].replace("\n", " ")
            
            print("\n{}. Source: {}".format(i, source))
            print("   Original score: {:.4f}".format(original))
            print("   Rerank score: {:.4f}".format(rerank_score))
            print("   Combined score: {:.4f}".format(combined))
            print("   Content: {}...".format(content))
        
        return {
            "status": "SUCCESS",
            "count": len(reranked_docs),
            "documents": reranked_docs
        }
        
    except Exception as e:
        print("\nFAIL: Reranking error")
        print("Error: " + str(e))
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e)
        }


def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("MEDICAL RAG - DIRECT RETRIEVAL PIPELINE TEST")
    print("="*80)
    
    try:
        from config import Config
        
        config = Config()
        
        print("\nConfiguration loaded:")
        print("  Collection: " + config.rag.collection_name)
        print("  Embedding dim: {}".format(config.rag.embedding_dim))
        print("  Distance: " + config.rag.distance_metric)
        print("  Top-k: {}".format(config.rag.top_k))
        print("  Reranker: " + config.rag.reranker_model)
        
        # Test queries
        queries = [
            "What are the common types of brain tumors?",
            "What is diabetes mellitus?",
            "What information does the document provide about brain tumor diagnosis?"
        ]
        
        all_results = {}
        
        for query_idx, query in enumerate(queries, 1):
            print("\n\n" + "#"*80)
            print("# QUERY {}/{}".format(query_idx, len(queries)))
            print("#"*80)
            
            query_results = {
                "query": query,
                "stages": {}
            }
            
            # Dense retrieval
            dense_result = test_dense_retrieval(query, config)
            query_results["stages"]["dense_retrieval"] = dense_result
            
            retrieved_docs = dense_result.get("documents", [])
            
            # Reranking
            rerank_result = test_reranking(query, retrieved_docs, config)
            query_results["stages"]["reranking"] = rerank_result
            
            # Final results
            final_docs = rerank_result.get("documents", retrieved_docs)
            
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            print("Total final documents: {}".format(len(final_docs)))
            
            if final_docs:
                print("\nTop 3 Sources:")
                for i, doc in enumerate(final_docs[:3], 1):
                    print("{}. {} (score: {:.4f})".format(
                        i,
                        doc.get('source', 'unknown'),
                        doc.get('combined_score', doc.get('score', 0))
                    ))
            
            query_results["final_count"] = len(final_docs)
            query_results["top_sources"] = [doc.get('source', 'unknown') for doc in final_docs[:3]]
            
            all_results["Query_{}".format(query_idx)] = query_results
        
        # Summary
        print("\n\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        for key, results in all_results.items():
            print("\n{}: {}".format(key, results['query']))
            dense_status = results['stages']['dense_retrieval'].get('status', 'UNKNOWN')
            rerank_status = results['stages']['reranking'].get('status', 'UNKNOWN')
            print("  Dense: {} ({} docs)".format(
                dense_status,
                results['stages']['dense_retrieval'].get('count', 0)
            ))
            print("  Reranking: {}".format(rerank_status))
            print("  Final: {} docs".format(results.get('final_count', 0)))
            print("  Top sources: {}".format(', '.join(results.get('top_sources', []))))
        
        # Save results
        results_file = Path("test_retrieval_direct_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n\nResults saved to: " + str(results_file))
        
        return 0
        
    except Exception as e:
        print("\nFATAL ERROR: " + str(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print("Unexpected error: " + str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
