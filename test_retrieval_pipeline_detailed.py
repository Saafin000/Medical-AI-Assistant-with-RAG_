#!/usr/bin/env python3
"""
Detailed Execution Test of Medical RAG Retrieval Pipeline

Runs each stage separately and reports results:
1. Query expansion
2. Dense retrieval
3. Sparse retrieval  
4. Hybrid fusion
5. Reranking
6. Final results with sources

READ-ONLY execution - no modifications.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


def test_query_expansion(query: str, config):
    """Test query expansion stage."""
    
    print(f"\n{'='*80}")
    print(f"STAGE 1: QUERY EXPANSION")
    print(f"{'='*80}")
    
    print(f"Original query: '{query}'")
    
    try:
        from agents.rag_agent.query_expander import QueryExpander
        
        print("Initializing QueryExpander...")
        expander = QueryExpander(config)
        
        print("Executing expand_query()...")
        result = expander.expand_query(query)
        
        expanded_query = result.get("expanded_query", query)
        
        print(f"\n✓ Query expansion successful")
        print(f"  Original: '{query}'")
        print(f"  Expanded: '{expanded_query}'")
        
        return {
            "status": "SUCCESS",
            "original_query": query,
            "expanded_query": expanded_query
        }
        
    except Exception as e:
        print(f"\n✗ Query expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e),
            "expanded_query": query  # Fall back to original
        }


def test_dense_retrieval(query: str, config):
    """Test dense vector retrieval."""
    
    print(f"\n{'='*80}")
    print(f"STAGE 2: DENSE RETRIEVAL")
    print(f"{'='*80}")
    
    try:
        from agents.rag_agent.vectorstore_qdrant import VectorStore
        
        print(f"Query: '{query}'")
        print("Initializing VectorStore...")
        
        vector_store = VectorStore(config)
        
        print("Loading vectorstore and docstore...")
        vectorstore, docstore = vector_store.load_vectorstore()
        
        print(f"Executing retrieve_relevant_chunks()...")
        print(f"  (top_k = {config.rag.top_k})")
        
        retrieved_docs = vector_store.retrieve_relevant_chunks(
            query=query,
            vectorstore=vectorstore,
            docstore=docstore
        )
        
        print(f"\n✓ Dense retrieval successful")
        print(f"  Retrieved {len(retrieved_docs)} documents")
        
        # Show top 3
        for i, doc in enumerate(retrieved_docs[:3], 1):
            score = doc.get("score", 0)
            source = doc.get("source", "unknown")
            content_preview = doc.get("content", "")[:100].replace("\n", " ")
            
            print(f"\n  Document {i}:")
            print(f"    Source: {source}")
            print(f"    Score: {score:.4f}")
            print(f"    Content: {content_preview}...")
        
        return {
            "status": "SUCCESS",
            "count": len(retrieved_docs),
            "top_k": config.rag.top_k,
            "documents": retrieved_docs
        }
        
    except Exception as e:
        print(f"\n✗ Dense retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e),
            "count": 0,
            "documents": []
        }


def test_sparse_retrieval(query: str, config):
    """Test sparse retrieval if available."""
    
    print(f"\n{'='*80}")
    print(f"STAGE 3: SPARSE RETRIEVAL")
    print(f"{'='*80}")
    
    print(f"Query: '{query}'")
    
    try:
        from qdrant_client import QdrantClient
        from langchain_qdrant import FastEmbedSparse, RetrievalMode
        
        print("Checking sparse vector configuration...")
        
        client = QdrantClient(path=config.rag.vector_local_path)
        collection = client.get_collection(config.rag.collection_name)
        
        # Check if sparse vectors exist
        sparse_config = collection.config.params.sparse_vectors
        
        if sparse_config is None:
            print("⚠ Sparse vectors not configured")
            return {
                "status": "SKIPPED",
                "reason": "Sparse vectors not configured in collection"
            }
        
        print(f"✓ Sparse vectors configured: {list(sparse_config.keys())}")
        
        # Test sparse search
        print("Initializing sparse embeddings (BM25)...")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        print("Executing sparse search...")
        sparse_results = client.search(
            collection_name=config.rag.collection_name,
            query_vector=sparse_embeddings.embed_query(query),
            query_vector_name="sparse",
            limit=config.rag.top_k
        )
        
        print(f"\n✓ Sparse retrieval successful")
        print(f"  Retrieved {len(sparse_results)} results")
        
        for i, result in enumerate(sparse_results[:3], 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result.score:.4f}")
            if result.payload:
                print(f"    Payload keys: {list(result.payload.keys())}")
        
        return {
            "status": "SUCCESS",
            "count": len(sparse_results),
            "results": sparse_results
        }
        
    except ImportError as e:
        if "qdrant_client" in str(e):
            print(f"✗ qdrant_client not installed: {e}")
            return {
                "status": "SKIPPED",
                "reason": "qdrant_client not installed"
            }
        else:
            raise
    except Exception as e:
        print(f"⚠ Sparse retrieval skipped: {e}")
        return {
            "status": "SKIPPED",
            "reason": str(e)
        }


def test_reranking(query: str, retrieved_docs: List[Dict], config):
    """Test reranking of retrieved documents."""
    
    print(f"\n{'='*80}")
    print(f"STAGE 4: RERANKING")
    print(f"{'='*80}")
    
    if not retrieved_docs:
        print("⚠ No documents to rerank")
        return {
            "status": "SKIPPED",
            "reason": "No documents to rerank"
        }
    
    print(f"Query: '{query}'")
    print(f"Documents to rerank: {len(retrieved_docs)}")
    
    try:
        from agents.rag_agent.reranker import Reranker
        
        print(f"Initializing Reranker...")
        print(f"  Model: {config.rag.reranker_model}")
        print(f"  Top-k: {config.rag.reranker_top_k}")
        
        reranker = Reranker(config)
        
        print("Executing rerank()...")
        
        reranked_docs, picture_refs = reranker.rerank(
            query=query,
            documents=retrieved_docs,
            parsed_content_dir=config.rag.parsed_content_dir
        )
        
        print(f"\n✓ Reranking successful")
        print(f"  Reranked {len(reranked_docs)} documents")
        print(f"  Picture references: {len(picture_refs)}")
        
        # Show top 3 with scores
        for i, doc in enumerate(reranked_docs[:3], 1):
            source = doc.get("source", "unknown")
            original_score = doc.get("score", 0)
            rerank_score = doc.get("rerank_score", 0)
            combined_score = doc.get("combined_score", 0)
            content_preview = doc.get("content", "")[:100].replace("\n", " ")
            
            print(f"\n  Document {i}:")
            print(f"    Source: {source}")
            print(f"    Original score: {original_score:.4f}")
            print(f"    Rerank score: {rerank_score:.4f}")
            print(f"    Combined score: {combined_score:.4f}")
            print(f"    Content: {content_preview}...")
        
        return {
            "status": "SUCCESS",
            "count": len(reranked_docs),
            "picture_refs": len(picture_refs),
            "documents": reranked_docs
        }
        
    except Exception as e:
        print(f"\n✗ Reranking failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e)
        }


def run_full_pipeline(query: str, config):
    """Run complete retrieval pipeline."""
    
    print(f"\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    results = {
        "query": query,
        "stages": {}
    }
    
    # Stage 1: Query Expansion
    expansion_result = test_query_expansion(query, config)
    results["stages"]["expansion"] = expansion_result
    
    expanded_query = expansion_result.get("expanded_query", query)
    
    # Stage 2: Dense Retrieval
    dense_result = test_dense_retrieval(expanded_query, config)
    results["stages"]["dense_retrieval"] = dense_result
    
    retrieved_docs = dense_result.get("documents", [])
    
    # Stage 3: Sparse Retrieval
    sparse_result = test_sparse_retrieval(expanded_query, config)
    results["stages"]["sparse_retrieval"] = sparse_result
    
    # Stage 4: Reranking
    rerank_result = test_reranking(expanded_query, retrieved_docs, config)
    results["stages"]["reranking"] = rerank_result
    
    # Final results
    final_docs = rerank_result.get("documents", retrieved_docs)
    
    print(f"\n" + "="*80)
    print(f"FINAL RESULTS")
    print("="*80)
    
    print(f"\nTop 3 Retrieved Chunks:")
    for i, doc in enumerate(final_docs[:3], 1):
        print(f"\n{i}. Source: {doc.get('source', 'unknown')}")
        print(f"   Score: {doc.get('combined_score', doc.get('score', 0)):.4f}")
        print(f"   Content: {doc.get('content', '')[:200]}...")
    
    results["final_count"] = len(final_docs)
    results["top_3_sources"] = [doc.get('source', 'unknown') for doc in final_docs[:3]]
    
    return results


def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("MEDICAL RAG - DETAILED RETRIEVAL PIPELINE EXECUTION")
    print("="*80)
    
    try:
        # Import config
        from config import Config
        
        config = Config()
        
        # Test queries
        queries = [
            "What are the common types of brain tumors?",
            "What is diabetes mellitus?",
            "What information does the document provide about brain tumor diagnosis?"
        ]
        
        all_results = {}
        
        for query_idx, query in enumerate(queries, 1):
            print(f"\n\n" + "#"*80)
            print(f"# QUERY {query_idx}/{len(queries)}")
            print(f"#"*80)
            
            query_results = run_full_pipeline(query, config)
            all_results[f"Query_{query_idx}"] = query_results
        
        # Summary
        print(f"\n\n" + "="*80)
        print(f"EXECUTION SUMMARY")
        print("="*80)
        
        for query_key, results in all_results.items():
            print(f"\n{query_key}: {results['query']}")
            print(f"  Expansion: {results['stages']['expansion']['status']}")
            print(f"  Dense: {results['stages']['dense_retrieval']['status']} ({results['stages']['dense_retrieval'].get('count', 0)} docs)")
            print(f"  Sparse: {results['stages']['sparse_retrieval']['status']}")
            print(f"  Reranking: {results['stages']['reranking']['status']}")
            print(f"  Final docs: {results.get('final_count', 0)}")
            print(f"  Top sources: {results.get('top_3_sources', [])}")
        
        # Save results
        results_file = Path("test_retrieval_detailed_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n\nResults saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
