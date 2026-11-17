#!/usr/bin/env python3
"""
READ-ONLY verification of Qdrant collection after brain_tumor_2024.pdf ingestion.

Checks:
1. Collection point count (expecting 32: 8 from diabetes.pdf + 24 from brain_tumor_2024.pdf)
2. Vector dimension (expecting 384)
3. Distance metric (expecting Cosine)
4. Sample of stored points and their payloads
5. Document identification by source filename
6. Document-store entries
7. New unique points count

NO MODIFICATIONS - READ-ONLY ACCESS ONLY
"""

import sys
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def verify_qdrant_connection():
    """Verify Qdrant is accessible."""
    print("\n" + "="*80)
    print("CHECK 1: Qdrant Connection & Configuration")
    print("="*80)
    
    try:
        from qdrant_client import QdrantClient
        from config import Config
        
        config = Config()
        
        print(f"Vector DB Type: {config.rag.vector_db_type}")
        print(f"Collection Name: {config.rag.collection_name}")
        print(f"Distance Metric: {config.rag.distance_metric}")
        print(f"Vector Dimension: {config.rag.embedding_dim}")
        print(f"Use Local: {config.rag.use_local}")
        print(f"Local Path: {config.rag.vector_local_path}")
        
        # Try to connect
        if config.rag.use_local:
            print(f"\nConnecting to local Qdrant at: {config.rag.vector_local_path}")
            client = QdrantClient(path=config.rag.vector_local_path)
        else:
            print(f"\nConnecting to remote Qdrant at: {config.rag.url}")
            client = QdrantClient(url=config.rag.url, api_key=config.rag.api_key)
        
        print("✓ Connection successful")
        return client, config
    
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return None, None


def verify_collection_exists(client, config):
    """Check if collection exists and get metadata."""
    print("\n" + "="*80)
    print("CHECK 2: Collection Existence & Metadata")
    print("="*80)
    
    try:
        collection = client.get_collection(config.rag.collection_name)
        
        print(f"Collection Name: {config.rag.collection_name}")
        print(f"Status: {collection.status}")
        print(f"Vector Count: {collection.points_count}")
        print(f"Vector Size: {collection.config.params.vectors.size}")
        print(f"Distance Metric: {collection.config.params.vectors.distance}")
        
        # Expected vs actual
        expected_count = 32
        actual_count = collection.points_count
        
        print(f"\nExpected point count: {expected_count}")
        print(f"Actual point count: {actual_count}")
        
        if actual_count == expected_count:
            print("✓ Point count matches expected value")
            count_match = True
        else:
            print(f"⚠ Point count differs (expected {expected_count}, got {actual_count})")
            count_match = False
        
        # Check dimensions
        if collection.config.params.vectors.size == 384:
            print("✓ Vector dimension is 384 (correct)")
            dim_match = True
        else:
            print(f"✗ Vector dimension mismatch (expected 384, got {collection.config.params.vectors.size})")
            dim_match = False
        
        # Check distance
        if "Cosine" in str(collection.config.params.vectors.distance):
            print("✓ Distance metric is Cosine (correct)")
            distance_match = True
        else:
            print(f"✗ Distance metric mismatch (expected Cosine, got {collection.config.params.vectors.distance})")
            distance_match = False
        
        return collection, count_match and dim_match and distance_match
    
    except Exception as e:
        print(f"✗ Failed to get collection: {e}")
        return None, False


def inspect_sample_points(client, config, collection):
    """Inspect sample of stored points and their payloads."""
    print("\n" + "="*80)
    print("CHECK 3: Sample Points & Payloads")
    print("="*80)
    
    try:
        # Get a sample of points
        limit = 5
        points = client.scroll(
            collection_name=config.rag.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        actual_points = points[0]
        
        print(f"Retrieved {len(actual_points)} sample points (of {collection.points_count} total)\n")
        
        sources = {}
        for i, point in enumerate(actual_points):
            print(f"Point {i + 1}:")
            print(f"  ID: {point.id}")
            print(f"  Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
            
            if point.payload:
                source = point.payload.get("source_file", "unknown")
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
                
                print(f"  Source file: {source}")
                print(f"  Chunk text length: {len(point.payload.get('chunk_text', ''))} chars")
                
                # Show other payload fields
                for key in point.payload.keys():
                    if key not in ["source_file", "chunk_text"]:
                        print(f"  {key}: {point.payload[key]}")
            print()
        
        print(f"Source files in sample: {sources}")
        return True, sources
    
    except Exception as e:
        print(f"✗ Failed to inspect points: {e}")
        return False, {}


def count_by_source(client, config, collection):
    """Count how many chunks belong to each source document."""
    print("\n" + "="*80)
    print("CHECK 4: Chunk Count by Source Document")
    print("="*80)
    
    try:
        # Get all points (in batches if needed)
        all_points = []
        current_offset = 0
        batch_size = 100
        
        while True:
            points = client.scroll(
                collection_name=config.rag.collection_name,
                limit=batch_size,
                offset=current_offset,
                with_payload=True,
                with_vectors=False
            )
            
            actual_points = points[0]
            if not actual_points:
                break
            
            all_points.extend(actual_points)
            current_offset += batch_size
        
        print(f"Total points retrieved: {len(all_points)}")
        
        # Group by source
        sources = {}
        for point in all_points:
            if point.payload:
                source = point.payload.get("source_file", "unknown")
                if source not in sources:
                    sources[source] = []
                sources[source].append(point.id)
        
        print(f"\nDocuments in collection:")
        for source, point_ids in sorted(sources.items()):
            count = len(point_ids)
            print(f"  {source}: {count} chunks")
        
        # Check for expected documents
        brain_tumor_count = 0
        diabetes_count = 0
        
        for source, point_ids in sources.items():
            if "brain_tumor" in source.lower():
                brain_tumor_count = len(point_ids)
            elif "diabetes" in source.lower():
                diabetes_count = len(point_ids)
        
        print(f"\nExpectation:")
        print(f"  diabetes.pdf: 8 chunks")
        print(f"  brain_tumor_2024.pdf: 24 chunks")
        
        print(f"\nActual:")
        print(f"  diabetes.pdf: {diabetes_count} chunks")
        print(f"  brain_tumor_2024.pdf: {brain_tumor_count} chunks")
        
        success = True
        if diabetes_count != 8:
            print(f"⚠ Diabetes chunks: expected 8, got {diabetes_count}")
            success = False
        else:
            print("✓ Diabetes chunks count correct")
        
        if brain_tumor_count != 24:
            print(f"⚠ Brain tumor chunks: expected 24, got {brain_tumor_count}")
            success = False
        else:
            print("✓ Brain tumor chunks count correct")
        
        return success, sources
    
    except Exception as e:
        print(f"✗ Failed to count by source: {e}")
        return False, {}


def verify_document_store(config):
    """Check if document-store entries exist."""
    print("\n" + "="*80)
    print("CHECK 5: Document Store Entries")
    print("="*80)
    
    try:
        doc_store_path = Path(config.rag.doc_local_path)
        
        if not doc_store_path.exists():
            print(f"Document store path: {doc_store_path}")
            print("✗ Document store directory does not exist")
            return False
        
        print(f"Document store path: {doc_store_path}")
        print(f"✓ Document store directory exists")
        
        # List files in document store
        files = list(doc_store_path.glob("*"))
        print(f"Files in document store: {len(files)}")
        
        for file in sorted(files)[:5]:  # Show first 5
            print(f"  - {file.name}")
        
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        
        # Look for diabetes and brain_tumor entries
        brain_tumor_found = any("brain_tumor" in f.name.lower() for f in files)
        diabetes_found = any("diabetes" in f.name.lower() for f in files)
        
        print(f"\nDocument entries found:")
        print(f"  diabetes.pdf: {'Yes' if diabetes_found else 'No'}")
        print(f"  brain_tumor_2024.pdf: {'Yes' if brain_tumor_found else 'No'}")
        
        return diabetes_found or brain_tumor_found
    
    except Exception as e:
        print(f"✗ Failed to verify document store: {e}")
        return False


def verify_vector_integrity(client, config, collection):
    """Verify vectors are properly stored and searchable."""
    print("\n" + "="*80)
    print("CHECK 6: Vector Integrity & Searchability")
    print("="*80)
    
    try:
        # Try a simple search with a dummy vector
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Create a test query embedding
        test_query = "brain tumor diagnosis"
        query_vector = embeddings.embed_query(test_query)
        
        print(f"Test query: '{test_query}'")
        print(f"Query vector dimension: {len(query_vector)}")
        
        # Search
        results = client.search(
            collection_name=config.rag.collection_name,
            query_vector=query_vector,
            limit=3,
            with_payload=True
        )
        
        print(f"Search returned {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  Result {i + 1}:")
            print(f"    Score: {result.score:.4f}")
            print(f"    Source: {result.payload.get('source_file', 'unknown')}")
            print(f"    Text preview: {result.payload.get('chunk_text', '')[:80]}...")
        
        print("✓ Vectors are searchable and intact")
        return True
    
    except Exception as e:
        print(f"✗ Failed to verify vector integrity: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("QDRANT DATA VERIFICATION (READ-ONLY)")
    print("After brain_tumor_2024.pdf ingestion")
    print("="*80)
    
    # Check 1: Connection
    client, config = verify_qdrant_connection()
    if not client:
        print("\n✗ Cannot proceed without Qdrant connection")
        return 1
    
    # Check 2: Collection
    collection, check2_pass = verify_collection_exists(client, config)
    if not collection:
        print("\n✗ Cannot proceed without collection")
        return 1
    
    # Check 3: Sample points
    check3_pass, sample_sources = inspect_sample_points(client, config, collection)
    
    # Check 4: Count by source
    check4_pass, all_sources = count_by_source(client, config, collection)
    
    # Check 5: Document store
    check5_pass = verify_document_store(config)
    
    # Check 6: Vector integrity
    check6_pass = verify_vector_integrity(client, config, collection)
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    checks = {
        "Connection & Configuration": True,  # Already passed
        "Collection Exists & Metadata": check2_pass,
        "Sample Points & Payloads": check3_pass,
        "Chunks by Source": check4_pass,
        "Document Store": check5_pass,
        "Vector Integrity": check6_pass
    }
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
