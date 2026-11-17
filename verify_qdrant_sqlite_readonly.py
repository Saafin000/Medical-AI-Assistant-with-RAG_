#!/usr/bin/env python3
"""
READ-ONLY verification of Qdrant SQLite database after brain_tumor_2024.pdf ingestion.

Directly reads the SQLite database to verify:
1. Total point count
2. Vector dimensions
3. Distance metric
4. Sample chunks by source
5. Distribution of documents
"""

import sys
import sqlite3
import json
from pathlib import Path
from collections import defaultdict

def verify_qdrant_sqlite():
    """Read Qdrant SQLite database directly."""
    
    db_path = Path("data/qdrant_db/collection/medical_assistance_rag_hf/storage.sqlite")
    
    print("\n" + "="*80)
    print("QDRANT COLLECTION VERIFICATION (SQLite READ-ONLY)")
    print("="*80)
    
    if not db_path.exists():
        print(f"\nDatabase file not found: {db_path}")
        return False
    
    print(f"\nDatabase file: {db_path}")
    print(f"File size: {db_path.stat().st_size} bytes")
    
    try:
        # Open database in read-only mode
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\nTables in database: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Try to find point storage table
        print("\n" + "="*80)
        print("CHECK 1: Point Count")
        print("="*80)
        
        point_count = 0
        try:
            # Common Qdrant table names
            for table_name in ['points', 'point_by_id', 'segments', 'payload']:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    count = cursor.fetchone()[0]
                    print(f"Table '{table_name}': {count} rows")
                    if table_name == 'points' and count > 0:
                        point_count = count
                except:
                    pass
        except Exception as e:
            print(f"Error querying tables: {e}")
        
        # Try to get schema info
        print("\n" + "="*80)
        print("CHECK 2: Collection Metadata")
        print("="*80)
        
        # Read from meta.json for collection info
        meta_path = Path("data/qdrant_db/meta.json")
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            if "medical_assistance_rag_hf" in meta.get("collections", {}):
                collection = meta["collections"]["medical_assistance_rag_hf"]
                dense = collection.get("vectors", {}).get("dense", {})
                
                print(f"Collection: medical_assistance_rag_hf")
                print(f"Vector size: {dense.get('size')} dimensions")
                print(f"Distance metric: {dense.get('distance')}")
                
                # Verify expected values
                if dense.get('size') == 384:
                    print("✓ Vector dimension is 384 (correct)")
                else:
                    print(f"✗ Vector dimension mismatch (expected 384, got {dense.get('size')})")
                
                if dense.get('distance') == 'Cosine':
                    print("✓ Distance metric is Cosine (correct)")
                else:
                    print(f"✗ Distance metric mismatch (expected Cosine, got {dense.get('distance')})")
        
        # Analyze document store
        print("\n" + "="*80)
        print("CHECK 3: Document Store Analysis")
        print("="*80)
        
        docs_dir = Path("data/docs_db")
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*"))
            doc_files = [f for f in doc_files if f.is_file() and f.name not in ['.lock', 'meta.json']]
            
            print(f"Document store files: {len(doc_files)}")
            
            # Sample some documents to determine source
            sources = defaultdict(int)
            sample_docs = {}
            
            for i, doc_file in enumerate(sorted(doc_files)):
                try:
                    with open(doc_file, 'r', errors='ignore') as f:
                        content = f.read(500)  # First 500 chars
                    
                    # Try to identify source
                    if "brain" in content.lower() or "tumor" in content.lower():
                        sources["brain_tumor_2024.pdf"] += 1
                        if "brain_tumor_2024.pdf" not in sample_docs and len(sample_docs) < 2:
                            sample_docs["brain_tumor_2024.pdf"] = doc_file
                    elif "diabetes" in content.lower() or "glucose" in content.lower():
                        sources["diabetes.pdf"] += 1
                        if "diabetes.pdf" not in sample_docs and len(sample_docs) < 2:
                            sample_docs["diabetes.pdf"] = doc_file
                    else:
                        # Check more of the file
                        with open(doc_file, 'r', errors='ignore') as f:
                            full_content = f.read()
                        
                        if "brain" in full_content.lower() or "tumor" in full_content.lower():
                            sources["brain_tumor_2024.pdf"] += 1
                        elif "diabetes" in full_content.lower():
                            sources["diabetes.pdf"] += 1
                        else:
                            sources["unknown"] += 1
                
                except Exception as e:
                    sources["unknown"] += 1
            
            print(f"\nDocument distribution (estimated by content):")
            for source, count in sorted(sources.items()):
                print(f"  {source}: {count}")
            
            print(f"\nExpected:")
            print(f"  diabetes.pdf: 8 chunks")
            print(f"  brain_tumor_2024.pdf: 24 chunks")
            print(f"  Total: 32 chunks")
            
            print(f"\nActual (from document store):")
            print(f"  diabetes.pdf: {sources.get('diabetes.pdf', 0)} chunks")
            print(f"  brain_tumor_2024.pdf: {sources.get('brain_tumor_2024.pdf', 0)} chunks")
            print(f"  unknown: {sources.get('unknown', 0)} chunks")
            print(f"  Total: {sum(sources.values())} chunks")
            
            total_expected = 32
            total_actual = sum(sources.values())
            
            if total_actual == total_expected:
                print(f"✓ Total count matches expected ({total_expected})")
            else:
                print(f"⚠ Total count differs (expected {total_expected}, got {total_actual})")
                if total_actual > total_expected:
                    print(f"  (+{total_actual - total_expected} extra chunks - possible duplicate ingestion)")
                else:
                    print(f"  ({total_expected - total_actual} missing chunks)")
        
        # Check parsed content
        print("\n" + "="*80)
        print("CHECK 4: Parsed Content Directory")
        print("="*80)
        
        parsed_dir = Path("data/parsed_docs")
        if parsed_dir.exists():
            parsed_files = list(parsed_dir.glob("*"))
            parsed_files = [f for f in parsed_files if f.is_file()]
            
            print(f"Parsed content files: {len(parsed_files)}")
            
            for parsed_file in sorted(parsed_files):
                print(f"  - {parsed_file.name}")
        else:
            print("Parsed content directory does not exist")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\nError reading database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification."""
    success = verify_qdrant_sqlite()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    if success:
        print("\n✓ Database is readable and accessible (READ-ONLY)")
        print("✓ No modifications were made")
        return 0
    else:
        print("\n✗ Could not fully verify database")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
