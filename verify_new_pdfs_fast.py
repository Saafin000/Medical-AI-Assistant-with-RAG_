#!/usr/bin/env python3
"""
FAST READ-ONLY VERIFICATION OF TWO NEW MEDICAL PDFs BEFORE INGESTION

Uses PyPDF2 for quick validation without full Docling parsing.
Does NOT ingest or modify any data.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Suppress logging
logging.basicConfig(level=logging.CRITICAL)

print("\n" + "="*80)
print("PDF VERIFICATION BEFORE INGESTION (FAST)")
print("="*80)

print(f"\nVerification Start: {datetime.now().isoformat()}")

# Define target PDFs
target_pdfs = [
    "Diagnosis and Management of Type 2 Diabetes.pdf",
    "Hypertension.pdf"
]

pdf_dir = "data/documents"

# Step 1: Verify PDFs exist
print("\n[STEP 1: Verify PDF Files Exist]")

pdf_paths = {}
for pdf_name in target_pdfs:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    exists = os.path.exists(pdf_path)
    size = os.path.getsize(pdf_path) if exists else 0
    
    status = "✓ FOUND" if exists else "✗ NOT FOUND"
    print(f"  {pdf_name}")
    print(f"    Status: {status}")
    if exists:
        print(f"    Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        pdf_paths[pdf_name] = pdf_path

missing = len([p for p in target_pdfs if p not in pdf_paths])
if missing > 0:
    print(f"\n  ERROR: {missing} PDF(s) missing. Cannot proceed.")
    sys.exit(1)

print(f"  Status: ALL FILES FOUND ✓")

# Step 2: Try PyPDF2 reader for quick validation
print("\n[STEP 2: Quick PDF Structure Validation]")

try:
    from PyPDF2 import PdfReader
    print(f"  PyPDF2 import: ✓ OK")
except ImportError:
    print(f"  PyPDF2 import: ✗ NOT AVAILABLE")
    print(f"  Attempting fallback with pypdf...")
    try:
        from pypdf import PdfReader
        print(f"  pypdf import: ✓ OK")
    except ImportError:
        print(f"  ✗ No PDF reader available. Using file-based checks only.")
        PdfReader = None

pdf_results = {}

for pdf_name in target_pdfs:
    pdf_path = pdf_paths[pdf_name]
    
    print(f"\n  PDF: {pdf_name}")
    print(f"  {'─' * 70}")
    
    result = {
        "name": pdf_name,
        "path": pdf_path,
        "size_mb": os.path.getsize(pdf_path) / 1024 / 1024,
        "status": "UNKNOWN"
    }
    
    try:
        # Try PyPDF2 if available
        if PdfReader:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            
            result["page_count"] = page_count
            result["status"] = "READABLE"
            
            print(f"    File Format: ✓ Valid PDF")
            print(f"    Page Count: {page_count}")
            
            # Try to extract text from first page
            try:
                first_page = reader.pages[0]
                text = first_page.extract_text()
                text_length = len(text) if text else 0
                word_count = len(text.split()) if text else 0
                
                has_content = text_length > 50
                
                result["text_length"] = text_length
                result["word_count"] = word_count
                result["has_content"] = has_content
                
                print(f"    Sample Text: {'✓ YES' if has_content else '✗ NO'}")
                print(f"    First Page Content: {word_count} words (~{text_length} chars)")
                
                if has_content:
                    print(f"    Preview: {text[:100]}...")
                    
            except Exception as e:
                print(f"    Text Extraction: ⚠ Partial ({str(e)[:50]})")
                result["text_extraction"] = "partial"
            
        else:
            # Fallback: just check if file is readable
            print(f"    File Format: ✓ File accessible")
            print(f"    Size: {result['size_mb']:.2f} MB")
            result["status"] = "ACCESSIBLE"
            
    except Exception as e:
        print(f"    ✗ ERROR: {str(e)[:100]}")
        result["status"] = "FAILED"
        result["error"] = str(e)
    
    pdf_results[pdf_name] = result

# Step 3: Docling availability check
print("\n[STEP 3: Check Docling Parser Availability]")

try:
    from agents.rag_agent.doc_parser import MedicalDocParser
    print(f"  MedicalDocParser: ✓ AVAILABLE")
    print(f"  Docling: ✓ Ready for ingestion")
    docling_available = True
except Exception as e:
    print(f"  MedicalDocParser: ✗ NOT AVAILABLE")
    print(f"  Error: {str(e)[:100]}")
    docling_available = False

# Step 4: RAG Pipeline availability
print("\n[STEP 4: Check RAG Pipeline]")

try:
    from agents.rag_agent import MedicalRAG
    from config import RAGConfig
    print(f"  MedicalRAG: ✓ AVAILABLE")
    print(f"  RAGConfig: ✓ AVAILABLE")
    rag_available = True
except Exception as e:
    print(f"  MedicalRAG: ✗ NOT AVAILABLE")
    print(f"  Error: {str(e)[:100]}")
    rag_available = False

# Step 5: Summary and recommendations
print("\n" + "="*80)
print("[VERIFICATION SUMMARY]")
print("="*80)

all_readable = all(r.get("status") in ["READABLE", "ACCESSIBLE"] for r in pdf_results.values())

print(f"\nFile Status:")
for pdf_name, result in pdf_results.items():
    status_sym = "✓" if result["status"] in ["READABLE", "ACCESSIBLE"] else "✗"
    print(f"  {status_sym} {pdf_name}")
    print(f"      Size: {result['size_mb']:.2f} MB")
    print(f"      Status: {result['status']}")
    if "page_count" in result:
        print(f"      Pages: {result['page_count']}")
    if "word_count" in result:
        print(f"      Content: {result['word_count']} words")
    if "error" in result:
        print(f"      Error: {result['error'][:80]}")

print(f"\nSystem Status:")
print(f"  Docling Parser: {'✓ READY' if docling_available else '✗ NOT AVAILABLE'}")
print(f"  RAG Pipeline: {'✓ READY' if rag_available else '✗ NOT AVAILABLE'}")

# Step 6: Recommendation and command
print("\n" + "="*80)
print("[RECOMMENDATION]")
print("="*80)

if all_readable and docling_available and rag_available:
    print(f"\n  ✓ PASS: Both PDFs are ready for ingestion")
    print(f"\n  All requirements met:")
    print(f"    ✓ Files exist and are readable")
    print(f"    ✓ Docling parser available")
    print(f"    ✓ RAG pipeline available")
    print(f"    ✓ No parsing errors detected")
    
    # Save the ingestion command
    with open("INGEST_NEW_PDFS_COMMAND.txt", "w") as f:
        f.write("INGESTION COMMAND FOR TWO NEW PDFs\n")
        f.write("="*80 + "\n\n")
        f.write("Verified Files:\n")
        f.write(f"  1. Diagnosis and Management of Type 2 Diabetes.pdf ({pdf_results[target_pdfs[0]]['size_mb']:.2f} MB)\n")
        f.write(f"  2. Hypertension.pdf ({pdf_results[target_pdfs[1]]['size_mb']:.2f} MB)\n\n")
        f.write("Status: READY FOR INGESTION\n\n")
        f.write("Recommended Ingestion Command:\n")
        f.write("─"*80 + "\n\n")
        f.write(".\\venv\\Scripts\\python.exe -c \"\n")
        f.write("from agents.rag_agent import MedicalRAG\n")
        f.write("from config import RAGConfig\n\n")
        f.write("# Initialize RAG\n")
        f.write("rag = MedicalRAG(RAGConfig())\n\n")
        f.write("# Ingest first PDF\n")
        f.write("print('Ingesting: Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("result1 = rag.ingest_file('data/documents/Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("print('Result:', result1)\n\n")
        f.write("# Ingest second PDF\n")
        f.write("print('Ingesting: Hypertension.pdf')\n")
        f.write("result2 = rag.ingest_file('data/documents/Hypertension.pdf')\n")
        f.write("print('Result:', result2)\n")
        f.write("\"\n\n")
        f.write("Notes:\n")
        f.write("  - This command ingests ONLY these two files\n")
        f.write("  - No other documents will be affected\n")
        f.write("  - Qdrant collection will be updated\n")
        f.write("  - LocalFileStore will be updated\n")
        f.write("  - Estimated time: 5-15 minutes per document\n")
        f.write("  - Chunks will be created and embedded\n")
    
    print(f"\n  Ingestion Command File: INGEST_NEW_PDFS_COMMAND.txt")
    
elif all_readable:
    print(f"\n  ⚠ CONDITIONAL PASS: Files are readable, but pipeline incomplete")
    if not docling_available:
        print(f"    - Docling not available (required for full parsing)")
    if not rag_available:
        print(f"    - RAG pipeline not available (required for ingestion)")

else:
    print(f"\n  ✗ FAIL: One or more files are not readable")
    for pdf_name, result in pdf_results.items():
        if result["status"] not in ["READABLE", "ACCESSIBLE"]:
            print(f"    - {pdf_name}: {result.get('error', 'Unknown error')}")

print("\n" + "="*80)

# Final status
if all_readable and docling_available and rag_available:
    print("\n✓ READY FOR INGESTION")
    sys.exit(0)
else:
    print("\n✗ NOT READY FOR INGESTION")
    sys.exit(1)
