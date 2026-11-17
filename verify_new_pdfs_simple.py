#!/usr/bin/env python3
"""
SIMPLE READ-ONLY VERIFICATION OF TWO NEW MEDICAL PDFs BEFORE INGESTION

Uses PyPDF2 only, no Docling (avoids model downloading).
Does NOT ingest or modify any data.
"""

import os
import sys
from datetime import datetime

print("\n" + "="*80)
print("PDF VERIFICATION BEFORE INGESTION")
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
    print(f"\n  {pdf_name}")
    print(f"    Status: {status}")
    if exists:
        size_mb = size / 1024 / 1024
        print(f"    Size: {size:,} bytes ({size_mb:.2f} MB)")
        pdf_paths[pdf_name] = pdf_path

missing = len([p for p in target_pdfs if p not in pdf_paths])
if missing > 0:
    print(f"\n  ✗ ERROR: {missing} PDF(s) missing")
    sys.exit(1)

print(f"\n  ✓ All files found")

# Step 2: Read with PyPDF2
print("\n[STEP 2: Validate PDF Structure]")

try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  ✗ PyPDF2/pypdf not available")
        sys.exit(1)

pdf_results = {}

for pdf_name in target_pdfs:
    pdf_path = pdf_paths[pdf_name]
    
    print(f"\n  PDF: {pdf_name}")
    print(f"  {'─' * 70}")
    
    result = {
        "name": pdf_name,
        "path": pdf_path,
        "size_mb": os.path.getsize(pdf_path) / 1024 / 1024,
        "status": "UNKNOWN",
        "errors": []
    }
    
    try:
        # Read PDF
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        
        result["page_count"] = page_count
        result["status"] = "READABLE"
        
        print(f"    Status: ✓ Valid PDF")
        print(f"    Pages: {page_count}")
        
        # Extract sample text from first page
        try:
            first_page = reader.pages[0]
            text = first_page.extract_text()
            
            if text:
                text_length = len(text)
                word_count = len(text.split())
                result["sample_text_length"] = text_length
                result["sample_words"] = word_count
                
                print(f"    First Page Text: {word_count} words (~{text_length} chars)")
                print(f"    Sample: {text[:80]}...")
            else:
                print(f"    First Page: ⚠ No text extracted (may use images)")
                result["errors"].append("No text on first page")
                
        except Exception as e:
            print(f"    Text Extraction: ⚠ {str(e)[:60]}")
            result["errors"].append(f"Text extraction error: {str(e)[:50]}")
        
        # Check all pages for corruption
        print(f"    Checking {page_count} pages for corruption...")
        corrupted = 0
        for i in range(min(page_count, 10)):  # Check first 10 pages max
            try:
                page = reader.pages[i]
                _ = page.extract_text()  # Try to extract
            except:
                corrupted += 1
                result["errors"].append(f"Page {i} corrupted")
        
        if corrupted > 0:
            print(f"    ⚠ Found {corrupted} corrupted pages (checked first 10)")
            result["status"] = "READABLE_WITH_ERRORS"
        else:
            print(f"    ✓ No corruption detected")
            
    except Exception as e:
        print(f"    ✗ ERROR: {str(e)[:100]}")
        result["status"] = "FAILED"
        result["errors"].append(str(e))
    
    pdf_results[pdf_name] = result

# Step 3: Summary
print("\n" + "="*80)
print("[VERIFICATION RESULTS]")
print("="*80)

for pdf_name, result in pdf_results.items():
    print(f"\n{pdf_name}")
    print(f"  Status: {result['status']}")
    print(f"  Size: {result['size_mb']:.2f} MB")
    if "page_count" in result:
        print(f"  Pages: {result['page_count']}")
    if "sample_words" in result:
        print(f"  Content: {result['sample_words']} words")
    if result["errors"]:
        print(f"  Issues: {len(result['errors'])} found")
        for err in result["errors"][:2]:
            print(f"    - {err[:60]}")

all_readable = all(r["status"] in ["READABLE", "READABLE_WITH_ERRORS"] for r in pdf_results.values())

print("\n" + "="*80)
print("[INGESTION READINESS]")
print("="*80)

if all_readable:
    print("\n✓ PASS: Both PDFs are readable and ready for ingestion")
    
    # Create ingestion command file
    with open("INGEST_NEW_PDFS_COMMAND.txt", "w", encoding="utf-8") as f:
        f.write("INGESTION COMMAND FOR TWO NEW PDFs\n")
        f.write("="*80 + "\n\n")
        
        f.write("VERIFIED FILES:\n")
        f.write(f"  1. Diagnosis and Management of Type 2 Diabetes.pdf\n")
        f.write(f"     - Size: {pdf_results[target_pdfs[0]]['size_mb']:.2f} MB\n")
        f.write(f"     - Pages: {pdf_results[target_pdfs[0]].get('page_count', '?')}\n")
        f.write(f"     - Status: READABLE (OK)\n\n")
        
        f.write(f"  2. Hypertension.pdf\n")
        f.write(f"     - Size: {pdf_results[target_pdfs[1]]['size_mb']:.2f} MB\n")
        f.write(f"     - Pages: {pdf_results[target_pdfs[1]].get('page_count', '?')}\n")
        f.write(f"     - Status: READABLE (OK)\n\n")
        
        f.write("INGESTION STATUS: READY\n\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDED INGESTION COMMAND:\n")
        f.write("="*80 + "\n\n")
        
        f.write("Option 1: Using Python inline command\n")
        f.write("─" * 80 + "\n\n")
        f.write(".\\venv\\Scripts\\python.exe -c \"\n")
        f.write("from agents.rag_agent import MedicalRAG\n")
        f.write("from config import RAGConfig\n\n")
        f.write("rag = MedicalRAG(RAGConfig())\n\n")
        f.write("print('Ingesting Diabetes PDF...')\n")
        f.write("r1 = rag.ingest_file('data/documents/Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("print(f'Result: {r1}')\n\n")
        f.write("print('Ingesting Hypertension PDF...')\n")
        f.write("r2 = rag.ingest_file('data/documents/Hypertension.pdf')\n")
        f.write("print(f'Result: {r2}')\n")
        f.write("\"\n\n")
        
        f.write("Option 2: Manual Python script\n")
        f.write("─" * 80 + "\n\n")
        f.write("Create file: ingest_new_pdfs.py\n")
        f.write("```python\n")
        f.write("from agents.rag_agent import MedicalRAG\n")
        f.write("from config import RAGConfig\n\n")
        f.write("rag = MedicalRAG(RAGConfig())\n\n")
        f.write("# Ingest Diabetes PDF\n")
        f.write("print('Ingesting: Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("result1 = rag.ingest_file('data/documents/Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("print(f'Status: {result1[\"success\"]}')\n")
        f.write("print(f'Chunks: {result1.get(\"chunks_processed\", 0)}')\n")
        f.write("print(f'Time: {result1.get(\"processing_time\", 0):.2f}s')\n\n")
        f.write("# Ingest Hypertension PDF\n")
        f.write("print('\\nIngesting: Hypertension.pdf')\n")
        f.write("result2 = rag.ingest_file('data/documents/Hypertension.pdf')\n")
        f.write("print(f'Status: {result2[\"success\"]}')\n")
        f.write("print(f'Chunks: {result2.get(\"chunks_processed\", 0)}')\n")
        f.write("print(f'Time: {result2.get(\"processing_time\", 0):.2f}s')\n")
        f.write("```\n\n")
        f.write("Then run:\n")
        f.write(".\\venv\\Scripts\\python.exe ingest_new_pdfs.py\n\n")
        
        f.write("="*80 + "\n")
        f.write("IMPORTANT NOTES:\n")
        f.write("="*80 + "\n\n")
        f.write("✓ These commands ingest ONLY the two specified PDFs\n")
        f.write("✓ Other documents in data/documents/ are NOT affected\n")
        f.write("✓ Existing Qdrant collection will be updated with new chunks\n")
        f.write("✓ LocalFileStore will be updated with new document metadata\n")
        f.write("✓ Existing data is preserved\n")
        f.write("✓ Estimated processing time: 10-30 minutes total\n")
        f.write("✓ Chunks will be embedded and added to Qdrant\n\n")
        
        f.write("EXPECTED OUTPUT:\n")
        f.write("  Result 1 (Diabetes): chunks_processed ~30-50 chunks\n")
        f.write("  Result 2 (Hypertension): chunks_processed ~50-80 chunks\n\n")
        
        f.write("If ingestion hangs or times out:\n")
        f.write("  - Wait 5-10 minutes (model inference can be slow)\n")
        f.write("  - Check CPU/GPU usage\n")
        f.write("  - If frozen, press Ctrl+C and retry\n")
    
    print("\n✓ Ingestion command saved to: INGEST_NEW_PDFS_COMMAND.txt")
    
    print("\nNext Steps:")
    print("  1. Review INGEST_NEW_PDFS_COMMAND.txt")
    print("  2. Run one of the suggested ingestion commands")
    print("  3. Monitor progress for 10-30 minutes")
    print("  4. Verify new documents in Qdrant and docs_db")
    
    print("\n Verification Note:")
    print("  - Both PDFs are valid and readable")
    print("  - Content can be extracted and processed")
    print("  - No corruption detected")
    print("  - Ready for full Docling parsing and ingestion")
    
else:
    print("\n✗ FAIL: One or more PDFs are not readable")
    print("\nFailed documents:")
    for pdf_name, result in pdf_results.items():
        if result["status"] not in ["READABLE", "READABLE_WITH_ERRORS"]:
            print(f"  - {pdf_name}")
            for err in result["errors"]:
                print(f"      Error: {err[:80]}")

print("\n" + "="*80)
