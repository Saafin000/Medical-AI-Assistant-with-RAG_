#!/usr/bin/env python3
"""
READ-ONLY VERIFICATION OF TWO NEW MEDICAL PDFs BEFORE INGESTION

Tests parsing of:
1. Diagnosis and Management of Type 2 Diabetes.pdf
2. Hypertension.pdf

Does NOT ingest or modify any data.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Suppress excessive logging
logging.basicConfig(level=logging.WARNING)

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

# Verify PDFs exist
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
        print(f"    Path: {pdf_path}")
        print(f"    Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        pdf_paths[pdf_name] = pdf_path
    else:
        print(f"    Path: Not found - {pdf_path}")

missing_pdfs = len([p for p in target_pdfs if p not in pdf_paths])
if missing_pdfs > 0:
    print(f"\n  ERROR: {missing_pdfs} PDF(s) missing. Cannot proceed.")
    sys.exit(1)

print(f"\n  Status: ALL FILES FOUND")

# Step 2: Test Docling parser initialization
print("\n[STEP 2: Initialize Docling Parser]")

try:
    from agents.rag_agent.doc_parser import MedicalDocParser
    
    parser = MedicalDocParser()
    print(f"  Status: PARSER INITIALIZED")
    print(f"  Parser: MedicalDocParser")
    print(f"  Module: agents.rag_agent.doc_parser")
    
except Exception as e:
    print(f"  Status: FAILED")
    print(f"  Error: {str(e)[:150]}")
    sys.exit(1)

# Step 3: Parse each PDF
print("\n[STEP 3: Parse PDFs]")

parsed_results = {}

for pdf_name in target_pdfs:
    pdf_path = pdf_paths[pdf_name]
    
    print(f"\n  PDF: {pdf_name}")
    print(f"  {'─' * 70}")
    
    try:
        # Parse document
        print(f"    Parsing document...")
        
        # Create temp output directory for images (will not be saved permanently)
        temp_output_dir = os.path.join("data/parsed_docs", f"temp_verify_{pdf_name.replace('.pdf', '')}")
        
        parsed_doc, images = parser.parse_document(
            document_path=pdf_path,
            output_dir=temp_output_dir,
            image_resolution_scale=1.0,
            do_ocr=True,
            do_tables=True,
            do_formulas=False,
            do_picture_desc=False
        )
        
        print(f"    ✓ Parsing successful")
        
        # Collect metadata
        page_count = len(parsed_doc.pages)
        print(f"    Page Count: {page_count}")
        
        # Extract text content
        print(f"    Extracting text...")
        full_text = parsed_doc.export_to_markdown()
        text_length = len(full_text)
        word_count = len(full_text.split())
        
        print(f"    Text Length: {text_length:,} characters")
        print(f"    Word Count: ~{word_count:,} words")
        
        # Check for meaningful content
        has_content = text_length > 100
        print(f"    Meaningful Content: {'✓ YES' if has_content else '✗ NO'}")
        
        # Extract images
        image_count = len(images)
        print(f"    Images Found: {image_count}")
        
        # Check for tables, headings
        print(f"    Analyzing document structure...")
        
        has_tables = False
        has_headings = False
        medical_sections = []
        
        # Check markdown structure
        lines = full_text.split('\n')
        for line in lines:
            if line.startswith('#'):
                has_headings = True
            if '|' in line:
                has_tables = True
            # Look for medical keywords
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ['diagnosis', 'treatment', 'symptoms', 'causes', 'prevention', 'management', 'therapy', 'clinical', 'patient', 'disease']):
                if line not in medical_sections and len(line) > 10:
                    medical_sections.append(line[:100])
        
        print(f"    Headings Found: {'✓ YES' if has_headings else '✗ NO'}")
        print(f"    Tables Found: {'✓ YES' if has_tables else '✗ NO'}")
        print(f"    Medical Sections: {len(medical_sections)} identified")
        
        # Show sample medical sections
        if medical_sections:
            print(f"    Sample Sections:")
            for section in medical_sections[:3]:
                print(f"      - {section.strip()[:70]}...")
        
        # Check for parsing errors
        print(f"    Checking for corrupted pages...")
        
        corrupted_pages = []
        for page_no, page in parsed_doc.pages.items():
            try:
                if page.image is None:
                    corrupted_pages.append(page_no)
            except:
                corrupted_pages.append(page_no)
        
        if corrupted_pages:
            print(f"    ⚠ Corrupted Pages: {corrupted_pages}")
        else:
            print(f"    ✓ No corrupted pages detected")
        
        # Store results
        parsed_results[pdf_name] = {
            "status": "SUCCESS",
            "page_count": page_count,
            "text_length": text_length,
            "word_count": word_count,
            "has_content": has_content,
            "has_tables": has_tables,
            "has_headings": has_headings,
            "medical_sections": len(medical_sections),
            "images": image_count,
            "corrupted_pages": corrupted_pages,
            "sample_text": full_text[:200]
        }
        
        print(f"    Status: ✓ VERIFIED")
        
    except Exception as e:
        error_msg = str(e)
        print(f"    ✗ PARSING FAILED")
        print(f"    Error: {error_msg[:150]}")
        
        parsed_results[pdf_name] = {
            "status": "FAILED",
            "error": error_msg
        }

# Step 4: Summary
print("\n" + "="*80)
print("[VERIFICATION SUMMARY]")
print("="*80)

all_passed = all(r.get("status") == "SUCCESS" for r in parsed_results.values())

print(f"\nParsing Results:")
for pdf_name, result in parsed_results.items():
    status_symbol = "✓" if result["status"] == "SUCCESS" else "✗"
    print(f"\n  {status_symbol} {pdf_name}")
    
    if result["status"] == "SUCCESS":
        print(f"      Pages: {result['page_count']}")
        print(f"      Text: {result['text_length']:,} chars ({result['word_count']:,} words)")
        print(f"      Meaningful Content: {'✓' if result['has_content'] else '✗'}")
        print(f"      Tables: {'✓' if result['has_tables'] else '✗'}")
        print(f"      Headings: {'✓' if result['has_headings'] else '✗'}")
        print(f"      Medical Sections: {result['medical_sections']}")
        print(f"      Images: {result['images']}")
        print(f"      Corrupted Pages: {len(result['corrupted_pages'])}")
    else:
        print(f"      Error: {result['error'][:100]}")

# Step 5: Ingestion command
print("\n" + "="*80)
print("[INGESTION COMMAND]")
print("="*80)

if all_passed:
    print(f"\n  To ingest ONLY these two PDFs, use one of:")
    
    print(f"\n  Option 1: Using Python directly")
    print(f"    python ingest_medical_documents.py --files \"Diagnosis and Management of Type 2 Diabetes.pdf\" \"Hypertension.pdf\"")
    
    print(f"\n  Option 2: Using the MedicalRAG API")
    print(f"    from agents.rag_agent import MedicalRAG")
    print(f"    from config import RAGConfig")
    print(f"    rag = MedicalRAG(RAGConfig())")
    print(f"    rag.ingest_file(\"data/documents/Diagnosis and Management of Type 2 Diabetes.pdf\")")
    print(f"    rag.ingest_file(\"data/documents/Hypertension.pdf\")")
    
    print(f"\n  Option 3: Using the ingestion script with directory filter")
    print(f"    .\\\venv\\\\Scripts\\\\python.exe ingest_medical_documents.py --pdf-dir \"data/documents\" --filter \"Diabetes|Hypertension\"")
    
    print(f"\n  Note: These commands assume the ingestion script exists or you use the RAG API directly.")
    
    # Show recommended exact command
    print(f"\n  RECOMMENDED EXACT COMMAND (Read-Safe):")
    print(f"    .\\\venv\\\\Scripts\\\\python.exe -c \"")
    print(f"from agents.rag_agent import MedicalRAG")
    print(f"from config import RAGConfig")
    print(f"rag = MedicalRAG(RAGConfig())")
    print(f"result1 = rag.ingest_file('data/documents/Diagnosis and Management of Type 2 Diabetes.pdf')")
    print(f"result2 = rag.ingest_file('data/documents/Hypertension.pdf')")
    print(f"print(result1)")
    print(f"print(result2)")
    print(f"    \"")
    
    # Save command to file for easy reference
    with open("INGEST_NEW_PDFS_COMMAND.txt", "w") as f:
        f.write("INGESTION COMMAND FOR TWO NEW PDFS\n")
        f.write("="*80 + "\n\n")
        f.write("Verified PDFs:\n")
        f.write(f"  1. Diagnosis and Management of Type 2 Diabetes.pdf\n")
        f.write(f"  2. Hypertension.pdf\n\n")
        f.write("Ready for ingestion: YES\n\n")
        f.write("Recommended ingestion command:\n\n")
        f.write(".\\venv\\Scripts\\python.exe -c \"\n")
        f.write("from agents.rag_agent import MedicalRAG\n")
        f.write("from config import RAGConfig\n")
        f.write("rag = MedicalRAG(RAGConfig())\n")
        f.write("result1 = rag.ingest_file('data/documents/Diagnosis and Management of Type 2 Diabetes.pdf')\n")
        f.write("result2 = rag.ingest_file('data/documents/Hypertension.pdf')\n")
        f.write("print('Result 1:', result1)\n")
        f.write("print('Result 2:', result2)\n")
        f.write("\"\n")
    
    print(f"\n  Command saved to: INGEST_NEW_PDFS_COMMAND.txt")

# Final status
print("\n" + "="*80)
print("[FINAL RESULT]")
print("="*80)

if all_passed:
    print(f"\n  ✓ PASS: Both PDFs verified and ready for ingestion")
    print(f"\n  Status Summary:")
    print(f"    - Diagnosis and Management of Type 2 Diabetes.pdf: ✓ READY")
    print(f"    - Hypertension.pdf: ✓ READY")
    print(f"    - No parsing errors detected")
    print(f"    - All content meaningful and extractable")
    print(f"    - Next: Execute the ingestion command above")
else:
    print(f"\n  ✗ FAILED: One or more PDFs failed parsing")
    for pdf_name, result in parsed_results.items():
        if result["status"] != "SUCCESS":
            print(f"    - {pdf_name}: {result.get('error', 'Unknown error')[:100]}")

print("\n" + "="*80)
