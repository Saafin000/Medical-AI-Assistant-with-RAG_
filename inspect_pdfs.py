#!/usr/bin/env python3
"""
Inspect PDF documents in data/raw and data/raw_extras
Extract metadata, titles, and content summaries
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
import json

def extract_pdf_info(pdf_path):
    """Extract metadata and first page content from PDF"""
    try:
        reader = PdfReader(pdf_path)
        
        # Get metadata
        metadata = reader.metadata or {}
        num_pages = len(reader.pages)
        
        # Get first page text
        first_page_text = ""
        if num_pages > 0:
            first_page_text = reader.pages[0].extract_text()[:500]
        
        return {
            "filename": os.path.basename(pdf_path),
            "pages": num_pages,
            "title": metadata.get("/Title", "Not set"),
            "author": metadata.get("/Author", "Not set"),
            "subject": metadata.get("/Subject", "Not set"),
            "first_page_snippet": first_page_text,
            "file_size_kb": os.path.getsize(pdf_path) / 1024
        }
    except Exception as e:
        return {
            "filename": os.path.basename(pdf_path),
            "error": str(e)
        }

# Scan both directories
raw_dir = "data/raw"
raw_extras_dir = "data/raw_extras"

print("=" * 100)
print("PDF INSPECTION REPORT")
print("=" * 100)

print("\n[1] DATA/RAW DIRECTORY")
print("-" * 100)

raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".pdf")])

for filename in raw_files:
    filepath = os.path.join(raw_dir, filename)
    info = extract_pdf_info(filepath)
    
    print(f"\n📄 {filename}")
    print(f"   Pages: {info.get('pages', 'N/A')}")
    print(f"   Title: {info.get('title', 'N/A')}")
    print(f"   Author: {info.get('author', 'N/A')}")
    print(f"   Subject: {info.get('subject', 'N/A')}")
    print(f"   Size: {info.get('file_size_kb', 'N/A'):.1f} KB")
    if "error" not in info:
        snippet = info.get('first_page_snippet', '')[:100]
        print(f"   First page snippet: {snippet}...")

print("\n\n[2] DATA/RAW_EXTRAS DIRECTORY")
print("-" * 100)

extras_files = sorted([f for f in os.listdir(raw_extras_dir) if f.endswith(".pdf")])

for filename in extras_files:
    filepath = os.path.join(raw_extras_dir, filename)
    info = extract_pdf_info(filepath)
    
    print(f"\n📄 {filename}")
    print(f"   Pages: {info.get('pages', 'N/A')}")
    print(f"   Title: {info.get('title', 'N/A')}")
    print(f"   Author: {info.get('author', 'N/A')}")
    print(f"   Subject: {info.get('subject', 'N/A')}")
    print(f"   Size: {info.get('file_size_kb', 'N/A'):.1f} KB")
    if "error" not in info:
        snippet = info.get('first_page_snippet', '')[:100]
        print(f"   First page snippet: {snippet}...")

print("\n\n" + "=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)
