#!/usr/bin/env python3
"""
READ-ONLY content verification of 5 PDFs
Inspect actual document contents, text extraction, and medical topics
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader

def verify_pdf(pdf_path):
    """Verify PDF contents and extract information"""
    
    print(f"\n{'='*100}")
    print(f"📄 {os.path.basename(pdf_path)}")
    print(f"{'='*100}")
    
    try:
        # Check file exists and size
        if not os.path.exists(pdf_path):
            print(f"❌ ERROR: File not found - {pdf_path}")
            return
        
        file_size_kb = os.path.getsize(pdf_path) / 1024
        print(f"\n[1] FILE METADATA")
        print(f"    Path: {pdf_path}")
        print(f"    Size: {file_size_kb:.1f} KB")
        
        # Read PDF
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        print(f"\n[2] PDF METADATA")
        print(f"    Pages: {num_pages}")
        
        metadata = reader.metadata or {}
        print(f"    Title: {metadata.get('/Title', 'Not set')}")
        print(f"    Author: {metadata.get('/Author', 'Not set')}")
        print(f"    Subject: {metadata.get('/Subject', 'Not set')}")
        print(f"    Creator: {metadata.get('/Creator', 'Not set')}")
        print(f"    Producer: {metadata.get('/Producer', 'Not set')}")
        
        # Extract text from all pages
        print(f"\n[3] TEXT EXTRACTION")
        all_text = ""
        extraction_success = True
        pages_with_text = 0
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 0:
                    pages_with_text += 1
                all_text += page_text + "\n"
            except Exception as e:
                extraction_success = False
                print(f"    ⚠️  Page {page_num+1}: Extraction failed - {e}")
        
        total_chars = len(all_text)
        print(f"    Pages with extractable text: {pages_with_text}/{num_pages}")
        print(f"    Total extracted characters: {total_chars:,}")
        
        if total_chars > 0:
            print(f"    ✅ Text extraction: SUCCESS")
        else:
            print(f"    ❌ Text extraction: FAILED (possibly scanned/image-based)")
        
        # Analyze text content
        print(f"\n[4] CONTENT ANALYSIS")
        
        # First page analysis
        first_page_text = ""
        if num_pages > 0:
            first_page_text = reader.pages[0].extract_text()
        
        first_page_snippet = first_page_text[:300] if first_page_text else "(No text extracted)"
        print(f"    First page (first 300 chars):")
        print(f"    {first_page_snippet[:300]}")
        
        # Check for medical topics
        print(f"\n[5] MEDICAL TOPICS DETECTED")
        
        medical_keywords = {
            "COVID": ["covid", "coronavirus", "sars-cov", "pandemic", "covid-19"],
            "Brain Tumor": ["brain tumor", "glioma", "glioblastoma", "brain cancer", "tumor"],
            "Skin Lesion": ["melanoma", "skin lesion", "dermatology", "pigmented", "mole"],
            "Chest X-ray": ["chest x-ray", "chest xray", "pneumonia", "lung", "radiograph"],
            "Diabetes": ["diabetes", "glucose", "insulin", "hyperglycemia", "diabetic"],
            "Deep Learning": ["deep learning", "neural network", "cnn", "machine learning", "ai", "artificial intelligence"],
            "Classification": ["classification", "detect", "diagnosis", "predict", "algorithm"],
            "Medical Imaging": ["imaging", "ct scan", "mri", "ultrasound", "radiograph", "radiology"],
        }
        
        detected_topics = set()
        text_lower = all_text.lower()
        
        for topic, keywords in medical_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_topics.add(topic)
                    break
        
        if detected_topics:
            for topic in sorted(detected_topics):
                print(f"    ✓ {topic}")
        else:
            print(f"    ⚠️  No medical topics detected")
        
        # Document type analysis
        print(f"\n[6] DOCUMENT TYPE")
        
        doc_type = "Unknown"
        if "abstract" in text_lower and "introduction" in text_lower:
            doc_type = "Research Paper"
        elif "clinical guideline" in text_lower or "recommendation" in text_lower:
            doc_type = "Clinical Guideline"
        elif "dataset" in text_lower or "images" in text_lower:
            doc_type = "Dataset/Review"
        elif "overview" in text_lower or "summary" in text_lower:
            doc_type = "Educational Overview"
        
        print(f"    Likely type: {doc_type}")
        
        # Check for images (by looking for image objects in PDF)
        print(f"\n[7] CONTENT TYPE")
        
        has_text = total_chars > 1000
        
        # Try to detect if PDF has images
        has_images = False
        try:
            for page in reader.pages:
                if "/XObject" in page["/Resources"]:
                    has_images = True
                    break
        except:
            pass
        
        if has_text and has_images:
            print(f"    Type: MIXED (text + images)")
        elif has_text:
            print(f"    Type: TEXT-BASED")
        elif has_images:
            print(f"    Type: IMAGE/SCANNED")
        else:
            print(f"    Type: UNKNOWN")
        
        # Suitability for RAG
        print(f"\n[8] SUITABILITY FOR RAG")
        
        suitability_score = 0
        reasons = []
        
        if total_chars > 1000:
            suitability_score += 2
            reasons.append("✓ Sufficient text content")
        elif total_chars > 100:
            suitability_score += 1
            reasons.append("⚠️  Limited text content")
        else:
            reasons.append("❌ No readable text")
        
        if detected_topics:
            suitability_score += 2
            reasons.append(f"✓ Medical topics: {', '.join(list(detected_topics)[:3])}")
        
        if num_pages > 3:
            suitability_score += 1
            reasons.append("✓ Sufficient depth (multiple pages)")
        
        if has_text:
            suitability_score += 1
            reasons.append("✓ Text is extractable")
        
        for reason in reasons:
            print(f"    {reason}")
        
        if suitability_score >= 5:
            print(f"\n    ✅ SUITABLE FOR RAG (Score: {suitability_score}/6)")
        elif suitability_score >= 3:
            print(f"\n    ⚠️  PARTIALLY SUITABLE (Score: {suitability_score}/6)")
        else:
            print(f"\n    ❌ NOT SUITABLE FOR RAG (Score: {suitability_score}/6)")
        
        return {
            "filename": os.path.basename(pdf_path),
            "pages": num_pages,
            "text_chars": total_chars,
            "topics": list(detected_topics),
            "suitable": suitability_score >= 4,
            "doc_type": doc_type
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# Verify all 5 PDFs
pdfs_to_verify = [
    "data/raw/brain_tumor_2024.pdf",
    "data/raw/brain_tumors_ucni.pdf",
    "data/raw/covid_chest_xray_2024.pdf",
    "data/raw/skin_lesion_2023.pdf",
    "data/raw_extras/diabetes.pdf"
]

print("\n\n")
print("█" * 100)
print("█ PDF CONTENT VERIFICATION - 5 DOCUMENTS".ljust(99) + "█")
print("█" * 100)

results = []
for pdf_path in pdfs_to_verify:
    result = verify_pdf(pdf_path)
    if result:
        results.append(result)

# Summary
print(f"\n\n{'='*100}")
print("VERIFICATION SUMMARY")
print(f"{'='*100}")

print(f"\n{len(results)} PDFs verified:\n")

for result in results:
    suitable_mark = "✅" if result["suitable"] else "⚠️"
    print(f"{suitable_mark} {result['filename']}")
    print(f"   Pages: {result['pages']}, Text: {result['text_chars']:,} chars")
    print(f"   Type: {result['doc_type']}")
    print(f"   Topics: {', '.join(result['topics']) if result['topics'] else 'None detected'}")
    print()

suitable_count = sum(1 for r in results if r["suitable"])
print(f"Suitable for RAG: {suitable_count}/{len(results)}")

print(f"\n{'='*100}")
