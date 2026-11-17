# Implementation Quick Reference: Embedding Migration

**Purpose:** Single-page reference for exact code locations and modifications needed  
**Target:** HuggingFace all-MiniLM-L6-v2 (384-dim) support with parallel Azure collection  
**Status:** Ready for implementation

---

## Files to Modify: 2

### File 1: `config.py`

#### Change 1: Line 48 - Embedding Dimension
```python
# BEFORE:
self.embedding_dim = 1536  # Add the embedding dimension here

# AFTER (for separate 384-dim collection):
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
self.embedding_dim = 384 if use_local_embeddings else 1536
```

#### Change 2: Line 56 - Collection Name
```python
# BEFORE:
self.collection_name = "medical_assistance_rag"  # Ensure a valid name

# AFTER (for separate collection):
self.collection_name = "medical_assistance_rag_hf" if use_local_embeddings else "medical_assistance_rag"
```

#### Change 3: Lines 62-72 - Add HuggingFace Initialization
```python
# ADD BEFORE EXISTING CODE (line 62):
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

# AFTER EXISTING Azure initialization (line 72), ADD:
if use_local_embeddings:
    # Initialize HuggingFace embeddings for 384-dim collection
    from langchain_huggingface import HuggingFaceEmbeddings
    self.embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # or "cuda" for GPU
        cache_folder="./models/embeddings"  # optional local cache
    )
elif embedding_model_name:
    # EXISTING Azure path - unchanged
    self.embedding_model = AzureOpenAIEmbeddings(...)
else:
    self.embedding_model = None
```

---

### File 2: `requirements.txt`

#### Addition: Add LangChain HuggingFace Package
```
# ADD after line 7 (langchain-community):
langchain-huggingface==0.1.0
```

**Note:** sentence-transformers==3.4.1 already present (line 30)

---

## Files NOT Requiring Changes: 7+

```
agents/rag_agent/vectorstore_qdrant.py     (abstraction layer - works with any embedding)
agents/rag_agent/__init__.py                (uses VectorStore - works with any embedding)
agents/rag_agent/content_processor.py       (uses Groq chunker - no embedding dependency)
agents/rag_agent/reranker.py                (uses CrossEncoder - no embedding dependency)
agents/rag_agent/doc_parser.py              (document parsing - no embedding dependency)
agents/rag_agent/query_expander.py          (query expansion - no embedding dependency)
agents/rag_agent/response_generator.py      (response generation - no embedding dependency)
ingest_rag_data.py                          (uses config - no direct changes)
```

---

## Verification: Environment Variables

### To Enable HuggingFace Embeddings
```bash
# Set environment variable:
export USE_LOCAL_EMBEDDINGS=true

# OR in .env file:
USE_LOCAL_EMBEDDINGS=true

# Then run:
python ingest_rag_data.py --dir ./data/documents/
```

### To Use Azure Embeddings (default)
```bash
# Option 1: Don't set USE_LOCAL_EMBEDDINGS
python ingest_rag_data.py --dir ./data/documents/

# Option 2: Explicitly set to false
export USE_LOCAL_EMBEDDINGS=false
python ingest_rag_data.py --dir ./data/documents/
```

---

## Data Flow After Implementation

### Azure (Default - USE_LOCAL_EMBEDDINGS=false or not set)
```
embedding_dim = 1536
collection_name = "medical_assistance_rag"
embedding_model = AzureOpenAIEmbeddings (if credentials set) or None
```

### HuggingFace (USE_LOCAL_EMBEDDINGS=true)
```
embedding_dim = 384
collection_name = "medical_assistance_rag_hf"
embedding_model = HuggingFaceEmbeddings (all-MiniLM-L6-v2)
```

---

## Code Locations Reference

| What | File | Lines | Details |
|------|------|-------|---------|
| Embedding init | config.py | 62-90 (modified) | Add HF conditional |
| Collection name | config.py | 56 | Change conditionally |
| Embedding dim | config.py | 48 | Change conditionally |
| VectorStore init | vectorstore_qdrant.py | 15-28 | NO CHANGE (uses config.embedding_model) |
| Collection create | vectorstore_qdrant.py | 40-50 | NO CHANGE (uses config params) |
| Collection load | vectorstore_qdrant.py | 52-77 | NO CHANGE (uses config.embedding_model) |
| RAG init | agents/rag_agent/__init__.py | 17-28 | NO CHANGE (passes config) |

---

## Test Commands (After Implementation)

### Test 1: Configuration Only
```bash
python -c "
from config import Config
import os
os.environ['USE_LOCAL_EMBEDDINGS'] = 'true'
config = Config()
print(f'HF Mode:')
print(f'  embedding_dim: {config.rag.embedding_dim}')
print(f'  collection_name: {config.rag.collection_name}')
print(f'  embedding_model type: {type(config.rag.embedding_model).__name__}')
"
```

### Test 2: Re-ingest with HuggingFace
```bash
export USE_LOCAL_EMBEDDINGS=true
python ingest_rag_data.py --dir ./data/documents/
```

### Test 3: Query with HuggingFace
```bash
export USE_LOCAL_EMBEDDINGS=true
python -c "
from config import Config
from agents.rag_agent import MedicalRAG

config = Config()
rag = MedicalRAG(config)
result = rag.process_query('What is COVID-19?')
print(result)
"
```

---

## Rollback (If Issues Occur)

### Option 1: Config Variable
```bash
# Unset environment variable to revert to Azure
unset USE_LOCAL_EMBEDDINGS
# OR
export USE_LOCAL_EMBEDDINGS=false

# Application automatically uses old collection and Azure embeddings
```

### Option 2: Manual Collection Switch
```python
# In code:
config.rag.collection_name = "medical_assistance_rag"  # Revert to Azure collection
config.rag.embedding_dim = 1536
```

### Option 3: Qdrant Reset (Last Resort)
```bash
# Restore from backup
rm -rf ./data/qdrant_db/
cp -r ./backups/qdrant_db.backup ./data/qdrant_db/
```

---

## Success Criteria

After implementation, verify:

- [ ] `config.rag.embedding_dim = 384` (when USE_LOCAL_EMBEDDINGS=true)
- [ ] `config.rag.collection_name = "medical_assistance_rag_hf"` (when USE_LOCAL_EMBEDDINGS=true)
- [ ] `type(config.rag.embedding_model).__name__ = "HuggingFaceEmbeddings"` (when USE_LOCAL_EMBEDDINGS=true)
- [ ] New collection created in Qdrant with 384-dim vectors
- [ ] Old collection still accessible (backward compatible)
- [ ] Re-ingestion completes without errors
- [ ] Query retrieval returns results (verify quality manually)
- [ ] Both Azure and HF modes can coexist

---

## Common Issues & Fixes

### Issue 1: "Model download failed"
```
Error: Failed to download sentence-transformers/all-MiniLM-L6-v2
Fix: 
  1. Check internet connection
  2. Pre-download: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
  3. Use cache_folder in config for offline mode
```

### Issue 2: "Dimension mismatch 1536 vs 384"
```
Error: Invalid query vector dimension
Fix:
  1. Verify embedding_dim correctly set to 384
  2. Use separate collection_name to avoid mixing dimensions
  3. Check USE_LOCAL_EMBEDDINGS environment variable is set correctly
```

### Issue 3: "HuggingFaceEmbeddings not found"
```
Error: ImportError: cannot import name 'HuggingFaceEmbeddings'
Fix:
  1. Verify langchain-huggingface installed: pip install langchain-huggingface
  2. Try: from langchain_community.embeddings import HuggingFaceEmbeddings (fallback)
  3. Update requirements.txt and reinstall
```

### Issue 4: "Collection already exists" error
```
Error: Collection medical_assistance_rag_hf already exists
Fix:
  This is normal if re-running ingestion
  Action: Either
    - Delete collection and re-ingest: client.delete_collection("medical_assistance_rag_hf")
    - Update existing collection: code will upsert by default
```

---

## Timeline Estimate

| Task | Duration | Notes |
|------|----------|-------|
| Update config.py | 10 min | Add 20 lines of code |
| Update requirements.txt | 2 min | Add 1 line |
| Test locally | 5 min | Verify imports work |
| Re-ingest documents | 15-120 min | Depends on corpus size |
| Validate results | 30-60 min | Manual quality check |
| **TOTAL** | **1-3 hours** | Mostly automated |

---

## Minimal Implementation (Core Changes)

If time-constrained, minimum viable changes:

```python
# config.py - RAGConfig.__init__() - Lines to ADD/CHANGE:

# Line 48 - CHANGE:
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
self.embedding_dim = 384 if use_local_embeddings else 1536

# Line 56 - CHANGE:
self.collection_name = "medical_assistance_rag_hf" if use_local_embeddings else "medical_assistance_rag"

# After Line 72 - ADD:
if use_local_embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

## Files Snapshot for Reference

### config.py - RAGConfig Initialization
- **Starts:** Line 46 (`class RAGConfig:`)
- **Ends:** Line 115 (`self.context_limit = 20`)
- **Key lines:** 48 (dim), 56 (collection), 62-72 (embedding init)

### vectorstore_qdrant.py - VectorStore Class
- **Starts:** Line 13 (`class VectorStore:`)
- **Key methods:**
  - `__init__()` lines 15-28 (receives config)
  - `_create_collection()` lines 40-50 (creates 1536-dim collection)
  - `load_vectorstore()` lines 52-77 (loads existing collection)
  - `create_vectorstore()` lines 79-150 (ingests documents)

### agents/rag_agent/__init__.py - MedicalRAG Class
- **Starts:** Line 13 (`class MedicalRAG:`)
- **Key methods:**
  - `__init__()` lines 17-28 (initializes VectorStore)
  - `process_query()` lines 162-213 (retrieves from vectorstore)

---

**Last Updated:** November 16, 2026  
**Implementation Status:** READY ✓  
**Estimated Effort:** 2-3 hours (mostly automated)
