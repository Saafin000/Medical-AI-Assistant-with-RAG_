# Embedding & Qdrant Implementation: Strict Read-Only Inspection

**Date:** November 16, 2026  
**Scope:** Inspect exact code locations for Azure embeddings, Qdrant initialization, collection management  
**Status:** READ-ONLY INSPECTION (no modifications)  
**Inspected Files:**
1. `config.py`
2. `agents/rag_agent/vectorstore_qdrant.py`
3. `agents/rag_agent/__init__.py`

---

## 1. Azure Embeddings Initialization

### Location: `config.py` - `RAGConfig.__init__()` (Lines 46-72)

**Initialization Method:**

```python
class RAGConfig:
    def __init__(self):
        # ... [other config]
        
        # Initialize Azure OpenAI Embeddings (lazy-loaded if vars are set)
        embedding_model_name = os.getenv("embedding_model_name")
        if embedding_model_name:
            self.embedding_model = AzureOpenAIEmbeddings(
                deployment = os.getenv("embedding_deployment_name"),
                model = embedding_model_name,
                azure_endpoint = os.getenv("embedding_azure_endpoint"),
                openai_api_key = os.getenv("embedding_openai_api_key"),
                openai_api_version = os.getenv("embedding_openai_api_version")
            )
        else:
            self.embedding_model = None
```

**Key Details:**

| Aspect | Value | Location |
|--------|-------|----------|
| **Class** | `AzureOpenAIEmbeddings` | Imported line 14 from `langchain_openai` |
| **Instance Variable** | `self.embedding_model` | Line 63 assignment |
| **Initialization Pattern** | Lazy-loaded (conditional) | Lines 62-72 |
| **Condition** | `if embedding_model_name:` | Line 62 |
| **Fallback** | `None` | Line 72 |
| **Environment Variables Used** | 5 variables | Lines 64-68 |

**Environment Variable Dependencies:**

```
embedding_model_name          (line 62, condition)
embedding_deployment_name     (line 64, deployment param)
embedding_model_name          (line 65, model param - reused)
embedding_azure_endpoint      (line 66, azure_endpoint param)
embedding_openai_api_key      (line 67, openai_api_key param)
embedding_openai_api_version  (line 68, openai_api_version param)
```

**Current State:** Not initialized (environment variables not set)
- During Step 2 Groq migration, embedding_model is None (lines 62-72)

---

## 2. Embedding Dimension and Collection Configuration

### Location A: `config.py` - `RAGConfig.__init__()` (Lines 47-60)

**Configuration Properties:**

```python
class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"              # Line 47
        self.embedding_dim = 1536                   # Line 48 - HARDCODED
        self.distance_metric = "Cosine"             # Line 49 - HARDCODED
        self.use_local = True                       # Line 50
        self.vector_local_path = "./data/qdrant_db" # Line 51
        self.doc_local_path = "./data/docs_db"      # Line 52
        self.parsed_content_dir = "./data/parsed_docs" # Line 53
        self.url = os.getenv("QDRANT_URL")          # Line 54 - Optional
        self.api_key = os.getenv("QDRANT_API_KEY")  # Line 55 - Optional
        self.collection_name = "medical_assistance_rag" # Line 56 - HARDCODED
        self.chunk_size = 512                       # Line 57
        self.chunk_overlap = 50                      # Line 58
```

**Critical Configuration:**

| Parameter | Value | Type | Modifiable | Usage |
|-----------|-------|------|-----------|-------|
| `embedding_dim` | 1536 | int (hardcoded) | Requires code change | Passed to Qdrant VectorParams |
| `distance_metric` | "Cosine" | string (hardcoded) | Requires code change | Passed to Distance.COSINE |
| `collection_name` | "medical_assistance_rag" | string (hardcoded) | Requires code change | Used in all Qdrant operations |
| `vector_local_path` | "./data/qdrant_db" | string | Via config change | Qdrant client storage path |

**Storage Location:**
- These properties are stored in `RAGConfig` instance
- Accessed by `VectorStore` via `config.rag` (line 57 in vectorstore_qdrant.py)

---

## 3. Qdrant Collection Creation and Loading

### Location B: `agents/rag_agent/vectorstore_qdrant.py` - `VectorStore` class

#### 3.1 Collection Existence Check

**Method:** `_does_collection_exist()` (Lines 31-38)

```python
def _does_collection_exist(self) -> bool:
    """Check if the collection already exists in Qdrant."""
    try:
        collection_info = self.client.get_collections()
        collection_names = [collection.name for collection in collection_info.collections]
        return self.collection_name in collection_names
    except Exception as e:
        self.logger.error(f"Error checking for collection existence: {e}")
        return False
```

**Uses:**
- `self.collection_name` (from config, line 57 vectorstore_qdrant.py)
- `self.client.get_collections()` (Qdrant API)

**Returns:** Boolean (True/False)

---

#### 3.2 Collection Creation

**Method:** `_create_collection()` (Lines 40-50)

```python
def _create_collection(self):
    """Create a new collection with dense and sparse vectors."""
    try:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        self.logger.info(f"Created new collection: {self.collection_name}")
    except Exception as e:
        self.logger.error(f"Error creating collection: {e}")
        raise e
```

**Vector Configuration Details:**

| Vector Type | Configuration | Parameters |
|-------------|---|---|
| **Dense** | `VectorParams(size=self.embedding_dim, distance=Distance.COSINE)` | `size=1536`, `distance=Cosine` |
| **Sparse** | `SparseVectorParams(index=models.SparseIndexParams(on_disk=False))` | `on_disk=False` (in-memory) |

**Parameters Used:**
- `self.collection_name` - "medical_assistance_rag" (line 44)
- `self.embedding_dim` - 1536 (line 45)
- `Distance.COSINE` - hardcoded distance metric (line 45)

---

#### 3.3 Collection Loading

**Method:** `load_vectorstore()` (Lines 52-77)

```python
def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
    """
    Load existing vectorstore and docstore for retrieval operations without ingesting new documents.
    
    Returns:
        Tuple containing (vectorstore, docstore)
    """
    # Check if collection exists
    if not self._does_collection_exist():
        self.logger.error(f"Collection {self.collection_name} does not exist. Please ingest documents first.")
        raise ValueError(f"Collection {self.collection_name} does not exist")
        
    # Setup sparse embeddings
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Initialize vector store
    qdrant_vectorstore = QdrantVectorStore(
        client=self.client,
        collection_name=self.collection_name,
        embedding=self.embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    
    # Document storage
    docstore = LocalFileStore(self.docstore_local_path)
    
    self.logger.info(f"Successfully loaded existing vectorstore and docstore")
    return qdrant_vectorstore, docstore
```

**Critical Operations:**

| Line | Operation | Detail |
|------|-----------|--------|
| 57 | Existence check | Calls `_does_collection_exist()` |
| 65 | Sparse embeddings setup | `FastEmbedSparse(model_name="Qdrant/bm25")` (BM25 keyword index) |
| 68-77 | QdrantVectorStore init | Creates vector store instance |

---

## 4. Dense and Sparse Vector Configuration

### Location C: `agents/rag_agent/vectorstore_qdrant.py` - Collection Creation & Loading

**Dense Vector Configuration (Line 45):**

```python
vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
```

**Components:**
- **Key:** `"dense"` - Named vector configuration
- **Size:** `self.embedding_dim` = 1536 dimensions
- **Distance:** `Distance.COSINE` - Cosine similarity metric

**Sparse Vector Configuration (Lines 46-48):**

```python
sparse_vectors_config={
    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
}
```

**Components:**
- **Key:** `"sparse"` - Named sparse vector configuration
- **Index:** `SparseIndexParams(on_disk=False)` - BM25 in-memory index

**Hybrid Retrieval Setup (Lines 71-76 in load_vectorstore):**

```python
qdrant_vectorstore = QdrantVectorStore(
    client=self.client,
    collection_name=self.collection_name,
    embedding=self.embedding_model,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)
```

**Hybrid Configuration Details:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `embedding` | `self.embedding_model` | Dense vector embedder (Azure) |
| `sparse_embedding` | `FastEmbedSparse(...)` | Sparse vector (BM25) |
| `retrieval_mode` | `RetrievalMode.HYBRID` | Use both dense + sparse |
| `vector_name` | `"dense"` | Reference to dense config |
| `sparse_vector_name` | `"sparse"` | Reference to sparse config |

---

## 5. Embedding Model Propagation into QdrantVectorStore

### Location D: VectorStore initialization and QdrantVectorStore usage

#### 5.1 Embedding Model Assignment in `__init__`

**File:** `agents/rag_agent/vectorstore_qdrant.py` - `VectorStore.__init__()` (Lines 15-28)

```python
def __init__(self, config):
    self.logger = logging.getLogger(__name__)
    self.collection_name = config.rag.collection_name  # Line 17
    self.embedding_dim = config.rag.embedding_dim      # Line 18
    self.distance_metric = config.rag.distance_metric  # Line 19
    self.embedding_model = config.rag.embedding_model  # Line 20 - CRITICAL
    self.retrieval_top_k = config.rag.top_k            # Line 21
    self.vector_search_type = config.rag.vector_search_type # Line 22
    self.vectorstore_local_path = config.rag.vector_local_path # Line 23
    self.docstore_local_path = config.rag.doc_local_path      # Line 24
    
    self.client = QdrantClient(path=self.vectorstore_local_path) # Line 27
```

**Critical Line:** Line 20
```python
self.embedding_model = config.rag.embedding_model
```

**Current State:** `self.embedding_model = None` (because Azure env vars not set)

---

#### 5.2 Embedding Model Usage in `create_vectorstore()`

**File:** `agents/rag_agent/vectorstore_qdrant.py` - `create_vectorstore()` (Lines 130-145)

```python
# Initialize vector store
qdrant_vectorstore = QdrantVectorStore(
    client=self.client,
    collection_name=self.collection_name,
    embedding=self.embedding_model,              # Line 140 - PASSED HERE
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

# Ingest documents into vector and doc stores
qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)  # Line 147
```

**Flow:**
1. `self.embedding_model` (from config, line 20)
2. Passed to `QdrantVectorStore(embedding=...)` (line 140)
3. Used internally by `add_documents()` to generate embeddings (line 147)

---

#### 5.3 Embedding Model Usage in `load_vectorstore()`

**File:** `agents/rag_agent/vectorstore_qdrant.py` - `load_vectorstore()` (Lines 68-77)

```python
qdrant_vectorstore = QdrantVectorStore(
    client=self.client,
    collection_name=self.collection_name,
    embedding=self.embedding_model,              # Line 74 - PASSED HERE
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)
```

**Flow:**
1. `self.embedding_model` (from config, line 20)
2. Passed to `QdrantVectorStore(embedding=...)` (line 74)
3. Used internally by `similarity_search_with_score()` to embed queries

---

#### 5.4 Query Embedding in `retrieve_relevant_chunks()`

**File:** `agents/rag_agent/vectorstore_qdrant.py` - `retrieve_relevant_chunks()` (Lines 173-181)

```python
def retrieve_relevant_chunks(
        self,
        query: str,
        vectorstore: QdrantVectorStore,
        docstore: LocalFileStore,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Retrieve relevant chunks based on a query.
    """
    # Use similarity_search_with_score to get documents and scores
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=self.retrieval_top_k
    )
```

**Embedding Usage:**
- `vectorstore.similarity_search_with_score()` internally calls `embedding.embed_query(query)`
- Uses the same embedding model instance passed in `load_vectorstore()`

**Current State:** `self.embedding_model = None` → Will fail at query time

---

## 6. Methods Requiring Modification for HuggingFace + 384-dim Support

### 6.1 Configuration Changes Required

**File:** `config.py` - `RAGConfig.__init__()` (Lines 46-72)

**Method/Section:** RAGConfig initialization

**Current Code:**
```python
self.embedding_dim = 1536  # Line 48 - HARDCODED
self.collection_name = "medical_assistance_rag"  # Line 56 - HARDCODED

# Lines 62-72
embedding_model_name = os.getenv("embedding_model_name")
if embedding_model_name:
    self.embedding_model = AzureOpenAIEmbeddings(
        deployment = os.getenv("embedding_deployment_name"),
        model = embedding_model_name,
        azure_endpoint = os.getenv("embedding_azure_endpoint"),
        openai_api_key = os.getenv("embedding_openai_api_key"),
        openai_api_version = os.getenv("embedding_openai_api_version")
    )
else:
    self.embedding_model = None
```

**Required Modifications:**

| Line | Change | Reason |
|------|--------|--------|
| 48 | Change `embedding_dim` from 1536 → 384 (or make dynamic) | Support different embedding dimensions |
| 56 | Change `collection_name` to "medical_assistance_rag_hf" (or make dynamic) | Distinguish from Azure collection |
| 62-72 | Add conditional branch for HuggingFace embeddings | Support local embeddings |
| N/A (new) | Add `use_local_embeddings` configuration flag | Switch between Azure and HF |

**Modification Strategy:**
```python
# Pseudocode - Not implementing, only describing
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false") == "true"

if use_local_embeddings:
    self.embedding_dim = 384
    self.collection_name = "medical_assistance_rag_hf"
    # Initialize HuggingFaceEmbeddings
    self.embedding_model = HuggingFaceEmbeddings(...)
else:
    self.embedding_dim = 1536
    self.collection_name = "medical_assistance_rag"
    # Initialize AzureOpenAIEmbeddings (existing code)
```

---

### 6.2 VectorStore Code - No Modifications Needed

**File:** `agents/rag_agent/vectorstore_qdrant.py`

**Finding:** This file requires **ZERO CODE CHANGES** for HuggingFace support.

**Why:** The VectorStore class uses abstraction:
- Line 20: `self.embedding_model = config.rag.embedding_model` (accepts any model)
- Line 45: Uses generic `self.embedding_dim` (works with any dimension)
- Line 44: Uses generic `self.collection_name` (works with any name)
- Lines 140, 74: Pass embedding model to QdrantVectorStore (accepts any model)

**LangChain Abstraction:**
- `QdrantVectorStore` accepts any object with `embed_documents()` and `embed_query()` methods
- Both Azure and HuggingFace embeddings implement this interface
- No code changes needed in VectorStore class

---

### 6.3 RAG Agent Initialization - No Modifications Needed

**File:** `agents/rag_agent/__init__.py` - `MedicalRAG.__init__()` (Lines 17-28)

**Current Code:**
```python
def __init__(self, config):
    """Initialize the RAG Agent."""
    self.logger = logging.getLogger(f"{self.__module__}")
    self.logger.info("Initializing Medical RAG system")
    self.config = config
    self.doc_parser = MedicalDocParser()
    self.content_processor = ContentProcessor(config)
    self.vector_store = VectorStore(config)  # Line 25 - Passes config
    self.reranker = Reranker(config)
    self.query_expander = QueryExpander(config)
    self.response_generator = ResponseGenerator(config)
```

**Finding:** This file requires **ZERO CODE CHANGES**.

**Why:**
- Line 25: Passes `config` to VectorStore
- VectorStore handles all embedding logic
- MedicalRAG is agnostic to embedding model type

---

### 6.4 RAG Agent Query Processing - No Modifications Needed

**File:** `agents/rag_agent/__init__.py` - `process_query()` (Lines 162-213)

**Current Code (excerpt):**
```python
# Step 2: Retrieval
vectorstore, docstore = self.vector_store.load_vectorstore()
retrieved_documents = self.vector_store.retrieve_relevant_chunks(
    query=query,
    vectorstore=vectorstore,
    docstore=docstore,
)
```

**Finding:** This file requires **ZERO CODE CHANGES**.

**Why:**
- Calls generic `load_vectorstore()` method
- Calls generic `retrieve_relevant_chunks()` method
- Both methods work with any embedding model (abstracted in VectorStore)

---

## 7. Summary: Modification Map

### Files That MUST Be Modified

| File | Class/Method | Lines | Changes | Reason |
|------|---|---|---|---|
| `config.py` | `RAGConfig.__init__()` | 48, 56, 62-72 | Add HF conditional, change dim/collection | Primary configuration |
| `requirements.txt` | N/A | N/A | Add `langchain-huggingface` | New dependency |

**Total lines to modify:** ~15-20 lines (in config.py)

---

### Files That DO NOT Require Modification

| File | Reason |
|---|---|
| `agents/rag_agent/vectorstore_qdrant.py` | Uses generic config params and LangChain abstraction |
| `agents/rag_agent/__init__.py` | Uses VectorStore abstraction, no embedding logic |
| `agents/rag_agent/content_processor.py` | Uses chunker model (Groq), not embeddings |
| `agents/rag_agent/reranker.py` | Uses CrossEncoder, independent of embeddings |
| `agents/rag_agent/doc_parser.py` | Document parsing, no embedding dependency |
| All other files | No embedding dependencies |

**Total files requiring changes:** 2 (config.py + requirements.txt)

---

## 8. Data Flow: Embedding Model Lifecycle

### Current State (Azure, None)

```
Config Initialization
    ↓
RAGConfig.__init__() [config.py:62-72]
    ↓
Check embedding_model_name env var [config.py:62]
    ↓
NOT SET → self.embedding_model = None [config.py:72]
    ↓
VectorStore.__init__() [vectorstore_qdrant.py:20]
    ↓
self.embedding_model = config.rag.embedding_model = None
    ↓
create_vectorstore() OR load_vectorstore()
    ↓
QdrantVectorStore(embedding=None) → WILL FAIL
```

### Proposed State (HuggingFace, 384-dim)

```
Config Initialization
    ↓
RAGConfig.__init__() [config.py - MODIFIED]
    ↓
Check USE_LOCAL_EMBEDDINGS env var
    ↓
IF true:
    ├─ self.embedding_dim = 384 [CHANGED from 1536]
    ├─ self.collection_name = "medical_assistance_rag_hf" [CHANGED]
    └─ self.embedding_model = HuggingFaceEmbeddings(...) [NEW]
    
    OR
    
IF false (default):
    ├─ self.embedding_dim = 1536 [UNCHANGED]
    ├─ self.collection_name = "medical_assistance_rag" [UNCHANGED]
    └─ self.embedding_model = AzureOpenAIEmbeddings(...) [UNCHANGED]
    ↓
VectorStore.__init__() [vectorstore_qdrant.py - NO CHANGE]
    ↓
self.embedding_model = config.rag.embedding_model [Generic]
    ↓
create_vectorstore() OR load_vectorstore()
    ↓
QdrantVectorStore(embedding=<HF or Azure>) → Works for both
```

---

## 9. Critical Dependencies and Coupling Points

### Dimension Coupling

**Points where 1536 dimension is used:**

1. **config.py, line 48:**
   ```python
   self.embedding_dim = 1536  # HARDCODED
   ```
   
2. **vectorstore_qdrant.py, line 45:**
   ```python
   vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
   ```

**Impact if not updated:** Dimension mismatch → Qdrant collection creation fails

---

### Collection Name Coupling

**Points where collection name is used:**

1. **config.py, line 56:**
   ```python
   self.collection_name = "medical_assistance_rag"  # HARDCODED
   ```

2. **vectorstore_qdrant.py, line 44:**
   ```python
   collection_name=self.collection_name,
   ```

3. **vectorstore_qdrant.py, lines 57, 65, 71:**
   ```python
   return self.collection_name in collection_names  # Line 35
   self.client.create_collection(collection_name=self.collection_name, ...)  # Line 44
   qdrant_vectorstore = QdrantVectorStore(collection_name=self.collection_name, ...)  # Line 71
   ```

**Impact if not changed:** All operations query same collection (no separation)

---

### Embedding Model Interface

**Points where embedding model is used:**

1. **config.py, lines 62-72:** Initialization
2. **vectorstore_qdrant.py, line 20:** Assignment
3. **vectorstore_qdrant.py, line 140:** Passed to QdrantVectorStore (create)
4. **vectorstore_qdrant.py, line 74:** Passed to QdrantVectorStore (load)
5. **vectorstore_qdrant.py, line 179:** Used indirectly via `similarity_search_with_score()`

**LangChain Interface Required:**
```python
class EmbeddingModel:
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
```

Both Azure and HuggingFace implementations support this.

---

## 10. Exact Modification Points (Implementation Reference)

### Point 1: config.py - Collection Name Configuration

**Current:**
```python
# Line 56
self.collection_name = "medical_assistance_rag"  # HARDCODED - change this
```

**For separate 384-dim collection:**
```python
# Option A: Hardcode new name
self.collection_name = "medical_assistance_rag_hf"

# Option B: Dynamic based on environment variable
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
self.collection_name = "medical_assistance_rag_hf" if use_local_embeddings else "medical_assistance_rag"

# Option C: Allow override
self.collection_name = os.getenv("COLLECTION_NAME", "medical_assistance_rag")
```

---

### Point 2: config.py - Embedding Dimension Configuration

**Current:**
```python
# Line 48
self.embedding_dim = 1536  # HARDCODED - change this for HF support
```

**For 384-dim embeddings:**
```python
# Option A: Hardcode new dimension
self.embedding_dim = 384

# Option B: Dynamic based on embedding model
if use_local_embeddings:
    self.embedding_dim = 384
else:
    self.embedding_dim = 1536

# Option C: Auto-detect from model (if supported)
# self.embedding_dim = self.embedding_model.embedding_dimension
```

---

### Point 3: config.py - Embedding Model Initialization

**Current:**
```python
# Lines 62-72
embedding_model_name = os.getenv("embedding_model_name")
if embedding_model_name:
    self.embedding_model = AzureOpenAIEmbeddings(...)
else:
    self.embedding_model = None
```

**For HuggingFace support (add to existing code):**
```python
# After line 72, add:
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

if use_local_embeddings:
    # NEW: Initialize HuggingFace embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    self.embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # or "cuda"
    )
elif embedding_model_name:
    # EXISTING: Azure path unchanged
    self.embedding_model = AzureOpenAIEmbeddings(...)
else:
    self.embedding_model = None
```

---

### Point 4: requirements.txt - Add HuggingFace Integration

**Current:**
```
langchain-community==0.3.19
```

**Add:**
```
langchain-huggingface==0.1.0  # (or latest compatible version)
```

**Note:** sentence-transformers already present (v3.4.1)

---

## 11. Verification Checklist for Implementation

### Pre-Implementation
- [ ] Backup existing Qdrant collection (medical_assistance_rag)
- [ ] Verify sentence-transformers installed (v3.4.1)
- [ ] Verify langchain-huggingface availability or plan to add

### During Implementation
- [ ] Update config.py embedding_dim (line 48)
- [ ] Update config.py collection_name (line 56)
- [ ] Add HuggingFace initialization (after line 72)
- [ ] Add use_local_embeddings flag handling
- [ ] Update requirements.txt with langchain-huggingface

### Post-Implementation
- [ ] Verify config initialization with new embeddings
- [ ] Verify VectorStore accepts HF embeddings (no errors)
- [ ] Verify new collection created (384-dim)
- [ ] Verify old collection still accessible
- [ ] Test re-ingestion with HF embeddings
- [ ] Test query retrieval with HF embeddings

---

## 12. Code Snippets Ready for Reference

### To Copy (when implementing):

```python
# HuggingFace initialization snippet
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # or "cuda"
    cache_folder="./models/embeddings"  # optional
)
```

```python
# Collection name selection
use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
collection_name = "medical_assistance_rag_hf" if use_local_embeddings else "medical_assistance_rag"
embedding_dim = 384 if use_local_embeddings else 1536
```

---

## Conclusion

### Summary

**Azure Embedding Initialization:** config.py lines 62-72 (currently lazy-loaded as None)  
**Collection Configuration:** config.py lines 48, 56 (hardcoded 1536-dim, "medical_assistance_rag")  
**Qdrant Operations:** vectorstore_qdrant.py (all methods use generic config params)  
**Embedding Propagation:** config → VectorStore → QdrantVectorStore (clean abstraction)

### Modification Requirements

| Change | File | Lines | Effort |
|--------|------|-------|--------|
| Add HF embedding init | config.py | ~62-90 | ~15 new lines |
| Update collection name | config.py | 56 | 1 line change |
| Update embedding_dim | config.py | 48 | 1 line change |
| Add HuggingFace pkg | requirements.txt | N/A | 1 line add |
| **Total** | **2 files** | **~20 lines** | **~20 minutes** |

### Files NOT Requiring Changes

All other files (vectorstore_qdrant.py, __init__.py, etc.) work with any embedding model due to LangChain abstraction.

---

**Inspection Status:** ✅ COMPLETE  
**Modifications Made:** NONE (read-only as requested)  
**Ready for Implementation:** YES  
**Documentation:** Complete with code references and line numbers
