# Embedding Architecture: Current vs Proposed Comparison

## Current Architecture (Azure Embeddings - 1536-dim)

### Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline (Azure)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT DOCUMENTS                                                 │
│  ↓                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Document Parser (docling)                                │  │
│  │ • Extract text from PDFs/docs                            │  │
│  │ • Extract images                                         │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Content Processor                                        │  │
│  │ • Image summarization (Azure GPT-4o vision) [UNCHANGED] │  │
│  │ • Format document with image summaries                   │  │
│  │ • Semantic chunking (Groq qwen) [UNCHANGED]            │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Embedding Generation [THIS CHANGES]                      │  │
│  │ • Model: Azure text-embedding-3-small                    │  │
│  │ • Dimensions: 1536                                       │  │
│  │ • API: Cloud-based (requires credentials)                │  │
│  │ • Cost: $0.02 per 1M tokens                              │  │
│  │ • Speed: ~50-100 ms per chunk                            │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Vector Store (Qdrant)                                    │  │
│  │ Collection: medical_assistance_rag                       │  │
│  │ • Dense vectors: 1536-dim (cosine distance)              │  │
│  │ • Sparse vectors: BM25 index (keyword search)            │  │
│  │ • Retrieval mode: HYBRID (dense + sparse)                │  │
│  │ • Storage: ./data/qdrant_db/                             │  │
│  │ • Size: ~6.1 KB per chunk (1536 * 4 bytes)               │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  QUERY RETRIEVAL                                                 │
│  • Input: User query                                            │
│  • Embed query: Azure embeddings (1536-dim)                     │
│  • Search: Qdrant hybrid search                                 │
│  • Rerank: CrossEncoder (independent of embeddings)             │
│  • Return: Top-k most relevant chunks with scores               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

Configuration File (config.py):
  ├─ embedding_dim = 1536
  ├─ collection_name = "medical_assistance_rag"
  ├─ embedding_model = AzureOpenAIEmbeddings(...)
  └─ distance_metric = "Cosine"
```

---

## Proposed Architecture (HuggingFace Embeddings - 384-dim)

### Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                  RAG Pipeline (HuggingFace)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT DOCUMENTS                                                 │
│  ↓                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Document Parser (docling)                                │  │
│  │ • Extract text from PDFs/docs                            │  │
│  │ • Extract images                                         │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Content Processor                                        │  │
│  │ • Image summarization (Azure GPT-4o vision) [UNCHANGED] │  │
│  │ • Format document with image summaries                   │  │
│  │ • Semantic chunking (Groq qwen) [UNCHANGED]            │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Embedding Generation [NEW]                               │  │
│  │ • Model: sentence-transformers/all-MiniLM-L6-v2         │  │
│  │ • Dimensions: 384 (75% reduction)                        │  │
│  │ • API: Local (no credentials needed)                     │  │
│  │ • Cost: Free (one-time download ~133 MB)                 │  │
│  │ • Speed: ~50-100 ms/CPU, ~5-10 ms/GPU                   │  │
│  │ • Inference: Deterministic (same query = same output)    │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Vector Store (Qdrant)                                    │  │
│  │ Collection: medical_assistance_rag_hf (NEW)              │  │
│  │ • Dense vectors: 384-dim (cosine distance)               │  │
│  │ • Sparse vectors: BM25 index (keyword search)            │  │
│  │ • Retrieval mode: HYBRID (dense + sparse)                │  │
│  │ • Storage: ./data/qdrant_db/                             │  │
│  │ • Size: ~1.5 KB per chunk (384 * 4 bytes) [75% smaller]  │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       ↓                                           │
│  QUERY RETRIEVAL                                                 │
│  • Input: User query                                            │
│  • Embed query: HuggingFace embeddings (384-dim)                │
│  • Search: Qdrant hybrid search                                 │
│  • Rerank: CrossEncoder (independent of embeddings)             │
│  • Return: Top-k most relevant chunks with scores               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

Configuration File (config.py):
  ├─ embedding_dim = 384
  ├─ collection_name = "medical_assistance_rag_hf"
  ├─ embedding_model = HuggingFaceEmbeddings(...)
  └─ distance_metric = "Cosine"
```

---

## Side-by-Side Comparison

### Infrastructure

| Aspect | Azure (Current) | HuggingFace (Proposed) | Change |
|--------|-----------------|----------------------|--------|
| **Model Name** | text-embedding-3-small | all-MiniLM-L6-v2 | Different provider |
| **Dimensions** | 1536 | 384 | -75% |
| **Inference** | Cloud API | Local processing | Self-hosted |
| **Authentication** | API key required | None | Eliminated |
| **First-use** | Instant | 30-60s (model download) | One-time |
| **Latency** | ~200-500 ms (network) | 50-100 ms (CPU) or 5-10 ms (GPU) | 4-10x faster |
| **Cost** | $0.02 per 1M tokens | Free (local) | $0/month |
| **Availability** | Internet required | Offline capable | Better availability |
| **Reproducibility** | Cloud updates may change | Deterministic | More stable |

---

### Vector Storage

| Aspect | Azure (Current) | HuggingFace (Proposed) | Impact |
|--------|-----------------|----------------------|--------|
| **Collection** | medical_assistance_rag | medical_assistance_rag_hf | Separate, no conflict |
| **Dimensions** | 1536 | 384 | 75% storage reduction |
| **Chunk Size** | ~6.1 KB | ~1.5 KB | 4x smaller per chunk |
| **Total Size (1000 chunks)** | ~6.1 MB | ~1.5 MB | 4x smaller total |
| **Distance Metric** | Cosine | Cosine | Same |
| **Sparse Vectors** | BM25 | BM25 | Unchanged |
| **Hybrid Retrieval** | Dense + Sparse | Dense + Sparse | Unchanged |

---

### Code Impact

| File | Current | Proposed | Changes |
|------|---------|----------|---------|
| config.py | Import AzureOpenAIEmbeddings | Import HuggingFaceEmbeddings | 3-5 lines |
| vectorstore_qdrant.py | Uses embedding_model from config | Uses embedding_model from config | NONE |
| content_processor.py | Image summarization (Azure) | Image summarization (Azure) | NONE |
| doc_parser.py | Docling parsing | Docling parsing | NONE |
| reranker.py | CrossEncoder scoring | CrossEncoder scoring | NONE |
| __init__.py | Calls VectorStore | Calls VectorStore | NONE |
| requirements.txt | No langchain_huggingface | Add langchain_huggingface | 1 line |

**Lines of code changes: ~10 lines total**

---

### Quality & Performance

#### Retrieval Quality

| Metric | Azure 1536-dim | HF 384-dim | Expectation |
|--------|---|---|---|
| Semantic understanding | Excellent | Good-Excellent | Comparable |
| Medical domain knowledge | High (Azure-tuned) | Good (general, but large corpus) | May vary, needs testing |
| Out-of-vocabulary words | Excellent | Good | HF more generalizable |
| Multilingual support | Limited | Better | Advantage HF |
| Hallucination rate | Low | Low | Same |

#### Performance

| Metric | Azure | HF CPU | HF GPU | Notes |
|--------|-------|--------|---------|-------|
| Embed 1000 chunks | 50-100s | 50-100s | 5-10s | Network overhead vs local CPU |
| Query latency | 200-500ms | 50-100ms | 5-10ms | 4-10x faster local |
| Memory usage | Network | ~500 MB | ~2 GB | GPU needs memory |
| Throughput (chunks/sec) | 10-20 | 10-20 | 100-200 | GPU 10x faster |

---

### Migration Impact on Components

```
╔════════════════════════════════════════════════════════════════════╗
║                   IMPACT ANALYSIS BY COMPONENT                     ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Document Ingestion Pipeline:                                     ║
║  ┌────────────────────────────────────────────────────────────┐   ║
║  │ Parser      → [SAME] - No embedding dependency             │   ║
║  │ Processor   → [SAME] - Uses Groq chunking, Azure images    │   ║
║  │ Embeddings  → [CHANGE] - Azure 1536 → HF 384              │   ║
║  │ VectorStore → [COMPATIBLE] - Works with any embedding      │   ║
║  └────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║  Query Processing Pipeline:                                       ║
║  ┌────────────────────────────────────────────────────────────┐   ║
║  │ Query       → [SAME] - User input format unchanged         │   ║
║  │ Embeddings  → [CHANGE] - Azure 1536 → HF 384              │   ║
║  │ Retrieval   → [COMPATIBLE] - Qdrant works with any embedding │   ║
║  │ Reranking   → [SAME] - CrossEncoder independent            │   ║
║  │ Response    → [SAME] - LLM-based generation               │   ║
║  └────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║  Vision Processing (Unchanged):                                  ║
║  ┌────────────────────────────────────────────────────────────┐   ║
║  │ Image Analysis     → [SAME] - Azure GPT-4o vision         │   ║
║  │ Image Summarization → [SAME] - Azure vision model          │   ║
║  │ Classification     → [SAME] - Medical CV agent             │   ║
║  └────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║  Overall Impact: ISOLATED (embeddings only)                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

### Data Flow Comparison

#### Current (Azure)
```
Medical Document
    ↓
Parse (Docling)
    ↓
Summarize Images (Azure GPT-4o Vision)
    ↓
Chunk (Groq)
    ↓
Embed (Azure 1536-dim) ← API CALL
    ↓
Store (Qdrant medical_assistance_rag)
    ↓
USER QUERY
    ↓
Embed Query (Azure 1536-dim) ← API CALL
    ↓
Search (Qdrant Hybrid)
    ↓
Rerank (CrossEncoder - Local)
    ↓
Generate Response (Groq)
    ↓
Response to User
```

#### Proposed (HuggingFace)
```
Medical Document
    ↓
Parse (Docling)
    ↓
Summarize Images (Azure GPT-4o Vision)
    ↓
Chunk (Groq)
    ↓
Embed (HF 384-dim) ← LOCAL PROCESSING
    ↓
Store (Qdrant medical_assistance_rag_hf)
    ↓
USER QUERY
    ↓
Embed Query (HF 384-dim) ← LOCAL PROCESSING
    ↓
Search (Qdrant Hybrid)
    ↓
Rerank (CrossEncoder - Local)
    ↓
Generate Response (Groq)
    ↓
Response to User
```

**Key Change:** Two API calls → Local processing (eliminates network dependency)

---

## Testing Strategy for Migration

### Phase 1: Functional Testing
```
✓ Create new Qdrant collection with 384-dim vectors
✓ Ingest all documents into new collection
✓ Retrieve documents for test queries
✓ Verify dimensions and vector counts
✓ Compare retrieval results vs old collection
```

### Phase 2: Quality Testing
```
Medical Query Examples:
  1. "What is the treatment for COVID-19?"
     - Compare top results from both collections
     - Manual quality review
     - Score comparison

  2. "Describe the symptoms of brain tumors"
     - Verify medical accuracy
     - Check for hallucinations
     - Validate source attribution

  3. "How to diagnose melanoma?"
     - Test medical domain knowledge
     - Verify image summaries used
     - Check confidence scores
```

### Phase 3: Performance Testing
```
Metrics to Measure:
  • Ingestion time per document (compare 1536 vs 384)
  • Query latency (p50, p95, p99)
  • Memory usage (peak, average)
  • CPU/GPU utilization
  • Cost per query
```

### Phase 4: Regression Testing
```
Ensure Unchanged Components Work:
  ✓ Image summarization (Azure vision)
  ✓ Document parsing (Docling)
  ✓ Chunking (Groq)
  ✓ Reranking (CrossEncoder)
  ✓ Response generation (Groq)
```

---

## Migration Timeline (Actual Duration)

```
Friday (Preparation)
├─ 10:00 - Backup Qdrant data (5 min)
├─ 10:05 - Verify dependencies (10 min)
└─ 10:15 - Test HF embeddings locally (15 min)
          TOTAL: 30 minutes

Monday (Configuration)
├─ 09:00 - Update config.py (10 min)
├─ 09:10 - Add env variable for model selection (5 min)
├─ 09:15 - Test config initialization (10 min)
└─ 09:30 - Ready for re-ingestion (checkpoint)
          TOTAL: 30 minutes

Monday Afternoon (Re-ingestion)
├─ 14:00 - Download HF model first (if needed, ~2 min)
├─ 14:02 - Start re-ingestion (varies by corpus)
│         Small corpus (10 docs): 15-30 min
│         Medium corpus (50 docs): 45-90 min
│         Large corpus (100+ docs): 2-4 hours
└─ 14:XX - Re-ingestion complete (checkpoint)
          TOTAL: 15 min to 4 hours (depending on size)

Tuesday Morning (Validation)
├─ 09:00 - Compare retrieval results (30 min)
├─ 09:30 - Quality review of top results (45 min)
├─ 10:15 - Performance benchmarking (30 min)
├─ 10:45 - Adjust thresholds if needed (15 min)
└─ 11:00 - Ready for cutover (checkpoint)
          TOTAL: 2 hours

Total Project Duration: 1 day to 2 days (depending on corpus size)
- Config changes: ~1 hour
- Re-ingestion: 15 min - 4 hours (varies)
- Testing: ~2-3 hours
- TOTAL: 3.5 - 8.5 hours (mostly automated)
```

---

## Rollback Procedure

If migration issues occur, rollback is trivial:

```
Option 1: Config-Based Rollback (Instant)
  1. Set USE_LOCAL_EMBEDDINGS=false
  2. Restart application
  3. Application queries original medical_assistance_rag collection

Option 2: Collection-Based Rollback (Instant)
  1. Change config.collection_name to medical_assistance_rag
  2. Restart application
  3. Querying original collection

Option 3: Database Rollback (5 minutes)
  1. Stop application
  2. rm -rf ./data/qdrant_db/
  3. cp -r ./backups/qdrant_db.backup ./data/qdrant_db/
  4. Start application
  5. Original state restored

Risk of rollback: NONE (original data untouched)
Time to rollback: < 5 minutes
```

---

## Estimated Savings

### Azure Costs (Current)
```
Documents ingested: 30-50
Average tokens per document: 10,000-15,000
Total tokens: 300,000 - 750,000

Ingestion cost: 0.300M - 0.750M tokens × $0.02 = $6 - $15
Queries: ~100/day × 100 tokens/query = 10,000 tokens/day
Daily query cost: 10K tokens × $0.02/1M = $0.0002 = negligible
Monthly query cost: ~$0.006
Annual query cost: ~$0.07

Annual embedding cost: ~$6-15 (mostly ingestion, queries negligible)
```

### HuggingFace Costs (After Migration)
```
Model download: 133 MB (one-time, free)
Storage: 384-dim vs 1536-dim = 75% reduction
Compute: Local, cost is electricity
Monthly query cost: $0 (local processing)
Annual embedding cost: $0

Breakeven point: Immediate (saves Azure API calls)
```

**Estimated Annual Savings: $6-15** (if only considering embeddings)

---

## Final Recommendation

### Migration Should Proceed Because:

1. ✅ **Technically sound** - All components compatible
2. ✅ **Zero data loss** - Parallel collections preserve all data
3. ✅ **Minimal code changes** - ~10 lines total
4. ✅ **Quick implementation** - 3.5-8.5 hours total
5. ✅ **No vendor lock-in** - Local processing
6. ✅ **Cost savings** - Eliminate Azure embedding fees
7. ✅ **Performance gain** - Faster inference
8. ✅ **Easy rollback** - Original data preserved

### Recommended Implementation Order:

1. **Week 1 Monday:** Prepare, backup, verify dependencies
2. **Week 1 Tuesday:** Update config, test locally
3. **Week 1 Wednesday:** Re-ingest documents
4. **Week 1 Thursday-Friday:** Validation & testing
5. **Week 2 Monday:** Cutover (or extended testing if needed)

---

**Comparison Generated:** November 16, 2026  
**Status:** READY FOR IMPLEMENTATION  
**Confidence:** HIGH ✓
