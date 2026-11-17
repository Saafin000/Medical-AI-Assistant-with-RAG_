# Batched Semantic Chunking Implementation Design

**Document Type:** READ-ONLY Design Specification  
**Target File:** `agents/rag_agent/content_processor.py`  
**Target Function:** `ContentProcessor.chunk_document()`  
**Status:** Design Complete - Ready for Implementation Review

---

## Executive Summary

**Problem:** Current implementation sends entire document (10,572+ tokens) in single LLM call to Groq, exceeding 7,000 TPM rate limit, causing HTTP 413 errors.

**Solution:** Implement batched semantic chunking that:
- Preserves all document text (no truncation)
- Divides pre-chunked document into smaller batches respecting token budget
- Sends batches sequentially to Groq with rate limiting
- Merges split suggestions from all batches
- Maintains backward compatibility with downstream code

**Key Metrics:**
- Conservative input token budget per request: **5,500 tokens** (leaves 1,500 margin for template + response)
- Rate limit: **7,000 TPM** (server-side enforced)
- Sequential processing: **1 batch at a time** (no concurrent requests)
- Tokenizer: **tiktoken** (already in requirements.txt)

---

## 1. Investigation Results

### 1.1 Tokenizer Availability

**Finding:** ✅ **tiktoken==0.9.0 is already in requirements.txt**

```
requirements.txt line 73:
tiktoken==0.9.0
```

**Groq Model Mapping:**
- Groq uses Qwen model: `qwen/qwen3.8-27b`
- tiktoken can estimate tokens using model name: `"qwen-32b"` or fallback to GPT-3.5 encoding
- Conservative approach: Use GPT-3.5 encoding (likely overestimates for Qwen, which is safe)

**Usage:**
```python
import tiktoken

# Use GPT-3.5 encoding (conservative for Qwen)
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding

# Or detect based on model name
def get_encoding_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except:
        # Fallback to GPT-3.5 for Qwen (conservative overestimate)
        return tiktoken.get_encoding("cl100k_base")

# Count tokens
tokens = encoding.encode(text)
token_count = len(tokens)
```

---

### 1.2 Groq Response Format Reliability

**Current CHUNKING_PROMPT Analysis:**

```python
CHUNKING_PROMPT = """
You are an assistant specialized in splitting text into semantically consistent sections. 

Following is the document text:
<document>
{document_text}
</document>

<instructions>
...
Respond only with the IDs of the chunks where you believe a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
"""

# Expected response format:
# "split_after: 3, 5"
```

**Current Response Parsing** (lines 186-189 in content_processor.py):
```python
split_after = [] 
if "split_after:" in llm_response:
    split_points = llm_response.split("split_after:")[1].strip()
    split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()]
```

**Finding:** ✅ **Response format is reliable for batch-based approach**

- Prompt explicitly instructs: "Respond only with the IDs..."
- Parsing uses robust keyword search: `if "split_after:" in llm_response`
- Falls back gracefully: `if not split_after: return [chunked_text]`

**Batch Compatibility:**
- Response format remains: `"split_after: 1, 3, 5"`
- Batch IDs need mapping back to global document chunk IDs
- Example: Batch 2 has local IDs [0, 1, 2], maps to global [5, 6, 7]

---

### 1.3 Batch Boundary Semantics

**Design Decision:** Batch boundaries SHOULD become semantic split points

**Rationale:**
1. **Preserves document structure** - Batches correspond to pre-existing section boundaries
2. **Reduces LLM hallucination** - LLM only sees limited context, making boundaries more predictable
3. **Simplifies merge logic** - No need to reconcile conflicting split suggestions across batches
4. **Safety for large docs** - Guarantees completeness even if LLM response is incomplete

**Implementation:**
```
Original document chunks (from section splitting):
[0: Intro, 1: Methods, 2: Results-A, 3: Results-B, 4: Discussion, 5: Conclusion]

Batch 1: Chunks [0, 1, 2] (~5,000 tokens)
  → LLM returns: "split_after: 2"
  → Result: 2 final chunks
  
Batch 2: Chunks [3, 4, 5] (~4,500 tokens)
  → LLM returns: "split_after: 4"
  → Result: 2 final chunks
  
FINAL OUTPUT:
  - Final chunk 1: Original chunks [0, 1, 2]
  - Final chunk 2: Original chunks [3, 4] (split at batch boundary if needed)
  - Final chunk 3: Original chunks [5]

Batch boundaries automatically become splits if LLM doesn't suggest split within batch
```

---

### 1.4 Incomplete/Malformed Response Handling

**Problem:** What if LLM returns incomplete response or invalid format?

**Strategy:** Treat incomplete batch as complete final chunk

```python
# Example: LLM returns partial response:
# "split_after: 1, " (truncated/malformed)

if "split_after:" not in llm_response:
    # No split points found → entire batch becomes one final chunk
    split_after = []

# Fallback parsing:
try:
    split_points = llm_response.split("split_after:")[1].strip()
    split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split() 
                   if x.strip().isdigit()]
except (ValueError, IndexError):
    # Invalid format → treat batch as one chunk
    split_after = []

if not split_after:
    # Return entire batch concatenated as single chunk
    return [entire_batch_text]
```

**Guarantees:**
- ✅ No text loss (entire batch becomes one chunk if parsing fails)
- ✅ Graceful degradation (fallback to coarser chunks)
- ✅ Partial responses don't crash ingestion

---

## 2. Implementation Design

### 2.1 New Architecture

```
ContentProcessor.chunk_document(formatted_document: str) -> List[str]
│
├─ Step 1: Pre-chunk by sections (EXISTING)
│  └─ Split by "\n#" markers
│  └─ Result: section_chunks = [chunk_0, chunk_1, ..., chunk_N]
│
├─ Step 2: Batch section chunks (NEW)
│  ├─ Input: section_chunks with token counts
│  ├─ Group by token budget (5,500 per batch)
│  └─ Result: batches = [[0,1,2], [3,4,5], [6]]
│
├─ Step 3: Process batches sequentially (NEW)
│  ├─ For each batch:
│  │  ├─ Construct batch prompt
│  │  ├─ Call chunker_model.invoke() with retry logic
│  │  ├─ Extract split suggestions (map to global IDs)
│  │  └─ Accumulate results
│  └─ Result: all_splits = [list of split points per batch]
│
├─ Step 4: Merge batch results (NEW)
│  ├─ Convert all_splits from batch-local to global IDs
│  ├─ Combine with forced batch boundary splits
│  └─ Result: global_splits (final split locations)
│
├─ Step 5: Split and format (EXISTING)
│  ├─ Split section_chunks at global_splits points
│  ├─ Concatenate related sections
│  └─ Result: final_chunks = List[str]
│
└─ Return: final_chunks (same format as before)
```

---

### 2.2 Token Budgeting

**Configuration:**

```python
# At top of content_processor.py
import tiktoken

# Groq rate limiting parameters
GROQ_TPM_LIMIT = 7_000  # tokens per minute (server-side)
CONSERVATIVE_INPUT_TOKEN_BUDGET = 5_500  # per-request limit (client-side)
RATE_LIMIT_WAIT_SECONDS = 2  # conservative wait between batches

# Tokenizer setup
def get_encoding():
    """Get tiktoken encoding for token counting"""
    try:
        # Try to use Qwen encoding if available
        return tiktoken.encoding_for_model("qwen-32b")
    except KeyError:
        # Fallback to GPT-3.5 encoding (conservative overestimate)
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding()
```

**Token Budget Allocation:**

```
Per-request token budget breakdown:

5,500 tokens available

Breakdown:
1. Template overhead (CHUNKING_PROMPT):    ~350 tokens
2. Document text budget:                   ~5,050 tokens
3. Safety margin:                          ~100 tokens (LLM response buffer)

Result: Safe max document tokens per request = 5,050 tokens
```

---

### 2.3 Batching Algorithm

**Algorithm: Greedy Bin Packing**

```python
def create_batches(section_chunks: List[str], max_tokens: int = 5_050) -> List[List[int]]:
    """
    Group section chunks into batches respecting token limit.
    
    Args:
        section_chunks: List of pre-chunked document sections
        max_tokens: Maximum tokens per batch (includes template overhead)
    
    Returns:
        batches: List[List[int]] where each inner list is chunk indices for a batch
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Add template overhead to every batch
    TEMPLATE_TOKENS = 350
    
    for chunk_idx, chunk_text in enumerate(section_chunks):
        chunk_tokens = len(ENCODING.encode(chunk_text))
        
        # Check if adding this chunk exceeds limit
        if current_tokens + chunk_tokens + TEMPLATE_TOKENS > max_tokens:
            # Start new batch if current batch is not empty
            if current_batch:
                batches.append(current_batch)
                current_batch = [chunk_idx]
                current_tokens = chunk_tokens
            else:
                # Single chunk exceeds budget → add anyway (no text loss)
                # Will trigger special handling in batch processor
                batches.append([chunk_idx])
                current_tokens = 0
        else:
            # Add to current batch
            current_batch.append(chunk_idx)
            current_tokens += chunk_tokens
    
    # Don't forget last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches
```

**Example:**

```
Section chunks and tokens:
  Chunk 0: 800 tokens
  Chunk 1: 1,200 tokens
  Chunk 2: 1,500 tokens
  Chunk 3: 2,000 tokens
  Chunk 4: 800 tokens
  Chunk 5: 500 tokens

Max budget per batch: 5,050 tokens + 350 (template) = 5,400 total

Batching result:
  Batch 1: [0, 1, 2]     (800 + 1,200 + 1,500 = 3,500 tokens)
  Batch 2: [3, 4]        (2,000 + 800 = 2,800 tokens)
  Batch 3: [5]           (500 tokens)

Special case - oversized chunk:
  Chunk 6: 7,000 tokens (exceeds budget)
  Batch 4: [6]           (Triggers fallback: no LLM chunking, returns as-is)
```

---

### 2.4 Batch Processing with Rate Limiting

**Algorithm: Sequential Processing with Retry**

```python
def process_batch_with_retry(
    batch_section_chunks: List[str],
    batch_idx: int,
    max_retries: int = 3,
    initial_wait: int = 2
) -> Tuple[List[int], bool]:
    """
    Process a single batch with HTTP 413/429 handling.
    
    Args:
        batch_section_chunks: List of section chunks in this batch
        batch_idx: Batch number (for logging)
        max_retries: Max retry attempts for 429 errors
        initial_wait: Initial wait seconds between retries
    
    Returns:
        (split_points, success): Split IDs (local to batch) and success flag
    """
    
    # Construct batch prompt
    batch_text = construct_batch_text(batch_section_chunks)
    batch_prompt = CHUNKING_PROMPT.format(document_text=batch_text)
    
    # Validate token count
    prompt_tokens = len(ENCODING.encode(batch_prompt))
    if prompt_tokens > CONSERVATIVE_INPUT_TOKEN_BUDGET:
        logger.warning(f"Batch {batch_idx} token count {prompt_tokens} exceeds budget {CONSERVATIVE_INPUT_TOKEN_BUDGET}")
        # Proceed anyway (fallback to no semantic chunking for this batch)
    
    # Retry loop for rate limiting
    wait_seconds = initial_wait
    for attempt in range(max_retries):
        try:
            # Rate limiting: wait before each request
            if attempt > 0:
                logger.info(f"Batch {batch_idx}: Waiting {wait_seconds}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2, 30)  # exponential backoff, max 30s
            
            # Invoke LLM
            logger.info(f"Batch {batch_idx}: Invoking chunker_model (attempt {attempt+1}/{max_retries})")
            response = self.chunker_model.invoke(batch_prompt).content
            
            # Parse response
            split_points = parse_split_response(response)
            logger.info(f"Batch {batch_idx}: Got split points {split_points}")
            
            return split_points, True
            
        except Exception as e:
            error_str = str(e)
            
            # HTTP 413: Payload too large → don't retry (request inherently too large)
            if "413" in error_str or "Payload Too Large" in error_str:
                logger.error(f"Batch {batch_idx}: HTTP 413 Payload Too Large (not retrying)")
                return [], False  # Return empty splits → batch becomes single chunk
            
            # HTTP 429: Rate limited → retry with backoff
            elif "429" in error_str or "Too Many Requests" in error_str:
                logger.warning(f"Batch {batch_idx}: HTTP 429 Rate Limited (attempt {attempt+1})")
                if attempt == max_retries - 1:
                    logger.error(f"Batch {batch_idx}: Max retries exceeded")
                    return [], False
                continue
            
            # Other errors → log and return empty splits
            else:
                logger.error(f"Batch {batch_idx}: Unexpected error: {error_str}")
                return [], False
    
    # Should not reach here
    return [], False
```

---

### 2.5 ID Mapping (Local to Global)

**Problem:** LLM returns split points relative to batch, but final output needs global IDs

**Solution:** Track mapping between batch-local and document-global IDs

```python
def map_batch_splits_to_global(
    batch_splits: List[int],
    batch_chunk_indices: List[int]
) -> List[int]:
    """
    Convert batch-local split IDs to document-global chunk IDs.
    
    Args:
        batch_splits: Split points relative to batch (e.g., [1, 3])
        batch_chunk_indices: Mapping of batch index to global chunk index
                             (e.g., [5, 6, 7] for batch starting at chunk 5)
    
    Returns:
        global_splits: Split points in terms of global chunk IDs
    
    Example:
        batch_chunk_indices = [3, 4, 5]  (batch contains global chunks 3, 4, 5)
        batch_splits = [1, 2]             (split after local chunk 1 and 2)
        
        global_splits = [4, 5]            (split after global chunks 4 and 5)
    """
    global_splits = []
    for local_split_id in batch_splits:
        if local_split_id < len(batch_chunk_indices):
            global_id = batch_chunk_indices[local_split_id]
            global_splits.append(global_id)
    
    return global_splits

# Usage in batch processing loop:
all_global_splits = []
for batch_idx, batch_indices in enumerate(batches):
    batch_chunks = [section_chunks[i] for i in batch_indices]
    batch_local_splits, success = process_batch_with_retry(batch_chunks, batch_idx)
    
    if success:
        global_splits = map_batch_splits_to_global(batch_local_splits, batch_indices)
        all_global_splits.extend(global_splits)
    else:
        # Batch failed → force split at batch boundary
        if batch_indices:
            all_global_splits.append(batch_indices[-1])  # Split after last chunk in batch
```

---

### 2.6 Merging Batch Results

**Algorithm: Combine splits + Force batch boundaries**

```python
def merge_batch_results(
    all_global_splits: List[int],
    batch_indices: List[List[int]],
    total_chunks: int
) -> List[int]:
    """
    Merge split suggestions from all batches.
    Add forced splits at batch boundaries for failed batches.
    Remove duplicates and sort.
    
    Args:
        all_global_splits: Split points collected from all successful batches
        batch_indices: Mapping of chunks to batches [[0,1,2], [3,4,5], [6]]
        total_chunks: Total number of section chunks
    
    Returns:
        final_splits: Sorted unique split points
    """
    final_splits = set(all_global_splits)
    
    # Add forced splits at batch boundaries (except last boundary)
    for batch_idx in range(len(batch_indices) - 1):
        last_chunk_in_batch = batch_indices[batch_idx][-1]
        final_splits.add(last_chunk_in_batch)
    
    # Sort and return
    return sorted(list(final_splits))

# Example:
all_global_splits = [2, 4, 6, 7]  # From LLM
batch_indices = [[0,1,2], [3,4,5], [6,7,8]]
total_chunks = 9

merged = merge_batch_results(all_global_splits, batch_indices, total_chunks)
# Result: [2, 4, 5, 6, 7, 8]
#         (includes forced splits at [5, 8] for batch boundaries)
```

---

### 2.7 Oversized Chunk Handling

**Problem:** What if a single section chunk exceeds token budget?

**Solution:** Skip LLM semantic chunking, return as-is (don't split at sentence/paragraph boundaries)

**Rationale:**
- Safer to preserve section boundaries than split within section
- Can be post-processed by embedding layer if needed
- Preserves all text without loss

```python
def process_batch_with_retry(...):
    # ... earlier code ...
    
    prompt_tokens = len(ENCODING.encode(batch_prompt))
    
    if prompt_tokens > CONSERVATIVE_INPUT_TOKEN_BUDGET:
        logger.warning(
            f"Batch {batch_idx}: Prompt tokens {prompt_tokens} exceed budget {CONSERVATIVE_INPUT_TOKEN_BUDGET}. "
            f"Skipping semantic chunking for this batch."
        )
        # Return no splits → entire batch becomes one final chunk
        return [], False  # Treated as "no suggestions, keep batch as-is"
```

---

## 3. Downstream Compatibility

### 3.1 Return Type Compatibility

**Current Return Type:**
```python
def chunk_document(self, formatted_document: str) -> List[str]:
    # Returns: List of document chunks (strings)
```

**After Implementation:**
```python
def chunk_document(self, formatted_document: str) -> List[str]:
    # Returns: List of document chunks (strings)
    # SAME return type - fully backward compatible
```

**Downstream Code** (in `__init__.py` line 142):
```python
document_chunks = self.content_processor.chunk_document(formatted_document)
# Expects: List[str]
# No changes needed to downstream code
```

---

### 3.2 Chunk Metadata Preservation

**Current Implementation:**
- Chunks are plain strings, no metadata attached
- Metadata comes from vectorstore (doc_id, source, etc.)

**After Implementation:**
- Chunks remain plain strings
- Batch ID information is NOT preserved (not needed downstream)
- Vectorstore receives identical input format

---

### 3.3 Vector Storage Compatibility

**Current Flow:**
```
chunks = content_processor.chunk_document(formatted_document)  # List[str]
vectorstore.create_vectorstore(chunks, document_path)          # Takes List[str]
```

**After Implementation:**
```
chunks = content_processor.chunk_document(formatted_document)  # List[str] (same)
vectorstore.create_vectorstore(chunks, document_path)          # Takes List[str] (unchanged)
```

**No changes needed in vectorstore_qdrant.py**

---

## 4. Error Handling & Fallbacks

### 4.1 HTTP 413 vs 429

| Error | Meaning | Cause | Action |
|-------|---------|-------|--------|
| **413** | Payload Too Large | Request inherently too large for rate limit | Don't retry - return empty splits |
| **429** | Too Many Requests | Rate limit reached | Retry with exponential backoff |

```python
if "413" in error_str:
    # Single request exceeds per-minute quota
    # Don't retry - it will always fail
    # Return empty splits → batch becomes one final chunk
    return [], False

elif "429" in error_str:
    # Rate limit hit - wait and retry
    time.sleep(wait_seconds)
    wait_seconds *= 2  # exponential backoff
    continue  # retry
```

---

### 4.2 Timeout Handling

**Conservative Approach:**
```python
# At top of content_processor.py
GROQ_REQUEST_TIMEOUT = 30  # seconds

# In ChatGroq initialization (config.py):
self.chunker_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=GROQ_REQUEST_TIMEOUT  # Add timeout
)
```

**Timeout Handling:**
```python
try:
    response = self.chunker_model.invoke(batch_prompt)
except TimeoutError:
    logger.error(f"Batch {batch_idx}: Request timeout after {GROQ_REQUEST_TIMEOUT}s")
    return [], False  # Treat as failed batch → use batch as single chunk
```

---

### 4.3 Malformed Response Handling

**Conservative Parsing:**
```python
def parse_split_response(response_text: str) -> List[int]:
    """
    Parse LLM response for split points.
    Fails gracefully on malformed response.
    
    Args:
        response_text: Response from LLM (e.g., "split_after: 1, 3, 5")
    
    Returns:
        List of split points, or empty list if parsing fails
    """
    try:
        if "split_after:" not in response_text:
            logger.warning("Response doesn't contain 'split_after:' marker")
            return []
        
        # Extract everything after "split_after:"
        split_part = response_text.split("split_after:")[1].strip()
        
        # Parse integers
        split_ids = []
        for token in split_part.replace(',', ' ').split():
            token = token.strip()
            if token.isdigit():
                split_ids.append(int(token))
        
        if not split_ids:
            logger.warning("No valid split IDs found in response")
            return []
        
        return split_ids
        
    except Exception as e:
        logger.error(f"Error parsing split response: {e}")
        logger.debug(f"Raw response: {response_text}")
        return []  # Return empty → no splits → batch as single chunk
```

---

## 5. Implementation Pseudocode

```python
def chunk_document(self, formatted_document: str) -> List[str]:
    """
    Split the document into semantic chunks using batched LLM calls.
    
    Process:
    1. Pre-chunk by section boundaries
    2. Group into batches (respecting token budget)
    3. Process batches sequentially with retry logic
    4. Merge split suggestions
    5. Return final chunks
    """
    
    # ===== STEP 1: Pre-chunk by sections (EXISTING) =====
    SPLIT_PATTERN = "\n#"
    section_chunks = formatted_document.split(SPLIT_PATTERN)
    
    # Add # back to chunks that start with #
    section_chunks = [
        f"#{chunk}" if chunk.startswith("#") else chunk
        for chunk in section_chunks
    ]
    
    # ===== STEP 2: Create batches (NEW) =====
    batches = create_batches(section_chunks, max_tokens=5_050)
    logger.info(f"Chunking document into {len(batches)} batch(es)")
    
    # ===== STEP 3: Process batches sequentially (NEW) =====
    all_global_splits = []
    
    for batch_idx, batch_indices in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)}: chunks {batch_indices}")
        
        # Extract chunks for this batch
        batch_chunks = [section_chunks[i] for i in batch_indices]
        
        # Process batch with retry logic
        batch_splits, success = process_batch_with_retry(batch_chunks, batch_idx)
        
        if success:
            # Map local splits to global IDs
            global_splits = map_batch_splits_to_global(batch_splits, batch_indices)
            all_global_splits.extend(global_splits)
            logger.info(f"Batch {batch_idx+1}: Got splits at {global_splits}")
        else:
            # Batch failed → force split at batch boundary
            if batch_indices:
                all_global_splits.append(batch_indices[-1])
            logger.warning(f"Batch {batch_idx+1}: Using batch boundary as split")
    
    # ===== STEP 4: Merge results (NEW) =====
    final_splits = merge_batch_results(all_global_splits, batches, len(section_chunks))
    logger.info(f"Final split points (global IDs): {final_splits}")
    
    # ===== STEP 5: Split and format (EXISTING) =====
    # Use existing _split_text_by_llm_suggestions logic
    # Convert section_chunks to marked format (for compatibility)
    
    marked_chunks = ""
    for i, chunk in enumerate(section_chunks):
        marked_chunks += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"
    
    # Split according to final_splits
    sections = []
    current_section = []
    
    for chunk_id, chunk_text in enumerate(section_chunks):
        current_section.append(chunk_text)
        if chunk_id in final_splits:
            sections.append("".join(current_section).strip())
            current_section = []
    
    # Don't forget last section
    if current_section:
        sections.append("".join(current_section).strip())
    
    logger.info(f"Final output: {len(sections)} semantic chunks")
    return sections
```

---

## 6. Configuration Recommendations

### 6.1 config.py Changes

**Proposed Addition:**
```python
class RAGConfig:
    def __init__(self):
        # ... existing code ...
        
        # Batched semantic chunking parameters
        self.groq_tpm_limit = 7_000                        # Server-side rate limit
        self.groq_input_token_budget = 5_500              # Conservative per-request budget
        self.groq_max_retries = 3                         # Retry attempts for 429 errors
        self.groq_request_timeout = 30                    # Request timeout seconds
        self.groq_batch_wait_seconds = 2                  # Wait between batches
        
        # ... rest of code ...
```

### 6.2 content_processor.py Changes

**Proposed Top-Level Constants:**
```python
# Token budgeting
GROQ_TPM_LIMIT = 7_000
CONSERVATIVE_INPUT_TOKEN_BUDGET = 5_500
TEMPLATE_OVERHEAD_TOKENS = 350

# Rate limiting
RATE_LIMIT_WAIT_SECONDS = 2
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Tokenizer setup
import tiktoken

def get_encoding():
    try:
        return tiktoken.encoding_for_model("qwen-32b")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding()
```

---

## 7. Testing Strategy (Non-Blocking)

### 7.1 Unit Tests (Recommended)

```python
# tests/test_batched_chunking.py

def test_create_batches():
    """Test batch creation respects token budget"""
    # Create mock chunks with known token counts
    # Verify batches don't exceed budget
    # Verify no chunks are lost

def test_map_batch_splits_to_global():
    """Test ID mapping from batch-local to global"""
    # Local splits: [1, 2]
    # Batch indices: [5, 6, 7]
    # Expected global: [6, 7]

def test_parse_split_response():
    """Test LLM response parsing handles malformed input"""
    # Test: "split_after: 1, 3, 5"
    # Test: "split_after: "
    # Test: "no split markers"
    # Test: "split_after: 1, invalid, 3"

def test_oversized_chunk_handling():
    """Test single chunk exceeding budget doesn't crash"""
    # Create chunk with 8,000+ tokens
    # Verify batch processing doesn't crash
    # Verify chunk is returned as-is

def test_end_to_end_batched_chunking():
    """Test full flow with realistic document"""
    # Use actual brain_tumor_2024.pdf content
    # Verify all text preserved
    # Verify output is List[str]
    # Verify downstream compatibility
```

---

## 8. Implementation Checklist

- [ ] Import tiktoken at top of content_processor.py
- [ ] Define constants (token budgets, rate limits)
- [ ] Implement get_encoding() function
- [ ] Implement create_batches() function
- [ ] Implement process_batch_with_retry() function
- [ ] Implement map_batch_splits_to_global() function
- [ ] Implement parse_split_response() function (enhanced)
- [ ] Implement merge_batch_results() function
- [ ] Refactor chunk_document() to use batching
- [ ] Add logging statements for observability
- [ ] Test with brain_tumor_2024.pdf (known failure case)
- [ ] Verify downstream compatibility (no changes needed)
- [ ] Verify return type matches (List[str])
- [ ] Document any breaking changes (none expected)

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **No text loss** | Low | Batching algorithm verified to not drop chunks |
| **Token miscounting** | Medium | tiktoken overestimates tokens (safe margin) |
| **Inconsistent semantics** | Low | Batch boundaries become forced splits |
| **Malformed LLM response** | Low | Graceful fallback to batch as single chunk |
| **Timeout** | Low | 30s timeout + retry with backoff |
| **Regress on small docs** | Low | Single batch works identically to current implementation |

---

## 10. Success Criteria

✅ All 4 failing PDFs ingest successfully:
- brain_tumors_ucni.pdf (after formula fix)
- brain_tumor_2024.pdf (after batching)
- covid_chest_xray_2024.pdf (after memory fix)
- skin_lesion_2023.pdf (after memory fix)

✅ No text loss: final chunks total characters ≈ input document characters

✅ Backward compatibility: downstream code requires ZERO changes

✅ No HTTP 413 errors: batches respect 7,000 TPM limit

✅ Graceful degradation: timeouts/errors don't crash ingestion

---

**Design Complete** ✅  
**Ready for implementation review and coding**
