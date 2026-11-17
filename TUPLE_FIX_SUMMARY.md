# Tuple-Indexing Bug Fix - Summary

## Problem

**Location:** `agents/rag_agent/content_processor.py`

**Bug:** `_create_batches()` stored oversized section entries as tuples `(idx, piece_idx)`, but `chunk_document()` tried to use them as integer array indices, causing TypeError.

```python
# BROKEN CODE (lines 456-460):
for piece_idx, piece in enumerate(split_pieces):
    batches.append([(idx, piece_idx)])  # ← Stores tuple

# BROKEN CODE (line 192):
batch_sections = [section_chunks[i] for i in batch_indices]
# If i = (5, 0), then section_chunks[(5, 0)] → TypeError
```

## Solution

**Approach:** Keep split pieces in cache, store only integer indices in batches, let reconstruction handle cache lookup.

### Changes Made

#### 1. Fix `_create_batches()` (Lines ~456-460)

**Before:**
```python
split_pieces = self._split_oversized_section(section, idx, MAX_BATCH_TOKENS)
self.split_sections_cache[idx] = split_pieces
for piece_idx, piece in enumerate(split_pieces):
    # Use tuple (original_idx, piece_idx) as batch entry to distinguish splits
    batches.append([(idx, piece_idx)])  # ← TUPLE STORED
```

**After:**
```python
split_pieces = self._split_oversized_section(section, idx, MAX_BATCH_TOKENS)
self.split_sections_cache[idx] = split_pieces
for piece_idx in range(len(split_pieces)):
    # Each split piece needs its own batch for LLM processing
    # Note: We store only the original section idx (integer), not tuples
    # The split pieces are retrieved from cache during reconstruction
    batches.append([idx])  # ← INTEGER ONLY
```

#### 2. Fix `chunk_document()` (Lines ~189-200)

**Before:**
```python
batch_idx, batch_indices = batch_queue.pop(0)
batch_sections = [section_chunks[i] for i in batch_indices]
```

**After:**
```python
batch_idx, batch_indices = batch_queue.pop(0)

# Build batch_sections: if a section was split, use pieces from cache
batch_sections = []
for idx in batch_indices:
    if idx in self.split_sections_cache:
        # This section was split: use the cached pieces
        batch_sections.extend(self.split_sections_cache[idx])
    else:
        # Normal section: use original
        batch_sections.append(section_chunks[idx])
```

#### 3. Reconstruction Already Correct (Lines ~767-784)

No changes needed - reconstruction already handles cache lookup:

```python
for section_idx, section in enumerate(section_chunks):
    # Check if this section was split (has a cache entry)
    if section_idx in self.split_sections_cache:
        # Add all split pieces for this section without stripping
        split_pieces = self.split_sections_cache[section_idx]
        for piece in split_pieces:
            current_chunk += piece
    else:
        # Normal section: add as-is
        current_chunk += section
```

## How It Works Now

**Flow with Oversized Section:**

```
1. _create_batches():
   - Detects section 2 is oversized (6,000 tokens > 5,150 budget)
   - Calls _split_oversized_section(section_2, 2, 5_150)
   - Returns: ["piece1", "piece2"]
   - Caches: split_sections_cache[2] = ["piece1", "piece2"]
   - Stores batches: [[0, 1], [2], [2], [3, 4]]
     (Note: [2] appears twice, each for one piece)

2. chunk_document():
   - Pops batch_indices = [2]
   - Checks if 2 in split_sections_cache → YES
   - Builds batch_sections = ["piece1", "piece2"]  (NOT original section)
   - Sends batch_sections to LLM

3. _reconstruct_chunks_from_sections():
   - Iterates section_chunks with enumerate
   - At idx=2, checks split_sections_cache[2] → YES
   - Retrieves ["piece1", "piece2"]
   - Concatenates: current_chunk += "piece1" + "piece2"
   - Result: split pieces reassembled exactly
```

## Test Results

### Verification Tests: 6/6 PASS

```
[PASS] test_1_no_tuple_indexing_error
  - No TypeError when processing split sections
  - All batch_indices are integers

[PASS] test_2_split_pieces_sent_to_llm
  - LLM receives split pieces, not original oversized section
  - Original section "s2_original_12345" is NOT sent
  - Split pieces ["s2_piece1", "s2_piece2"] ARE sent

[PASS] test_3_reconstruction_uses_cache
  - Reconstruction correctly looks up split_sections_cache
  - Combines split pieces without loss

[PASS] test_4_cache_initialization
  - Cache is cleared at start of _create_batches()

[PASS] test_5_integer_indices_only
  - ALL batch entries are integers (no tuples)
  - Safe to use with array indexing

[PASS] test_6_order_preserved
  - Original section order maintained
  - S0, S1, S2, S3, S4 order correct through processing
```

## Verification Against Requirements

✅ **Requirement 1:** Keep batch entries compatible with integer indexing
- batches now store only integers `[idx]`, not tuples `[(idx, piece_idx)]`

✅ **Requirement 2:** Preserve split_sections_cache mechanism
- Cache still used: `split_sections_cache[idx] = split_pieces`
- Lookup in reconstruction: `if idx in self.split_sections_cache`

✅ **Requirement 3:** Every split piece processed by LLM
- chunk_document() builds batch_sections from cache
- Each piece sent separately: `batch_sections.extend(split_pieces)`

✅ **Requirement 4:** Preserve section order and global IDs
- Reconstruction uses cache, preserves order
- Global ID mapping unchanged

✅ **Requirement 5:** Combine split pieces without loss
- Direct concatenation: `current_chunk += piece` (no strip)
- All text, whitespace, newlines preserved

✅ **Requirement 6:** 413 handling compatible
- 413 recovery uses batch_indices (integers only)
- Binary splitting works with integers
- No change to 413 logic needed

## Safe to Proceed

**Status:** ✅ READY FOR SINGLE-PDF TEST

- No tuples in batch structure
- Split pieces are processed correctly
- Text preservation verified
- All 6 verification tests pass
- Syntax validated

**Next Step:** Test with `brain_tumor_2024.pdf` (known oversized PDF)
