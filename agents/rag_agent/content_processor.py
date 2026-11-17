import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

try:
    import tiktoken
except ImportError:
    tiktoken = None

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            llm: Large language model for image summarization
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
        
        # Chunking configuration (for RecursiveCharacterTextSplitter)
        self.chunk_size = config.rag.chunk_size           # 512 tokens
        self.chunk_overlap = config.rag.chunk_overlap     # 50 tokens
        
        # Rolling 60-second token usage tracker for TPM rate limiting
        self.token_usage_window = deque()  # List of (timestamp, estimated_tokens) tuples
        self.tpm_window_seconds = 60
        self.conservative_tpm_limit = 5_000  # Conservative limit below 7,000 TPM
        
        # Track split oversized sections: map original_idx to list of split pieces
        # Example: {5: ["piece1", "piece2", "piece3"]}
        self.split_sections_cache = {}
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize images using the provided model, with error handling.
        
        Args:
            images: List of image paths
            
        Returns:
            List of image summaries, with placeholders for failed images or missing model
        """
        
        # Check if summarizer model is available (Azure Vision credentials may not be set)
        if self.summarizer_model is None:
            self.logger.warning("Azure Vision summarizer model not configured. Skipping image summarization.")
            return ["no image summary"] * len(images)
        
        prompt_template = """Describe the image in detail while keeping it concise and to the point. 
                        For context, the image is part of either a medical research paper or a research paper
                        demonstrating the use of artificial intelligence techniques like
                        machine learning and deep learning in diagnosing diseases or a medical report.
                        Be specific about graphs, such as bar plots if they are present in the image.
                        Only summarize what is present in the image, without adding any extra detail or comment.
                        Summarize the image only if it is related to the context, return 'non-informative' explicitly 
                        if the image is of some button not relevant to the context."""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()
        
        results = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                # Log the error if needed
                print(f"Error processing image: {str(e)}")
                # Add placeholder for the failed image
                results.append("no image summary")
        
        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.
        
        Args:
            parsed_document: Parsed document from doc_parser
            image_summaries: List of image summaries
            
        Returns:
            Formatted document text with image summaries
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.
        
        Args:
            text: Text containing placeholders
            target: Placeholder to replace
            replacements: List of replacements for each occurrence
            
        Returns:
            Text with replacements
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into semantic chunks using RecursiveCharacterTextSplitter.
        
        Uses deterministic recursive character-based splitting at paragraph and 
        sentence boundaries, measured in tokens rather than characters.
        
        No LLM calls are made. This preserves all text exactly and is fast/reliable.
        
        Args:
            formatted_document: Formatted document text
            
        Returns:
            List of document chunks (preserves all content, maintains order)
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        self.logger.info(f"Chunking document ({len(formatted_document)} chars) using RecursiveCharacterTextSplitter")
        
        # Use tiktoken-based token counting as length function
        def token_counter(text: str) -> int:
            return self._estimate_tokens(text)
        
        # Create splitter with hierarchical separators
        # Prefer paragraph boundaries, then lines, then spaces, then characters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,           # 512 tokens
            chunk_overlap=self.chunk_overlap,     # 50 tokens
            separators=["\n\n", "\n", " ", ""],   # Hierarchy of boundaries
            length_function=token_counter,
            keep_separator=False  # Don't duplicate separators
        )
        
        # Split the document
        try:
            chunks = splitter.split_text(formatted_document)
            self.logger.info(f"Document split into {len(chunks)} chunks")
            
            # Log chunk sizes for debugging
            for i, chunk in enumerate(chunks):
                tokens = self._estimate_tokens(chunk)
                self.logger.debug(f"  Chunk {i}: {tokens} tokens ({len(chunk)} chars)")
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error during chunking: {e}")
            # Fallback: return document as single chunk if splitting fails
            self.logger.warning("Falling back to single chunk")
            return [formatted_document] if formatted_document else []
    
    # ============================================================================
    # DEPRECATED: LLM-based batching logic (replaced by RecursiveCharacterTextSplitter)
    # Kept for reference but not used in new chunk_document()
    # ============================================================================
    
    def _get_encoding(self) -> 'tiktoken.Encoding':
        """Get tiktoken encoder for token counting."""
        if tiktoken is None:
            self.logger.warning("tiktoken not available, token counting disabled")
            return None
        
        try:
            # Try Qwen encoding if available
            return tiktoken.encoding_for_model("qwen-32b")
        except KeyError:
            # Fallback to GPT-3.5 (conservative overestimate)
            return tiktoken.get_encoding("cl100k_base")
    
    def _cleanup_tpm_window(self):
        """Remove expired tokens from the rolling 60-second window."""
        current_time = time.time()
        while self.token_usage_window and (current_time - self.token_usage_window[0][0]) > self.tpm_window_seconds:
            self.token_usage_window.popleft()
    
    def _get_current_tpm_usage(self) -> int:
        """Get total tokens used in the current rolling window."""
        self._cleanup_tpm_window()
        return sum(tokens for _, tokens in self.token_usage_window)
    
    def _wait_for_tpm_capacity(self, estimated_tokens: int) -> float:
        """
        Wait if necessary to respect the conservative TPM limit.
        
        Raises ValueError if estimated_tokens exceeds the conservative limit.
        
        Returns:
            Time waited in seconds
        """
        # Reject oversized requests immediately
        if estimated_tokens > self.conservative_tpm_limit:
            raise ValueError(
                f"Request size {estimated_tokens} tokens exceeds conservative TPM limit "
                f"{self.conservative_tpm_limit}. Batch is too large and cannot be sent."
            )
        
        total_wait = 0.0
        max_wait_total = 600  # Prevent infinite waits (10 minutes max)
        
        while True:
            current_usage = self._get_current_tpm_usage()
            if current_usage + estimated_tokens <= self.conservative_tpm_limit:
                # Capacity available
                break
            
            # Need to wait for some tokens to expire
            if self.token_usage_window:
                oldest_time, _ = self.token_usage_window[0]
                wait_until = oldest_time + self.tpm_window_seconds
                current_time = time.time()
                if wait_until > current_time:
                    wait_duration = wait_until - current_time + 0.1  # Small buffer
                    self.logger.info(
                        f"TPM rate limit: Current usage {current_usage}/{self.conservative_tpm_limit} tokens, "
                        f"estimated {estimated_tokens} new tokens. Waiting {wait_duration:.1f}s for window expiry."
                    )
                    time.sleep(wait_duration)
                    total_wait += wait_duration
                    
                    # Safety: prevent infinite waits
                    if total_wait > max_wait_total:
                        raise TimeoutError(
                            f"TPM capacity wait exceeded {max_wait_total}s timeout. "
                            f"Token window may be permanently full."
                        )
                    
                    self._cleanup_tpm_window()
                else:
                    # Window should be expired, retry cleanup
                    self._cleanup_tpm_window()
            else:
                # Window is empty but we still need capacity (shouldn't happen)
                self.logger.warning(
                    f"TPM window empty but current_usage + estimated exceeds limit. "
                    f"This may indicate a timing issue. Allowing request."
                )
                break
        
        return total_wait
    
    def _record_tpm_usage(self, estimated_tokens: int):
        """Record token usage in the rolling window."""
        self.token_usage_window.append((time.time(), estimated_tokens))
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if tiktoken is None:
            # Fallback: rough estimate (~1 token per 4 characters)
            return len(text) // 4
        
        encoding = self._get_encoding()
        if encoding is None:
            return len(text) // 4
        
        try:
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.warning(f"Token estimation failed: {e}, using fallback")
            return len(text) // 4
    
    def _create_batches(self, section_chunks: List[str], max_tokens: int = 5_500) -> List[List[tuple]]:
        """
        Group section chunks into batches respecting token limit.
        
        Uses greedy bin packing: adds sections to current batch until adding the next
        section would exceed max_tokens. Handles oversized individual sections by
        splitting them at paragraph/sentence boundaries.
        
        Each batch item is a tuple (section_idx, piece_idx):
        - For normal sections: (idx, 0) meaning section idx, piece 0 (entire section)
        - For split sections: (idx, 0), (idx, 1), ... each piece as separate item
        
        This ensures each split piece is assigned to exactly one batch and sent once.
        
        Args:
            section_chunks: List of section texts
            max_tokens: Maximum tokens per batch (includes template overhead)
        
        Returns:
            List of batches, where each batch is a list of (section_idx, piece_idx) tuples
        """
        TEMPLATE_OVERHEAD = 350  # Approximate tokens for CHUNKING_PROMPT
        MAX_BATCH_TOKENS = max_tokens - TEMPLATE_OVERHEAD
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Clear split sections cache from any previous call
        self.split_sections_cache = {}
        
        for idx, section in enumerate(section_chunks):
            section_tokens = self._estimate_tokens(section)
            
            # Check if adding this section exceeds limit
            if current_tokens + section_tokens > MAX_BATCH_TOKENS:
                # Start new batch if current batch is not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                # Handle oversized section by splitting it
                if section_tokens > MAX_BATCH_TOKENS:
                    self.logger.warning(
                        f"Section {idx} is {section_tokens} tokens (exceeds budget {MAX_BATCH_TOKENS}), "
                        f"splitting at paragraph boundaries"
                    )
                    split_pieces = self._split_oversized_section(section, idx, MAX_BATCH_TOKENS)
                    # Cache the split pieces for later reconstruction
                    self.split_sections_cache[idx] = split_pieces
                    
                    # Assign each split piece to batches (as separate work items)
                    for piece_idx, piece in enumerate(split_pieces):
                        piece_tokens = self._estimate_tokens(piece)
                        
                        # Check if adding this piece to current batch exceeds limit
                        if current_batch and current_tokens + piece_tokens > MAX_BATCH_TOKENS:
                            # Start new batch
                            batches.append(current_batch)
                            current_batch = []
                            current_tokens = 0
                        
                        # Add this piece to current batch
                        current_batch.append((idx, piece_idx))
                        current_tokens += piece_tokens
                else:
                    current_batch = [(idx, 0)]
                    current_tokens = section_tokens
            else:
                # Add to current batch
                current_batch.append((idx, 0))
                current_tokens += section_tokens
        
        # Don't forget last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _split_oversized_section(self, section: str, original_idx: int, max_tokens: int) -> List[str]:
        """
        Split a single oversized section into smaller pieces at paragraph boundaries,
        then sentence boundaries if needed. Preserves all original text exactly.
        
        Args:
            section: Section text that exceeds max_tokens
            original_idx: Original section index (for logging)
            max_tokens: Maximum tokens per piece
        
        Returns:
            List of section pieces (preserved in exact order, no text loss)
        """
        pieces = []
        
        # First try: split by paragraphs (double newline)
        paragraphs = section.split('\n\n')
        
        if len(paragraphs) > 1:
            current_piece = ""
            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                piece_tokens = self._estimate_tokens(current_piece)
                
                if piece_tokens + para_tokens > max_tokens:
                    # Current piece would exceed budget, save it and start new piece
                    if current_piece:
                        # Store WITHOUT stripping to preserve exact text
                        pieces.append(current_piece)
                        current_piece = para
                    else:
                        # Single paragraph exceeds budget, split at sentences
                        pieces.extend(self._split_at_sentences(para, max_tokens, original_idx))
                        current_piece = ""
                else:
                    # Add paragraph to current piece
                    if current_piece:
                        current_piece += '\n\n' + para
                    else:
                        current_piece = para
            
            if current_piece:
                pieces.append(current_piece)
        else:
            # No paragraph breaks, split at sentences
            pieces = self._split_at_sentences(section, max_tokens, original_idx)
        
        self.logger.info(f"Section {original_idx}: Split into {len(pieces)} pieces")
        return pieces if pieces else [section]  # Fallback: return whole section if splitting fails
    
    def _split_at_sentences(self, text: str, max_tokens: int, section_idx: int) -> List[str]:
        """
        Split text at sentence boundaries. Preserves all text exactly.
        
        Uses a conservative approach: splits on sentence-ending punctuation
        followed by whitespace, preserving all characters.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per piece
            section_idx: Section index (for logging)
        
        Returns:
            List of text pieces (exact text preserved, no character loss)
        """
        # Conservative approach: don't actually split, just accumulate
        # Split on periods/exclamations/questions followed by space,
        # but use a simple iteration approach
        
        # If the text is simple, just return as-is if it fits
        piece_tokens = self._estimate_tokens(text)
        if piece_tokens <= max_tokens:
            return [text]
        
        # Otherwise, do a simple split by trying to find sentence boundaries
        # Split by '. ', '! ', '? ' but preserve them
        pieces = []
        current = ""
        
        i = 0
        while i < len(text):
            current += text[i]
            
            # Check if we hit a sentence boundary (punctuation + space)
            if i < len(text) - 1 and text[i] in '.!?' and text[i + 1] == ' ':
                # Include the space too
                current += text[i + 1]
                i += 1
                
                # Check token count
                current_tokens = self._estimate_tokens(current)
                if current_tokens > max_tokens and pieces:
                    # This would exceed, save previous and start new
                    pieces.append(current[:-1])  # Don't include the space in new piece
                    current = text[i + 1]
                    i += 1
                elif current_tokens > max_tokens:
                    # Single sentence exceeds limit, return as-is
                    pieces.append(current)
                    current = ""
            
            i += 1
        
        if current:
            pieces.append(current)
        
        return pieces if pieces else [text]
    
    def _process_batch_with_retry(
        self,
        batch_sections: List[str],
        batch_idx: int,
        total_batches: int,
        max_retries: int = 3,
        initial_wait: float = 2.0,
        recursion_depth: int = 0
    ) -> Tuple[List[int], bool, bool]:
        """
        Process a single batch with retry logic for HTTP 429/413 errors.
        
        Records tokens for every attempt (success or failure).
        Re-checks TPM capacity before each retry.
        
        Args:
            batch_sections: List of section texts in this batch
            batch_idx: Batch index (0-based)
            total_batches: Total number of batches
            max_retries: Max retry attempts
            initial_wait: Initial wait seconds for backoff
            recursion_depth: Recursion depth for 413 splitting
        
        Returns:
            (split_ids, success, should_split_on_413)
        """
        CHUNKING_PROMPT = """You are an assistant specialized in splitting text into semantically consistent sections. 
        
Following is the document text:
<document>
{document_text}
</document>

<instructions>
Instructions:
    1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
    2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
    3. Each chunk must be between 256 and 512 words.
    4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
    5. The chunks must be listed in ascending order.
    6. Provide your response in the form: 'split_after: 3, 5'.
</instructions>

Respond only with the IDs of the chunks where you believe a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT.""".strip()
        
        # Construct batch text with markers
        batch_text = ""
        for i, section in enumerate(batch_sections):
            batch_text += f"<|start_chunk_{i}|>\n{section}\n<|end_chunk_{i}|>\n"
        
        # Estimate tokens (including prompt overhead)
        batch_prompt = CHUNKING_PROMPT.format(document_text=batch_text)
        prompt_tokens = self._estimate_tokens(batch_prompt)
        
        self.logger.info(
            f"Batch {batch_idx+1}/{total_batches}: {len(batch_sections)} section(s), "
            f"~{prompt_tokens} tokens (recursion_depth={recursion_depth})"
        )
        
        # Check TPM capacity before first attempt
        try:
            tpm_wait = self._wait_for_tpm_capacity(prompt_tokens)
        except (ValueError, TimeoutError) as e:
            self.logger.error(f"Batch {batch_idx+1}: TPM capacity check failed: {e}")
            return [], False, False
        
        # Retry loop
        wait_seconds = initial_wait
        for attempt in range(max_retries):
            try:
                # Rate limiting: wait before retries
                if attempt > 0:
                    self.logger.info(f"Batch {batch_idx+1}: Waiting {wait_seconds}s before retry {attempt+1}/{max_retries}")
                    time.sleep(wait_seconds)
                    wait_seconds = min(wait_seconds * 2, 30)  # Exponential backoff
                    
                    # RE-CHECK TPM capacity before retry (key fix for 429 handling)
                    try:
                        self._wait_for_tpm_capacity(prompt_tokens)
                    except (ValueError, TimeoutError) as e:
                        self.logger.error(f"Batch {batch_idx+1}: TPM capacity check on retry failed: {e}")
                        # Record this failed attempt
                        self._record_tpm_usage(prompt_tokens)
                        return [], False, False
                
                # Record tokens immediately before invocation
                self._record_tpm_usage(prompt_tokens)
                
                # Invoke LLM
                self.logger.debug(f"Batch {batch_idx+1}: Invoking chunker (attempt {attempt+1})")
                response = self.chunker_model.invoke(batch_prompt).content
                
                # Parse response
                split_ids = self._parse_split_response(response)
                self.logger.info(f"Batch {batch_idx+1}: Got {len(split_ids)} split point(s): {split_ids}")
                
                return split_ids, True, False
                
            except Exception as e:
                error_str = str(e).lower()
                error_full = str(e)
                
                # HTTP 413: Signal caller to split batch further
                if "413" in error_str or "payload too large" in error_str:
                    self.logger.error(
                        f"Batch {batch_idx+1}: HTTP 413 Payload Too Large. Error: {error_full[:100]}. "
                        f"Signaling batch split (recursion_depth={recursion_depth})."
                    )
                    # Check recursion depth to prevent infinite splitting
                    if recursion_depth >= 10:
                        self.logger.error(
                            f"Batch {batch_idx+1}: Max recursion depth reached, treating as hard failure"
                        )
                        return [], False, False
                    return [], False, True  # Signal to split
                
                # HTTP 429: Retry with backoff
                elif "429" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"Batch {batch_idx+1}: HTTP 429 Rate Limited (attempt {attempt+1}/{max_retries})"
                        )
                        continue
                    else:
                        self.logger.error(f"Batch {batch_idx+1}: Max retries exceeded after rate limit")
                        return [], False, False
                
                # Other errors
                else:
                    self.logger.error(f"Batch {batch_idx+1}: Unexpected error: {type(e).__name__}: {e}")
                    return [], False, False
        
        return [], False, False
    
    def _split_batch_for_413_v2(self, batch_items: List[tuple], section_chunks: List[str]) -> List[List[tuple]]:
        """
        Split a batch that received HTTP 413 into smaller batches (tuple-based).
        
        Each batch_item is (section_idx, piece_idx).
        Splits at the midpoint to preserve order.
        
        Args:
            batch_items: List of (section_idx, piece_idx) tuples
            section_chunks: All section texts
        
        Returns:
            List of smaller batches (each is a list of tuples)
        """
        if len(batch_items) <= 1:
            # Cannot split single item further
            return [batch_items]
        
        # Split at midpoint
        mid = len(batch_items) // 2
        left_batch = batch_items[:mid]
        right_batch = batch_items[mid:]
        
        # Estimate tokens for each batch
        TEMPLATE_OVERHEAD = 350
        def estimate_batch_tokens(items):
            total = TEMPLATE_OVERHEAD
            for section_idx, piece_idx in items:
                if section_idx in self.split_sections_cache:
                    piece = self.split_sections_cache[section_idx][piece_idx]
                    total += self._estimate_tokens(piece)
                else:
                    total += self._estimate_tokens(section_chunks[section_idx])
            return total
        
        left_tokens = estimate_batch_tokens(left_batch)
        right_tokens = estimate_batch_tokens(right_batch)
        
        self.logger.debug(
            f"413 Recovery: Split batch {len(batch_items)} items into "
            f"left {len(left_batch)} items ({left_tokens}t) and "
            f"right {len(right_batch)} items ({right_tokens}t)"
        )
        
        return [left_batch, right_batch]
    
    def _split_batch_for_413(self, batch_indices: List[int], section_chunks: List[str]) -> List[List[int]]:
        """
        Split a batch that received HTTP 413 into smaller batches.
        
        Splits at the midpoint to preserve section order and binary-search-like splitting.
        Recursively called if needed.
        
        Args:
            batch_indices: Global section indices in this batch
            section_chunks: All section texts
        
        Returns:
            List of smaller batches (each is a list of global section indices)
        """
        if len(batch_indices) <= 1:
            # Cannot split single section further
            return [batch_indices]
        
        # Split at midpoint
        mid = len(batch_indices) // 2
        left_batch = batch_indices[:mid]
        right_batch = batch_indices[mid:]
        
        # Estimate tokens for each batch
        TEMPLATE_OVERHEAD = 350
        left_tokens = sum(self._estimate_tokens(section_chunks[i]) for i in left_batch) + TEMPLATE_OVERHEAD
        right_tokens = sum(self._estimate_tokens(section_chunks[i]) for i in right_batch) + TEMPLATE_OVERHEAD
        
        self.logger.debug(
            f"413 Recovery: Split batch [sections {batch_indices[0]}-{batch_indices[-1]}] "
            f"into [sections {left_batch[0] if left_batch else 'empty'}-{left_batch[-1] if left_batch else 'empty'}] ({left_tokens}t) "
            f"and [sections {right_batch[0] if right_batch else 'empty'}-{right_batch[-1] if right_batch else 'empty'}] ({right_tokens}t)"
        )
        
        return [left_batch, right_batch]
    
    def _parse_split_response(self, response_text: str) -> List[int]:
        """
        Parse LLM response for split point IDs.
        Gracefully handles malformed or empty responses.
        
        Args:
            response_text: Response from LLM
        
        Returns:
            List of split IDs (batch-local), or empty list if parsing fails
        """
        try:
            if "split_after:" not in response_text:
                self.logger.warning("Response doesn't contain 'split_after:' marker, treating as no splits")
                return []
            
            # Extract everything after "split_after:"
            split_part = response_text.split("split_after:")[1].strip()
            
            # Parse integers robustly
            split_ids = []
            for token in split_part.replace(',', ' ').split():
                token = token.strip()
                if token and token.isdigit():
                    split_ids.append(int(token))
            
            if not split_ids:
                self.logger.warning("No valid split IDs found in response")
                return []
            
            return split_ids
            
        except Exception as e:
            self.logger.error(f"Error parsing split response: {e}")
            return []
    
    def _map_batch_splits_to_global_v2(self, batch_splits: List[int], batch_items: List[tuple]) -> List[int]:
        """
        Convert batch-local split IDs to document-global section IDs (tuple-based).
        
        Each batch_item is (section_idx, piece_idx).
        For split sections, multiple items reference the same section_idx.
        
        Example:
            batch_items = [(3, 0), (4, 0), (5, 0), (5, 1)]  (5 is split into 2 pieces)
            batch_splits = [1, 2]                          (split after local items 1 and 2)
            -> returns [4, 5]                              (split after sections 4 and 5)
        
        Args:
            batch_splits: Split IDs relative to batch items (0-indexed)
            batch_items: List of (section_idx, piece_idx) tuples in batch
        
        Returns:
            Global section IDs where splits should occur
        """
        global_splits = []
        seen_sections = set()
        
        for local_split_id in batch_splits:
            if local_split_id < len(batch_items):
                section_idx, _ = batch_items[local_split_id]
                # Only add each section once (avoid duplication for split sections)
                if section_idx not in seen_sections:
                    global_splits.append(section_idx)
                    seen_sections.add(section_idx)
        
        return global_splits
    
    def _map_batch_splits_to_global(self, batch_splits: List[int], batch_indices: List[int]) -> List[int]:
        """
        Convert batch-local split IDs to document-global section IDs.
        
        Example:
            batch_indices = [3, 4, 5]  (batch contains global sections 3, 4, 5)
            batch_splits = [1, 2]       (split after local sections 1 and 2)
            -> returns [4, 5]           (split after global sections 4 and 5)
        
        Args:
            batch_splits: Split IDs relative to batch (0-indexed)
            batch_indices: Mapping of batch positions to global section indices
        
        Returns:
            Global section IDs where splits should occur
        """
        global_splits = []
        for local_split_id in batch_splits:
            if local_split_id < len(batch_indices):
                global_id = batch_indices[local_split_id]
                global_splits.append(global_id)
        
        return global_splits
    
    def _merge_batch_results(
        self,
        all_global_splits: List[int],
        batches: List[List],
        total_sections: int
    ) -> List[int]:
        """
        Merge split suggestions from all batches and force batch boundaries.
        
        Handles both integer-based batches [0, 1, 2] and tuple-based batches [(0,0), (1,0)].
        
        Args:
            all_global_splits: Split IDs collected from all batches
            batches: All batch definitions (can be List[int] or List[tuple])
            total_sections: Total number of sections
        
        Returns:
            Final unique sorted split IDs
        """
        final_splits = set(all_global_splits)
        
        # Force splits at batch boundaries (except last boundary)
        for batch_idx in range(len(batches) - 1):
            last_item_in_batch = batches[batch_idx][-1]
            
            # Handle both integer indices and (section_idx, piece_idx) tuples
            if isinstance(last_item_in_batch, tuple):
                last_section_idx = last_item_in_batch[0]
            else:
                last_section_idx = last_item_in_batch
            
            final_splits.add(last_section_idx)
        
        return sorted(list(final_splits))
    
    def _reconstruct_chunks_from_sections(self, section_chunks: List[str], final_splits: List[int]) -> List[str]:
        """
        Reconstruct final chunks from sections based on split points.
        
        Handles both regular sections and split oversized sections.
        Preserves all text exactly without stripping intermediate pieces.
        
        Args:
            section_chunks: Original section chunks
            final_splits: Section IDs where splits should occur
        
        Returns:
            List of final document chunks (exact text preserved)
        """
        chunks = []
        current_chunk = ""
        
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
            
            # Check if we should split after this section
            if section_idx in final_splits:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
        
        # Add last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(f"Reconstructed {len(chunks)} final chunks from {len(section_chunks)} sections")
        return chunks
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Split text according to LLM suggested split points.
        
        Args:
            chunked_text: Text with chunk markers
            llm_response: LLM response with split suggestions
            
        Returns:
            List of document chunks
        """
        # Extract split points from LLM response
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        # If no splits were suggested, return the whole text as one section
        if not split_after:
            return [chunked_text]

        # Find all chunk markers in the text
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks according to split points
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # Add the last section if it's not empty
        if current_section:
            sections.append("".join(current_section).strip())

        return sections