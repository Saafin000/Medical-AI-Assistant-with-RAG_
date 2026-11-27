import os
import re
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    Preserves source attribution throughout the pipeline.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def _extract_sources_from_results(self, web_results: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Extract sources (title and URL) from raw web search results.
        
        Raw format from DuckDuckGo:
        title: X - url: Y - content: Z - score: S
        
        Args:
            web_results: Raw web search results string
            
        Returns:
            Tuple of:
            - List of dicts with 'title' and 'url' keys (in order of appearance)
            - String containing only the content parts (for LLM processing)
        """
        sources = []
        content_parts = []
        
        # Split results by newline to process each result
        results_lines = web_results.strip().split('\n')
        
        for line in results_lines:
            if not line.strip() or "No relevant results found" in line.strip():
                continue
            
            # Parse: title: X - url: Y - content: Z - score: S
            # More robust regex matching to handle various formats
            title_match = re.search(r'title:\s*(.+?)\s*-\s*url:', line)
            url_match = re.search(r'url:\s*(.+?)\s*-\s*content:', line)
            content_match = re.search(r'content:\s*(.+?)\s*-\s*score:', line)
            
            if title_match and url_match:
                title = title_match.group(1).strip()
                url = url_match.group(1).strip()
                content = content_match.group(1).strip() if content_match else ""
                
                # Only add if we have both title and URL
                if title and url:
                    # Validate URL is not obviously fake - must start with http or https
                    if url.startswith('http://') or url.startswith('https://'):
                        sources.append({
                            'title': title,
                            'url': url
                        })
                        content_parts.append(f"{title}: {content}")
        
        # Reconstruct content string for LLM
        content_str = '\n'.join(content_parts) if content_parts else web_results
        
        return sources, content_str
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, any]:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response with sources.
        
        Returns:
            Dict with:
            - 'answer': LLM-generated answer text
            - 'sources': List of source dicts with 'title' and 'url'
            - 'formatted_response': Full response including sources section (for backward compatibility)
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results (raw format with titles and URLs)
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Extract sources from raw results BEFORE passing to LLM
        sources, content_for_llm = self._extract_sources_from_results(web_results)
        
        # Construct prompt to LLM for processing the results
        # Include content but NOT raw URLs (we'll add them back from sources list)
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{content_for_llm}\n\nResponse:"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        
        # Format final response with sources section
        formatted_response = answer_text
        if sources:
            formatted_response += "\n\n**Sources:**"
            for source in sources:
                formatted_response += f"\n- [{source['title']}]({source['url']})"
        
        return {
            'answer': answer_text,
            'sources': sources,
            'formatted_response': formatted_response
        }

    def stream_web_search_results(self, query: str, chat_history: Optional[str] = None):
        """
        Execute web search and stream final summary response tokens, followed by sources.
        Yields (event_type, event_data):
          - ("token", token_text)
          - ("complete", {"answer": answer_text, "sources": sources, "formatted_response": formatted_response})
        """
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        search_query_str = web_search_query.content if hasattr(web_search_query, 'content') else str(web_search_query)

        web_results = self.web_search_agent.search(search_query_str)
        sources, content_for_llm = self._extract_sources_from_results(web_results)

        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{content_for_llm}\n\nResponse:"
        )

        full_answer = ""
        for chunk in self.llm.stream(llm_prompt):
            chunk_str = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if chunk_str:
                full_answer += chunk_str
                yield ("token", chunk_str)

        sources_text = ""
        if sources:
            sources_text += "\n\n**Sources:**"
            for source in sources:
                sources_text += f"\n- [{source['title']}]({source['url']})"

        if sources_text:
            full_answer += sources_text
            yield ("token", sources_text)

        yield ("complete", {
            "answer": full_answer,
            "sources": sources,
            "formatted_response": full_answer
        })
