from typing import List, Dict, Any, Optional
from .web_search_processor import WebSearchProcessor

class WebSearchProcessorAgent:
    """
    Agent responsible for processing web search results and routing them to the appropriate LLM for response generation.
    Preserves source attribution throughout the pipeline.
    """
    
    def __init__(self, config):
        self.web_search_processor = WebSearchProcessor(config)
    
    def process_web_search_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Processes web search results and returns a user-friendly response with source attribution.
        
        Returns:
            Dict with:
            - 'answer': LLM-generated answer text
            - 'sources': List of source dicts with 'title' and 'url'
            - 'formatted_response': Full response including sources section
        """
        return self.web_search_processor.process_web_results(query, chat_history)

    def stream_web_search_results(self, query: str, chat_history: Optional[str] = None):
        """
        Stream tokens from web search results processing.
        """
        return self.web_search_processor.stream_web_search_results(query, chat_history)