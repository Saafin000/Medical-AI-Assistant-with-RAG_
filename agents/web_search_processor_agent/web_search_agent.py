import requests
from typing import Dict

from .pubmed_search import PubmedSearchAgent
from .duckduckgo_search import DuckDuckGoSearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time medical information from web sources.
    """
    
    def __init__(self, config):
        self.duckduckgo_search_agent = DuckDuckGoSearchAgent()
        
        # self.pubmed_search_agent = PubmedSearchAgent()
        # self.pubmed_api_url = config.pubmed_api_url
    
    def search(self, query: str) -> str:
        """
        Perform both general and medical-specific searches.
        """
        # print(f"[WebSearchAgent] Searching for: {query}")
        
        web_results = self.duckduckgo_search_agent.search_duckduckgo(query=query)
        # pubmed_results = self.pubmed_search_agent.search_pubmed(self.pubmed_api_url, query)
        
        return f"Web Search Results:\n{web_results}\n"
        # \nPubMed Results:\n{pubmed_results}"
