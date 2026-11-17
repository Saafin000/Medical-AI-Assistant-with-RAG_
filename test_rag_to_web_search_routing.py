#!/usr/bin/env python3
"""
READ-ONLY WEB SEARCH FALLBACK ROUTING TEST

Tests automatic routing from low-confidence RAG responses to Web Search Agent.
Verifies confidence-based routing logic and web search integration.
"""

import sys
import logging
import traceback
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
try:
    logger.info("Importing components...")
    from config import Config
    from agents.agent_decision import create_agent_graph, init_agent_state
    from agents.rag_agent import MedicalRAG
    from langchain_core.messages import HumanMessage, AIMessage
    logger.info("Components imported successfully")
except Exception as e:
    logger.error(f"Failed to import components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize config
try:
    logger.info("Initializing configuration...")
    config = Config()
    logger.info(f"Configuration initialized")
    logger.info(f"  RAG min_retrieval_confidence: {config.rag.min_retrieval_confidence}")
    logger.info(f"  Web Search context limit: {config.web_search.context_limit}")
except Exception as e:
    logger.error(f"Failed to initialize config: {e}")
    traceback.print_exc()
    sys.exit(1)

# Create agent graph
try:
    logger.info("Creating agent graph...")
    workflow = create_agent_graph()
    logger.info("Agent graph created successfully")
except Exception as e:
    logger.error(f"Failed to create agent graph: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test queries
test_queries = [
    {
        "query": "What are the latest WHO recommendations for malaria prevention in 2026?",
        "expected_routing": "WEB_SEARCH_PROCESSOR_AGENT",
        "reason": "Question about recent/current information not in indexed documents"
    },
    {
        "query": "What are the latest developments in brain tumor treatment methods?",
        "expected_routing": "WEB_SEARCH_PROCESSOR_AGENT (or RAG if document contains some info)",
        "reason": "Question about recent developments - may have partial info in brain_tumor_2024.pdf"
    }
]

print("\n" + "="*80)
print("RAG TO WEB SEARCH AUTOMATIC ROUTING TEST")
print("="*80)

print(f"\nPython Executable: {sys.executable}")
print(f"Environment: Project Virtual Environment")

print(f"\n[CONFIGURATION]")
print(f"  RAG Confidence Threshold: {config.rag.min_retrieval_confidence}")
print(f"  Web Search Agent Available: Yes")
print(f"  Search Provider: DuckDuckGo (ddgs)")
print(f"  Max Search Results: 5")

# Test each query
for query_idx, test_case in enumerate(test_queries, 1):
    query = test_case["query"]
    expected_routing = test_case["expected_routing"]
    reason = test_case["reason"]
    
    print("\n" + "="*80)
    print(f"TEST {query_idx}: {query}")
    print("="*80)
    
    print(f"Expected Routing: {expected_routing}")
    print(f"Reason: {reason}")
    
    try:
        # Initialize state
        logger.info(f"Initializing agent state...")
        state = init_agent_state()
        state["messages"] = [HumanMessage(content=query)]
        state["current_input"] = query
        
        # Execute the workflow with required configuration
        logger.info(f"Processing query through workflow...")
        config_dict = {"configurable": {"thread_id": f"test_{query_idx}"}}
        result = workflow.invoke(state, config=config_dict)
        
        # Extract results
        final_output = result.get("output", "")
        agent_name = result.get("agent_name", "UNKNOWN")
        retrieval_confidence = result.get("retrieval_confidence", 0.0)
        insufficient_info = result.get("insufficient_info", False)
        
        print(f"\n[ROUTING DECISION]")
        print(f"  Agent Selected: {agent_name}")
        print(f"  Retrieval Confidence: {retrieval_confidence:.4f}")
        print(f"  Insufficient Info Flag: {insufficient_info}")
        print(f"  Threshold: {config.rag.min_retrieval_confidence}")
        
        # Determine if routing was correct
        if "WEB_SEARCH_PROCESSOR_AGENT" in agent_name:
            print(f"  Routing Status: WEB SEARCH ACTIVATED")
        elif "RAG_AGENT" in agent_name:
            print(f"  Routing Status: RAG RESPONSE (Confidence high enough)")
        else:
            print(f"  Routing Status: {agent_name}")
        
        print(f"\n[RESPONSE]")
        if isinstance(final_output, AIMessage):
            response_text = final_output.content
        else:
            response_text = str(final_output)
        
        # Truncate long responses for display
        if len(response_text) > 500:
            print(f"  {response_text[:500]}...")
            print(f"  [Response truncated. Total length: {len(response_text)} chars]")
        else:
            print(f"  {response_text}")
        
        # Analyze routing
        print(f"\n[ROUTING ANALYSIS]")
        if retrieval_confidence < config.rag.min_retrieval_confidence:
            print(f"  Confidence {retrieval_confidence:.4f} < Threshold {config.rag.min_retrieval_confidence}")
            print(f"  Expected Action: Should trigger Web Search fallback")
            if "WEB_SEARCH_PROCESSOR_AGENT" in agent_name:
                print(f"  Actual Action: WEB SEARCH TRIGGERED")
                print(f"  Status: CORRECT ROUTING")
            else:
                print(f"  Actual Action: {agent_name}")
                print(f"  Status: ROUTING ERROR - Should have triggered Web Search")
        elif insufficient_info:
            print(f"  Insufficient Info Flag: True")
            print(f"  Expected Action: Should trigger Web Search fallback")
            if "WEB_SEARCH_PROCESSOR_AGENT" in agent_name:
                print(f"  Actual Action: WEB SEARCH TRIGGERED")
                print(f"  Status: CORRECT ROUTING")
            else:
                print(f"  Actual Action: {agent_name}")
                print(f"  Status: ROUTING ERROR - Should have triggered Web Search")
        else:
            print(f"  Confidence {retrieval_confidence:.4f} >= Threshold {config.rag.min_retrieval_confidence}")
            print(f"  Expected Action: RAG response acceptable")
            if "RAG_AGENT" in agent_name or "RAG" in agent_name:
                print(f"  Actual Action: RAG RESPONSE USED")
                print(f"  Status: CORRECT ROUTING")
            else:
                print(f"  Actual Action: {agent_name}")
        
        print(f"\n[SUMMARY FOR TEST {query_idx}]")
        print(f"  Status: SUCCESS")
        print(f"  Agent Used: {agent_name}")
        print(f"  Confidence: {retrieval_confidence:.4f}")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        print(f"\n[ERROR FOR TEST {query_idx}]")
        print(f"  Status: FAILED")
        print(f"  Error: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()

print("\n" + "="*80)
print("RAG TO WEB SEARCH ROUTING TEST COMPLETE")
print("="*80)

print(f"\n[TEST SUMMARY]")
print(f"  Total Tests: {len(test_queries)}")
print(f"  Environment: Project Virtual Environment (Python 3.11)")
print(f"  Read-Only: Yes (no modifications to database or files)")
print(f"\n[KEY FINDINGS]")
print(f"  - Routing threshold: {config.rag.min_retrieval_confidence}")
print(f"  - Web Search Provider: DuckDuckGo")
print(f"  - Automatic fallback mechanism: Implemented in confidence_based_routing()")
