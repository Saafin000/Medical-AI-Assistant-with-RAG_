"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
import re
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config
from agents.clinical_pre_consultation_agent import ClinicalPreConsultationAgent
from agents.provider_error_handler import error_classifier, state_preserver
import threading

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Thread config will be set per session in process_query function
# This is now dynamic and session-specific
def get_thread_config(session_id: Optional[str] = None) -> dict:
    """
    Generate a thread config for LangGraph based on session_id.
    Ensures each session has independent conversation history.
    
    Args:
        session_id: Unique identifier for the session (UUID)
        
    Returns:
        Thread config dict for LangGraph graph.invoke()
    """
    # Use session_id if provided, otherwise use a default
    thread_id = session_id if session_id else "default"
    return {"configurable": {"thread_id": thread_id}}

# Clinical Agent Registry: maps session_id -> ClinicalPreConsultationAgent instance
_clinical_agent_registry = {}
_registry_lock = threading.Lock()


def get_clinical_agent(session_id: str) -> ClinicalPreConsultationAgent:
    """Get or create a clinical agent for the given session."""
    with _registry_lock:
        if not session_id or session_id in ("default", "none", "null"):
            # Prevent un-sessioned queries from polluting or sharing an agent instance
            return ClinicalPreConsultationAgent(config)

        if session_id not in _clinical_agent_registry:
            agent = ClinicalPreConsultationAgent(config)
            # Restore state if preserved from a previous provider failure or persistent SQLite DB
            preserved = state_preserver.restore_session_state(session_id)
            if preserved and hasattr(agent, 'patient_info_collector'):
                agent.patient_info_collector.patient_data = preserved
            else:
                try:
                    from storage.session_db import session_db
                    saved_state = session_db.get_clinical_state(session_id)
                    if saved_state and hasattr(agent, 'patient_info_collector'):
                        if saved_state.get("patient_data"):
                            agent.patient_info_collector.patient_data = saved_state["patient_data"]
                            fc_status = saved_state["patient_data"].get("final_confirmation_status")
                            if fc_status == "awaiting_response":
                                agent._awaiting_final_confirmation = True
                                agent._final_confirmation_asked = True
                            elif fc_status == "confirmed":
                                agent._final_confirmation_asked = True
                                agent._final_confirmation_completed = True
                        latest_rep = saved_state.get("latest_report") or saved_state.get("latest_report_text")
                        if latest_rep:
                            agent._latest_report = latest_rep
                        if saved_state.get("formatted_report"):
                            agent._latest_formatted_report = saved_state["formatted_report"]
                        if saved_state.get("emergency_flag"):
                            agent._emergency_detected = True
                        if saved_state.get("current_stage") in ("completed", "emergency"):
                            agent._session_active = False
                except Exception:
                    pass
            _clinical_agent_registry[session_id] = agent
        return _clinical_agent_registry[session_id]


def reset_clinical_agent_session(session_id: str) -> None:
    """Reset (remove) the clinical agent for a session."""
    with _registry_lock:
        if not session_id or session_id in ("*", "all"):
            _clinical_agent_registry.clear()
        elif session_id in _clinical_agent_registry:
            del _clinical_agent_registry[session_id]
        try:
            state_preserver.clear_session_state(session_id)
        except Exception:
            pass


def build_conversation_prompt(input_text: str, recent_context: str) -> str:
    """Build standardized conversation prompt for both streaming and non-streaming modes."""
    return f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CLINICAL_PRE_CONSULTATION_AGENT - For structured patient intake, symptom collection, preliminary assessment, and pre-consultation report generation. Route here for: explicit requests to start pre-consultation or patient intake, symptom-focused medical histories, or requests for a doctor-facing clinical report. This agent conducts adaptive follow-up questions until sufficient information is collected or emergency signs are detected. If the user is already in a clinical consultation session, continue routing to this agent unless they explicitly exit or complete the consultation.
    2. CONVERSATION_AGENT - For general chat, greetings, casual medical questions, and non-clinical dialogue.
    3. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    4. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    5. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    6. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    7. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    Make your decision based on these guidelines:
    - **Workspace Boundary Rules (STRICT):**
      - If Active Workspace is "Medical AI Conversation": CLINICAL_PRE_CONSULTATION_AGENT is PROHIBITED. The user is in a general conversational workspace. All symptom inquiries (such as fever, cough, runny nose, headache, pain, cold), general medical questions, and health advice requests must be routed to CONVERSATION_AGENT (or RAG_AGENT / WEB_SEARCH_PROCESSOR_AGENT if asking for medical literature or recent health news). Never route to CLINICAL_PRE_CONSULTATION_AGENT when in Medical AI Conversation.
      - If Active Workspace is "Clinical Pre-Consultation": The user is in structured pre-consultation. Route to CLINICAL_PRE_CONSULTATION_AGENT for clinical intake, symptom collection, adaptive follow-ups, and report generation.
    - **Clinical Pre-Consultation Priority:** In the Clinical Pre-Consultation workspace, or if the user explicitly requests to start a pre-consultation, begin a patient intake, start a medical consultation, or asks for a pre-consultation report, route to CLINICAL_PRE_CONSULTATION_AGENT.
    - **Session Continuity:** If the user is already in an active clinical consultation (indicated in conversation history and clinical workspace), continue routing follow-up answers and symptoms to CLINICAL_PRE_CONSULTATION_AGENT unless the user explicitly exits or completes the consultation.
    - **Image Handling:** If the user has uploaded a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - **Recent Medical Information:** If the user asks about recent medical developments or current health situations, use the web search processor agent.
    - **Medical Knowledge Questions:** If the user asks specific medical knowledge questions, use the RAG agent.
    - **General Conversation:** For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    session_id: Optional[str]  # Session ID for tracking clinical agent state
    workspace: Optional[str]  # Active workspace boundary ('conversation', 'clinical_pre_consultation', 'image_analysis')
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        workspace = state.get("workspace")
        session_id = state.get("session_id")

        # Determine workspace from session_id if not present
        if not workspace and session_id:
            if session_id.startswith("clin_"):
                workspace = "clinical_pre_consultation"
            elif session_id.startswith("conv_"):
                workspace = "conversation"
            elif session_id.startswith("img_"):
                workspace = "image_analysis"
        if not workspace:
            workspace = "conversation"

        workspace_display = "Clinical Pre-Consultation" if workspace == "clinical_pre_consultation" else "Medical AI Conversation"
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Active Workspace: {workspace_display}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information and the workspace boundary rules, which agent should handle this query?
        """
        
        # Make the decision - wrap with provider error classification
        try:
            decision = decision_chain.invoke({"input": decision_input})
        except Exception as e:
            error_msg = str(e)
            print(f"Routing decision error: {error_msg}")
            classification = error_classifier.classify_error(e)
            print(f"Routing error classified as: {classification['category']} ({classification['error_code']})")
            # Re-raise to be caught by /chat endpoint
            raise

        decided_agent = decision.get("agent")
        confidence = decision.get("confidence", 0.95)
        print(f"Decision: {decided_agent} (confidence: {confidence}, workspace: {workspace})")

        # Workspace boundary enforcement
        if workspace == "conversation" and decided_agent == "CLINICAL_PRE_CONSULTATION_AGENT":
            print(f"Workspace boundary enforcement: CLINICAL_PRE_CONSULTATION_AGENT is prohibited in Medical AI Conversation workspace. Re-routing to CONVERSATION_AGENT.")
            decided_agent = "CONVERSATION_AGENT"
            confidence = 0.95
        elif workspace == "clinical_pre_consultation" and decided_agent not in ("CLINICAL_PRE_CONSULTATION_AGENT", "BRAIN_TUMOR_AGENT", "CHEST_XRAY_AGENT", "SKIN_LESION_AGENT"):
            print(f"Workspace boundary enforcement: Clinical Pre-Consultation workspace requires structured intake. Re-routing to CLINICAL_PRE_CONSULTATION_AGENT.")
            decided_agent = "CLINICAL_PRE_CONSULTATION_AGENT"
            confidence = 0.95

        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decided_agent,
            "workspace": workspace
        }
        
        # Route based on agent name and confidence
        if confidence < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decided_agent}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_clinical_pre_consultation_agent(state: AgentState) -> AgentState:
        """
        Handle clinical pre-consultation intake and report generation.
        
        Ensures patient data is persisted even if LLM fails.
        Distinguishes provider failures from successful responses.
        """
        
        print(f"Selected agent: CLINICAL_PRE_CONSULTATION_AGENT")
        
        session_id = state.get("session_id")
        current_input = state["current_input"]
        
        # Prepare input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Get or create clinical agent for this session
        clinical_agent = get_clinical_agent(session_id)
        
        # Process patient info through the clinical agent
        # The agent maintains its own state, so each call accumulates information
        try:
            clinical_response = clinical_agent.process_patient_info(input_text)
        except Exception as e:
            error_msg = str(e)
            print(f"Clinical agent error: {error_msg}")
            
            # Preserve patient data already collected in clinical_agent
            try:
                if hasattr(clinical_agent, 'patient_info_collector'):
                    patient_data = clinical_agent.patient_info_collector.get_patient_data()
                    if session_id and patient_data:
                        state_preserver.save_session_state(session_id, patient_data)
            except Exception as pe:
                print(f"Failed to preserve patient data: {pe}")
            
            # Do NOT append a fake assistant response to LangGraph state.
            # Re-raise so /chat endpoint handles provider failure and preserves session state.
            raise
        
        # Extract response data from clinical agent
        status = clinical_response.get("status")  # "active", "completed", "partial", "emergency_escalation"
        next_question = clinical_response.get("next_question")
        report = clinical_response.get("report")
        formatted_report = clinical_response.get("formatted_report")
        requires_immediate_attention = clinical_response.get("requires_immediate_attention", False)
        questions_asked_count = clinical_response.get("questions_asked_count", 0)
        
        # Prepare response content
        response_content = ""
        
        if status == "emergency_escalation":
            # Emergency detected - provide guidance and report
            response_content = formatted_report or clinical_response.get("message", "Emergency detected. Seeking immediate medical attention.")
            needs_validation = True
        elif status == "completed":
            # Sufficient information collected - provide final report
            response_content = formatted_report or clinical_response.get("message", "Pre-consultation intake completed.")
            needs_validation = True
        elif status == "partial":
            # Question limit reached - provide partial report
            response_content = formatted_report or clinical_response.get("message", "Pre-consultation intake completed (partial).")
            needs_validation = True
        else:  # status == "active"
            # Still collecting information - ask next question
            if next_question and next_question.get("question"):
                response_content = next_question["question"]
            else:
                response_content = clinical_response.get("message", "Gathering patient information...")
            needs_validation = False
        
        # Create AI message from response
        response_message = AIMessage(content=response_content)
        
        # Update state with new message
        updated_messages = state["messages"] + [response_message]
        
        # Return updated state
        return {
            **state,
            "messages": updated_messages,
            "output": response_message,
            "agent_name": "CLINICAL_PRE_CONSULTATION_AGENT",
            "needs_human_validation": needs_validation
        }
    
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (capped to config.max_conversation_history to prevent oversized context)
        recent_context = ""
        for msg in messages[-config.max_conversation_history:]:
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        conversation_prompt = build_conversation_prompt(input_text, recent_context)

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, generates a refined response, and attaches sources."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        # process_web_search_results now returns a dict with 'answer', 'sources', 'formatted_response'
        search_result = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # Extract formatted response (includes sources) or fallback to answer + formatted sources
        if isinstance(search_result, dict):
            # Use the formatted response which includes sources section
            response_text = search_result.get('formatted_response', search_result.get('answer', str(search_result)))
        else:
            # Backward compatibility: if result is string, use as-is
            response_text = str(search_result)

        # print("######### DEBUG WEB SEARCH:", response_text)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Return response with sources attached
        return {
            **state,
            "output": response_text,  # Now includes sources in formatted_response
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None) if isinstance(current_input, dict) else None

        print(f"Selected agent: CHEST_XRAY_AGENT")

        if not image_path:
            input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
            response = AIMessage(content=f"Validation recorded: {input_text}")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "HUMAN_VALIDATION"
            }

        # classify chest x-ray into covid or normal
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
        elif predicted_class == "normal":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None) if isinstance(current_input, dict) else None

        print(f"Selected agent: SKIN_LESION_AGENT")

        if not image_path:
            input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
            response = AIMessage(content=f"Validation recorded: {input_text}")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "HUMAN_VALIDATION"
            }

        # classify chest x-ray into covid or normal
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        if predicted_mask:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        
        # CRITICAL FIX: Remove internal safety-filter prompt instructions from user-facing output
        # These instructions are meant for LLM processing, not for display in chat/reports
        clean_output = guardrails.remove_safety_filter_instructions(sanitized_output)
        
        # For non-validation cases, add the cleaned output to messages
        clean_message = AIMessage(content=clean_output) if isinstance(output, AIMessage) else clean_output
        
        return {
            **state,
            "messages": clean_message,
            "output": clean_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CLINICAL_PRE_CONSULTATION_AGENT", run_clinical_pre_consultation_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CLINICAL_PRE_CONSULTATION_AGENT": "CLINICAL_PRE_CONSULTATION_AGENT",
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CLINICAL_PRE_CONSULTATION_AGENT", "check_validation")
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state(session_id: Optional[str] = None, workspace: Optional[str] = None) -> AgentState:
    """Initialize the agent state with default values and resolved workspace boundary."""
    # Resolve workspace boundary: if not explicitly provided, infer from session_id prefix
    if not workspace and session_id:
        if session_id.startswith("clin_"):
            workspace = "clinical_pre_consultation"
        elif session_id.startswith("conv_"):
            workspace = "conversation"
        elif session_id.startswith("img_"):
            workspace = "image_analysis"
    if not workspace:
        workspace = "conversation"

    return {
        "messages": [],
        "session_id": session_id,
        "workspace": workspace,
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False
    }


def process_query(
    query: Union[str, Dict], 
    conversation_history: List[BaseMessage] = None, 
    session_id: Optional[str] = None,
    workspace: Optional[str] = None
) -> str:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages
        session_id: Optional session ID for maintaining clinical agent state across requests and ensuring conversation isolation
        workspace: Optional explicit workspace boundary ('conversation', 'clinical_pre_consultation', 'image_analysis')
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # Reset Groq API query metrics tracker
    try:
        from groq_llm_provider import usage_tracker
        usage_tracker.reset_query_metrics()
    except Exception:
        pass

    # Initialize state with session_id and workspace
    state = init_agent_state(session_id=session_id, workspace=workspace)
    # if conversation_history:
    #     state["messages"] = conversation_history
    
    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]

    # Generate session-aware thread config (CRITICAL FIX: use session_id instead of hardcoded "1")
    thread_config = get_thread_config(session_id=session_id)
    
    # Invoke graph with session-specific thread configuration
    result = graph.invoke(state, thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # Log per-query Groq API usage
    try:
        from groq_llm_provider import usage_tracker
        metrics = usage_tracker.get_query_metrics()
        print(f"Groq API Query Usage: {metrics['requests']} requests, {metrics['total_tokens']} tokens (prompt: {metrics['prompt_tokens']}, completion: {metrics['completion_tokens']})")
    except Exception:
        pass

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result


def process_query_stream(
    query: Union[str, Dict], 
    conversation_history: List[BaseMessage] = None, 
    session_id: Optional[str] = None,
    workspace: Optional[str] = None
):
    """
    Process a user query and stream real-time tokens across agents.
    Yields event dictionaries with keys 'event' and 'data'.
    """
    # 1. Reset metrics
    try:
        from groq_llm_provider import usage_tracker
        usage_tracker.reset_query_metrics()
    except Exception:
        pass

    # 2. Resolve workspace
    if not workspace and session_id:
        if session_id.startswith("clin_"):
            workspace = "clinical_pre_consultation"
        elif session_id.startswith("conv_"):
            workspace = "conversation"
        elif session_id.startswith("img_"):
            workspace = "image_analysis"
    if not workspace:
        workspace = "conversation"

    workspace_display = "Clinical Pre-Consultation" if workspace == "clinical_pre_consultation" else "Medical AI Conversation"

    # 3. Extract input text & check for images
    input_text = ""
    has_image = False
    image_type = None
    image_path = None
    if isinstance(query, str):
        input_text = query
    elif isinstance(query, dict):
        input_text = query.get("text", "")
        if "image" in query:
            has_image = True
            image_path = query.get("image")
            try:
                img_resp = AgentConfig.image_analyzer.analyze_image(image_path)
                image_type = img_resp.get("image_type")
            except Exception as ie:
                print(f"Notice: Image analyzer error: {ie}")

    # 4. Input guardrail check
    guardrails = LocalGuardrails(config.rag.llm)
    if input_text:
        try:
            is_allowed, guardrail_msg = guardrails.check_input(input_text)
            if not is_allowed:
                g_text = guardrail_msg[0].content if isinstance(guardrail_msg, list) and hasattr(guardrail_msg[0], 'content') else str(guardrail_msg)
                clean_g = guardrails.remove_safety_filter_instructions(g_text)
                yield {"event": "start", "data": {"session_id": session_id, "agent": "INPUT_GUARDRAILS", "workspace": workspace}}
                yield {"event": "token", "data": {"chunk": clean_g}}
                yield {"event": "complete", "data": {"response": clean_g, "agent": "INPUT_GUARDRAILS", "session_id": session_id}}
                return
        except Exception as ge:
            print(f"Guardrail check error: {ge}")

    # 5. Build recent context for decision model
    recent_context = ""
    try:
        from storage.session_db import session_db
        if session_id:
            db_msgs = session_db.get_messages(session_id, limit=config.max_conversation_history)
            for m in db_msgs[-6:]:
                sender_label = "User" if m.get("sender") == "user" else "Assistant"
                recent_context += f"{sender_label}: {m.get('content', '')}\n"
    except Exception:
        pass

    # 6. Agent decision triage
    if workspace == "clinical_pre_consultation" and not has_image:
        decided_agent = "CLINICAL_PRE_CONSULTATION_AGENT"
        confidence = 1.0
    else:
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        decision_chain = decision_prompt | config.agent_decision.llm | JsonOutputParser(pydantic_object=AgentDecision)

        decision_input = f"""
        User query: {input_text}

        Active Workspace: {workspace_display}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information and the workspace boundary rules, which agent should handle this query?
        """

        try:
            decision = decision_chain.invoke({"input": decision_input})
            decided_agent = decision.get("agent", "CONVERSATION_AGENT")
            confidence = decision.get("confidence", 0.95)
        except Exception as e:
            print(f"Routing decision notice (defaulting to workspace agent): {e}")
            decided_agent = "CLINICAL_PRE_CONSULTATION_AGENT" if workspace == "clinical_pre_consultation" else "CONVERSATION_AGENT"
            confidence = 0.95

        # Workspace boundary enforcement
        if workspace == "conversation" and decided_agent == "CLINICAL_PRE_CONSULTATION_AGENT":
            print("Workspace boundary enforcement: CLINICAL_PRE_CONSULTATION_AGENT prohibited in conversation. Re-routing to CONVERSATION_AGENT.")
            decided_agent = "CONVERSATION_AGENT"
            confidence = 0.95
        elif workspace == "clinical_pre_consultation" and decided_agent not in ("CLINICAL_PRE_CONSULTATION_AGENT", "BRAIN_TUMOR_AGENT", "CHEST_XRAY_AGENT", "SKIN_LESION_AGENT"):
            print("Workspace boundary enforcement: Re-routing to CLINICAL_PRE_CONSULTATION_AGENT in clinical workspace.")
            decided_agent = "CLINICAL_PRE_CONSULTATION_AGENT"
            confidence = 0.95

        if confidence < AgentConfig.CONFIDENCE_THRESHOLD and decided_agent not in ("BRAIN_TUMOR_AGENT", "CHEST_XRAY_AGENT", "SKIN_LESION_AGENT"):
            decided_agent = "RAG_AGENT"

    # Emit start event
    yield {"event": "start", "data": {"session_id": session_id, "agent": decided_agent, "workspace": workspace}}

    # 7. Execute Agent Stream
    try:
        if decided_agent == "CONVERSATION_AGENT":
            conversation_prompt = build_conversation_prompt(input_text, recent_context)
            full_response = ""
            for chunk in config.conversation.llm.stream(conversation_prompt):
                txt = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if txt:
                    full_response += txt
                    yield {"event": "token", "data": {"chunk": txt}}

            cleaned = guardrails.remove_safety_filter_instructions(full_response)
            yield {
                "event": "complete",
                "data": {
                    "response": cleaned,
                    "agent": "CONVERSATION_AGENT",
                    "session_id": session_id
                }
            }

        elif decided_agent == "RAG_AGENT":
            rag_agent = MedicalRAG(config)
            expansion_result = rag_agent.query_expander.expand_query(input_text)
            search_query = expansion_result.get("expanded_query", input_text) or input_text
            vectorstore, docstore = rag_agent.vector_store.load_vectorstore()
            retrieved_documents = rag_agent.vector_store.retrieve_relevant_chunks(
                query=search_query,
                vectorstore=vectorstore,
                docstore=docstore,
            )
            retrieval_confidence = rag_agent.response_generator._calculate_confidence(retrieved_documents)

            if retrieval_confidence < config.rag.min_retrieval_confidence or not retrieved_documents:
                # Fallback to web search
                yield {"event": "status", "data": {"message": "Searching external medical literature..."}}
                web_search_processor = WebSearchProcessorAgent(config)
                for item in web_search_processor.stream_web_search_results(query=input_text, chat_history=recent_context):
                    if item[0] == "token":
                        yield {"event": "token", "data": {"chunk": item[1]}}
                    elif item[0] == "complete":
                        cleaned = guardrails.remove_safety_filter_instructions(item[1]["formatted_response"])
                        yield {
                            "event": "complete",
                            "data": {
                                "response": cleaned,
                                "agent": "RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT",
                                "session_id": session_id
                            }
                        }
            else:
                if rag_agent.reranker and len(retrieved_documents) > 1:
                    reranked_documents, reranked_top_k_picture_paths = rag_agent.reranker.rerank(
                        search_query, retrieved_documents, rag_agent.parsed_content_dir
                    )
                else:
                    reranked_documents = retrieved_documents
                    reranked_top_k_picture_paths = []

                full_rag = ""
                for item in rag_agent.response_generator.generate_response_stream(
                    query=input_text,
                    retrieved_docs=reranked_documents,
                    picture_paths=reranked_top_k_picture_paths,
                    chat_history=recent_context
                ):
                    if item[0] == "token":
                        full_rag += item[1]
                        yield {"event": "token", "data": {"chunk": item[1]}}
                    elif item[0] == "complete":
                        # Check insufficient info
                        if any(marker in full_rag.lower() for marker in ["don't have enough information", "insufficient information", "cannot answer", "unable to answer"]):
                            yield {"event": "status", "data": {"message": "Expanding search to web medical literature..."}}
                            web_search_processor = WebSearchProcessorAgent(config)
                            for w_item in web_search_processor.stream_web_search_results(query=input_text, chat_history=recent_context):
                                if w_item[0] == "token":
                                    yield {"event": "token", "data": {"chunk": w_item[1]}}
                                elif w_item[0] == "complete":
                                    cleaned = guardrails.remove_safety_filter_instructions(w_item[1]["formatted_response"])
                                    yield {
                                        "event": "complete",
                                        "data": {
                                            "response": cleaned,
                                            "agent": "RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT",
                                            "session_id": session_id
                                        }
                                    }
                        else:
                            cleaned = guardrails.remove_safety_filter_instructions(item[1]["response"])
                            yield {
                                "event": "complete",
                                "data": {
                                    "response": cleaned,
                                    "agent": "RAG_AGENT",
                                    "session_id": session_id
                                }
                            }

        elif decided_agent == "WEB_SEARCH_PROCESSOR_AGENT":
            yield {"event": "status", "data": {"message": "Searching external medical literature..."}}
            web_search_processor = WebSearchProcessorAgent(config)
            for item in web_search_processor.stream_web_search_results(query=input_text, chat_history=recent_context):
                if item[0] == "token":
                    yield {"event": "token", "data": {"chunk": item[1]}}
                elif item[0] == "complete":
                    cleaned = guardrails.remove_safety_filter_instructions(item[1]["formatted_response"])
                    yield {
                        "event": "complete",
                        "data": {
                            "response": cleaned,
                            "agent": "WEB_SEARCH_PROCESSOR_AGENT",
                            "session_id": session_id
                        }
                    }

        elif decided_agent == "CLINICAL_PRE_CONSULTATION_AGENT":
            clinical_agent = get_clinical_agent(session_id)
            clinical_response = clinical_agent.process_patient_info(input_text)
            status = clinical_response.get("status")
            next_question = clinical_response.get("next_question")
            formatted_report = clinical_response.get("formatted_report")
            if status == "emergency_escalation":
                response_content = formatted_report or clinical_response.get("message", "Emergency detected. Seeking immediate medical attention.")
                needs_validation = True
            elif status in ("completed", "partial"):
                response_content = formatted_report or clinical_response.get("message", "Pre-consultation intake completed.")
                needs_validation = True
            else:
                if next_question and next_question.get("question"):
                    response_content = next_question["question"]
                else:
                    response_content = clinical_response.get("message", "Gathering patient information...")
                needs_validation = False

            if needs_validation:
                response_content = f"{response_content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

            clean_content = guardrails.remove_safety_filter_instructions(response_content)
            yield {"event": "token", "data": {"chunk": clean_content}}

            agent_name = "CLINICAL_PRE_CONSULTATION_AGENT"
            if needs_validation:
                agent_name += ", HUMAN_VALIDATION"

            yield {
                "event": "complete",
                "data": {
                    "response": clean_content,
                    "agent": agent_name,
                    "session_id": session_id
                }
            }

        elif decided_agent in ("CHEST_XRAY_AGENT", "BRAIN_TUMOR_AGENT", "SKIN_LESION_AGENT"):
            result_image = None
            if decided_agent == "CHEST_XRAY_AGENT":
                if not image_path:
                    resp_text = "No image was provided for chest X-ray analysis."
                else:
                    predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)
                    if predicted_class == "covid19":
                        resp_text = "The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**."
                    elif predicted_class == "normal":
                        resp_text = "The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**."
                    else:
                        resp_text = "The uploaded image is not clear enough to make a diagnosis / the image is not a medical image."
                agent_name = "CHEST_XRAY_AGENT, HUMAN_VALIDATION"
            elif decided_agent == "SKIN_LESION_AGENT":
                if not image_path:
                    resp_text = "No image was provided for skin lesion analysis."
                else:
                    predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)
                    if predicted_mask:
                        resp_text = "Following is the analyzed **segmented** output of the uploaded skin lesion image:"
                        seg_path = os.path.join("uploads/skin_lesion_output", "segmentation_plot.png")
                        if os.path.exists(seg_path):
                            result_image = "/uploads/skin_lesion_output/segmentation_plot.png"
                    else:
                        resp_text = "The uploaded image is not clear enough to make a diagnosis / the image is not a medical image."
                agent_name = "SKIN_LESION_AGENT, HUMAN_VALIDATION"
            else:  # BRAIN_TUMOR_AGENT
                resp_text = "This would be handled by the brain tumor agent, analyzing the MRI image."
                agent_name = "BRAIN_TUMOR_AGENT, HUMAN_VALIDATION"

            resp_text += "\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."
            clean_text = guardrails.remove_safety_filter_instructions(resp_text)
            yield {"event": "token", "data": {"chunk": clean_text}}

            comp_data = {
                "response": clean_text,
                "agent": agent_name,
                "session_id": session_id
            }
            if result_image:
                comp_data["result_image"] = result_image
            yield {"event": "complete", "data": comp_data}

    except Exception as stream_err:
        print(f"Error during stream generation: {stream_err}")
        yield {"event": "error", "data": {"message": str(stream_err), "session_id": session_id}}