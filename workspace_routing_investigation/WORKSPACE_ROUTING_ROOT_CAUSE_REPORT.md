# WORKSPACE ROUTING INVESTIGATION: ROOT CAUSE REPORT

**Project:** Medical AI Assistant with Multi-Agent Orchestration & RAG  
**Date:** 2026-11-25  
**Investigation Focus:** Premature / Incorrect Invocation of Clinical Pre-Consultation Agent inside Medical AI Conversation Workspace  

---

## 1. Executive Summary

A critical workspace routing anomaly was reported in the Multi-Agent Medical Assistant:
When interacting within the **Medical AI Conversation** workspace, a user submitted the general medical query:
> *"I have fever from 2 days and I also have runny nose and feeling body temperature high."*

Instead of receiving general medical information, explanations of potential causes, and appropriate health guidance via the Conversational/RAG/Web agents, the application responded with:
> *"Have you taken any fever-reducing medication, such as paracetamol or ibuprofen, and if so, did it help lower your temperature?"*

This response originated directly from `ClinicalPreConsultationAgent`, initiating an adaptive clinical pre-consultation intake interview inside what was selected as the **Medical AI Conversation** workspace.

Our comprehensive source-code and runtime investigation has identified the precise root cause:
1. **Workspace Context Blindness in Decision Input**: Although the frontend distinguishes workspaces and transmits workspace-prefixed session IDs (`conv_<uuid>` vs `clin_<uuid>`), neither the prompt provided to the LLM decision chain (`decision_input`) nor the state evaluated by `route_to_agent()` (`agents/agent_decision.py`, lines 247–307) included the active workspace context.
2. **Symptom-Biased Routing Prompt**: The decision system prompt (`AgentConfig.DECISION_SYSTEM_PROMPT`, lines 122–149) explicitly instructs the LLM to route any *"symptom collection, preliminary assessment, symptom-focused medical histories"* to `CLINICAL_PRE_CONSULTATION_AGENT`. Because the query described common symptoms (*"fever"*, *"runny nose"*, *"body temperature high"*), the decision model selected `CLINICAL_PRE_CONSULTATION_AGENT` regardless of workspace.
3. **Absence of Workspace Boundary Enforcement**: The routing node in LangGraph (`route_to_agent()`) executed the LLM's raw decision without checking whether `CLINICAL_PRE_CONSULTATION_AGENT` is permissible in the active workspace. As a result, symptom queries in Medical AI Conversation bypassed general conversation and activated the clinical intake engine.

---

## 2. Actual Frontend-to-Backend Request Flow

The end-to-end request flow was traced across the codebase:

```
[User in React Frontend: /conversation]
                │
                ▼ (User types symptom query & clicks Send)
[frontend/src/context/ConversationContext.jsx: sendMessage()]
                │
                ├── Active Session ID: "conv_<uuid>"
                ├── Headers: { "Content-Type": "application/json", "X-Session-ID": "conv_<uuid>" }
                └── Payload: { "query": "I have fever...", "conversation_history": [] }
                │
                ▼ (POST /chat)
[FastAPI Backend: app.py: chat()]
                │
                ├── resolve_session_id() validates session "conv_<uuid>" against allowed_workspaces
                ├── session_db.save_message() stores user message under session "conv_<uuid>"
                └── Calls process_query(request.query, session_id=session_id)
                │
                ▼ (agents/agent_decision.py: process_query())
[init_agent_state(session_id=session_id)]
                │
                ├── State initialized with session_id="conv_<uuid>", but NO workspace field
                └── Graph invoked: graph.invoke(state, thread_config)
                │
                ▼ (LangGraph Node 1: analyze_input())
[analyze_input(): Checks guardrails & image attachments -> No image, passes guardrails]
                │
                ▼ (LangGraph Node 2: route_to_agent())
[route_to_agent(): Constructs decision_input for LLM decision chain]
                │
                ├── decision_input contains ONLY: query, recent history, has_image, image_type
                ├── Workspace context ("conversation") is COMPLETELY OMITTED from decision_input
                └── decision_chain.invoke({"input": decision_input})
                │
                ▼ (Groq LLM evaluates DECISION_SYSTEM_PROMPT)
[LLM Decision Model: Evaluates query against DECISION_SYSTEM_PROMPT]
                │
                ├── Prompt says: "Route here for: symptom collection, symptom-focused medical histories"
                ├── Query contains: "fever", "runny nose", "body temperature high"
                └── LLM returns: { "agent": "CLINICAL_PRE_CONSULTATION_AGENT", "confidence": 0.95 }
                │
                ▼ (LangGraph Edge: route_to_agent next step)
[route_to_agent() returns {"agent_state": updated_state, "next": "CLINICAL_PRE_CONSULTATION_AGENT"}]
                │
                ├── Zero check performed on session_id prefix ("conv_") or workspace boundary!
                └── Graph edge unconditionally routes to CLINICAL_PRE_CONSULTATION_AGENT node
                │
                ▼ (LangGraph Node 3: run_clinical_pre_consultation_agent())
[run_clinical_pre_consultation_agent()]
                │
                ├── Calls get_clinical_agent("conv_<uuid>") -> creates/caches instance for conv_ session
                ├── Calls clinical_agent.process_patient_info(input_text)
                ├── clinical_agent treats query as Turn 1 of adaptive clinical intake
                ├── Generates next adaptive follow-up question:
                │   "Have you taken any fever-reducing medication, such as paracetamol or ibuprofen..."
                └── State updated with agent_name="CLINICAL_PRE_CONSULTATION_AGENT"
                │
                ▼ (app.py: response handling)
[app.py: chat() response packaging]
                │
                ├── session_db.save_message() stores clinical question under conv_ session
                ├── session_db.save_clinical_state() saves intake state to database for conv_ session
                └── Returns JSON { "agent": "CLINICAL_PRE_CONSULTATION_AGENT", "response": "Have you taken..." }
                │
                ▼
[User in Medical AI Conversation receives clinical intake question instead of medical info]
```

---

## 3. Detailed Answers to Investigation Questions

### 1. Is the selected workspace sent with every message?
**Finding:** Partially.
- **Header:** Both `ConversationContext.jsx` (line 310) and `ClinicalPreConsultationContext.jsx` (line 346) send the `X-Session-ID` HTTP header containing the active workspace prefix: `conv_<uuid>` for Medical AI Conversation, and `clin_<uuid>` for Clinical Pre-Consultation.
- **Body:** Neither frontend context currently includes `workspace` or `workspace_type` in the JSON request body sent to `/chat` (`{ query, conversation_history }`).
- **Verdict:** Workspace identity is present in the `X-Session-ID` header prefix, but was never passed into the downstream graph state or decision prompt.

### 2. Does the backend receive and preserve it?
**Finding:** Yes in FastAPI, but dropped before LangGraph routing.
- In `app.py` lines 295–301, `resolve_session_id()` successfully parses and validates the session ID prefix (`conv_` or `clin_`), verifies it against `allowed_workspaces=("conversation", "clinical_pre_consultation")`, registers it in SQLite, and sets response headers/cookies.
- However, when `app.py` calls `process_query(request.query, session_id=session_id)` at line 315, it passes only `session_id`. `process_query` places `session_id` into `AgentState`, but does not derive or store `workspace` in `AgentState`, and `route_to_agent()` ignores `session_id` when making routing decisions.

### 3. Can symptom-related content override the selected workspace?
**Finding:** **YES. This is the direct bug mechanism.**
- Because `route_to_agent()` does not supply workspace context to `decision_chain`, and does not restrict candidate agents based on workspace, the LLM evaluates the query purely based on content semantics.
- In `AgentConfig.DECISION_SYSTEM_PROMPT` (`agents/agent_decision.py`, line 127):
  ```text
  1. CLINICAL_PRE_CONSULTATION_AGENT - For structured patient intake, symptom collection, preliminary assessment, and pre-consultation report generation. Route here for: explicit requests to start pre-consultation or patient intake, symptom-focused medical histories, or requests for a doctor-facing clinical report.
  ```
- Any query describing symptoms (*"fever"*, *"cough"*, *"headache"*, *"body aches"*) matches *"symptom collection"* and *"symptom-focused medical histories"*, prompting the LLM to output `CLINICAL_PRE_CONSULTATION_AGENT`. Symptom keywords completely overrode the user's selected workspace.

### 4. Can the LLM decision chain select the Clinical Agent while the user is in Medical AI Conversation?
**Finding:** **YES.**
- `DECISION_SYSTEM_PROMPT` presents `CLINICAL_PRE_CONSULTATION_AGENT` as candidate agent #1 to the LLM on every query, regardless of which workspace generated the request.
- Without a workspace boundary constraint, the LLM decision chain freely selected `CLINICAL_PRE_CONSULTATION_AGENT` while the user was interacting in `MedicalConversationPage`.

### 5. Is clinical invocation controlled by workspace context, message content, or both?
**Finding:** Currently controlled **exclusively by message content** through the LLM prompt.
- Workspace context has 0% influence on the agent selection in `route_to_agent()`.
- The architectural requirement mandates that clinical invocation MUST be bounded by workspace context:
  - In **Medical AI Conversation**: `CLINICAL_PRE_CONSULTATION_AGENT` must be prohibited.
  - In **Clinical Pre-Consultation**: The structured clinical intake engine must be guaranteed.

### 6. Are workspace identifiers consistent between frontend and backend?
**Finding:** **YES.**
- Frontend contexts:
  - `ConversationContext.jsx`: `workspace_type: "conversation"`, prefix `"conv_"`
  - `ClinicalPreConsultationContext.jsx`: `workspace_type: "clinical_pre_consultation"`, prefix `"clin_"`
  - `ImageAnalysisContext.jsx`: `workspace_type: "image_analysis"`, prefix `"img_"`
- Backend `app.py`:
  - `WORKSPACE_PREFIX_MAP = {"conversation": "conv_", "image_analysis": "img_", "clinical_pre_consultation": "clin_"}`
  - `PREFIX_WORKSPACE_MAP = {"conv_": "conversation", "img_": "image_analysis", "clin_": "clinical_pre_consultation"}`
- Backend SQLite `storage/session_db.py`:
  - `VALID_WORKSPACE_TYPES = ("conversation", "image_analysis", "clinical_pre_consultation")`
- The nomenclature and prefix mapping are 100% unified and consistent across all layers.

### 7. Can session restoration or history switching cause workspace leakage?
**Finding:** **YES, as a secondary side effect of the routing bug.**
- When `CLINICAL_PRE_CONSULTATION_AGENT` was erroneously triggered in a `conv_` session:
  1. `app.py` line 347 saved assistant messages into `session_messages` with `agent_name="CLINICAL_PRE_CONSULTATION_AGENT"`.
  2. `app.py` line 378 saved intake data into `clinical_intake_state` for `session_id="conv_..."`.
  3. `agents/agent_decision.py` line 93 registered a `ClinicalPreConsultationAgent` in `_clinical_agent_registry["conv_..."]`.
- When the user refreshed or reopened that conversation session, the conversation history contained clinical questions. If the user replied, `DECISION_SYSTEM_PROMPT` line 137 (`"If the user is already in an active clinical consultation (indicated in conversation history), continue routing follow-up answers... to CLINICAL_PRE_CONSULTATION_AGENT"`) cemented the session into an inescapable clinical interview loop!

### 8. Are there fallback branches that invoke the Clinical Agent unexpectedly?
**Finding:** **NO.**
- In `create_agent_graph()`:
  - `check_if_bypassing()` routes to `apply_guardrails` or `route_to_agent`.
  - `needs_validation` (confidence < 0.85) defaults to `RAG_AGENT` (line 820).
  - `confidence_based_routing()` in `RAG_AGENT` routes to `WEB_SEARCH_PROCESSOR_AGENT` or `check_validation` (line 623).
  - `check_validation` routes to `human_validation` or `apply_guardrails`.
- No fallback branch directs to `CLINICAL_PRE_CONSULTATION_AGENT`. The unexpected invocation is caused solely by `route_to_agent()` selecting it.

### 9. Does the same `/chat` endpoint handle multiple workspaces, and if so, how is workspace context preserved?
**Finding:**
- Yes. `POST /chat` is the shared endpoint for both `conversation` and `clinical_pre_consultation`.
- `app.py` line 295 resolves the workspace through `resolve_session_id()`, which inspects the `X-Session-ID` header prefix (`conv_` vs `clin_`).
- However, `app.py` failed to pass the resolved workspace to `process_query()`, and `agents/agent_decision.py` failed to extract the workspace from `session_id`, dropping the workspace boundary before routing.

---

## 4. Source-Code Evidence

### Evidence A: `app.py` (lines 287–316)
```python
@app.post("/chat")
def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None),
    x_session_id: Optional[str] = Header(None)
):
    """Process user text query through the multi-agent system."""
    session_id = resolve_session_id(
        x_session_id=x_session_id,
        session_cookie=session_id,
        response=response,
        allowed_workspaces=("conversation", "clinical_pre_consultation"),
        default_workspace="conversation"
    )
    ...
    # BUG: Only request.query and session_id are passed; workspace is not passed
    response_data = process_query(request.query, session_id=session_id)
```

### Evidence B: `agents/agent_decision.py` (lines 269–307)
```python
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        ...
        # BUG: decision_input has no workspace awareness
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        decision = decision_chain.invoke({"input": decision_input})
        ...
        # BUG: Raw decision["agent"] is returned with zero workspace boundary check
        return {"agent_state": updated_state, "next": decision["agent"]}
```

### Evidence C: `agents/agent_decision.py` (lines 127–137)
```text
    Available agents:
    1. CLINICAL_PRE_CONSULTATION_AGENT - For structured patient intake, symptom collection, preliminary assessment, and pre-consultation report generation. Route here for: explicit requests to start pre-consultation or patient intake, symptom-focused medical histories, or requests for a doctor-facing clinical report...
    
    Make your decision based on these guidelines:
    - Clinical Pre-Consultation Priority: If the user explicitly requests to start a pre-consultation, begin a patient intake, start a medical consultation, or asks for a pre-consultation report, route to CLINICAL_PRE_CONSULTATION_AGENT.
```
*Note:* The prompt lacked any instruction clarifying that when interacting in the Medical AI Conversation workspace, `CLINICAL_PRE_CONSULTATION_AGENT` is prohibited and symptom discussions belong to `CONVERSATION_AGENT`.

### Evidence D: `CONVERSATION_AGENT` Already Built to Handle Symptoms (lines 473–475)
```python
        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."
```
`CONVERSATION_AGENT` was explicitly designed, prompted, and calibrated to answer symptom questions with balanced medical explanations and non-diagnostic disclaimers, but was being starved of these queries by the unconstrained decision router.

---

## 5. Expected vs. Actual Behavior

| Dimension | Expected Behavior | Actual Current Behavior |
| :--- | :--- | :--- |
| **Workspace: Medical AI Conversation** | Answering symptom query (*"I have fever from 2 days..."*) with conversational medical information, potential non-diagnostic causes, home care, and warning signs via `CONVERSATION_AGENT` (or literature via RAG/Web). | Activates `CLINICAL_PRE_CONSULTATION_AGENT`. Begins structured patient intake interview asking: *"Have you taken any fever-reducing medication..."*. |
| **Decision Prompt Context** | Decision prompt informs the LLM of the active workspace and its allowed agent boundaries. | Decision prompt contains only user query and recent messages, without workspace context. |
| **Routing Boundary** | `route_to_agent()` strictly enforces that `CLINICAL_PRE_CONSULTATION_AGENT` can NEVER be invoked in `conversation` workspace. | `route_to_agent()` blindly executes whatever agent string the LLM returns. |
| **Workspace: Clinical Pre-Consultation** | Retains full adaptive questioning, symptom extraction, red-flag detection, and physician report generation. | Functional, but susceptible to cross-pollution if session IDs lack workspace isolation. |
| **Session Isolation** | `conv_` sessions contain only conversational/RAG/Web messages. `clin_` sessions contain intake messages and report state. | `conv_` sessions were polluted with clinical agent messages and `clinical_intake_state` records. |

---

## 6. Minimal Targeted Fix Strategy

To adhere strictly to all architectural constraints (no hardcoded keyword routing, decision chain executes once per message, no LangGraph bypass, no agent rewrites, backward-compatible API contracts):

1. **Frontend Contexts (`ConversationContext.jsx`, `ClinicalPreConsultationContext.jsx`)**:
   - Add explicit `workspace_type: "conversation"` and `workspace_type: "clinical_pre_consultation"` to the JSON body sent to `POST /chat` (alongside existing `X-Session-ID` headers).
2. **Backend Request Model & Endpoint (`app.py`)**:
   - Update `QueryRequest` to accept optional `workspace: Optional[str] = None` and `workspace_type: Optional[str] = None`.
   - In `/chat`, derive `resolved_workspace`:
     1. From `session_id` prefix (`conv_` -> `"conversation"`, `clin_` -> `"clinical_pre_consultation"`).
     2. From request body `workspace_type` or `workspace`.
     3. Fallback to `"conversation"`.
   - Pass `workspace=resolved_workspace` to `process_query()`.
3. **Agent State & Decision Routing (`agents/agent_decision.py`)**:
   - Update `AgentState` and `init_agent_state()` to accept and store `workspace: Optional[str]` (auto-derived from `session_id` prefix if not provided).
   - In `DECISION_SYSTEM_PROMPT`: Add workspace boundary rules clarifying that in Medical AI Conversation, `CLINICAL_PRE_CONSULTATION_AGENT` is prohibited and symptom discussions belong to `CONVERSATION_AGENT`.
   - In `route_to_agent()`:
     - Include `Active workspace: {workspace_display}` in `decision_input`.
     - Execute `decision_chain.invoke()` once.
     - Enforce the workspace boundary:
       - If `workspace == "conversation"` and `decision["agent"] == "CLINICAL_PRE_CONSULTATION_AGENT"`:
         Redirect to `CONVERSATION_AGENT` (the legitimate conversational agent for symptom guidance).
       - If `workspace == "clinical_pre_consultation"`:
         Ensure the target remains `CLINICAL_PRE_CONSULTATION_AGENT`.

---

## 7. Potential Regression Risks & Mitigation

| Potential Risk | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **Breaking RAG / Web Search routing in Conversation** | High | The LLM decision chain continues to evaluate every message. Queries regarding brain tumors or COVID-19 chest x-rays still route to `RAG_AGENT`; recent medical queries still route to `WEB_SEARCH_PROCESSOR_AGENT`. Only `CLINICAL_PRE_CONSULTATION_AGENT` is restricted from the conversation workspace. |
| **Breaking Clinical Pre-Consultation adaptive intake** | High | In `clinical_pre_consultation` workspace, `CLINICAL_PRE_CONSULTATION_AGENT` remains the dedicated agent; all adaptive questioning, symptom extraction, emergency red flags, and report generation workflows remain 100% intact. |
| **API Backward Compatibility** | Low | New `workspace` fields on `QueryRequest` and `process_query()` are completely optional with default fallbacks to `session_id` prefixes and default `"conversation"`. |
| **Image Analysis Interruption** | None | Image analysis continues to use `/upload-image` and `/analyze-image` with `img_` sessions, completely isolated from `/chat`. |

---

## 8. Proposed Verification Test Cases

- **Test 1: Medical Conversation — Fever Query**: Verify `"I have fever from 2 days and I also have runny nose and feeling body temperature high."` invokes `CONVERSATION_AGENT`, produces general non-diagnostic medical information, and never invokes `CLINICAL_PRE_CONSULTATION_AGENT`.
- **Test 2: Medical Conversation — Headache Query**: Verify `"I have had a headache since yesterday. What could be the common reasons?"` produces a general informational response without starting clinical intake.
- **Test 3: Medical Conversation — Direct Clinical Question**: Verify `"What information should a doctor know before evaluating a patient with fever?"` answers informationally.
- **Test 4: Clinical Pre-Consultation — Fever Intake**: Verify `"I have fever from 2 days and runny nose."` in `clinical_pre_consultation` workspace invokes `CLINICAL_PRE_CONSULTATION_AGENT` and asks an adaptive follow-up question.
- **Test 5: Clinical Pre-Consultation — Adaptive Follow-up**: Verify multiple turns collect information and generate report without repeating questions.
- **Test 6: Clinical Red Flag**: Verify severe chest pain or emergency symptom triggers immediate emergency escalation.
- **Test 7: Workspace Switching**: Verify switching between `/conversation` and `/clinical-pre-consultation` keeps state and history strictly isolated.
- **Test 8: Session Restoration**: Verify loading saved sessions restores the correct workspace and messages.
- **Test 9: Full Regression Suite**: Verify RAG, Web Search, SQLite storage, and frontend build succeed.
