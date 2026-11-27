import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class FollowupQuestionGenerator:
    """
    Generates adaptive, patient-specific follow-up questions based on collected clinical context.
    
    Responsibilities:
    - Truly adaptive questioning via Groq LLM (qwen/qwen3.8-27b) or complaint-specific clinical fallback
    - Evaluates complete clinical context (chief complaint, symptom details, progression, history)
    - Generates one clear, patient-friendly question at a time
    - Strictly avoids repeating previously asked or answered questions
    - Evaluates when sufficient clinical information is collected for doctor review
    - Supports dynamic early completion and emergency safety escalations
    """
    
    # Priority weighting for clinical areas
    QUESTION_CATEGORIES = {
        "chief_complaint": 1,
        "symptom_details": 2,
        "associated_symptoms": 3,
        "medical_history": 4,
        "medications": 5,
        "allergies": 6,
        "vitals": 7,
        "lifestyle": 8,
        "demographics": 9,
    }
    
    def __init__(self, config, llm=None):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing FollowupQuestionGenerator")
        
        if hasattr(config, 'clinical_consultation'):
            self.clinical_config = config.clinical_consultation
        else:
            self.clinical_config = config
        
        if llm is not None:
            self.llm = llm
        else:
            if hasattr(self.clinical_config, 'llm'):
                self.llm = self.clinical_config.llm
            else:
                self.llm = None
        
        if hasattr(self.clinical_config, 'max_followup_questions'):
            self.max_followup_questions = self.clinical_config.max_followup_questions
        else:
            self.max_followup_questions = 7
        
        # Track previously asked questions: list of dicts {"question": str, "category": str, "turn": int}
        self.asked_questions = {}  # category -> list of asked questions
        self._asked_question_list: List[str] = []
        
        self.logger.info(f"FollowupQuestionGenerator initialized successfully (max_questions={self.max_followup_questions})")
    
    def get_all_asked_questions(self) -> List[str]:
        """Return a flat list of all questions asked so far."""
        return list(self._asked_question_list)
    
    def reset_asked_questions(self) -> None:
        """Reset the tracking of previously asked questions."""
        self.asked_questions = {}
        self._asked_question_list = []
        self.logger.info("Cleared asked questions history")
    
    def get_next_question(
        self,
        patient_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        questions_asked_count: int = 0
    ) -> Dict[str, Any]:
        """
        Generate the next clinically relevant, non-repetitive follow-up question.
        """
        if not patient_data or not isinstance(patient_data, dict):
            raise ValueError("patient_data must be a non-empty dictionary")
        
        self.logger.info(f"Generating next question (asked: {questions_asked_count}/{self.max_followup_questions})")
        
        # Operational safeguard: question limit reached
        if questions_asked_count >= self.max_followup_questions:
            self.logger.info(f"Maximum questions reached ({self.max_followup_questions})")
            return {
                "question": None,
                "category": "limit_reached",
                "priority": 999,
                "reason": "Maximum follow-up questions limit reached",
                "needs_more_questions": False,
                "confidence": 1.0,
                "sufficient_info": True
            }
        
        # Primary: Adaptive question generation via LLM if available
        if self.llm is not None:
            try:
                llm_result = self._generate_adaptive_question_llm(
                    patient_data=patient_data,
                    conversation_history=conversation_history,
                    questions_asked_count=questions_asked_count
                )
                if llm_result:
                    return llm_result
            except Exception as e:
                self.logger.warning(f"LLM question generation failed ({e}). Activating complaint-specific clinical fallback.")
        
        # Secondary / Fallback: Complaint-specific adaptive clinical rule engine
        return self._generate_complaint_specific_fallback(
            patient_data=patient_data,
            conversation_history=conversation_history,
            questions_asked_count=questions_asked_count
        )
    
    def _generate_adaptive_question_llm(
        self,
        patient_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        questions_asked_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate next question using Groq LLM (qwen/qwen3.8-27b) with structured clinical prompt and JSON output.
        """
        prompt = self._build_clinical_llm_prompt(patient_data, conversation_history)
        
        # Invoke LLM
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON
        parsed = self._extract_json_from_response(content)
        if not parsed:
            self.logger.warning("Could not parse JSON from LLM response")
            return None
        
        decision = parsed.get("decision", "ASK_QUESTION")
        reason = parsed.get("reason", "Gathering relevant clinical context")
        target_area = parsed.get("target_area", "symptom_details")
        question_text = parsed.get("question")
        is_sufficient = parsed.get("is_sufficient_for_report", False)
        
        # Check for safety escalation decision first
        if decision == "ESCALATE_SAFETY":
            return {
                "question": None,
                "category": "emergency_escalation",
                "priority": 0,
                "reason": reason,
                "needs_more_questions": False,
                "confidence": 1.0,
                "sufficient_info": False,
                "should_escalate": True,
                "source": "groq"
            }
        
        # Check for report generation decision
        if decision == "GENERATE_REPORT" or is_sufficient or not question_text:
            return {
                "question": None,
                "category": "sufficient_info",
                "priority": 999,
                "reason": reason,
                "needs_more_questions": False,
                "confidence": 0.95,
                "sufficient_info": True,
                "source": "groq"
            }
        
        # Anti-repetition validation: ensure question has not already been asked
        clean_q = question_text.strip()
        if self._is_question_repeated(clean_q):
            self.logger.warning(f"LLM generated a repeated question: '{clean_q}'. Falling back to alternate clinical area.")
            return None
        
        # Record asked question
        self._record_asked_question(target_area, clean_q, patient_data)
        
        return {
            "question": clean_q,
            "category": target_area,
            "priority": self.QUESTION_CATEGORIES.get(target_area, 3),
            "reason": reason,
            "needs_more_questions": True,
            "confidence": 0.92,
            "sufficient_info": False,
            "extracted_updates": parsed.get("extracted_updates"),
            "source": "groq"
        }
    
    def _build_clinical_llm_prompt(
        self,
        patient_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construct a structured clinical prompt for Groq LLM."""
        cc = patient_data.get("chief_complaint", {})
        symp = patient_data.get("symptoms", {})
        details = patient_data.get("symptom_details", {})
        mh = patient_data.get("medical_history", {})
        meds = patient_data.get("medications", {})
        allergies = patient_data.get("allergies", {})
        vitals = patient_data.get("vitals", {})
        contradictions = patient_data.get("contradictions", [])
        clarifications = patient_data.get("unresolved_clarifications", [])
        
        already_asked = "\n".join([f"- {q}" for q in self._asked_question_list]) if self._asked_question_list else "None yet."
        
        history_formatted = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-10:]:
                role = "Patient" if msg.get("role") in ["user", "patient"] else "Intake Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_formatted = "\n".join(history_lines)
        else:
            history_formatted = "Initial conversation turn."
        
        contradiction_notes = ""
        if contradictions:
            c_lines = [f"- {c.get('note', str(c))}" for c in contradictions]
            contradiction_notes = "\nNOTED CONTRADICTIONS / CORRECTIONS:\n" + "\n".join(c_lines)
        
        clarification_notes = ""
        if clarifications:
            cl_lines = [f"- {c.get('note', str(c))}" for c in clarifications]
            clarification_notes = "\nUNRESOLVED AMBIGUITIES NEEDING CLARIFICATION:\n" + "\n".join(cl_lines)
        
        return f"""You are an expert Clinical Intake Assistant conducting an adaptive pre-consultation interview for a physician.
Your role is to collect factual, clinically relevant information about the patient's complaint.
DO NOT diagnose or prescribe treatment. Collect and organize data for doctor review.

CURRENT PATIENT INFORMATION:
- Chief Complaint: {cc.get('reason', 'Not yet clear')} (Duration: {cc.get('duration', 'Unknown')})
- Reported Symptoms: {', '.join(symp.get('list', [])) if symp.get('list') else 'None'}
- Symptom Details: Location={details.get('location', 'Unknown')}, Severity={details.get('severity', 'Unknown')}, Character={details.get('character', 'Unknown')}, Onset={details.get('onset', 'Unknown')}, Timing={details.get('timing', 'Unknown')}, Frequency={details.get('frequency', 'Unknown')}, Progression={details.get('progression', 'Unknown')}
- Explicitly Denied (Negative) Symptoms: {', '.join(details.get('negative_symptoms', [])) if details.get('negative_symptoms') else 'None'}
- Historical (Past/Resolved) Symptoms: {', '.join(details.get('historical_symptoms', [])) if details.get('historical_symptoms') else 'None'}
- Uncertain / Possible Symptoms: {', '.join(details.get('uncertain_symptoms', [])) if details.get('uncertain_symptoms') else 'None'}
- Medical History: Conditions={', '.join(mh.get('conditions', [])) if mh.get('conditions') else 'None reported'}; Surgeries={', '.join(mh.get('surgeries', [])) if mh.get('surgeries') else 'None reported'}; Family History={', '.join(mh.get('family_history', [])) if mh.get('family_history') else 'None reported'}; Status={mh.get('status', 'unknown')}
- Current Medications: {', '.join(meds.get('list', [])) if meds.get('list') else 'None reported'} (Status: {meds.get('status', 'unknown')})
- Allergies: {', '.join(allergies.get('drug_allergies', [])) if allergies.get('drug_allergies') else 'None reported'} (Status: {allergies.get('status', 'unknown')})
- Measured Vitals: BP={vitals.get('blood_pressure', 'N/A')}, Temp={vitals.get('temperature', 'N/A')}, HR={vitals.get('heart_rate', 'N/A')}
{contradiction_notes}
{clarification_notes}

RECENT CONVERSATION HISTORY:
{history_formatted}

PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT ANY OF THESE):
{already_asked}

CLINICAL GUIDELINES:
1. Conduct a natural, empathetic, intelligent patient interview like an experienced clinician.
2. Naturally understand Hindi, English, and Hinglish patient answers without language barriers.
3. Adapt directly to the patient's specific complaint and previous answers. Do not follow a rigid checklist.
4. If a correction was already resolved (e.g. timing changed from morning to evening), do NOT ask to clarify it again.
5. Ask relevant follow-up questions tailored to the complaint (e.g. onset, frequency, timing, progression, triggers, severity, relevant associated symptoms) and perform relevant complaint-specific safety screening (e.g. chest pain, shortness of breath, vision changes, neck stiffness, severe sudden onset).
6. When clinically appropriate, inquire about relevant medications, allergies, or past history.
7. Ask EXACTLY ONE clear, concise question at a time.
8. If the patient exhibits emergency red flags (e.g. active crushing chest pain, acute respiratory distress, severe hemorrhage, stroke signs), output decision "ESCALATE_SAFETY".
9. If sufficient clinical context has been gathered for a physician handoff (or after 4-5 focused turns), output decision "GENERATE_REPORT".
10. NEVER invent or assume a duration or any clinical fact. If the patient has not explicitly stated a duration, 'duration' in extracted_updates MUST be null. Only extract what the patient explicitly stated.
11. DO NOT ask for symptom severity if Severity is already provided in CURRENT PATIENT INFORMATION or if a 1-10 severity rating was already given (e.g. '5/10'). Retain symptom-specific severity and ask about other uncollected areas instead.
12. If the patient denied or said 'no' to a symptom (e.g. 'no nausea', 'does not radiate to neck'), put it in 'negative_symptoms', NEVER in 'symptoms'.

Respond ONLY with a valid JSON object matching this schema:
{{
  "decision": "ASK_QUESTION" | "CLARIFY_ANSWER" | "ESCALATE_SAFETY" | "GENERATE_REPORT",
  "reason": "Brief clinical rationale",
  "target_area": "symptom_details" | "associated_symptoms" | "medical_history" | "medications" | "allergies" | "vitals",
  "question": "The single focused question to ask the patient (or null if GENERATE_REPORT or ESCALATE_SAFETY)",
  "is_sufficient_for_report": false,
  "extracted_updates": {{
    "chief_complaint": {{ "reason": null, "duration": null }},
    "symptoms": [],
    "negative_symptoms": [],
    "symptom_details": {{ "location": null, "severity": null, "character": null, "onset": null, "progression": null }},
    "medications": [],
    "allergies": {{ "drug_allergies": [], "food_allergies": [] }},
    "medical_history": {{ "conditions": [], "surgeries": [] }}
  }}
}}"""

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from model output."""
        if not text:
            return None
        clean = text.strip()
        # Strip markdown fences
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)
        
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: search for first { and last }
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(clean[start:end+1])
                except json.JSONDecodeError:
                    pass
        return None

    def _generate_complaint_specific_fallback(
        self,
        patient_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        questions_asked_count: int = 0
    ) -> Dict[str, Any]:
        """
        Intelligent complaint-specific adaptive clinical fallback.
        Ensures diverse, tailored questions for different conditions (headache, fever, abdominal, respiratory, etc.)
        and strictly prevents repeating any already-asked question.
        """
        cc = patient_data.get("chief_complaint", {})
        cc_text = (cc.get("reason") or "").lower()
        symp_list = [s.lower() for s in patient_data.get("symptoms", {}).get("list", [])]
        details = patient_data.get("symptom_details", {})
        all_symptoms_text = " ".join(symp_list) + " " + cc_text
        
        # Build candidate questions based on the clinical profile
        candidates: List[Tuple[str, str, str]] = []  # (category, reason, question_text)
        
        # 0. Ambiguities & Contradictions Clarification (Highest Priority)
        clarifications = patient_data.get("unresolved_clarifications", [])
        ambig_sev = next((c for c in clarifications if c.get("category") == "ambiguous_severity"), None)
        if ambig_sev:
            cand_sev = ambig_sev.get("candidate_severity", "changed")
            symps_str = " or ".join(ambig_sev.get("symptoms", []))
            candidates.append((
                "symptom_details",
                "Clarify which symptom the severity update applies to",
                f"You noted that the severity is {cand_sev}. To ensure the doctor has the exact details, could you clarify which symptom has become {cand_sev} ({symps_str})?"
            ))
        
        contradictions = patient_data.get("contradictions", [])
        if contradictions:
            first_c = contradictions[0]
            if first_c.get("category") == "symptom_denied":
                candidates.append((
                    "symptom_details",
                    "Clarify contradiction regarding previously reported symptom",
                    f"To make sure our intake records for the doctor are completely accurate: earlier you mentioned having {first_c.get('item')}, but now it was noted as absent. Could you clarify if you are currently experiencing any {first_c.get('item')}?"
                ))
            elif first_c.get("category") == "severity_change":
                candidates.append((
                    "symptom_details",
                    "Clarify change in symptom severity",
                    f"To ensure the doctor has the right details, you noted that the severity changed from {first_c.get('previous')} to {first_c.get('new')}. Could you confirm how severe it currently feels on a 1-10 scale?"
                ))
        
        # Historical chest pain safety check
        historical_symps = details.get("historical_symptoms", [])
        if any("chest pain" in h or "chest discomfort" in h for h in historical_symps):
            candidates.append((
                "symptom_details",
                "Verify full resolution of prior chest pain",
                "You mentioned having chest pain previously that resolved. Has that chest pain remained completely gone, or have you noticed any recurrence, tightness, or shortness of breath?"
            ))
        
        # A. HEADACHE PROFILE
        if "headache" in all_symptoms_text or "migraine" in all_symptoms_text:
            loc = details.get("location")
            char = details.get("character")
            sev = details.get("severity")
            meds_list = patient_data.get("medications", {}).get("list", [])
            
            if not loc:
                candidates.append((
                    "symptom_details",
                    "Determine headache location to help differentiate tension, migraine, or sinus etiology",
                    "Where exactly is the headache located? For example, is it on one side of your head, across your forehead, or at the back of your head?"
                ))
            
            neg_list = [n.lower() for n in details.get("negative_symptoms", [])]
            explored = set(symp_list + [n.replace("no ", "").strip() for n in neg_list])
            has_neck_rad_explored = any("neck" in n or "shoulder" in n for n in neg_list)
            has_screen_explored = any("screen" in n for n in neg_list)
            severity_already_addressed = bool(sev or any("scale of 1 to 10" in q.lower() or "how severe" in q.lower() for q in self._asked_question_list))

            # If bilateral / dull / tension (tension headache signs)
            if (loc and any(b in loc for b in ["both", "bilateral"])) or char in ["dull", "aching"] or "monitor" in all_symptoms_text or "screen" in all_symptoms_text or ("both" in all_symptoms_text and "temple" in all_symptoms_text):
                if not has_neck_rad_explored and not has_screen_explored:
                    if meds_list or "paracetamol" in all_symptoms_text:
                        candidates.append((
                            "symptom_details",
                            "Evaluate response to medication for tension headache",
                            "Did the medication you took help ease the dull headache, and are you noticing any tightness or stiffness in your neck or shoulders?"
                        ))
                    else:
                        candidates.append((
                            "symptom_details",
                            "Assess tension triggers and neck involvement",
                            "Does the dull ache spread down into your neck or shoulders, and is it worse after periods of screen time or stress?"
                        ))
            
            # If unilateral / throbbing (migraine signs)
            elif (loc and any(u in loc for u in ["right", "left", "one side"])) or char in ["throbbing", "pounding"]:
                if not any("nausea" in s or "light" in s or "visual" in s or "vision" in s for s in explored):
                    candidates.append((
                        "associated_symptoms",
                        "Assess associated migraine symptoms (nausea, photophobia, aura)",
                        "Since you're experiencing throbbing pain on one side, have you noticed any nausea, sensitivity to bright light or sound, or visual changes like spots or flashes?"
                    ))
            
            if not severity_already_addressed and not char:
                candidates.append((
                    "symptom_details",
                    "Assess headache severity and qualitative character",
                    "On a scale of 1 to 10, how severe is the headache, and does it feel throbbing, sharp, or like dull pressure?"
                ))
            elif not severity_already_addressed:
                candidates.append((
                    "symptom_details",
                    "Assess headache severity on numerical rating scale",
                    "On a scale of 1 to 10, how severe would you rate the pain?"
                ))
            elif not char:
                candidates.append((
                    "symptom_details",
                    "Assess qualitative character of headache pain",
                    "Does the headache feel throbbing, sharp, or like dull pressure?"
                ))
            
            if not any("nausea" in s or "light" in s or "visual" in s or "vision" in s for s in explored):
                candidates.append((
                    "associated_symptoms",
                    "Assess associated neurological symptoms",
                    "Have you experienced any nausea, sensitivity to bright lights or sounds, or changes in your vision?"
                ))
        
        # B. FEVER PROFILE
        elif "fever" in all_symptoms_text or "temperature" in all_symptoms_text or "chill" in all_symptoms_text or "bukhar" in all_symptoms_text:
            vitals = patient_data.get("vitals", {})
            temp_measured = vitals.get("temperature") or vitals.get("temperature_status") in ["not_measured", "provided", "not_provided"] or any(re.search(r'\b10\d(?:\.\d+)?\b', s) for s in symp_list + [all_symptoms_text])
            if not temp_measured:
                candidates.append((
                    "vitals",
                    "Collect measured body temperature",
                    "Have you measured your temperature with a thermometer? If so, what was the highest reading?"
                ))
            
            # Check for associated symptoms if not all explored
            has_chills = any("chill" in s for s in symp_list)
            has_throat = any("throat" in s for s in symp_list)
            has_body_pain = any("body pain" in s or "ache" in s for s in symp_list)
            has_headache = any("headache" in s for s in symp_list)
            
            if not has_body_pain and not has_headache and not has_chills and not has_throat:
                candidates.append((
                    "associated_symptoms",
                    "Check for systemic infectious signs",
                    "Along with the fever, are you experiencing body aches, a headache, chills, or a sore throat?"
                ))
            elif not has_chills:
                candidates.append((
                    "associated_symptoms",
                    "Check for chills and shivering",
                    "Have you been experiencing any chills, shivering, or feeling unusually cold?"
                ))
            elif not has_throat:
                candidates.append((
                    "associated_symptoms",
                    "Check for upper respiratory symptoms",
                    "Are you experiencing any sore throat, difficulty swallowing, or cough?"
                ))
            
            # Check fever medications taken & response
            meds_status = patient_data.get("medications", {}).get("status")
            if meds_status in ["unknown", "not_asked"]:
                candidates.append((
                    "medications",
                    "Check fever-reducing medications and response",
                    "Have you taken any fever-reducing medication, such as paracetamol or ibuprofen, and did it help lower your temperature?"
                ))
        
        # C. ABDOMINAL / STOMACH PAIN PROFILE
        elif "abdominal" in all_symptoms_text or "stomach" in all_symptoms_text or "belly" in all_symptoms_text:
            loc = details.get("location")
            if not loc:
                candidates.append((
                    "symptom_details",
                    "Localize abdominal pain quadrant",
                    "Where specifically in your abdomen or stomach is the pain located (for example, upper stomach, lower right side, or all over)?"
                ))
            if loc and "lower right" in loc:
                candidates.append((
                    "symptom_details",
                    "Check acute peritoneal and appendicitis warning signs",
                    "Does the lower right stomach pain get worse when you walk or cough, and have you had any fever or vomiting?"
                ))
            elif not any("nausea" in s or "vomiting" in s or "bowel" in s for s in symp_list):
                candidates.append((
                    "associated_symptoms",
                    "Assess gastrointestinal symptoms",
                    "Have you noticed any nausea, vomiting, loss of appetite, or changes in your bowel movements?"
                ))
        
        # D. COUGH / RESPIRATORY PROFILE
        elif "cough" in all_symptoms_text or "breathing" in all_symptoms_text or "throat" in all_symptoms_text:
            if not details.get("character"):
                candidates.append((
                    "symptom_details",
                    "Determine cough quality",
                    "Is your cough dry, or are you coughing up any mucus or phlegm?"
                ))
            if not any("shortness of breath" in s or "chest" in s for s in symp_list):
                candidates.append((
                    "associated_symptoms",
                    "Check for respiratory distress signs",
                    "Are you experiencing any shortness of breath, wheezing, or chest tightness when breathing?"
                ))
        
        # E. JOINT / MUSCLE PAIN PROFILE
        elif "joint" in all_symptoms_text or "muscle" in all_symptoms_text or "back pain" in all_symptoms_text:
            if not details.get("location"):
                candidates.append((
                    "symptom_details",
                    "Identify affected joints or muscle groups",
                    "Which specific joints or muscles are bothering you, and is there any noticeable swelling, redness, or warmth?"
                ))
            if not details.get("progression"):
                candidates.append((
                    "symptom_details",
                    "Assess pain progression and morning stiffness",
                    "Is the pain constant or does it come and go, and does moving around or resting make it feel better or worse?"
                ))
        
        # F. SKIN / RASH PROFILE
        elif "rash" in all_symptoms_text or "skin" in all_symptoms_text or "itching" in all_symptoms_text:
            if not details.get("location"):
                candidates.append((
                    "symptom_details",
                    "Identify rash distribution and characteristics",
                    "Where is the rash located on your body, and how does it appear (for example, red bumps, dry patches, or blisters)?"
                ))
            candidates.append((
                "associated_symptoms",
                "Assess itching and recent exposures",
                "Is the rash itchy or painful, and have you recently used any new skin products, soaps, or medications?"
            ))
        
        # G. GENERAL / SYSTEMIC ONSET (if not captured above)
        if not cc.get("duration") and not details.get("onset"):
            candidates.append((
                "symptom_details",
                "Clarify symptom onset and duration",
                "How long have you been experiencing these symptoms, and did they begin suddenly or develop gradually?"
            ))
        
        # H. GENERAL MEDICAL HISTORY, MEDICATIONS, ALLERGIES (once symptoms explored)
        mh_status = patient_data.get("medical_history", {}).get("status")
        if mh_status in ["unknown", "not_asked"]:
            candidates.append((
                "medical_history",
                "Check relevant past medical conditions",
                "Do you have any chronic medical conditions, past surgeries, or significant medical history we should be aware of?"
            ))
        
        meds_status = patient_data.get("medications", {}).get("status")
        if meds_status in ["unknown", "not_asked"]:
            candidates.append((
                "medications",
                "Identify current medications",
                "Are you currently taking any prescription medications, over-the-counter remedies, or supplements?"
            ))
        
        allergies_status = patient_data.get("allergies", {}).get("status")
        if allergies_status in ["unknown", "not_asked"]:
            candidates.append((
                "allergies",
                "Screen for drug or environmental allergies",
                "Do you have any known allergies to medications, foods, or environmental factors?"
            ))
        
        # Select the first candidate that HAS NOT been asked yet
        for category, reason, question_text in candidates:
            if not self._is_question_repeated(question_text):
                self._record_asked_question(category, question_text, patient_data)
                return {
                    "question": question_text,
                    "category": category,
                    "priority": self.QUESTION_CATEGORIES.get(category, 5),
                    "reason": reason,
                    "needs_more_questions": True,
                    "confidence": 0.88,
                    "sufficient_info": False,
                    "source": "clinical_fallback"
                }
        
        # If all candidates have been asked or sufficient info is collected, finish consultation
        self.logger.info("All key clinical areas explored. Concluding intake.")
        return {
            "question": None,
            "category": "sufficient_info",
            "priority": 999,
            "reason": "Comprehensive clinical information gathered for physician review",
            "needs_more_questions": False,
            "confidence": 0.95,
            "sufficient_info": True,
            "source": "clinical_fallback"
        }
    
    def _is_question_repeated(self, candidate_question: str) -> bool:
        """Check if candidate question is substantially similar to an already-asked question."""
        if not candidate_question:
            return True
        c_clean = candidate_question.lower().strip()
        c_words = set(re.findall(r'\b[a-z]{3,}\b', c_clean))
        
        for asked in self._asked_question_list:
            a_clean = asked.lower().strip()
            if c_clean == a_clean:
                return True
            a_words = set(re.findall(r'\b[a-z]{3,}\b', a_clean))
            if len(c_words) > 0 and len(a_words) > 0:
                intersection = c_words.intersection(a_words)
                overlap_ratio = len(intersection) / max(len(c_words), len(a_words))
                if overlap_ratio > 0.70:
                    return True
        return False
    
    def _record_asked_question(self, category: str, question: str, patient_data: Optional[Dict[str, Any]] = None) -> None:
        """Track asked question in category dictionary and list, and mark category as asked."""
        if category not in self.asked_questions:
            self.asked_questions[category] = []
        self.asked_questions[category].append(question)
        if question not in self._asked_question_list:
            self._asked_question_list.append(question)
        
        if patient_data and isinstance(patient_data, dict):
            if category == "vitals":
                if patient_data.get("vitals", {}).get("temperature_status") == "not_asked":
                    patient_data["vitals"]["temperature_status"] = "asked"
                if patient_data.get("vitals", {}).get("status") == "not_asked":
                    patient_data["vitals"]["status"] = "asked"
            elif category == "medical_history":
                if patient_data.get("medical_history", {}).get("conditions_status") == "not_asked":
                    patient_data["medical_history"]["conditions_status"] = "asked"
                if patient_data.get("medical_history", {}).get("status") == "not_asked":
                    patient_data["medical_history"]["status"] = "asked"
            elif category == "medications":
                if patient_data.get("medications", {}).get("status") == "not_asked":
                    patient_data["medications"]["status"] = "asked"
            elif category == "allergies":
                if patient_data.get("allergies", {}).get("status") == "not_asked":
                    patient_data["allergies"]["status"] = "asked"
            elif category == "lifestyle":
                if patient_data.get("lifestyle", {}).get("status") == "not_asked":
                    patient_data["lifestyle"]["status"] = "asked"


__all__ = ['FollowupQuestionGenerator']
