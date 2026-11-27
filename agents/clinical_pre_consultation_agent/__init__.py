import logging
from typing import Dict, List, Optional, Any
from .patient_info_collector import PatientInfoCollector
from .red_flag_detector import RedFlagDetector
from .followup_question_generator import FollowupQuestionGenerator
from .report_generator import PreConsultationReportGenerator


class ClinicalPreConsultationAgent:
    """
    Clinical Pre-Consultation Agent orchestrating an adaptive, patient-specific intake workflow:
    - PatientInfoCollector: Extracts and accumulates structured clinical context
    - RedFlagDetector: Identifies emergency warning signs on every turn
    - FollowupQuestionGenerator: Generates adaptive, context-aware follow-up questions
    - PreConsultationReportGenerator: Formats doctor-facing structured pre-consultation reports
    
    Each agent instance maintains independent state and conversational history for one session.
    """
    
    def __init__(self, config):
        """
        Initialize the Clinical Pre-Consultation Agent.
        """
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Clinical Pre-Consultation Agent")
        
        self.config = config
        
        if not hasattr(config, 'clinical_consultation'):
            raise AttributeError(
                "config.clinical_consultation not found. "
                "ClinicalConsultationConfig must be added to Config class."
            )
        
        self.clinical_config = config.clinical_consultation
        self.llm = self.clinical_config.llm
        
        self.context_limit = self.clinical_config.context_limit
        self.max_followup_questions = self.clinical_config.max_followup_questions
        self.emergency_keywords = self.clinical_config.emergency_keywords
        self.patient_db_path = self.clinical_config.patient_db_path
        
        # Initialize modules
        self.patient_info_collector = PatientInfoCollector(config=config, llm=self.llm)
        self.red_flag_detector = RedFlagDetector(config=config, llm=self.llm)
        self.followup_question_generator = FollowupQuestionGenerator(config, llm=self.llm)
        self.report_generator = PreConsultationReportGenerator(llm=self.llm)
        
        # Session State
        self._questions_asked_count = 0
        self._session_active = True
        self._emergency_detected = False
        self._latest_report = None
        self._latest_formatted_report = None
        self._awaiting_final_confirmation = False
        self._final_confirmation_asked = False
        self._final_confirmation_completed = False
        self.conversation_history: List[Dict[str, str]] = []
        
        self.logger.info(
            f"Clinical Pre-Consultation Agent initialized (max_questions={self.max_followup_questions})"
        )
    
    def process_patient_info(
        self, 
        user_input: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process patient information intake through an adaptive, orchestrated workflow.
        
        Execution Order:
        1. Input validation & conversation tracking
        2. Check for feedback/corrections to an existing completed report
        3. Extract patient information & update clinical context
        4. Detect emergency red flags (HIGHEST PRIORITY)
        5. Check for patient-controlled early completion requests
        6. Generate next adaptive follow-up question
        7. If sufficient info or limit reached: generate report
        """
        if not user_input or not isinstance(user_input, str):
            raise ValueError("user_input must be a non-empty string")
        
        trimmed_input = user_input.strip()
        if len(trimmed_input) == 0:
            raise ValueError("user_input cannot be empty or whitespace-only")
        
        # Maintain internal conversation history
        if conversation_history:
            for item in conversation_history:
                if isinstance(item, dict) and item not in self.conversation_history:
                    self.conversation_history.append(item)
        
        self.conversation_history.append({"role": "user", "content": trimmed_input})
        self.logger.info(f"Processing patient input: {trimmed_input[:100]}")

        # Case A: Patient Feedback / Correction to an already-generated report
        # Check BEFORE raw extraction so negated/removed items in corrections are never added as positives
        is_feedback_turn = (self._latest_report is not None)
        if is_feedback_turn:
            self.logger.info("Patient feedback received for existing report. Processing corrections.")
            self._session_active = False
            self._apply_patient_corrections(trimmed_input)
            patient_data = self.patient_info_collector.get_patient_data()
            from datetime import datetime
            patient_data.setdefault("correction_history", []).append({
                "timestamp": datetime.now().isoformat(),
                "input": trimmed_input
            })
            patient_data["patient_review_status"] = "corrected_pending_confirmation"
            red_flag_result = self.red_flag_detector.detect_emergencies(patient_data)
            
            updated_report = self.report_generator.generate_report(
                patient_data,
                red_flag_result,
                self.conversation_history
            )
            updated_formatted_report = self.report_generator.format_report_text(updated_report)
            self._latest_report = updated_report
            self._latest_formatted_report = updated_formatted_report
            
            return {
                "status": "completed",
                "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
                "message": "Report updated with patient corrections.",
                "patient_data": patient_data,
                "red_flag_result": red_flag_result,
                "next_question": None,
                "report": updated_report,
                "formatted_report": updated_formatted_report,
                "questions_asked_count": self._questions_asked_count,
                "sufficient_info": True,
                "requires_immediate_attention": bool(red_flag_result.get("should_escalate")),
                "session_active": False
            }
        
        # Find last assistant question for short contextual reply extraction ("No", "Yes", etc.)
        last_assistant_q = None
        for msg in reversed(self.conversation_history[:-1]):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_q = msg["content"]
                break

        # Extract patient information
        try:
            extracted = self.patient_info_collector.extract_from_patient_text(trimmed_input, last_question=last_assistant_q)
            patient_data = self.patient_info_collector.get_patient_data()
        except ValueError as e:
            self.logger.error(f"Error extracting patient information: {e}")
            raise
        
        # Emergency Detection (Evaluated on every turn before questions)
        try:
            red_flag_result = self.red_flag_detector.detect_emergencies(patient_data)
        except ValueError as e:
            self.logger.error(f"Error detecting emergencies: {e}")
            raise
        
        if red_flag_result.get("should_escalate"):
            self.logger.warning("EMERGENCY ESCALATION: Halting follow-up questions immediately")
            self._emergency_detected = True
            self._session_active = False
            
            report = self.report_generator.generate_report(
                patient_data, 
                red_flag_result, 
                self.conversation_history
            )
            formatted_report = self.report_generator.format_report_text(report)
            self._latest_report = report
            self._latest_formatted_report = formatted_report
            
            return {
                "status": "emergency_escalation",
                "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
                "message": "EMERGENCY: Patient requires immediate medical attention. Follow-up questions halted.",
                "patient_data": patient_data,
                "red_flag_result": red_flag_result,
                "next_question": None,
                "report": report,
                "formatted_report": formatted_report,
                "questions_asked_count": self._questions_asked_count,
                "sufficient_info": False,
                "requires_immediate_attention": True,
                "guidance": red_flag_result.get("guidance"),
                "session_active": False
            }
        
        # Case B0: Patient is responding to the Mandatory Final Confirmation Question
        if self._awaiting_final_confirmation:
            if self._is_final_confirmation_negative(trimmed_input):
                self.logger.info("Patient confirmed no additional details. Generating final pre-consultation report.")
                self._awaiting_final_confirmation = False
                self._final_confirmation_completed = True
                patient_data["final_confirmation_status"] = "confirmed"
                patient_data["patient_review_status"] = "confirmed"
                self._session_active = False

                report = self.report_generator.generate_report(
                    patient_data,
                    red_flag_result,
                    self.conversation_history
                )
                formatted_report = self.report_generator.format_report_text(report)
                self._latest_report = report
                self._latest_formatted_report = formatted_report

                return {
                    "status": "completed",
                    "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
                    "message": "Pre-consultation intake completed after final patient confirmation.",
                    "patient_data": patient_data,
                    "red_flag_result": red_flag_result,
                    "next_question": None,
                    "report": report,
                    "formatted_report": formatted_report,
                    "questions_asked_count": self._questions_asked_count,
                    "sufficient_info": True,
                    "requires_immediate_attention": False,
                    "session_active": False
                }
            else:
                # Case B/C: Patient added new symptoms or corrected previous information -> reopen interview
                self.logger.info("Patient provided additional info or correction at final confirmation; reopening interview.")
                self._awaiting_final_confirmation = False
                self._final_confirmation_asked = False
                self._final_confirmation_completed = False
                patient_data["final_confirmation_status"] = "not_asked"
                # Grant additional follow-up budget for the newly provided info
                if self._questions_asked_count >= self.followup_question_generator.max_followup_questions:
                    self.followup_question_generator.max_followup_questions = self._questions_asked_count + 2

        # Case B: Patient-Controlled Early Completion Request ("that's all", "done", "finish", "give me report")
        # Rule (Case D): Do NOT immediately generate the final report before asking mandatory final confirmation.
        if self.patient_info_collector.detect_early_completion_request(trimmed_input):
            if not self._final_confirmation_asked and not self._final_confirmation_completed:
                return self._ask_final_confirmation(patient_data, red_flag_result)
            report = self.report_generator.generate_report(
                patient_data,
                red_flag_result,
                self.conversation_history
            )
            formatted_report = self.report_generator.format_report_text(report)
            self._latest_report = report
            self._latest_formatted_report = formatted_report
            self._session_active = False
            
            return {
                "status": "completed",
                "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
                "message": "Pre-consultation intake completed upon patient request.",
                "patient_data": patient_data,
                "red_flag_result": red_flag_result,
                "next_question": None,
                "report": report,
                "formatted_report": formatted_report,
                "questions_asked_count": self._questions_asked_count,
                "sufficient_info": True,
                "requires_immediate_attention": False,
                "session_active": False
            }
        
        # Case C: Adaptive Follow-up Question Generation
        next_question = None
        sufficient_info = False
        
        try:
            question_result = self.followup_question_generator.get_next_question(
                patient_data=patient_data,
                conversation_history=self.conversation_history,
                questions_asked_count=self._questions_asked_count
            )
            
            # Merge any structured updates extracted by LLM
            if question_result.get("extracted_updates"):
                self.patient_info_collector.merge_structured_updates(question_result["extracted_updates"])
                patient_data = self.patient_info_collector.get_patient_data()
            
            if question_result.get("question") is not None:
                self._questions_asked_count += 1
                next_question = question_result
                sufficient_info = False
                self.conversation_history.append({"role": "assistant", "content": question_result["question"]})
                self.logger.info(f"Adaptive question {self._questions_asked_count} generated: {question_result['category']}")
            else:
                sufficient_info = question_result.get("sufficient_info", False)
                self.logger.info(f"Interview conclusion triggered: sufficient_info={sufficient_info}")
        except Exception as e:
            self.logger.error(f"Error generating follow-up question: {e}")
            sufficient_info = True  # Fallback to report generation rather than stalling
        
        # Case D: Conclusion -> Mandatory Final Patient Confirmation BEFORE Report Generation
        report = None
        formatted_report = None
        final_status = "active"
        
        if sufficient_info or next_question is None:
            if not self._final_confirmation_asked and not self._final_confirmation_completed:
                return self._ask_final_confirmation(patient_data, red_flag_result)
            try:
                report = self.report_generator.generate_report(
                    patient_data, 
                    red_flag_result, 
                    self.conversation_history
                )
                formatted_report = self.report_generator.format_report_text(report)
                self._latest_report = report
                self._latest_formatted_report = formatted_report
                
                if sufficient_info:
                    final_status = "completed"
                    self.logger.info("Consultation completed: sufficient clinical information collected")
                else:
                    final_status = "partial"
                    self.logger.info("Consultation partial: question limit reached")
                
                self._session_active = False
            except ValueError as e:
                self.logger.error(f"Error generating report: {e}")
                raise
        
        return {
            "status": final_status,
            "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
            "message": self._build_response_message(final_status, sufficient_info, next_question),
            "patient_data": patient_data,
            "red_flag_result": red_flag_result,
            "next_question": next_question,
            "report": report,
            "formatted_report": formatted_report,
            "questions_asked_count": self._questions_asked_count,
            "sufficient_info": sufficient_info,
            "requires_immediate_attention": False,
            "session_active": self._session_active
        }
    
    def _build_response_message(
        self, 
        status: str, 
        sufficient_info: bool, 
        next_question: Optional[Dict[str, Any]]
    ) -> str:
        if status == "emergency_escalation":
            return "EMERGENCY: Seeking immediate medical attention required"
        elif status == "completed":
            return f"Pre-consultation intake completed after {self._questions_asked_count} follow-up question(s)"
        elif status == "partial":
            return f"Intake concluded after {self._questions_asked_count} follow-up question(s). Report generated with collected information"
        elif status == "active":
            if next_question and next_question.get("question"):
                return f"Follow-up question {self._questions_asked_count} ready"
            return "Gathering patient information"
        return "Processing patient information"
    
    def reset_session(self) -> None:
        """Reset session state for a new patient."""
        self.logger.info("Resetting session state")
        self.patient_info_collector.clear_patient_data()
        self.followup_question_generator.reset_asked_questions()
        self._questions_asked_count = 0
        self._session_active = True
        self._emergency_detected = False
        self._latest_report = None
        self._latest_formatted_report = None
        self._awaiting_final_confirmation = False
        self._final_confirmation_asked = False
        self._final_confirmation_completed = False
        self.conversation_history = []

    @staticmethod
    def _is_final_confirmation_negative(text: str) -> bool:
        """Return True when patient confirms they have nothing else to add before report generation."""
        import re
        t = (text or "").strip().lower().rstrip(".!,?")
        if not t:
            return False
        # If the patient added "no, I also have ..." or "yes, ...", it is NOT a simple negative confirmation
        if any(w in t for w in ["also have", "forgot", "actually", "but i", "and i", "pain", "fever", "headache", "dizziness", "cough", "nausea", "vomiting", "allergy", "medication", "medicine", "taking"]):
            return False
        neg_exact = {
            "no", "nope", "nah", "nothing", "nothing else", "that's all", "thats all",
            "that is all", "no that's all", "no, that's all", "no, nothing else",
            "no that's everything", "no, that's everything", "that's everything",
            "that is everything", "none", "all good", "done", "finish", "generate report",
            "nahi", "kuch nahi", "bas", "bas itna hi", "aur kuch nahi"
        }
        if t in neg_exact:
            return True
        return bool(re.match(r'^(?:no|nope|nahi)(?:\s*,?\s*(?:nothing\s+else|that\'?s\s+(?:all|everything)|kuch\s+nahi|bas))?$', t))

    def _ask_final_confirmation(self, patient_data: Dict[str, Any], red_flag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the mandatory final patient confirmation question once before generating the report."""
        question_text = (
            "Is there anything else about your symptoms or health that you would like to tell me "
            "before I prepare your pre-consultation report?"
        )
        self._awaiting_final_confirmation = True
        self._final_confirmation_asked = True
        patient_data["final_confirmation_status"] = "awaiting_response"
        self.conversation_history.append({"role": "assistant", "content": question_text})
        self.logger.info("Mandatory final patient confirmation question asked")
        return {
            "status": "active",
            "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
            "message": "Awaiting final patient confirmation before report generation",
            "patient_data": patient_data,
            "red_flag_result": red_flag_result,
            "next_question": {
                "question": question_text,
                "category": "final_confirmation",
                "priority": 100,
                "reason": "Mandatory final confirmation before preparing clinical report",
                "needs_more_questions": True,
                "confidence": 1.0,
                "sufficient_info": False
            },
            "report": None,
            "formatted_report": None,
            "questions_asked_count": self._questions_asked_count,
            "sufficient_info": False,
            "requires_immediate_attention": False,
            "session_active": True
        }

    def conclude_incomplete_session(self) -> Dict[str, Any]:
        """Case E: End an unresponsive session cleanly and mark report as incomplete/unconfirmed."""
        patient_data = self.patient_info_collector.get_patient_data()
        patient_data["final_confirmation_status"] = "incomplete_unconfirmed"
        self._awaiting_final_confirmation = False
        self._session_active = False
        red_flag_result = self.red_flag_detector.detect_emergencies(patient_data)
        report = self.report_generator.generate_report(
            patient_data,
            red_flag_result,
            self.conversation_history
        )
        formatted_report = self.report_generator.format_report_text(report)
        self._latest_report = report
        self._latest_formatted_report = formatted_report
        return {
            "status": "partial",
            "agent": "CLINICAL_PRE_CONSULTATION_AGENT",
            "message": "Session ended without final patient confirmation. Report marked incomplete.",
            "patient_data": patient_data,
            "red_flag_result": red_flag_result,
            "next_question": None,
            "report": report,
            "formatted_report": formatted_report,
            "questions_asked_count": self._questions_asked_count,
            "sufficient_info": False,
            "requires_immediate_attention": bool(red_flag_result.get("should_escalate")),
            "session_active": False
        }

    def _apply_patient_corrections(self, user_input: str) -> None:
        """
        Apply patient corrections to existing clinical data using contextual LLM reasoning
        when available, supported by deterministic fallback.
        """
        import json
        import re
        p_data = self.patient_info_collector.get_patient_data()
        
        # 1. If LLM is available, use LLM to understand corrections in clinical context
        if self.llm is not None:
            try:
                cc = p_data.get("chief_complaint", {})
                symps = p_data.get("symptoms", {}).get("list", [])
                meds = p_data.get("medications", {}).get("list", [])
                negs = p_data.get("symptom_details", {}).get("negative_symptoms", [])
                mh = p_data.get("medical_history", {})
                
                prompt = f"""You are a clinical assistant reviewing a patient's correction or update to their pre-consultation report.
CURRENT PATIENT DATA:
- Chief Complaint: {cc.get('reason')} (Duration: {cc.get('duration')})
- Symptoms: {symps}
- Global Symptom Character: {p_data.get('symptom_details', {}).get('character')}
- Global Symptom Location: {p_data.get('symptom_details', {}).get('location')}
- Explicitly Denied Symptoms: {negs}
- Medications: {meds}
- Medical History Conditions: {mh.get('conditions', [])}
- Denied Medical Conditions: {mh.get('denied_conditions', [])}

PATIENT'S CORRECTION / UPDATE:
"{user_input}"

Identify ALL changes the patient is requesting:
1. Chief complaint reason or duration corrections.
2. Symptoms to remove/deny (put in remove_symptoms and negative_symptoms).
3. Symptoms to add (put in symptoms, and per-symptom timelines in per_symptom_details).
4. Incorrect symptom character or location to remove (e.g. "remove throat tightness" -> remove_character: ["tightness"], remove_location: ["throat"]).
5. Medical conditions added or explicitly denied (put denied conditions in medical_history.denied_conditions, NEVER in conditions).
6. Medications added/updated or removed (medications_to_remove).
7. Allergies, vitals (including temperature_status="not_measured"), or exposure history (recent_travel, sick_contacts).

Respond ONLY with a valid JSON object matching this schema:
{{
  "chief_complaint": {{"reason": null, "duration": null, "is_correction": false}},
  "symptoms": [],
  "remove_symptoms": [],
  "negative_symptoms": [],
  "per_symptom_details": {{}},
  "symptom_details": {{"timing": null, "frequency": null}},
  "remove_character": [],
  "remove_location": [],
  "vitals": {{"temperature": null, "temperature_status": null}},
  "medications": [],
  "medications_status": null,
  "medications_to_remove": [],
  "medical_history": {{"conditions": [], "denied_conditions": [], "surgeries": []}},
  "allergies": {{"drug_allergies": [], "food_allergies": [], "environmental_allergies": [], "status": null}},
  "lifestyle": {{"recent_travel": null, "sick_contacts": null}}
}}"""
                resp = self.llm.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                clean = content.strip()
                if clean.startswith("```"):
                    clean = re.sub(r'^```(?:json)?\s*', '', clean)
                    clean = re.sub(r'\s*```$', '', clean)
                start = clean.find("{")
                end = clean.rfind("}")
                if start != -1 and end != -1 and end > start:
                    clean = clean[start:end+1]
                parsed = json.loads(clean)
                if isinstance(parsed, dict):
                    has_meaningful = any(
                        bool(v) if not isinstance(v, dict) else any(bool(sv) for sv in v.values())
                        for v in parsed.values()
                    )
                    if has_meaningful:
                        self.patient_info_collector.merge_structured_updates(parsed)
                        self.logger.info(f"LLM-based corrections merged into patient data: {parsed}")
                        return
            except Exception as e:
                self.logger.warning(f"LLM correction extraction failed ({e}), falling back to deterministic extraction")
        
        # 2. Deterministic fallback / augmentation:
        text_lower = user_input.lower()

        # Check for character/location/timing/frequency corrections
        if re.search(r'\b(?:remove|no|not|without|never\s+had)\b.*?\btightness\b', text_lower):
            self.patient_info_collector.merge_structured_updates({"remove_character": ["tightness"]})
        if re.search(r'\b(?:remove|not)\b.*?\bthroat\b', text_lower):
            self.patient_info_collector.merge_structured_updates({"remove_location": ["throat"]})
        for t_pat, t_val in [
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?evenings?\b|\bshaam\s+(?:ko|mein)\b', "evening"),
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?mornings?\b|\bsubah\s+(?:ko|mein)\b', "morning"),
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?afternoons?\b|\bdopahar\s+(?:ko|mein)\b', "afternoon"),
            (r'\b(?:at\s+|every\s+|during\s+(?:the\s+)?)?nights?\b|\braat\s+(?:ko|mein)\b', "night"),
        ]:
            if re.search(t_pat, text_lower) and not re.search(r'\b(?:not|no|never)\s+(?:in\s+the\s+)?' + t_val + r'\b', text_lower):
                self.patient_info_collector.merge_structured_updates({"symptom_details": {"timing": t_val}})
                break

        # Check for denied chronic conditions (diabetes, asthma, heart disease, hypertension, etc.)
        mh_ext = self.patient_info_collector._extract_medical_history(user_input, text_lower)
        if mh_ext:
            mh_updates = {}
            if mh_ext.get("denied_conditions"):
                mh_updates["denied_conditions"] = mh_ext["denied_conditions"]
            if mh_ext.get("conditions_mentioned"):
                mh_updates["conditions"] = mh_ext["conditions_mentioned"]
            if mh_updates:
                self.patient_info_collector.merge_structured_updates({"medical_history": mh_updates})

        # Check for lifestyle / travel / sick contacts updates
        ls_ext = self.patient_info_collector._extract_lifestyle(user_input, text_lower)
        if ls_ext:
            self.patient_info_collector.merge_structured_updates({"lifestyle": ls_ext})
        
        # Check for symptom removal / negation across all current symptoms and chief complaint
        current_symps = list(p_data.get("symptoms", {}).get("list", []))
        cc_cur = p_data.get("chief_complaint", {}).get("reason")
        check_terms = list(set(current_symps + ([cc_cur] if cc_cur else [])))
        for s in check_terms:
            if not s:
                continue
            s_low = s.lower()
            if self.patient_info_collector.is_term_negated(s, text_lower) or any(
                p in text_lower for p in [
                    f"remove {s_low}", f"no {s_low}", f"don't have {s_low}", f"dont have {s_low}", 
                    f"not {s_low}", f"{s_low} nahi hai", f"{s_low} nahi", f"without {s_low}"
                ]
            ):
                self.patient_info_collector.merge_structured_updates({
                    "remove_symptoms": [s],
                    "negative_symptoms": [f"no {s}"]
                })
        
        # Check for newly added symptoms in correction
        added_symps = []
        for kw in ["fever", "headache", "migraine", "stomach pain", "abdominal pain", "nausea", "vomiting", "cough", "sore throat", "chills", "dizziness", "chest pain", "body pain"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                if not self.patient_info_collector.is_term_negated(kw, text_lower) and not any(
                    p in text_lower for p in [f"no {kw}", f"not {kw}", f"remove {kw}", f"dont have {kw}", f"{kw} nahi"]
                ):
                    added_symps.append(kw)
        if added_symps:
            self.patient_info_collector.merge_structured_updates({
                "symptoms": added_symps
            })

        # Check for explicit Chief Complaint change
        cc_match = re.search(r'\b(?:chief\s+complaint\s+is|complaint\s+is|actually\s+(?:it\'s|its|have)|change\s+chief\s+complaint\s+to|change\s+complaint\s+to|instead\s+of\s+[a-z\s]+?\s+(?:it\'s|is))\s+([a-z\s]+)', text_lower)
        if cc_match:
            cand_cc = cc_match.group(1).strip().split(".")[0].split(",")[0].strip()
            if cand_cc and len(cand_cc) > 2:
                self.patient_info_collector.merge_structured_updates({
                    "chief_complaint": {"reason": cand_cc.title(), "is_correction": True}
                })
        
        # Check for duration update (e.g. "3 days", "3 din", "since yesterday", "since this morning")
        dur_match = re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(days?|weeks?|months?|hours?|din|hafte|mahine)\b', text_lower)
        if dur_match:
            n_raw = dur_match.group(1)
            num_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            n_val = int(n_raw) if n_raw.isdigit() else num_words.get(n_raw, n_raw)
            unit = dur_match.group(2)
            unit_map = {"din": "days", "hafte": "weeks", "mahine": "months"}
            eng_unit = unit_map.get(unit, unit)
            if not eng_unit.endswith('s') and n_val != 1:
                eng_unit += 's'
            self.patient_info_collector.merge_structured_updates({
                "duration": f"{n_val} {eng_unit}"
            })
        elif "since this morning" in text_lower or re.search(r'\bsubah\s+se\b', text_lower):
            self.patient_info_collector.merge_structured_updates({
                "duration": "Since this morning"
            })
        elif "since yesterday" in text_lower or re.search(r'\bkal\s+se\b', text_lower):
            self.patient_info_collector.merge_structured_updates({
                "duration": "1 day (since yesterday)"
            })
        
        # Check for vitals update
        ext_vitals = self.patient_info_collector._extract_vitals(user_input, text_lower)
        if ext_vitals:
            p_curr = self.patient_info_collector.get_patient_data()
            for vk, vv in ext_vitals.items():
                p_curr["vitals"][vk] = vv
            p_curr["vitals"]["status"] = "provided"

        # Check for medications update (positive OR explicit denial)
        ext_meds = self.patient_info_collector._extract_medications(user_input, text_lower)
        if ext_meds:
            meds_list = ext_meds.get("medications_mentioned") or ext_meds.get("medications_found")
            if meds_list:
                self.patient_info_collector.merge_structured_updates({"medications": meds_list})
            elif ext_meds.get("explicitly_none"):
                self.patient_info_collector.merge_structured_updates({"medications_status": "patient_denied"})

        # Check for allergies update (positive OR explicit denial)
        ext_alg = self.patient_info_collector._extract_allergies(user_input, text_lower)
        if ext_alg:
            if ext_alg.get("explicitly_none"):
                self.patient_info_collector.merge_structured_updates({"allergies": {"status": "explicitly_denied"}})
            elif ext_alg.get("allergies_mentioned"):
                self.patient_info_collector.merge_structured_updates({
                    "allergies": {"drug_allergies": ext_alg["allergies_mentioned"], "status": "provided"}
                })

        self.logger.info("Patient corrections successfully processed and merged")


__all__ = ['ClinicalPreConsultationAgent']
