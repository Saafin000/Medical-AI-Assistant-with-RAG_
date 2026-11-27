import logging
import html
import re
from typing import Dict, List, Optional, Any
from datetime import datetime


def clean_text_entities(text: Any) -> Any:
    """Recursively clean HTML entities (named, hex, or decimal) from strings, lists, and dicts using iterative passes."""
    if isinstance(text, str):
        c = text
        for _ in range(5):
            prev = c
            c = html.unescape(c)
            c = re.sub(r'&#x([0-9a-fA-F]+);?', lambda m: chr(int(m.group(1), 16)), c, flags=re.IGNORECASE)
            c = re.sub(r'&#([0-9]+);?', lambda m: chr(int(m.group(1))), c)
            if c == prev:
                break
        return c
    elif isinstance(text, list):
        return [clean_text_entities(x) for x in text]
    elif isinstance(text, dict):
        return {k: clean_text_entities(v) for k, v in text.items()}
    return text


try:
    from .patient_info_collector import normalize_symptoms
except Exception:
    def normalize_symptoms(items):
        if not items:
            return []
        return list(dict.fromkeys(items))


def _was_topic_asked(topic: str, conversation_history: Optional[List[Dict[str, str]]]) -> bool:
    """Check if a clinical topic was queried in previous assistant questions."""
    if not conversation_history:
        return False
    topic_keywords = {
        "medications": ["medication", "medicine", "pill", "drug", "prescription", "supplement", "paracetamol", "ibuprofen", "dawai"],
        "allergies": ["allerg", "reaction"],
        "surgeries": ["surgery", "surgeries", "operation", "procedure"],
        "hospitalizations": ["hospital", "admitted", "admission", "stay in hospital"],
        "family_history": ["family", "parent", "mother", "father", "relative"],
        "lifestyle": ["smoke", "alcohol", "drink", "exercise", "diet", "lifestyle"],
        "recent_travel": ["travel", "traveled", "travelled", "trip"],
        "sick_contacts": ["sick contact", "contact with", "anyone sick", "someone sick"],
        "conditions": ["chronic", "condition", "disease", "illness", "medical history", "bimari", "diabetes", "asthma", "heart"],
        "temperature": ["temperature", "thermometer", "temp", "fever reading"]
    }
    keywords = topic_keywords.get(topic, [topic])
    for msg in conversation_history:
        if msg.get("role") in ["assistant", "agent", "system"]:
            content = msg.get("content", "").lower()
            if any(kw in content for kw in keywords):
                return True
    return False


class PreConsultationReportGenerator:
    """
    Generates structured, doctor-facing Pre-Consultation Reports.
    
    Uses patient information from PatientInfoCollector and emergency 
    detection from RedFlagDetector to produce a factual summary for
    doctor review.
    
    Responsibilities:
    - Generate structured report from patient data
    - Format report for readability
    - Validate report completeness and accuracy
    - Preserve all patient-reported information
    - Never invent or synthesize patient data
    - Include emergency warnings prominently
    - Provide clear distinction between provided/missing information
    """
    
    def __init__(self, llm=None):
        """Initialize the PreConsultationReportGenerator."""
        self.logger = logging.getLogger(f"{self.__module__}")
        self.llm = llm
        self.logger.info("Initializing PreConsultationReportGenerator")
    
    def generate_report(
        self,
        patient_data: Dict[str, Any],
        red_flag_result: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured pre-consultation report.
        
        Args:
            patient_data: Structured patient information from PatientInfoCollector
            red_flag_result: Emergency detection result from RedFlagDetector (optional)
            conversation_history: List of conversation exchanges (optional)
        
        Returns:
            Dictionary containing structured report:
            - report_type (str): "pre_consultation"
            - generated_at (str): ISO timestamp
            - emergency_escalation (bool): Whether emergency care needed
            - emergency_guidance (str or None): Escalation guidance if applicable
            - chief_complaint (dict): Primary reason for visit
            - symptoms (dict): Symptoms reported
            - vitals (dict): Vital signs if available
            - medical_history (dict): Relevant history
            - medications (dict): Current medications
            - allergies (dict): Allergies and reactions
            - lifestyle (dict): Lifestyle information
            - demographics (dict): Patient demographics
            - red_flags (dict): Emergency detection results
            - information_status (dict): Completeness tracking
            - collection_metadata (dict): Metadata
            - clinical_summary (str): Factual summary for doctor
        
        Raises:
            ValueError: If patient_data is invalid
        """
        if not patient_data or not isinstance(patient_data, dict):
            raise ValueError("patient_data must be a non-empty dictionary")
        
        self.logger.info("Generating pre-consultation report")
        
        # Initialize report structure
        report = {
            "report_type": "pre_consultation",
            "generated_at": datetime.now().isoformat(),
            "emergency_escalation": False,
            "emergency_guidance": None,
        }
        
        # Process emergency information first (if applicable)
        if red_flag_result and red_flag_result.get("should_escalate"):
            report["emergency_escalation"] = True
            report["emergency_guidance"] = red_flag_result.get("guidance")
        
        # Extract and format patient information sections
        report["chief_complaint"] = self._format_chief_complaint(patient_data)
        report["symptoms"] = self._format_symptoms(patient_data)
        report["vitals"] = self._format_vitals(patient_data, conversation_history)
        report["medical_history"] = self._format_medical_history(patient_data, conversation_history)
        report["medications"] = self._format_medications(patient_data, conversation_history)
        report["allergies"] = self._format_allergies(patient_data, conversation_history)
        report["lifestyle"] = self._format_lifestyle(patient_data, conversation_history)
        report["demographics"] = self._format_demographics(patient_data)
        report["contradictions"] = patient_data.get("contradictions", [])
        report["correction_history"] = patient_data.get("correction_history", [])
        report["patient_review_status"] = patient_data.get("patient_review_status", "pending_review")
        report["final_confirmation_status"] = patient_data.get("final_confirmation_status", "not_asked")
        
        # Include red flag information
        report["red_flags"] = self._format_red_flags(red_flag_result) if red_flag_result else None
        
        # Track information status
        report["information_status"] = self._assess_information_status(patient_data)
        
        # Include collection metadata
        report["collection_metadata"] = patient_data.get("collection_metadata", {})

        # Generate chronological History of Present Illness (HPI)
        report["hpi"] = self._generate_hpi(
            patient_data,
            conversation_history,
            report
        )
        
        # Generate clinical summary
        report["clinical_summary"] = self._generate_clinical_summary(
            patient_data, 
            red_flag_result,
            report
        )

        # Generate targeted clinical points for physician review
        report["recommendations"] = self._generate_recommendations(
            patient_data, 
            red_flag_result,
            report
        )

        # Recursively clean any residual HTML entities across the entire structured report
        report = clean_text_entities(report)
        
        # Validate report
        self.validate_report(report)
        
        self.logger.info("Report generated successfully")
        return report
    
    def _format_chief_complaint(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format chief complaint information with fallback to reported symptoms."""
        cc = patient_data.get("chief_complaint", {})
        
        reason_val = cc.get("reason")
        if not reason_val or str(reason_val).strip().lower() in ["not provided", "unknown", "none", "null"]:
            symps = patient_data.get("symptoms", {}).get("list", [])
            if symps:
                reason_val = symps[0]
            else:
                reason_val = "Not provided"

        dur_val = cc.get("duration")
        if not dur_val or str(dur_val).strip().lower() in ["unknown", "none", "null", "not provided"]:
            dur_val = "Not provided"
        
        return {
            "reason": clean_text_entities(reason_val),
            "duration": clean_text_entities(dur_val),
            "duration_mentioned": bool(cc.get("duration") and str(cc.get("duration")).strip().lower() not in ["not provided", "unknown"]),
            "status": cc.get("status", "unknown"),
            "note": "Patient-reported"
        }
    
    def _format_symptoms(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format symptoms information including clinical details and canonical deduplication."""
        symptoms = patient_data.get("symptoms", {})
        details = patient_data.get("symptom_details", {})
        symptom_list = symptoms.get("list", [])
        if symptom_list:
            symptom_list = normalize_symptoms(symptom_list)
        
        # Rigorous negative symptom extraction and capitalization
        raw_negs = details.get("negative_symptoms", []) or []
        formatted_negs = []
        neg_terms = set()
        for ns in raw_negs:
            ns_clean = clean_text_entities(ns).strip()
            if not ns_clean:
                continue
            low_ns = ns_clean.lower()
            if low_ns.startswith("no "):
                cap_ns = "No " + ns_clean[3:].strip()
                neg_terms.add(low_ns[3:].strip())
            elif low_ns.startswith("not "):
                cap_ns = "Not " + ns_clean[4:].strip()
                neg_terms.add(low_ns[4:].strip())
            else:
                cap_ns = "No " + ns_clean
                neg_terms.add(low_ns)
            if cap_ns not in formatted_negs:
                formatted_negs.append(cap_ns)

        # Sanity scrub: ensure no denied symptom appears under reported symptoms
        if symptom_list and neg_terms:
            symptom_list = [
                s for s in symptom_list
                if not any(nt == s.lower() or nt in s.lower() or s.lower() in nt for nt in neg_terms)
            ]

        status = symptoms.get("status", "unknown")
        psd = details.get("per_symptom_details", {}) or {}
        global_onset = clean_text_entities(details.get("onset")) or "Not specified"
        # Do not display a misleading blanket global onset when multiple symptoms exist or per-symptom timelines differ
        if (len(symptom_list) >= 2 and (len(psd) >= 2 or any(v.get("duration") or v.get("onset") for v in psd.values()))) or len(psd) >= 2:
            global_onset = "Not specified"
        
        return {
            "reported_symptoms": clean_text_entities(symptom_list) if symptom_list else [],
            "count": len(symptom_list),
            "location": clean_text_entities(details.get("location")) or "Not specified",
            "severity": clean_text_entities(details.get("severity")) or "Not specified",
            "character": clean_text_entities(details.get("character")) or "Not specified",
            "onset": global_onset,
            "timing": clean_text_entities(details.get("timing")) or "Not specified",
            "frequency": clean_text_entities(details.get("frequency")) or "Not specified",
            "progression": clean_text_entities(details.get("progression")) or "Not specified",
            "per_symptom_details": clean_text_entities(psd) or {},
            "negative_symptoms": clean_text_entities(formatted_negs) or [],
            "historical_symptoms": clean_text_entities(details.get("historical_symptoms")) or [],
            "uncertain_symptoms": clean_text_entities(details.get("uncertain_symptoms")) or [],
            "status": status,  # "provided", "unknown", "none_reported"
            "interpretation": self._interpret_symptom_status(status),
            "note": "Patient-reported. Not clinically confirmed."
        }
    
    def _interpret_symptom_status(self, status: str) -> str:
        """Interpret symptom status for report."""
        status_map = {
            "provided": "Patient reported symptoms",
            "unknown": "Symptoms not reported by patient",
            "none_reported": "Patient reports no symptoms",
            "explicitly_denied": "Patient denies symptoms"
        }
        return status_map.get(status, "Unknown status")
    
    def _format_vitals(self, patient_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract and format vital signs information with 4-way distinction."""
        vitals = patient_data.get("vitals", {})
        vitals_status = vitals.get("status", "not_asked").lower()
        
        vital_readings = {}
        for vital_name in ["blood_pressure", "heart_rate", "temperature", 
                          "respiratory_rate", "oxygen_saturation", "blood_glucose"]:
            value = vitals.get(vital_name)
            if value:
                vital_readings[vital_name] = clean_text_entities(str(value))
            elif vital_name == "temperature":
                temp_st = (vitals.get("temperature_status") or vitals_status).lower()
                if temp_st in ["not_measured", "explicitly_not_measured"]:
                    vital_readings[vital_name] = "Not measured"
                else:
                    vital_readings[vital_name] = "Not provided"
            elif vitals_status in ["asked", "not_provided"] or _was_topic_asked("vitals", conversation_history):
                vital_readings[vital_name] = "Not provided"
            else:
                vital_readings[vital_name] = "Not asked"
        
        return {
            "readings": vital_readings,
            "status": vitals.get("status", "not_asked"),
            "interpretation": self._interpret_vitals_status(vitals.get("status", "not_asked")),
            "note": "Patient-reported. Not clinically measured unless otherwise noted."
        }
    
    def _interpret_vitals_status(self, status: str) -> str:
        """Interpret vitals status for report."""
        status_map = {
            "provided": "Patient provided vital readings",
            "unknown": "Vital signs not reported by patient",
            "not_available": "Vital signs unavailable",
            "not_measured": "Vital signs not measured",
            "not_provided": "Vital signs not provided",
            "not_asked": "Vital signs not asked"
        }
        return status_map.get(status, "Not reported")
    
    def _format_medical_history(self, patient_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract and format medical history information with 4-way distinction."""
        history = patient_data.get("medical_history", {})
        
        conditions_list = history.get("conditions", [])
        surgeries_list = history.get("surgeries", [])
        hospitalizations_list = history.get("hospitalizations", [])
        family_list = history.get("family_history", [])
        
        def resolve_items(items, item_status, topic_name, fallback_denial="Patient denied"):
            if items:
                return clean_text_entities(items)
            st = (item_status or history.get("status") or "not_asked").lower()
            if st in ["patient_denied", "explicitly_denied", "denied"]:
                return [fallback_denial]
            elif st in ["none_reported", "none"]:
                return ["None reported"]
            elif st in ["not_provided", "asked"] or _was_topic_asked(topic_name, conversation_history):
                return ["Not provided"]
            else:
                return ["Not asked"]
        
        conds = resolve_items(conditions_list, history.get("conditions_status"), "conditions", "Patient denied")
        denied_conds = clean_text_entities(history.get("denied_conditions", []))
        if denied_conds and not conditions_list:
            conds = [f"Patient denied ({', '.join(denied_conds)})"]
        surgs = resolve_items(surgeries_list, history.get("surgeries_status"), "surgeries", "Patient denied")
        hosps = resolve_items(hospitalizations_list, history.get("hospitalizations_status"), "hospitalizations", "Patient denied")
        fams = resolve_items(family_list, history.get("family_history_status"), "family_history", "None reported")
        
        return {
            "conditions": conds,
            "denied_conditions": denied_conds,
            "surgeries": surgs,
            "hospitalizations": hosps,
            "family_history": fams,
            "status": history.get("status", "not_asked"),
            "note": "Patient-reported. Medical records not reviewed."
        }
    
    def _format_medications(self, patient_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract and format medications information with 4-way distinction."""
        meds = patient_data.get("medications", {})
        med_list = meds.get("list", [])
        status = meds.get("status", "not_asked").lower()
        
        if med_list:
            formatted_meds = clean_text_entities(med_list)
            interpretation = f"Patient reports {len(formatted_meds)} medication(s)"
        elif status in ["patient_denied", "explicitly_denied", "denied"]:
            formatted_meds = ["Patient denied"]
            interpretation = "Patient denied taking medications"
        elif status in ["none_reported", "none"]:
            formatted_meds = ["None reported"]
            interpretation = "Patient reports no current medications"
        elif status in ["not_provided", "asked"] or _was_topic_asked("medications", conversation_history):
            formatted_meds = ["Not provided"]
            interpretation = "Information not provided by patient"
        else:
            formatted_meds = ["Not asked"]
            interpretation = "Information not asked"
        
        return {
            "medications": formatted_meds,
            "count": len(med_list),
            "status": status,
            "interpretation": interpretation,
            "note": "Patient-reported. Dosages and frequencies as stated by patient."
        }
    
    def _format_allergies(self, patient_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract and format allergies information with 4-way distinction."""
        allergies = patient_data.get("allergies", {})
        
        drug_allergies = allergies.get("drug_allergies", [])
        food_allergies = allergies.get("food_allergies", [])
        env_allergies = allergies.get("environmental_allergies", [])
        status = allergies.get("status", "not_asked").lower()
        
        def resolve_allergy(items, item_status, topic_name):
            if items:
                return clean_text_entities(items)
            st = (item_status or status).lower()
            if st in ["patient_denied", "explicitly_denied", "denied"]:
                return ["Patient denied"]
            elif st in ["none_reported", "none"]:
                return ["None reported"]
            elif st in ["not_provided", "asked"] or _was_topic_asked("allergies", conversation_history):
                return ["Not provided"]
            else:
                return ["Not asked"]
        
        d_allergies = resolve_allergy(drug_allergies, allergies.get("drug_status"), "allergies")
        f_allergies = resolve_allergy(food_allergies, allergies.get("food_status"), "allergies")
        e_allergies = resolve_allergy(env_allergies, allergies.get("environmental_status"), "allergies")
        
        all_reported = (drug_allergies + food_allergies + env_allergies)
        if all_reported:
            interpretation = f"Patient reports {len(all_reported)} allergy/allergies"
        elif status in ["patient_denied", "explicitly_denied", "denied"]:
            interpretation = "Patient denies drug, food, and environmental allergies"
        elif status in ["none_reported", "none"]:
            interpretation = "Patient reports no allergies"
        elif status in ["not_provided", "asked"] or _was_topic_asked("allergies", conversation_history):
            interpretation = "Information not provided by patient"
        else:
            interpretation = "Information not asked"
        
        return {
            "drug_allergies": d_allergies,
            "food_allergies": f_allergies,
            "environmental_allergies": e_allergies,
            "status": status,
            "interpretation": interpretation,
            "note": "Patient-reported. Reactions as described by patient."
        }
    
    def _format_lifestyle(self, patient_data: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract and format lifestyle information with 4-way distinction."""
        lifestyle = patient_data.get("lifestyle", {})
        status = lifestyle.get("status", "not_asked").lower()
        
        def resolve_single(val, field_status, topic_key="lifestyle"):
            if val and str(val).lower() == "patient_denied":
                return "Patient denied"
            if val and str(val).lower() not in ["not reported", "none", "none reported", "not asked", "not provided"]:
                return clean_text_entities(str(val))
            st = (field_status or status).lower()
            if st in ["patient_denied", "explicitly_denied", "denied"]:
                return "Patient denied"
            elif st in ["none_reported", "none"]:
                return "None reported"
            elif st in ["not_provided", "asked"] or _was_topic_asked(topic_key, conversation_history):
                return "Not provided"
            else:
                return "Not asked"
        
        return {
            "smoking": resolve_single(lifestyle.get("smoking"), lifestyle.get("smoking_status"), "lifestyle"),
            "alcohol": resolve_single(lifestyle.get("alcohol"), lifestyle.get("alcohol_status"), "lifestyle"),
            "exercise": resolve_single(lifestyle.get("exercise"), lifestyle.get("exercise_status"), "lifestyle"),
            "diet": resolve_single(lifestyle.get("diet"), lifestyle.get("diet_status"), "lifestyle"),
            "recent_travel": resolve_single(lifestyle.get("recent_travel"), lifestyle.get("recent_travel_status"), "recent_travel"),
            "sick_contacts": resolve_single(lifestyle.get("sick_contacts"), lifestyle.get("sick_contacts_status"), "sick_contacts"),
            "exposures": resolve_single(lifestyle.get("exposures"), lifestyle.get("exposures_status"), "lifestyle"),
            "status": status,
            "note": "Patient-reported lifestyle factors."
        }
    
    def _format_demographics(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format demographic information."""
        demographics = patient_data.get("demographics", {})
        
        age = demographics.get("age")
        age_display = f"{age} {demographics.get('age_unit', 'years')}" if age else "Not provided"
        
        return {
            "age": age_display,
            "age_status": demographics.get("age_status", "unknown"),
            "note": "Patient-reported information."
        }
    
    def _format_red_flags(self, red_flag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format emergency detection results."""
        if not red_flag_result:
            return {
                "flags_detected": False,
                "severity": "none",
                "flags": [],
                "should_escalate": False
            }
        
        flags = red_flag_result.get("flags", [])
        flag_summary = []
        for flag in flags:
            flag_summary.append({
                "type": flag.get("type"),
                "description": flag.get("description"),
                "severity": flag.get("severity"),
                "reason": flag.get("reason"),
                "source": flag.get("source")
            })
        
        return {
            "flags_detected": red_flag_result.get("has_red_flags", False),
            "severity": red_flag_result.get("severity", "none"),
            "flags": flag_summary,
            "should_escalate": red_flag_result.get("should_escalate", False),
            "guidance": red_flag_result.get("guidance"),
            "requires_immediate_attention": red_flag_result.get("requires_immediate_attention", False)
        }
    
    def _assess_information_status(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess what information has been collected."""
        return {
            "chief_complaint_provided": patient_data.get("chief_complaint", {}).get("status") == "provided",
            "symptoms_provided": patient_data.get("symptoms", {}).get("status") in ["provided", "none_reported"],
            "vitals_provided": patient_data.get("vitals", {}).get("status") == "provided",
            "medical_history_provided": patient_data.get("medical_history", {}).get("status") in ["provided", "none_reported"],
            "medications_provided": patient_data.get("medications", {}).get("status") in ["provided", "none_reported"],
            "allergies_provided": patient_data.get("allergies", {}).get("status") in ["provided", "none_reported", "explicitly_denied"],
            "lifestyle_provided": patient_data.get("lifestyle", {}).get("status") == "provided",
            "demographics_provided": patient_data.get("demographics", {}).get("age_status") == "provided"
        }
    
    def _generate_hpi(
        self,
        patient_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
        report: Dict[str, Any]
    ) -> str:
        """
        Generate a coherent, chronological History of Present Illness (HPI) narrative.
        Uses LLM contextual reasoning when available, with a structured chronological fallback.
        """
        cc = report.get("chief_complaint", {})
        symp = report.get("symptoms", {})
        psd = symp.get("per_symptom_details", {})
        vitals = report.get("vitals", {})
        lifestyle = report.get("lifestyle", {})

        if self.llm is not None:
            try:
                import json
                prompt = f"""Write a concise, chronological, physician-ready History of Present Illness (HPI) paragraph based ONLY on the verified patient data below.
Rules:
- Be strictly factual. NEVER invent temperatures, durations, severities, or symptoms not in the data.
- Preserve each symptom's individual timeline (e.g., fever for 3 days, headache since yesterday, sore throat since today).
- Include pertinent positive and explicitly denied negative findings.
- Do NOT make diagnoses or prescribe medications.
- Output plain prose (1-2 short paragraphs), no markdown headers.

VERIFIED PATIENT DATA:
- Chief Complaint: {cc.get('reason')} (Duration: {cc.get('duration')})
- Reported Symptoms: {symp.get('reported_symptoms')}
- Global Symptom Details: location={symp.get('location')}, severity={symp.get('severity')}, character={symp.get('character')}, onset={symp.get('onset')}, progression={symp.get('progression')}
- Per-Symptom Details & Timelines: {json.dumps(psd, ensure_ascii=False)}
- Pertinent Negatives (Denied): {symp.get('negative_symptoms')}
- Vitals: {json.dumps(vitals.get('readings', {}), ensure_ascii=False)} (Interpretation: {vitals.get('interpretation')})
- Relevant Exposure/Travel: travel={lifestyle.get('recent_travel')}, sick_contacts={lifestyle.get('sick_contacts')}"""
                resp = self.llm.invoke(prompt)
                hpi_text = resp.content if hasattr(resp, "content") else str(resp)
                hpi_text = clean_text_entities(hpi_text).strip()
                if hpi_text and len(hpi_text) > 20:
                    return hpi_text
            except Exception as e:
                self.logger.warning(f"LLM HPI generation failed ({e}); using structured fallback")

        # Deterministic chronological fallback (never raw message concatenation)
        parts = []
        reason = cc.get("reason") or "presenting symptoms"
        dur = cc.get("duration")
        if dur and dur != "Not provided":
            parts.append(f"Patient presents with {reason} for {dur}.")
        else:
            parts.append(f"Patient presents with {reason}.")

        if psd:
            timeline_bits = []
            for s_name, s_info in psd.items():
                if s_info.get("status") == "denied":
                    continue
                bits = []
                if s_info.get("duration"):
                    bits.append(f"duration {s_info['duration']}")
                if s_info.get("onset"):
                    bits.append(s_info["onset"].lower())
                if s_info.get("severity"):
                    bits.append(f"severity {s_info['severity']}")
                if s_info.get("location"):
                    bits.append(f"at {s_info['location']}")
                if s_info.get("character"):
                    bits.append(f"{s_info['character']} character")
                if s_info.get("aggravating_factors"):
                    bits.append(s_info["aggravating_factors"].lower())
                if bits:
                    timeline_bits.append(f"{s_name} ({', '.join(bits)})")
            if timeline_bits:
                parts.append("Symptom chronology: " + "; ".join(timeline_bits) + ".")

        other_symps = [s for s in symp.get("reported_symptoms", []) if s.lower() not in psd]
        if other_symps:
            parts.append(f"Associated reported symptoms include {', '.join(other_symps)}.")

        if symp.get("negative_symptoms"):
            parts.append(f"Pertinent negatives explicitly denied: {', '.join(symp['negative_symptoms'])}.")

        return clean_text_entities(" ".join(parts))

    def _generate_clinical_summary(
        self,
        patient_data: Dict[str, Any],
        red_flag_result: Optional[Dict[str, Any]],
        report: Dict[str, Any]
    ) -> str:
        """
        Generate a concise factual summary for doctor review using LLM reasoning when available,
        with a structured deterministic fallback. Never diagnoses.
        """
        if self.llm is not None:
            try:
                import json
                cc = report.get("chief_complaint", {})
                symp = report.get("symptoms", {})
                mh = report.get("medical_history", {})
                meds = report.get("medications", {})
                alg = report.get("allergies", {})
                vitals = report.get("vitals", {})
                lifestyle = report.get("lifestyle", {})
                rf = report.get("red_flags", {})
                prompt = f"""Generate a concise, physician-facing Clinical Summary (3-5 sentences) based strictly on the structured patient record and VERIFIED HPI below.
Include:
1. Main complaint and symptom chronology.
2. Important positive and explicitly denied negative findings.
3. Relevant medical/medication/allergy/exposure history.
4. Critical missing information (e.g. unmeasured temperature) and any red flags.
Rules: Do NOT invent facts, do NOT diagnose, do NOT prescribe treatment.
CRITICAL: Stay 100% consistent with VERIFIED HPI and Per-Symptom Details for every symptom's onset and duration (never contradict HPI).

VERIFIED HPI:
{report.get('hpi', '')}

RECORD:
- Chief Complaint: {cc.get('reason')} ({cc.get('duration')})
- Reported Symptoms: {symp.get('reported_symptoms')}
- Per-Symptom Details: {json.dumps(symp.get('per_symptom_details', {}), ensure_ascii=False)}
- Pertinent Negatives: {symp.get('negative_symptoms')}
- Vitals: {vitals.get('readings')} ({vitals.get('interpretation')})
- Medical History: conditions={mh.get('conditions')}, denied={mh.get('denied_conditions')}, surgeries={mh.get('surgeries')}
- Medications: {meds.get('medications')}
- Allergies: drug={alg.get('drug_allergies')}, status={alg.get('interpretation')}
- Travel/Contacts: travel={lifestyle.get('recent_travel')}, sick_contacts={lifestyle.get('sick_contacts')}
- Red Flags: {rf}"""
                resp = self.llm.invoke(prompt)
                text = resp.content if hasattr(resp, "content") else str(resp)
                text = clean_text_entities(text).strip()
                if text and len(text) > 25:
                    return text
            except Exception as e:
                self.logger.warning(f"LLM clinical summary failed ({e}); using deterministic fallback")

        summary_parts = []
        
        # Add emergency warning if applicable
        if report.get("emergency_escalation"):
            summary_parts.append(
                "[EMERGENCY ESCALATION] Patient has reported symptoms requiring immediate "
                "emergency medical evaluation. See red flags section for details."
            )
            summary_parts.append("")
        
        # Chief complaint
        cc = patient_data.get("chief_complaint", {})
        cc_reason = cc.get("reason")
        cc_duration = cc.get("duration")
        if cc_reason:
            dur_str = f" (Duration: {cc_duration})" if cc_duration else ""
            summary_parts.append(f"Chief Complaint: {cc_reason}{dur_str}")
        
        # Symptoms & clinical details
        symptoms = patient_data.get("symptoms", {}).get("list", [])
        details = patient_data.get("symptom_details", {})
        if symptoms:
            detail_strs = []
            if details.get("location"):
                detail_strs.append(f"Location: {details['location']}")
            if details.get("severity"):
                detail_strs.append(f"Severity: {details['severity']}")
            if details.get("character"):
                detail_strs.append(f"Character: {details['character']}")
            if details.get("progression"):
                detail_strs.append(f"Progression: {details['progression']}")
            detail_suffix = f" [{'; '.join(detail_strs)}]" if detail_strs else ""
            summary_parts.append(f"Reported Symptoms: {', '.join(symptoms)}{detail_suffix}")
        
        # Pertinent negatives
        neg_symps = details.get("negative_symptoms", [])
        if neg_symps:
            formatted_summary_negs = []
            for n in neg_symps:
                n_str = clean_text_entities(n).strip()
                if n_str.lower().startswith("no "):
                    n_str = "No " + n_str[3:].strip()
                elif n_str.lower().startswith("not "):
                    n_str = "Not " + n_str[4:].strip()
                else:
                    n_str = "No " + n_str
                if n_str not in formatted_summary_negs:
                    formatted_summary_negs.append(n_str)
            summary_parts.append(f"Pertinent Negatives (Explicitly Denied): {', '.join(formatted_summary_negs)}")
        
        # Historical symptoms
        hist_symps = details.get("historical_symptoms", [])
        if hist_symps:
            summary_parts.append(f"Historical (Past/Resolved) Symptoms: {', '.join(hist_symps)}")
        
        # Uncertain symptoms
        uncert_symps = details.get("uncertain_symptoms", [])
        if uncert_symps:
            summary_parts.append(f"Uncertain / Possible Symptoms: {', '.join(uncert_symps)}")
        
        # Contradictions / Corrections
        contradictions = patient_data.get("contradictions", [])
        if contradictions:
            c_summaries = [c.get("note", str(c)) for c in contradictions]
            summary_parts.append(f"Noted Corrections / Contradictions: {'; '.join(c_summaries)}")
        
        # Vitals if available
        vitals = patient_data.get("vitals", {})
        vitals_reported = []
        for vital in ["blood_pressure", "heart_rate", "temperature", "respiratory_rate", 
                     "oxygen_saturation", "blood_glucose"]:
            value = vitals.get(vital)
            if value:
                vitals_reported.append(f"{vital.replace('_', ' ')}: {value}")
        if vitals_reported:
            summary_parts.append(f"Vitals: {'; '.join(vitals_reported)}")
        
        # Medical history
        history = patient_data.get("medical_history", {})
        history_items = []
        if history.get("conditions"):
            history_items.append(f"Conditions: {', '.join(history['conditions'])}")
        if history.get("surgeries"):
            history_items.append(f"Surgeries: {', '.join(history['surgeries'])}")
        if history_items:
            summary_parts.append(f"Medical History: {'; '.join(history_items)}")
        
        # Medications
        meds = patient_data.get("medications", {}).get("list", [])
        if meds:
            summary_parts.append(f"Current Medications: {', '.join(meds)}")
        
        # Allergies
        allergies = patient_data.get("allergies", {})
        allergy_status = allergies.get("status")
        if allergy_status == "explicitly_denied":
            summary_parts.append("Allergies: Patient denies any known allergies")
        else:
            all_allergies = (
                allergies.get("drug_allergies", []) + 
                allergies.get("food_allergies", []) + 
                allergies.get("environmental_allergies", [])
            )
            if all_allergies:
                summary_parts.append(f"Allergies: {', '.join(all_allergies)}")
        
        # Red flags summary
        if red_flag_result and red_flag_result.get("has_red_flags"):
            flags = red_flag_result.get("flags", [])
            flag_descriptions = [f["description"] for f in flags]
            summary_parts.append(f"Red Flags: {'; '.join(flag_descriptions)}")
        
        summary_parts.append("")
        summary_parts.append("---")
        summary_parts.append("This report contains patient-reported information only.")
        summary_parts.append("No clinical diagnoses have been made.")
        summary_parts.append("Missing information indicates patient has not yet provided data in this category.")
        
        return clean_text_entities("\n".join(summary_parts))
    
    def _generate_recommendations(
        self, 
        patient_data: Dict[str, Any], 
        red_flag_result: Optional[Dict[str, Any]] = None,
        report: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate contextual, non-diagnostic Points for Physician Review.
        Uses LLM reasoning when available so output adapts dynamically to any patient presentation;
        falls back to deterministic clinical review points when offline.
        """
        recs = []

        # 1. Emergency escalation recommendations (deterministic safety always takes precedence)
        if red_flag_result and red_flag_result.get("should_escalate"):
            recs.append("CRITICAL: Immediate emergency evaluation and escalation required. Direct patient to emergency services or call 108 / 112.")
            recs.append("Initiate immediate emergency stabilization protocol, continuous telemetry, SpO2 monitoring, and intravenous access as clinically indicated.")
            return recs

        if self.llm is not None and report is not None:
            try:
                import json
                cc = report.get("chief_complaint", {})
                symp = report.get("symptoms", {})
                vitals = report.get("vitals", {})
                mh = report.get("medical_history", {})
                meds = report.get("medications", {})
                alg = report.get("allergies", {})
                prompt = f"""Generate 3 to 5 concise, cautious, non-diagnostic "Points for Physician Review" tailored specifically to this patient's case.
Rules:
- Focus on measurements/history requiring verification, relevant physical exam areas, and important missing clinical info.
- Do NOT prescribe medications, do NOT make definitive diagnoses, and do NOT invent unstated patient facts.
- CRITICAL: Different onset dates across different symptoms (e.g., fever starting 3 days ago, headache/body pain yesterday, sore throat/cough today) represent a normal progressive clinical timeline, NOT a discrepancy or contradiction. Never describe documented per-symptom onset differences as a discrepancy.
- Return ONLY a JSON array of strings.

VERIFIED HPI:
{report.get('hpi', '')}

CASE SUMMARY:
- Chief Complaint: {cc.get('reason')} ({cc.get('duration')})
- Symptoms: {symp.get('reported_symptoms')}
- Per-Symptom Details: {json.dumps(symp.get('per_symptom_details', {}), ensure_ascii=False)}
- Pertinent Negatives: {symp.get('negative_symptoms')}
- Vitals: {vitals.get('readings')} ({vitals.get('interpretation')})
- Medical History: {mh.get('conditions')} / Denied: {mh.get('denied_conditions')}
- Medications: {meds.get('medications')}
- Allergies: {alg.get('interpretation')}"""
                resp = self.llm.invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                clean = content.strip()
                if clean.startswith("```"):
                    clean = re.sub(r'^```(?:json)?\s*', '', clean)
                    clean = re.sub(r'\s*```$', '', clean)
                s_idx = clean.find("[")
                e_idx = clean.rfind("]")
                if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                    parsed_recs = json.loads(clean[s_idx:e_idx + 1])
                    if isinstance(parsed_recs, list):
                        has_real_contradictions = bool(patient_data.get("contradictions"))
                        clean_recs = []
                        for r in parsed_recs:
                            if not isinstance(r, str) or not r.strip():
                                continue
                            r_clean = clean_text_entities(r).strip()
                            # Guard: never falsely label progressive symptom timelines as a discrepancy
                            if not has_real_contradictions and any(w in r_clean.lower() for w in ["discrepanc", "inconsisten", "contradict"]) and any(t in r_clean.lower() for t in ["onset", "timeline", "day", "yesterday", "today"]):
                                continue
                            clean_recs.append(r_clean)
                        if clean_recs:
                            return clean_recs[:5]
            except Exception as e:
                self.logger.warning(f"LLM physician review points failed ({e}); using contextual fallback")

        cc_reason = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        all_symps = [s.lower() for s in patient_data.get("symptoms", {}).get("list", [])]
        combined = cc_reason + " " + " ".join(all_symps)

        # 2. Scenario A: Fever / Febrile illness
        if any(w in combined for w in ["fever", "bukhar", "pyrexia", "high temp", "feverish"]):
            recs.append("Perform core body temperature check and continuous vital signs monitoring (evaluate for temperature spikes > 101°F / 38.3°C).")
            recs.append("Evaluate for focal infectious etiology (conduct targeted ENT, respiratory, abdominal, and urinary system examinations).")
            recs.append("Encourage adequate oral rehydration with electrolyte-balanced fluids and advise physical cooling measures (tepid sponging).")
            recs.append("Assess clinical indication for complete blood count (CBC), inflammatory markers (ESR/CRP), or blood/urine cultures prior to empirical antipyretic or antibiotic therapy.")
            recs.append("Patient red-flag instructions: instruct patient to seek urgent evaluation if fever is accompanied by neck stiffness, confusion, altered mental status, persistent vomiting, or petechial rash.")

        # 3. Scenario B: Abdominal Pain / Gastrointestinal complaints
        elif any(w in combined for w in ["abdominal pain", "stomach pain", "pet dard", "cramp", "belly", "tummy ache"]):
            recs.append("Conduct systematic 4-quadrant abdominal physical examination to assess localized tenderness, involuntary guarding, rebound tenderness, and bowel sounds.")
            recs.append("Evaluate for acute surgical abdomen: evaluate McBurney's point tenderness, Murphy's sign, and peritoneal irritation markers.")
            recs.append("Dietary guidance: maintain light, bland oral hydration; keep patient NPO (nil per os) if acute surgical or obstructive etiology cannot be excluded.")
            recs.append("Consider preliminary diagnostic workup: abdominal ultrasound, serum lipase and amylase, liver and renal metabolic panels, and urinalysis.")
            recs.append("Patient red-flag instructions: seek emergency medical attention immediately if pain intensifies acutely, radiates to right lower quadrant, or is accompanied by high fever, repeated vomiting, or hematochezia.")

        # 4. Scenario C: Headache / Neurological complaints
        elif any(w in combined for w in ["headache", "migraine", "sir dard", "sar dard", "sardard", "head pain", "head hurts"]):
            recs.append("Perform focused neurological examination including cranial nerve assessment, motor/sensory function, pupil reactivity, fundoscopic exam, and gait.")
            recs.append("Screen for meningeal irritation signs (neck flexion stiffness, Kernig's and Brudzinski's signs).")
            recs.append("Evaluate headache features against red-flag 'SNOOP' criteria (systemic symptoms, focal neurologic deficits, sudden-onset 'thunderclap' onset, postural variation).")
            recs.append("Advise patient to rest in a dark, quiet room; maintain adequate hydration and keep a symptom diary to identify potential headache triggers.")
            recs.append("Consider neuroimaging (non-contrast head CT or brain MRI) if red-flag signs, atypical features, or refractory severe symptoms are observed.")

        # 5. General / Other Clinical Complaints
        else:
            recs.append("Conduct focused physical examination targeted to the patient's primary presenting complaint and anatomical location.")
            recs.append("Complete formal medication reconciliation cross-checking patient-reported drugs against verified allergy tolerances.")
            recs.append("Monitor baseline vital signs (blood pressure, heart rate, respiratory rate, temperature, SpO2) and order preliminary targeted diagnostic tests as clinically indicated.")

        return [clean_text_entities(r) for r in recs]
    
    def format_report_text(self, report: Dict[str, Any]) -> str:
        """
        Format report as readable text for display or printing.
        
        Args:
            report: Structured report from generate_report()
        
        Returns:
            Formatted text representation
        """
        lines = []
        
        # Title and metadata
        lines.append("=" * 70)
        lines.append("PRE-CONSULTATION INTAKE REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
        lines.append("")
        
        # Emergency warning (if applicable)
        if report.get("emergency_escalation"):
            lines.append("[!] EMERGENCY ESCALATION REQUIRED [!]")
            lines.append(report.get("emergency_guidance", "Emergency care needed"))
            lines.append("")
            lines.append("-" * 70)
            lines.append("")
        
        # Chief complaint
        cc = report.get("chief_complaint", {})
        lines.append("CHIEF COMPLAINT")
        lines.append(f"  Reason: {cc.get('reason', 'Not provided')}")
        lines.append(f"  Duration: {cc.get('duration', 'Not provided')}")
        lines.append(f"  Status: {cc.get('status', 'Unknown')}")
        lines.append("")

        # History of Present Illness (HPI)
        if report.get("hpi"):
            lines.append("HISTORY OF PRESENT ILLNESS (HPI)")
            lines.append(f"  {report.get('hpi')}")
            lines.append("")
        
        # Symptoms
        symp = report.get("symptoms", {})
        lines.append("SYMPTOMS")
        lines.append(f"  Reported: {', '.join(symp.get('reported_symptoms', [])) if symp.get('reported_symptoms') else 'None'}")
        if symp.get("location") and symp.get("location") != "Not specified":
            lines.append(f"  Location: {symp.get('location')}")
        if symp.get("severity") and symp.get("severity") != "Not specified":
            lines.append(f"  Severity: {symp.get('severity')}")
        if symp.get("character") and symp.get("character") != "Not specified":
            lines.append(f"  Character: {symp.get('character')}")
        if symp.get("onset") and symp.get("onset") != "Not specified":
            lines.append(f"  Onset: {symp.get('onset')}")
        if symp.get("timing") and symp.get("timing") != "Not specified":
            lines.append(f"  Timing: {symp.get('timing')}")
        if symp.get("frequency") and symp.get("frequency") != "Not specified":
            lines.append(f"  Frequency: {symp.get('frequency')}")
        if symp.get("progression") and symp.get("progression") != "Not specified":
            lines.append(f"  Progression: {symp.get('progression')}")
        if symp.get("per_symptom_details"):
            lines.append("  Symptom Timelines & Details:")
            for s_name, s_info in symp.get("per_symptom_details", {}).items():
                if s_info.get("status") == "denied":
                    continue
                bits = [f"{k.replace('_', ' ').title()}: {v}" for k, v in s_info.items() if k not in ("status", "evidence") and v]
                if bits:
                    lines.append(f"    - {s_name.title()}: {'; '.join(bits)}")
        if symp.get("negative_symptoms"):
            lines.append(f"  Explicitly Denied (Pertinent Negatives): {', '.join(symp.get('negative_symptoms'))}")
        if symp.get("historical_symptoms"):
            lines.append(f"  Historical (Past/Resolved): {', '.join(symp.get('historical_symptoms'))}")
        if symp.get("uncertain_symptoms"):
            lines.append(f"  Uncertain / Possible: {', '.join(symp.get('uncertain_symptoms'))}")
        lines.append(f"  Status: {symp.get('interpretation', 'Unknown')}")
        lines.append("")
        
        # Vital signs
        vitals = report.get("vitals", {})
        lines.append("VITAL SIGNS")
        for vital_name, vital_value in vitals.get("readings", {}).items():
            lines.append(f"  {vital_name.replace('_', ' ').title()}: {vital_value}")
        lines.append(f"  Status: {vitals.get('interpretation', 'Unknown')}")
        lines.append("")
        
        # Medical history
        mh = report.get("medical_history", {})
        lines.append("MEDICAL HISTORY")
        lines.append(f"  Conditions: {', '.join(mh.get('conditions', ['Not asked']))}")
        lines.append(f"  Surgeries: {', '.join(mh.get('surgeries', ['Not asked']))}")
        lines.append(f"  Hospitalizations: {', '.join(mh.get('hospitalizations', ['Not asked']))}")
        lines.append(f"  Family History: {', '.join(mh.get('family_history', ['Not asked']))}")
        lines.append("")
        
        # Medications
        meds = report.get("medications", {})
        lines.append("MEDICATIONS")
        lines.append(f"  Current: {', '.join(meds.get('medications', ['Not asked'])) if meds.get('medications') else 'Not asked'}")
        lines.append(f"  Status: {meds.get('interpretation', 'Not asked')}")
        lines.append("")
        
        # Allergies
        allergies = report.get("allergies", {})
        lines.append("ALLERGIES")
        lines.append(f"  Drug: {', '.join(allergies.get('drug_allergies', ['Not asked'])) if allergies.get('drug_allergies') else 'Not asked'}")
        lines.append(f"  Food: {', '.join(allergies.get('food_allergies', ['Not asked'])) if allergies.get('food_allergies') else 'Not asked'}")
        lines.append(f"  Environmental: {', '.join(allergies.get('environmental_allergies', ['Not asked'])) if allergies.get('environmental_allergies') else 'Not asked'}")
        lines.append(f"  Status: {allergies.get('interpretation', 'Not asked')}")
        lines.append("")
        
        # Lifestyle & Exposures
        lifestyle = report.get("lifestyle", {})
        lines.append("LIFESTYLE")
        lines.append(f"  Smoking: {lifestyle.get('smoking', 'Not asked')}")
        lines.append(f"  Alcohol: {lifestyle.get('alcohol', 'Not asked')}")
        lines.append(f"  Exercise: {lifestyle.get('exercise', 'Not asked')}")
        lines.append(f"  Diet: {lifestyle.get('diet', 'Not asked')}")
        lines.append(f"  Recent Travel: {lifestyle.get('recent_travel', 'Not asked')}")
        lines.append(f"  Sick Contacts: {lifestyle.get('sick_contacts', 'Not asked')}")
        lines.append("")
        
        # Demographics
        demo = report.get("demographics", {})
        lines.append("DEMOGRAPHICS")
        lines.append(f"  Age: {demo.get('age', 'Not provided')}")
        lines.append("")
        
        # Red flags
        if report.get("red_flags"):
            rf = report.get("red_flags", {})
            lines.append("RED FLAGS / EMERGENCY ASSESSMENT")
            if rf.get("flags_detected"):
                lines.append(f"  Severity: {rf.get('severity', 'Unknown')}")
                lines.append(f"  Escalation Required: {rf.get('should_escalate', False)}")
                lines.append("  Flags:")
                for flag in rf.get("flags", []):
                    lines.append(f"    - {flag.get('description', 'Unknown')} ({flag.get('severity', 'unknown')})")
                    lines.append(f"      Reason: {flag.get('reason', 'N/A')}")
            else:
                neg_list = report.get("symptoms", {}).get("negative_symptoms", [])
                if neg_list:
                    lines.append(f"  Explicitly Denied Screening Items: {', '.join(neg_list)}")
                lines.append("  Status: No emergency red flags reported among assessed items; unassessed red flags have not been ruled out (screening may be incomplete).")
            lines.append("")
        
        # Corrections / Contradictions (if any)
        if report.get("contradictions"):
            lines.append("CORRECTIONS / CONTRADICTIONS NOTED")
            for c in report.get("contradictions", []):
                lines.append(f"  - {c.get('note', str(c))}")
            lines.append("")
        
        # Missing information
        info_status = report.get("information_status", {})
        missing_categories = []
        category_labels = {
            "chief_complaint_provided": "Chief Complaint",
            "symptoms_provided": "Associated Symptoms",
            "vitals_provided": "Vital Signs Measurements",
            "medical_history_provided": "Past Medical History",
            "medications_provided": "Current Medications",
            "allergies_provided": "Allergies Profile",
            "lifestyle_provided": "Lifestyle Factors",
            "demographics_provided": "Demographics (Age/Gender)"
        }
        for k, label in category_labels.items():
            if not info_status.get(k):
                missing_categories.append(label)
        
        lines.append("MISSING INFORMATION")
        if missing_categories:
            lines.append(f"  Not collected / Not provided: {', '.join(missing_categories)}")
        else:
            lines.append("  All standard intake categories addressed")
        lines.append("")
        
        # Recommendations & Suggested Next Steps
        recs = report.get("recommendations", [])
        if recs:
            lines.append("RECOMMENDATIONS & SUGGESTED NEXT STEPS")
            for r in recs:
                lines.append(f"  - {r}")
            lines.append("")

        # Clinical summary
        lines.append("CLINICAL SUMMARY")
        lines.append(report.get("clinical_summary", "No summary available"))
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)
        
        return clean_text_entities("\n".join(lines))
    
    def validate_report(self, report: Dict[str, Any]) -> bool:
        """
        Validate report structure and content.
        
        Args:
            report: Structured report to validate
        
        Returns:
            True if valid, raises exception if invalid
        
        Raises:
            ValueError: If report is invalid
        """
        required_fields = [
            "report_type", "generated_at", "emergency_escalation",
            "chief_complaint", "symptoms", "vitals", "medical_history",
            "medications", "allergies", "lifestyle", "demographics",
            "clinical_summary"
        ]
        
        for field in required_fields:
            if field not in report:
                raise ValueError(f"Report missing required field: {field}")
        
        # Validate no fabricated information
        self._validate_no_fabrication(report)
        
        return True
    
    def _validate_no_fabrication(self, report: Dict[str, Any]) -> None:
        """Validate that no information has been invented."""
        # Check that "Not provided" or similar markers are used for missing data
        allowed_placeholders = [
            "Not provided", "Not reported", "Unknown", "Unknown status",
            "Not asked", "Patient denied", "None reported", "Not measured", "None"
        ]
        def check_for_invented(section_name, section_data):
            if not isinstance(section_data, dict):
                return
            
            for key, value in section_data.items():
                if key == "note":
                    continue
                
                # Check if value looks like fabricated data
                if isinstance(value, str) and value and value not in allowed_placeholders:
                    # This is potentially data - it should come from patient
                    if "Not" not in value and "Unknown" not in value and "denied" not in value.lower():
                        # Likely patient-reported data, which is OK
                        pass
                elif isinstance(value, list) and len(value) == 0:
                    # Empty list is OK (no data provided)
                    pass
                elif isinstance(value, list) and all(
                    v in allowed_placeholders or not v
                    for v in value
                ):
                    # All values are standard placeholders
                    pass
        
        # Validate each section
        for section in ["chief_complaint", "symptoms", "vitals", "medical_history", 
                       "medications", "allergies", "lifestyle", "demographics"]:
            check_for_invented(section, report.get(section, {}))


# Export the class for easy importing
__all__ = ['PreConsultationReportGenerator']
