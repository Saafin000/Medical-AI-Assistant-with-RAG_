import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Canonical symptom mapping to standardize terms and eliminate duplicate / synonymous entries
CANONICAL_SYMPTOMS = {
    # Abdominal / Gastrointestinal
    "stomach pain": "abdominal pain",
    "abdominal pain": "abdominal pain",
    "pet dard": "abdominal pain",
    "stomach ache": "abdominal pain",
    "belly ache": "abdominal pain",
    "tummy ache": "abdominal pain",
    "abdominal cramps": "abdominal pain",
    "stomach cramps": "abdominal pain",
    "cramps in stomach": "abdominal pain",
    "severe stomach cramps": "abdominal pain",
    "lower abdominal pain": "abdominal pain",
    "upper abdominal pain": "abdominal pain",
    "gastric pain": "abdominal pain",

    # Head / Neurological
    "headache": "headache",
    "sir dard": "headache",
    "sar dard": "headache",
    "sardard": "headache",
    "head pain": "headache",
    "head hurts": "headache",
    "head is hurting": "headache",
    "throbbing head": "headache",
    "throbbing headache": "headache",
    "tension headache": "headache",
    "migraine": "headache",
    "severe headache": "headache",
    "head ache": "headache",

    # Fever / Temperature
    "fever": "fever",
    "bukhar": "fever",
    "bukhaar": "fever",
    "tez bukhar": "fever",
    "high fever": "fever",
    "mild fever": "fever",
    "feeling hot / feverish": "fever",
    "feeling feverish": "fever",
    "feverish": "fever",
    "pyrexia": "fever",
    "high temp": "fever",
    "high temperature": "fever",

    # Chest / Cardiopulmonary
    "chest pain": "chest pain",
    "chhati mein dard": "chest pain",
    "chaati me dard": "chest pain",
    "chest discomfort": "chest pain",
    "shortness of breath": "shortness of breath",
    "difficulty breathing": "difficulty breathing",
    "trouble breathing": "difficulty breathing",

    # ENT / Respiratory
    "sore throat": "sore throat",
    "gale mein dard": "sore throat",
    "gale me dard": "sore throat",
    "gala kharab": "sore throat",
    "throat pain": "sore throat",
    "dry cough": "dry cough",
    "sukhi khansi": "dry cough",
    "cough": "cough",
    "halka cough": "cough",
    "mild cough": "cough",
    "khasi": "cough",
    "khaasi": "cough",
    "chest tightness": "chest tightness",
    "sneezing": "sneezing",
    "nasal congestion": "nasal congestion",
    "urinary difficulty": "urinary difficulty",
    "runny nose / cold": "runny nose / cold",
    "runny nose": "runny nose",
    "stuffy nose": "nasal congestion",
    "zukaam": "runny nose / cold",
    "zukam": "runny nose / cold",
    "sardi": "runny nose / cold",

    # GI other
    "nausea": "nausea",
    "vomiting": "vomiting",
    "nausea / vomiting": "nausea / vomiting",
    "ulti": "vomiting",
    "matli": "nausea",

    # General / Musculoskeletal
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "chakkar": "dizziness",
    "body pain": "body pain",
    "badan dard": "body pain",
    "body ache": "body pain",
    "joint pain": "joint pain",
    "muscle pain": "muscle pain",
    "back pain": "back pain",
    "neck pain": "neck pain",
    "chills": "chills",
    "feeling cold": "chills",
    "kapkapi": "chills",
    "shivering": "chills",
    "thand lagkar kapkapi": "chills",
    "hedache": "headache",
    "yellowish urine": "yellowish urine",
    "yellow urine": "yellowish urine",
    "dark urine": "dark urine",
    "fatigue": "fatigue",
    "weakness": "weakness",
    "tiredness": "fatigue",
}


def normalize_symptoms(symptoms_list: List[str]) -> List[str]:
    """
    Standardize symptom names using canonical taxonomy and deduplicate synonyms.
    If a specific regional pain (e.g. abdominal pain, headache) is present, generic 'pain' is dropped.
    If a specific cough type (e.g. 'dry cough') is present, bare 'cough' is dropped.
    """
    if not symptoms_list:
        return []
    res = []
    has_specific_pain = False
    has_specific_cough = False
    for s in symptoms_list:
        if not s or not isinstance(s, str):
            continue
        s_clean = s.strip()
        s_lower = s_clean.lower()
        canon = CANONICAL_SYMPTOMS.get(s_lower, s_lower)
        if canon not in res:
            res.append(canon)
        if any(p in canon.lower() for p in [
            "abdominal pain", "headache", "chest pain", "body pain", "joint pain", 
            "muscle pain", "back pain", "neck pain", "sore throat"
        ]):
            has_specific_pain = True
        if canon.lower() in ["dry cough", "productive cough", "wet cough"]:
            has_specific_cough = True
    
    # If a specific focal pain exists, drop bare generic "pain", "ache", "discomfort"
    if has_specific_pain:
        res = [s for s in res if s.lower() not in ["pain", "ache", "discomfort", "generalized pain"]]
    # If a specific cough type exists (e.g., dry cough), drop bare generic "cough"
    if has_specific_cough:
        res = [s for s in res if s.lower() != "cough"]
    return res


class PatientInfoCollector:
    """
    Collects and maintains structured patient information from patient-provided responses.
    
    Responsibilities:
    - Extract structured information from unstructured patient text
    - Maintain patient data across multiple interactions
    - Distinguish between unknown, not provided, and explicitly denied information
    - Preserve original patient-reported values
    - Track missing information
    - Validate values and preserve units
    
    Does NOT:
    - Generate follow-up questions (FollowupQuestionGenerator responsibility)
    - Diagnose or recommend treatment
    - Invent patient information
    - Independently access medical databases
    - Persist data to database (separate responsibility)
    """
    
    def __init__(self, config=None, llm=None):
        """Initialize an empty patient information structure with optional LLM reasoning support."""
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing PatientInfoCollector")
        self.config = config
        self.llm = llm
        
        # Initialize structured patient information
        self.patient_data = self._create_empty_patient_structure()
        
        self.logger.info("PatientInfoCollector initialized with empty patient structure")
    
    def _create_empty_patient_structure(self) -> Dict[str, Any]:
        """
        Create an empty structured patient information template.
        
        Returns:
            Dictionary with empty patient information structure
        """
        return {
            "demographics": {
                "age": None,  # in years
                "age_unit": "years",
                "age_status": "unknown",  # "provided", "unknown", "not_applicable"
            },
            "chief_complaint": {
                "reason": None,  # primary reason for visit
                "duration": None,  # how long it's been happening
                "urgency": None,  # "routine", "urgent", "emergency"
                "status": "unknown",  # "provided", "unknown"
            },
            "symptoms": {
                "list": [],  # List of symptoms
                "status": "unknown",  # "provided", "unknown", "none_reported"
            },
            "symptom_details": {
                "location": None,  # e.g., "right side", "frontal"
                "severity": None,  # e.g., "7/10", "mild", "severe"
                "character": None,  # e.g., "throbbing", "sharp", "dull"
                "onset": None,  # e.g., "sudden", "gradual"
                "progression": None,  # e.g., "worsening", "constant", "intermittent"
                "timing": None,  # e.g., "morning", "evening", "night"
                "frequency": None,  # e.g., "recurring", "daily", "occasional"
                "associated_symptoms": [],  # e.g., ["nausea", "light sensitivity"]
                "negative_symptoms": [],  # explicitly denied: ["no fever", "no dizziness"]
                "historical_symptoms": [],  # past/resolved: e.g., ["fever yesterday (resolved)"]
                "uncertain_symptoms": [],  # uncertain/possible: e.g., ["possible chills"]
                "per_symptom_details": {},  # symptom -> {status, onset, duration, location, character, severity, timing, frequency, progression, aggravating_factors, relieving_factors, radiation, context, evidence}
            },
            "contradictions": [],  # tracked contradictions/corrections across turns
            "correction_history": [],  # chronological record of patient corrections
            "patient_review_status": "pending_review",  # "pending_review", "confirmed", "corrected_pending_confirmation"
            "final_confirmation_status": "not_asked",  # "not_asked", "awaiting_response", "confirmed", "incomplete_unconfirmed"
            "unresolved_clarifications": [],
            "medical_history": {
                "conditions": [],  # Past/present medical conditions
                "denied_conditions": [],  # Explicitly denied medical conditions
                "surgeries": [],  # Past surgeries
                "hospitalizations": [],  # Past hospitalizations
                "family_history": [],  # Relevant family history
                "status": "not_asked",  # "not_asked", "not_provided", "patient_denied", "none_reported", "provided"
                "conditions_status": "not_asked",
                "surgeries_status": "not_asked",
                "hospitalizations_status": "not_asked",
                "family_history_status": "not_asked",
            },
            "medications": {
                "list": [],  # Current medications
                "status": "not_asked",  # "not_asked", "not_provided", "patient_denied", "none_reported", "provided"
            },
            "allergies": {
                "drug_allergies": [],  # Drug allergies
                "food_allergies": [],  # Food allergies
                "environmental_allergies": [],  # Environmental allergies
                "status": "not_asked",  # "not_asked", "not_provided", "patient_denied", "none_reported", "explicitly_denied", "provided"
                "drug_status": "not_asked",
                "food_status": "not_asked",
                "environmental_status": "not_asked",
            },
            "lifestyle": {
                "smoking": None,  # "never", "current", "former", "patient_denied", "none_reported", "not_provided", "not_asked"
                "alcohol": None,  # "none", "occasional", "regular", "patient_denied", "none_reported", "not_provided", "not_asked"
                "exercise": None,  # Brief description or frequency
                "diet": None,  # Brief description
                "recent_travel": None,  # Recent travel history or "patient_denied"
                "sick_contacts": None,  # Recent sick contact exposure or "patient_denied"
                "exposures": None,  # Environmental/occupational exposure
                "status": "not_asked",  # "not_asked", "not_provided", "patient_denied", "none_reported", "provided"
                "smoking_status": "not_asked",
                "alcohol_status": "not_asked",
                "exercise_status": "not_asked",
                "diet_status": "not_asked",
                "recent_travel_status": "not_asked",
                "sick_contacts_status": "not_asked",
                "exposures_status": "not_asked",
            },
            "vitals": {
                "blood_pressure": None,  # "mmHg" format
                "heart_rate": None,  # "bpm"
                "temperature": None,  # "celsius" or "fahrenheit"
                "respiratory_rate": None,  # "breaths/min"
                "oxygen_saturation": None,  # "%" (SpO2)
                "blood_glucose": None,  # "mg/dL"
                "status": "not_asked",  # "not_asked", "not_provided", "not_measured", "provided"
                "temperature_status": "not_asked",
            },
            "missing_info": {
                # Track explicitly missing information for later follow-up
                "chief_complaint": False,
                "symptoms": False,
                "medical_history": False,
                "medications": False,
                "allergies": False,
                "vitals": False,
            },
            "collection_metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "extraction_count": 0,  # Number of extraction attempts
            }
        }
    
    def initialize_patient_data(self) -> Dict[str, Any]:
        """
        Initialize a new empty patient information structure.
        Resets all previously collected data.
        
        Returns:
            Empty patient information dictionary
        """
        self.logger.info("Initializing new patient data structure")
        self.patient_data = self._create_empty_patient_structure()
        return self.get_patient_data()
    
    def extract_from_patient_text(self, text: str, last_question: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structured information from patient-provided text.
        Uses simple keyword matching to identify information categories.
        
        Args:
            text: Patient-provided text response
            
        Returns:
            Dictionary with extracted information
        
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Patient text must be a non-empty string")
        
        if len(text.strip()) == 0:
            raise ValueError("Patient text cannot be empty or whitespace-only")
        
        # Clean any HTML entities or hex entities from user text (multi-pass to eliminate nested entities)
        import html
        for _ in range(5):
            prev_text = text
            text = html.unescape(text)
            text = re.sub(r'&#x([0-9a-fA-F]+);?', lambda m: chr(int(m.group(1), 16)), text, flags=re.IGNORECASE)
            text = re.sub(r'&#([0-9]+);?', lambda m: chr(int(m.group(1))), text)
            if text == prev_text:
                break
        
        self.logger.info(f"Extracting patient information from {len(text)} characters")
        
        extracted = {}
        text_lower = text.lower()
        
        # Extract chief complaint indicators
        extracted["chief_complaint"] = self._extract_chief_complaint(text, text_lower)
        
        # Extract symptoms
        extracted["symptoms"] = self._extract_symptoms(text, text_lower)
        
        # Extract medical history
        extracted["medical_history"] = self._extract_medical_history(text, text_lower)
        
        # Extract medications
        extracted["medications"] = self._extract_medications(text, text_lower)
        
        # Extract allergies
        extracted["allergies"] = self._extract_allergies(text, text_lower)
        
        # Extract lifestyle information
        extracted["lifestyle"] = self._extract_lifestyle(text, text_lower)
        
        # Extract vitals (if mentioned)
        extracted["vitals"] = self._extract_vitals(text, text_lower)
        
        # Extract demographics (if mentioned)
        extracted["demographics"] = self._extract_demographics(text, text_lower)

        # Contextual short negative handling (e.g. patient answers "No" to "Do you have neck stiffness or photophobia?")
        if last_question and re.match(r'^(?:no|nope|nahi|none)\s*[.!]*$', text_lower.strip()):
            lq_lower = last_question.lower()
            ctx_negs = []
            for check_term, canon_neg in [
                ("neck stiffness", "no neck stiffness"),
                ("stiffness in your neck", "no neck stiffness"),
                ("photophobia", "no photophobia"),
                ("sensitivity to light", "no light sensitivity"),
                ("light sensitivity", "no light sensitivity"),
                ("rash", "no rash"),
                ("shortness of breath", "no shortness of breath"),
                ("difficulty breathing", "no difficulty breathing"),
                ("chest pain", "no chest pain"),
                ("nausea", "no nausea"),
                ("vomiting", "no vomiting"),
                ("vision changes", "no vision changes"),
                ("pain, burning", "no dysuria"),
                ("painful urination", "no dysuria"),
                ("burning", "no dysuria"),
                ("increased frequency", "no urinary frequency"),
            ]:
                if check_term in lq_lower and canon_neg not in ctx_negs:
                    ctx_negs.append(canon_neg)
            if ctx_negs:
                extracted.setdefault("symptoms", {})
                existing_n = extracted["symptoms"].get("negative_symptoms", [])
                extracted["symptoms"]["negative_symptoms"] = list(dict.fromkeys(existing_n + ctx_negs))
        
        # Update patient data with deterministic extracted information first (safety & baseline)
        self._update_patient_data(extracted)

        # If an LLM is configured, perform contextual extraction to capture Hindi/Hinglish/English nuances,
        # symptom-specific timelines, negations, sick contacts, travel, and complex clinical meaning.
        if self.llm is not None:
            try:
                llm_updates = self._llm_extract_clinical_info(text, last_question=last_question)
                if llm_updates:
                    self.merge_structured_updates(llm_updates)
            except Exception as llm_err:
                self.logger.warning(f"LLM contextual extraction failed ({llm_err}); retaining deterministic extraction")
        
        self.logger.info(f"Extraction complete. Categories with data: {len([k for k, v in extracted.items() if v])}")
        
        return extracted

    def _llm_extract_clinical_info(self, text: str, last_question: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Use the configured LLM to contextually extract structured clinical information
        from Hindi, Hinglish, or English patient statements.
        Returns validated structured updates or None on failure.
        """
        import json
        if self.llm is None:
            return None

        curr_cc = self.patient_data.get("chief_complaint", {})
        curr_symps = self.patient_data.get("symptoms", {}).get("list", [])
        curr_negs = self.patient_data.get("symptom_details", {}).get("negative_symptoms", [])
        curr_per_symp = self.patient_data.get("symptom_details", {}).get("per_symptom_details", {})

        prompt = f"""You are a clinical intake extraction engine. Extract ONLY explicitly stated facts from the patient's message in Hindi, Hinglish, or English.
NEVER invent durations, temperatures, locations, severities, or conditions.
NEVER infer a symptom just because it commonly occurs with another disease.
If the patient says temperature was not measured, set temperature_status="not_measured" and temperature=null (never treat feeling feverish as a measured temperature).
If the patient denies a symptom or condition (e.g. "no diabetes, asthma or heart disease", "no breathing difficulty, chest pain or chest tightness", "no recent sick contact or travel"), record it as denied—NEVER as positive.
Maintain individual timelines per symptom (e.g. fever for 3 days, headache since yesterday, sore throat since today).

EXISTING RECORD SUMMARY:
- Chief Complaint: {curr_cc.get('reason')} (Duration: {curr_cc.get('duration')})
- Reported Symptoms: {curr_symps}
- Denied Symptoms: {curr_negs}
- Per-symptom details: {curr_per_symp}

PREVIOUS ASSISTANT QUESTION:
"{last_question or 'None'}"

PATIENT MESSAGE:
"{text}"

Respond ONLY with a valid JSON object (omit or leave null/empty any field not mentioned in this message):
{{
  "chief_complaint": {{"reason": null, "duration": null}},
  "symptoms": [],
  "negative_symptoms": [],
  "historical_symptoms": [],
  "uncertain_symptoms": [],
  "per_symptom_details": {{
    "<symptom_name_lowercase>": {{
      "status": "reported",
      "onset": null,
      "duration": null,
      "location": null,
      "character": null,
      "severity": null,
      "timing": null,
      "frequency": null,
      "progression": null,
      "aggravating_factors": null,
      "relieving_factors": null,
      "radiation": null,
      "is_correction": false,
      "evidence": null
    }}
  }},
  "vitals": {{"temperature": null, "temperature_status": null, "blood_pressure": null, "heart_rate": null, "oxygen_saturation": null}},
  "medical_history": {{"conditions": [], "denied_conditions": [], "surgeries": [], "hospitalizations": [], "family_history": []}},
  "medications": [],
  "allergies": {{"drug_allergies": [], "food_allergies": [], "environmental_allergies": [], "status": null}},
  "lifestyle": {{"smoking": null, "alcohol": null, "recent_travel": null, "recent_travel_status": null, "sick_contacts": null, "sick_contacts_status": null, "exposures": null}},
  "demographics": {{"age": null}}
}}"""

        resp = self.llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        clean = content.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        parsed = json.loads(clean[start:end + 1])
        if not isinstance(parsed, dict):
            return None

        # Validate & sanitize LLM output against hallucinations / contradictions
        validated: Dict[str, Any] = {}
        allowed_statuses = {"reported", "denied", "resolved", "uncertain"}

        # Chief complaint
        cc = parsed.get("chief_complaint")
        if isinstance(cc, dict) and (cc.get("reason") or cc.get("duration")):
            # Only set chief complaint if not already established or if reason is valid
            validated_cc = {}
            if isinstance(cc.get("reason"), str) and cc["reason"].strip():
                if not curr_cc.get("reason"):
                    validated_cc["reason"] = cc["reason"].strip()
            if isinstance(cc.get("duration"), str) and cc["duration"].strip():
                if not curr_cc.get("duration"):
                    validated_cc["duration"] = cc["duration"].strip()
            if validated_cc:
                validated["chief_complaint"] = validated_cc

        # Negative symptoms
        neg_list = []
        if isinstance(parsed.get("negative_symptoms"), list):
            for ns in parsed["negative_symptoms"]:
                if isinstance(ns, str) and ns.strip():
                    ns_clean = ns.strip().lower()
                    if not ns_clean.startswith(("no ", "not ")):
                        ns_clean = f"no {ns_clean}"
                    neg_list.append(ns_clean)
        if neg_list:
            validated["negative_symptoms"] = neg_list

        # Positive symptoms (scrubbed against neg_list and deterministic negations)
        neg_roots = {n.replace("no ", "").replace("not ", "").strip() for n in neg_list}
        if isinstance(parsed.get("symptoms"), list):
            pos_list = []
            for s in parsed["symptoms"]:
                if isinstance(s, str) and s.strip():
                    sl = s.strip().lower()
                    if sl not in neg_roots and not self.is_term_negated(sl, text.lower()):
                        pos_list.append(sl)
            if pos_list:
                validated["symptoms"] = pos_list

        if isinstance(parsed.get("historical_symptoms"), list):
            hs = [s.strip() for s in parsed["historical_symptoms"] if isinstance(s, str) and s.strip()]
            if hs:
                validated["historical_symptoms"] = hs

        if isinstance(parsed.get("uncertain_symptoms"), list):
            us = [s.strip() for s in parsed["uncertain_symptoms"] if isinstance(s, str) and s.strip()]
            if us:
                validated["uncertain_symptoms"] = us

        # Per-symptom details
        if isinstance(parsed.get("per_symptom_details"), dict):
            psd_clean = {}
            for symp_key, details in parsed["per_symptom_details"].items():
                if not isinstance(symp_key, str) or not isinstance(details, dict):
                    continue
                sk = CANONICAL_SYMPTOMS.get(symp_key.strip().lower(), symp_key.strip().lower())
                if not sk:
                    continue
                st = str(details.get("status") or "reported").lower()
                if st not in allowed_statuses:
                    st = "reported"
                if st == "reported" and (sk in neg_roots or self.is_term_negated(sk, text.lower())):
                    st = "denied"
                entry = {"status": st}
                for f in ["onset", "duration", "location", "character", "severity", "timing", "frequency", "progression", "aggravating_factors", "relieving_factors", "radiation", "evidence"]:
                    val = details.get(f)
                    if isinstance(val, str) and val.strip() and val.strip().lower() not in ("null", "none", "unknown", "not specified"):
                        # Guard against negated character/location
                        if f == "character" and self.is_term_negated(val.strip().lower(), text.lower()):
                            continue
                        entry[f] = val.strip()
                if details.get("is_correction"):
                    entry["is_correction"] = True
                if len(entry) > 1 or st != "reported":
                    psd_clean[sk] = entry
            if psd_clean:
                validated["per_symptom_details"] = psd_clean

        # Vitals
        v = parsed.get("vitals")
        if isinstance(v, dict):
            v_clean = {}
            if v.get("temperature_status") == "not_measured":
                v_clean["temperature_status"] = "not_measured"
            elif isinstance(v.get("temperature"), str) and v["temperature"].strip():
                # Validate numeric temperature
                m_temp = re.search(r'(\d{2,3}(?:\.\d+)?)', v["temperature"])
                if m_temp:
                    tv = float(m_temp.group(1))
                    if (95 <= tv <= 106) or (35 <= tv <= 41):
                        v_clean["temperature"] = v["temperature"].strip()
            for vk in ["blood_pressure", "heart_rate", "oxygen_saturation"]:
                if isinstance(v.get(vk), str) and v[vk].strip() and v[vk].strip().lower() not in ("null", "none"):
                    v_clean[vk] = v[vk].strip()
            if v_clean:
                validated["vitals"] = v_clean

        # Medical history
        mh = parsed.get("medical_history")
        if isinstance(mh, dict):
            mh_clean = {}
            denied_conds = []
            if isinstance(mh.get("denied_conditions"), list):
                denied_conds = [c.strip().lower() for c in mh["denied_conditions"] if isinstance(c, str) and c.strip()]
            if isinstance(mh.get("conditions"), list):
                pos_conds = [
                    c.strip().lower() for c in mh["conditions"]
                    if isinstance(c, str) and c.strip()
                    and c.strip().lower() not in denied_conds
                    and not self.is_term_negated(c.strip().lower(), text.lower())
                ]
                if pos_conds:
                    mh_clean["conditions"] = pos_conds
            if denied_conds:
                mh_clean["denied_conditions"] = denied_conds
            for mk in ["surgeries", "hospitalizations", "family_history"]:
                if isinstance(mh.get(mk), list):
                    items = [x.strip() for x in mh[mk] if isinstance(x, str) and x.strip() and not self.is_term_negated(x.strip().lower(), text.lower())]
                    if items:
                        mh_clean[mk] = items
            if mh_clean:
                validated["medical_history"] = mh_clean

        # Medications
        if isinstance(parsed.get("medications"), list):
            meds = [m.strip() for m in parsed["medications"] if isinstance(m, str) and m.strip()]
            if meds:
                validated["medications"] = meds

        # Allergies
        alg = parsed.get("allergies")
        if isinstance(alg, dict):
            alg_clean = {}
            for ak in ["drug_allergies", "food_allergies", "environmental_allergies"]:
                if isinstance(alg.get(ak), list):
                    items = [x.strip() for x in alg[ak] if isinstance(x, str) and x.strip()]
                    if items:
                        alg_clean[ak] = items
            if alg.get("status") in ["explicitly_denied", "patient_denied", "none_reported", "provided"]:
                alg_clean["status"] = alg["status"]
            if alg_clean:
                validated["allergies"] = alg_clean

        # Lifestyle & exposures
        ls = parsed.get("lifestyle")
        if isinstance(ls, dict):
            ls_clean = {}
            for lk in ["smoking", "alcohol", "recent_travel", "recent_travel_status", "sick_contacts", "sick_contacts_status", "exposures"]:
                val = ls.get(lk)
                if isinstance(val, str) and val.strip() and val.strip().lower() not in ("null", "none", "not_asked"):
                    ls_clean[lk] = val.strip()
            if ls_clean:
                validated["lifestyle"] = ls_clean

        # Demographics
        demo = parsed.get("demographics")
        if isinstance(demo, dict) and demo.get("age") is not None:
            try:
                age_val = int(demo["age"])
                if 0 < age_val < 130:
                    validated["demographics"] = {"age": age_val}
            except (TypeError, ValueError):
                pass

        return validated if validated else None
    
    def is_term_negated(self, term: str, target_text: str) -> bool:
        """
        Check if a symptom or clinical term is explicitly negated across English,
        Hindi, and Hinglish phrasing (supporting both preceding and post-position negations).
        """
        if not term or not target_text:
            return False
        target_lower = target_text.lower()
        term_lower = term.lower().strip()
        
        # 1. Specific Breathing difficulty Hindi/Hinglish patterns
        if any(b in term_lower for b in ["breathing", "breath", "shortness of breath", "difficulty breathing"]):
            hindi_breathing_neg = [
                r'\bsaans\s+(?:lene\s+(?:mein|me)\s+)?(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue|pareshaani)\s+nahi\b',
                r'\b(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue)\s+nahi\s+(?:hai\s+)?saans\s+lene\s+(?:mein|me)\b',
                r'\bsaans\s+(?:phool\s+nahi|nahi\s+phool|theek\s+hai|normal\s+hai|sahi\s+hai)\b',
                r'\bbreathing\s+(?:is\s+)?(?:fine|normal|ok|okay|good)\b',
                r'\bno\s+(?:trouble|difficulty|issue|problem)\s+(?:with\s+)?breathing\b'
            ]
            if any(re.search(pat, target_lower) for pat in hindi_breathing_neg):
                return True

        # 2. Hindi / Hinglish post-position negations (e.g. "chest pain bhi nahi hai", "headache nahi hai")
        escaped_term = re.escape(term_lower)
        hindi_post_patterns = [
            escaped_term + r'\s+(?:bhi\s+|toh\s+|to\s+)?(?:nahi\s+hai|nahi\s+ho\s+raha|nahi\s+tha|nahi\s+hoga|nahi\s+rehta|nahi|na\s+hai)\b',
            escaped_term + r'\s+(?:mein\s+|me\s+)?(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue|dard)\s+nahi\b',
            escaped_term + r'\s+bilkul\s+(?:bhi\s+)?nahi\b',
            escaped_term + r'\s+(?:se\s+)?(?:koi\s+)?(?:problem|dikkat|takleef)\s+nahi\b',
            escaped_term + r'\s*[:\-]\s*(?:nahi|na)\b'
        ]
        if any(re.search(pat, target_lower) for pat in hindi_post_patterns):
            return True

        # 3. Hindi / Hinglish pre-position negations (e.g. "koi chest pain nahi", "nahi hai chest pain")
        hindi_pre_patterns = [
            r'\b(?:koi\s+|kuch\s+)?' + escaped_term + r'\s+(?:bhi\s+)?nahi\b',
            r'\b(?:nahi\s+hai|nahi)\s+(?:koi\s+|kuch\s+)?' + escaped_term + r'\b',
            r'\bna\s+to\s+' + escaped_term + r'\b'
        ]
        if any(re.search(pat, target_lower) for pat in hindi_pre_patterns):
            return True

        # 4. English succeeding / post-position negations (e.g. "chest pain: denied", "chest pain is absent")
        eng_post_patterns = [
            escaped_term + r'\s+(?:is\s+)?(?:absent|denied|ruled\s+out|none|zero|0|negative)\b',
            escaped_term + r'\s*[:\-]\s*(?:no|none|negative|denied|nil)\b',
            r'\b(?:remove|exclude|delete)\s+(?:the\s+)?' + escaped_term + r'\b'
        ]
        if any(re.search(pat, target_lower) for pat in eng_post_patterns):
            return True

        # 5. English preceding negations & auxiliary verbs
        adj = r'(?:(?:sudden|severe|mild|moderate|acute|chronic|constant|sharp|dull|throbbing|excruciating)\s+)*'
        eng_pre_patterns = [
            r'\b(?:do\s+not|don\'t|does\s+not|doesn\'t|did\s+not|didn\'t|have\s+not|haven\'t|has\s+not|hasn\'t|had\s+not|hadn\'t|am\s+not|is\s+not|isn\'t|was\s+not|wasn\'t)\s+(?:have|had|experience|experienced|feel|felt|notice|noticed|report|reporting)?\s*(?:a\s+|an\s+|the\s+|any\s+)?(?:signs?\s+of\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:have|has|had)\s+never\s+(?:had|experienced|felt|noticed)\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:no|not|never|without|denies|denied|deny|zero)\s+(?:having\s+)?(?:a\s+|an\s+|the\s+|any\s+)?(?:signs?\s+of\s+)?' + adj + escaped_term + r'\b',
            r'\bno\s+history\s+of\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\bnegative\s+for\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:free\s+of|rules?\s+out|ruled\s+out)\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?\band\s+(?:no\s+)?(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?,\s*(?:no\s+)?(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?\bor\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + escaped_term + r'\b',
        ]
        return any(re.search(pat, target_lower) for pat in eng_pre_patterns)

    def _extract_chief_complaint(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract chief complaint and duration, supporting English and Hindi/Hinglish."""
        result = {}
        
        # Check for patient corrections / changes to chief complaint
        correction_patterns = [
            r'\b(?:actually|correction|mistake|instead\s+of|rather\s+than|change\s+(?:to|my)|not\s+[a-z\s]+?\s+but|no\s+[a-z\s]+?\s+(?:only|just|it\'s|its|hai))\b',
            r'\b(?:nahi\s+hai\s+sirf|nahi\s+hai\s+bas|ke\s+badle|ki\s+jagah)\b'
        ]
        if any(re.search(cp, text_lower) for cp in correction_patterns):
            result["is_correction"] = True

        # Multi-word complaint phrases to check first (most specific)
        complaint_phrases = [
            "chest pain", "shortness of breath", "difficulty breathing",
            "joint pain", "stomach pain", "abdominal pain", "back pain",
            "neck pain", "throat pain", "sore throat", "muscle pain", "body pain",
            "gale mein dard", "gale me dard", "gala kharab", "sir dard", "sar dard",
            "badan dard", "pet dard", "chhati mein dard", "chaati me dard",
            "stomach ache", "tummy ache", "belly ache", "abdominal cramps", "stomach cramps",
            "severe stomach cramps", "cramps in stomach", "lower abdominal pain", "upper abdominal pain",
            "migraine", "severe headache", "head pain", "head hurts", "throbbing head",
            "throbbing headache", "tension headache", "head ache",
            "high fever", "mild fever", "tez bukhar", "feeling feverish", "high temperature"
        ]
        
        # Single-word complaint keywords (fallback)
        complaint_keywords = [
            "pain", "ache", "discomfort", "swelling", "redness", "bleeding",
            "fever", "cough", "fatigue", "weakness", "headache", "nausea", 
            "vomiting", "dizziness", "dizzy", "bukhar", "bukhaar", "zukaam", "zukam",
            "sardi", "khasi", "khaasi", "chakkar", "ulti", "matli", "sardard", "migraine"
        ]
        
        # Standardized chief complaint titles
        standard_names = {
            "fever": "Fever", "bukhar": "Fever", "bukhaar": "Fever", "tez bukhar": "Fever",
            "high fever": "Fever", "mild fever": "Fever", "feeling feverish": "Fever", "high temperature": "Fever",
            "headache": "Headache", "sir dard": "Headache", "sar dard": "Headache", "sardard": "Headache",
            "head pain": "Headache", "head hurts": "Headache", "throbbing head": "Headache",
            "throbbing headache": "Headache", "tension headache": "Headache", "migraine": "Headache / Migraine",
            "severe headache": "Headache", "head ache": "Headache",
            "chest pain": "Chest pain", "chhati mein dard": "Chest pain", "chaati me dard": "Chest pain",
            "shortness of breath": "Shortness of breath", "difficulty breathing": "Difficulty breathing",
            "joint pain": "Joint pain", "muscle pain": "Muscle pain",
            "body pain": "Body ache / Generalized body pain", "badan dard": "Body ache / Generalized body pain",
            "stomach pain": "Abdominal pain", "abdominal pain": "Abdominal pain", "pet dard": "Abdominal pain",
            "stomach ache": "Abdominal pain", "belly ache": "Abdominal pain", "tummy ache": "Abdominal pain",
            "abdominal cramps": "Abdominal pain", "stomach cramps": "Abdominal pain", "severe stomach cramps": "Abdominal pain",
            "cramps in stomach": "Abdominal pain", "lower abdominal pain": "Abdominal pain", "upper abdominal pain": "Abdominal pain",
            "back pain": "Back pain", "neck pain": "Neck pain",
            "throat pain": "Sore throat", "sore throat": "Sore throat", "gale mein dard": "Sore throat", "gale me dard": "Sore throat", "gala kharab": "Sore throat",
            "cough": "Cough", "khasi": "Cough", "khaasi": "Cough",
            "zukaam": "Common cold / Congestion", "zukam": "Common cold / Congestion", "sardi": "Common cold / Congestion",
            "dizziness": "Dizziness", "dizzy": "Dizziness", "chakkar": "Dizziness",
            "nausea": "Nausea", "vomiting": "Vomiting", "ulti": "Vomiting"
        }
        
        # Collect candidate complaint matches
        candidates = []
        for phrase in complaint_phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for m in re.finditer(pattern, text_lower):
                if self.is_term_negated(phrase, text_lower):
                    continue
                # If chest pain, check if historical/resolved
                if phrase in ["chest pain", "chhati mein dard", "chaati me dard"]:
                    hist_patterns = [
                        r'\b(?:yesterday|past|previously|prior|last\s+(?:week|month|year|night)|had|ago)\b.*?\b(?:but\s+not\s+now|no\s+longer|resolved|gone|subsided|improved|disappeared|stopped|not\s+anymore|not\s+now|not\s+today|none\s+today|completely\s+resolved|fully\s+resolved|went\s+away)\b',
                        r'\bchest\s+pain\b.*?\b(?:has\s+resolved|is\s+resolved|resolved|went\s+away|gone|subsided|stopped|not\s+having\s+it\s+now|no\s+longer\s+hurts|not\s+hurting\s+today|completely\s+resolved)\b'
                    ]
                    if any(re.search(hp, text_lower) for hp in hist_patterns):
                        continue
                candidates.append({
                    "start": m.start(),
                    "end": m.end(),
                    "term": phrase,
                    "is_phrase": True
                })

        for kw in complaint_keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            for m in re.finditer(pattern, text_lower):
                if self.is_term_negated(kw, text_lower):
                    continue
                candidates.append({
                    "start": m.start(),
                    "end": m.end(),
                    "term": kw,
                    "is_phrase": False
                })

        # Filter out keyword matches that are covered by an existing multi-word phrase (e.g. "pain" in "body pain")
        phrase_spans = [(c["start"], c["end"]) for c in candidates if c["is_phrase"]]
        valid_candidates = []
        for cand in candidates:
            if not cand["is_phrase"]:
                if any(ps <= cand["start"] and cand["end"] <= pe for ps, pe in phrase_spans):
                    continue
            valid_candidates.append(cand)

        if valid_candidates:
            # Sort order:
            # 1. Non-generic complaints prioritized over bare generic words ("pain", "ache", "discomfort")
            # 2. Earliest appearance in patient message (chief presenting reason)
            # 3. Longer match span first
            valid_candidates.sort(key=lambda c: (
                1 if c["term"] in {"pain", "ache", "discomfort"} else 0,
                c["start"],
                -(c["end"] - c["start"])
            ))
            best = valid_candidates[0]
            term = best["term"]
            result["has_complaint"] = True
            std_title = standard_names.get(term, term.title())
            result["primary_phrase"] = std_title
            result["reason"] = std_title
            phrase_start = best["start"]
            start = max(0, phrase_start - 50)
            end = min(len(text), best["end"] + 100)
            result["text"] = text[start:end].strip()
        
        # Look for duration indicators and extract time value (supporting English and Hindi/Hinglish)
        num_words = {
            "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
            "che": 6, "chhah": 6, "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10
        }
        
        dur_pattern = r'(?:for\s+|from\s+|se\s+)?(\d+|ek|do|teen|chaar|char|paanch|panch|che|chhah|saat|sat|aath|ath|nau|das|one|two|three|four|five|six|seven|eight|nine|ten)\s*(days?|din|weeks?|hafte|hafta|months?|mahine|mahina|hours?|ghante|ghanta|years?|saal)(?!\s*(?:old|of\s+age))\b'
        dur_match = None
        for dm in re.finditer(dur_pattern, text_lower):
            span_start = dm.start()
            pre_ctx = text_lower[max(0, span_start - 15):span_start]
            post_ctx = text_lower[dm.end():min(len(text_lower), dm.end() + 15)]
            if re.search(r'\b(?:i\s+am|i\'m|age)\b', pre_ctx) or re.search(r'\b(?:old|of\s+age)\b', post_ctx):
                continue
            dur_match = dm
            break
        if dur_match:
            raw_n = dur_match.group(1).lower()
            raw_u = dur_match.group(2).lower()
            n_val = int(raw_n) if raw_n.isdigit() else num_words.get(raw_n, 1)
            if any(u in raw_u for u in ["din", "day"]):
                unit_norm = "day" if n_val == 1 else "days"
            elif any(u in raw_u for u in ["haft", "week"]):
                unit_norm = "week" if n_val == 1 else "weeks"
            elif any(u in raw_u for u in ["mahin", "month"]):
                unit_norm = "month" if n_val == 1 else "months"
            elif any(u in raw_u for u in ["ghant", "hour"]):
                unit_norm = "hour" if n_val == 1 else "hours"
            elif any(u in raw_u for u in ["saal", "year"]):
                unit_norm = "year" if n_val == 1 else "years"
            else:
                unit_norm = raw_u
            result["duration_mentioned"] = True
            result["duration_value"] = f"{n_val} {unit_norm}"
        elif re.search(r'\b(?:aaj\s+)?subah\s+se\b', text_lower) or "since this morning" in text_lower:
            result["duration_mentioned"] = True
            result["duration_value"] = "Since this morning"
        elif re.search(r'\bkal\s+se\b', text_lower) or "since yesterday" in text_lower:
            result["duration_mentioned"] = True
            result["duration_value"] = "1 day (since yesterday)"
        elif re.search(r'\bparson\s+se\b', text_lower):
            result["duration_mentioned"] = True
            result["duration_value"] = "2 days"
        else:
            # Check for explicit standalone duration expressions with word boundaries
            standalone_dur = re.search(r'\b(?:for|since|se)\s+(\d+)\s*(days?|din|weeks?|hafte|months?|mahine|hours?|ghante)\b', text_lower)
            if standalone_dur:
                result["duration_mentioned"] = True
                result["duration_value"] = f"{standalone_dur.group(1)} {standalone_dur.group(2)}"
        
        return result if result else {}
    
    def _extract_symptoms(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract symptoms from patient text, distinguishing positive, negated, historical, uncertain, and family-attributed symptoms."""
        result = {}
        
        # Multi-word symptom phrases to match first (most clinically meaningful)
        symptom_phrases = [
            "chest pain", "chest tightness", "shortness of breath", "difficulty breathing", "trouble breathing",
            "sore throat", "dry cough", "productive cough", "runny nose", "stuffy nose", "nasal congestion",
            "urinary difficulty", "joint pain", "muscle pain", "body pain",
            "stomach pain", "abdominal pain", "back pain", "neck pain", "leg pain",
            "arm pain", "heart palpitations", "heart racing", "heart pounding",
            "light sensitivity", "sound sensitivity", "visual changes", "vision changes",
            "blurred vision", "blurry vision", "double vision", "loss of appetite",
            "head hurts", "head is hurting", "head ache", "head pain", "throbbing head", "throbbing headache",
            "stomach ache", "tummy ache", "belly ache", "severe stomach cramps", "abdominal cramps", "stomach cramps",
            "high fever", "mild fever", "tez bukhar", "feeling feverish", "feeling cold",
            "yellowish urine", "yellow urine", "dark urine"
        ]
        
        # Single-word symptoms (fallback if no phrases match)
        symptom_keywords = [
            "pain", "ache", "fever", "cough", "cold", "flu", "headache", "hedache",
            "nausea", "vomiting", "diarrhea", "fatigue", "tired", "weakness",
            "dizziness", "dizzy", "swelling", "rash", "itching", "burning",
            "breathing", "bleeding", "bruising", "sore", "throat", "congestion", 
            "sneeze", "sneezing", "chill", "chills", "shivering", "vision"
        ]
        
        found_symptoms = []
        negative_symptoms = []
        historical_symptoms = []
        uncertain_symptoms = []
        family_symptoms = []
        
        def _get_clause_context(term: str, full_text: str) -> str:
            """Find the specific sentence or clause containing the term."""
            clauses = re.split(r'[\.\;\!\n]|\b(?:but|however|although|whereas)\b', full_text)
            for clause in clauses:
                if term in clause:
                    return clause.strip()
            return full_text
        
        def is_term_family_attributed(term: str, clause: str) -> bool:
            """Check if symptom in clause is attributed to a family member, not patient."""
            fam_pattern = r'\b(?:father|mother|mom|dad|brother|sister|parent|parents|family|uncle|aunt|grand(?:father|mother|parent|ma|pa)|cousin|relative)\b'
            return bool(re.search(fam_pattern, clause))
        
        def is_term_negated(term: str, target_text: str) -> bool:
            """Check if a symptom term is explicitly negated using the unified negation engine."""
            return self.is_term_negated(term, target_text)
        
        def _get_sentence_context(term: str, full_text: str) -> str:
            """Find the full sentence containing the term without splitting on conjunctions."""
            sentences = re.split(r'[\.\;\!\n]', full_text)
            for s in sentences:
                if term in s:
                    return s.strip()
            return full_text
        
        def is_term_historical(term: str, clause: str) -> bool:
            """Check if a symptom is described as past, resolved, or historical."""
            sentence = _get_sentence_context(term, text_lower)
            
            # Check recurrence first: if patient states it returned or is still present, NOT historical
            recurrence_pattern = r'\b(?:came\s+back|returned|recurring|recurrent|still\s+hurts|still\s+having|started\s+again|hurting\s+again|happening\s+again|worse\s+today|persisting|continues?)\b'
            if re.search(recurrence_pattern, sentence):
                return False
            
            hist_patterns = [
                r'\b(?:yesterday|past|previously|prior|last\s+(?:week|month|year|night)|had|resolved|ago)\b.*?(?:but\s+not\s+now|no\s+longer|resolved|gone|subsided|improved|disappeared|stopped|not\s+anymore|not\s+now|not\s+today|none\s+today|completely\s+resolved|fully\s+resolved|went\s+away)\b',
                r'\b' + re.escape(term) + r'\b.*?\b(?:has\s+resolved|is\s+resolved|resolved|went\s+away|gone|subsided|stopped|not\s+having\s+it\s+now|no\s+longer\s+hurts|not\s+hurting\s+today|completely\s+resolved|disappeared|theek\s+ho\s+gaya|thik\s+ho\s+gaya|utar\s+gaya|band\s+ho\s+gaya|chala\s+gaya)\b',
                r'\b(?:had|experienced|felt)\s+(?:a\s+)?' + re.escape(term) + r'\s+(?:yesterday|previously|last\s+(?:week|month|night)|in\s+the\s+past)\b',
                r'\b' + re.escape(term) + r'\s+(?:yesterday|previously|last\s+(?:week|month|night)|in\s+the\s+past|has\s+resolved|is\s+resolved|is\s+gone|subsided)\b',
                r'\b(?:resolved|no\s+longer\s+having|used\s+to\s+have|prior|past)\s+(?:a\s+)?' + re.escape(term) + r'\b',
                r'\b(?:theek\s+ho\s+gaya|thik\s+ho\s+gaya|utar\s+gaya|band\s+ho\s+gaya)\s+(?:hai\s+)?' + re.escape(term) + r'\b',
            ]
            return any(re.search(p, clause) or re.search(p, sentence) for p in hist_patterns)
        
        def is_term_uncertain(term: str, clause: str) -> bool:
            """Check if a symptom is described with uncertainty or possibility."""
            uncert_patterns = [
                r'\b(?:not\s+sure\s+(?:whether|if)|unsure\s+(?:whether|if)|might\s+have|maybe|possibly|wondering\s+if|could\s+have)\s+(?:i\s+have\s+)?(?:a\s+)?' + re.escape(term) + r'\b',
                r'\b(?:not\s+sure|unsure)\s+(?:whether|if)\s+.*?\b' + re.escape(term) + r'\b',
            ]
            return any(re.search(p, clause) for p in uncert_patterns)
        
        # Helper to categorize a term (phrase or keyword)
        def categorize_term(term: str):
            clause = _get_clause_context(term, text_lower)
            # CRITICAL: Check historical/resolved BEFORE negation so resolved symptoms are not marked as denied
            if is_term_historical(term, clause):
                historical_symptoms.append(f"{term} (past/resolved)")
                return
            if is_term_negated(term, text_lower):
                negative_symptoms.append(f"no {term}")
                return
            if is_term_family_attributed(term, clause):
                family_symptoms.append(f"{clause.strip()}")
            elif is_term_uncertain(term, clause):
                uncertain_symptoms.append(f"possible/uncertain {term}")
            else:
                found_symptoms.append(term)
        
        # First, extract multi-word phrases
        for phrase in symptom_phrases:
            if phrase in text_lower:
                categorize_term(phrase)

        # Handle natural urine color phrases (without inventing dysuria or frequency)
        if re.search(r'\b(?:yellow(?:ish)?|dark|tea[- ]colored?)\b.*?\burine\b|\burine\b.*?\b(?:yellow(?:ish)?|dark|tea[- ]colored?)\b', text_lower):
            if "yellowish urine" not in found_symptoms and "dark urine" not in found_symptoms:
                found_symptoms.append("yellowish urine" if "yellow" in text_lower else "dark urine")
        
        # Check for "<body_part> hurts/aching/sore" natural expressions
        body_parts = ["head", "neck", "back", "stomach", "knee", "shoulder", "arm", "leg", "chest", "joint", "muscle"]
        for part in body_parts:
            part_pattern = r'\b' + part + r'\s+(?:hurts?|is\s+hurting|aches?|is\s+aching|is\s+sore)\b'
            if re.search(part_pattern, text_lower):
                pain_phrase = "headache" if part == "head" else f"{part} pain"
                if pain_phrase not in found_symptoms and not any(pain_phrase in p for p in found_symptoms):
                    categorize_term(pain_phrase)
        
        # Check Hindi/Hinglish symptom expressions
        hindi_symptom_mappings = [
            (r'\b(?:saans\s+(?:lene\s+(?:me(?:in)?)?\s+)?(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue|pareshaani)|saans\s+(?:phool|fool|chadh)|dum\s+ghut)\b', "difficulty breathing"),
            (r'\b(?:chhati|chaati|chest)\s+(?:me(?:in)?\s+)?(?:koi\s+|kuch\s+)?(?:dard|pain|dikkat|takleef)\b', "chest pain"),
            (r'\b(?:tez\s+)?(?:bukhar|bukhaar)\b', "fever"),
            (r'\b(?:thand\s+lag\s+rahi|thand\s+lagna|thand\s+hai|thand)\b', "chills"),
            (r'\b(?:gale\s+me(?:in)?\s+dard|gala\s+kharab|gale\s+kharash)\b', "sore throat"),
            (r'\b(?:badan\s+dard|body\s+pain|body\s+bahut\s+pain)\b', "body pain"),
            (r'\b(?:sir\s+dard|sar\s+dard|sardard)\b', "headache"),
            (r'\b(?:body\s+(?:bahut\s+)?hot\s+feel|hot\s+body|body\s+garam|feeling\s+hot)\b', "feeling hot / feverish"),
            (r'\b(?:khasi|khaasi)\b', "cough"),
            (r'\b(?:zukaam|zukam|sardi)\b', "runny nose / cold"),
            (r'\b(?:chakkar|chhakkar)\b', "dizziness"),
            (r'\b(?:ulti|matli)\b', "nausea / vomiting"),
            (r'\b(?:pet\s+dard)\b', "stomach pain"),
        ]
        for pat, eng_term in hindi_symptom_mappings:
            if re.search(pat, text_lower):
                if eng_term not in found_symptoms and not any(eng_term in p for p in found_symptoms) and not any(eng_term in ns for ns in negative_symptoms):
                    categorize_term(eng_term)
        
        # Then extract single-word keywords, but avoid if already captured in phrases
        all_captured_words = [
            p.lower() for p in (
                found_symptoms + 
                [ns.replace("no ", "") for ns in negative_symptoms] + 
                historical_symptoms + 
                uncertain_symptoms + 
                family_symptoms
            )
        ]
        
        for keyword in sorted(symptom_keywords, key=len, reverse=True):
            if keyword in text_lower:
                all_captured_words = [
                    p.lower() for p in (
                        found_symptoms + 
                        [ns.replace("no ", "") for ns in negative_symptoms] + 
                        historical_symptoms + 
                        uncertain_symptoms + 
                        family_symptoms
                    )
                ]
                # Skip if this keyword is already part of a phrase or term we handled
                skip = any(keyword in c for c in all_captured_words)
                if not skip:
                    categorize_term(keyword)
        
        # Extract explicit pertinent negatives (radiation, triggers, sensory changes)
        # 1. Radiation / Spreading denials
        rad_denial_pat = r'\b(?:does\s+not|doesn\'t|did\s+not|not|no)\s+(?:radiat(?:e|ing)?|spread(?:ing)?|travel(?:ing)?)\s+(?:down\s+)?(?:to\s+|into\s+)?(neck\s+or\s+shoulders?|shoulders?\s+or\s+neck|neck|shoulders?|arms?|back|chest)\b'
        rad_match = re.search(rad_denial_pat, text_lower)
        if not rad_match:
            rad_match = re.search(r'\bno\s+(?:radiation|spread|spreading)\s+(?:to\s+|into\s+)?(neck\s+or\s+shoulders?|shoulders?\s+or\s+neck|neck|shoulders?|arms?|back|chest)\b', text_lower)
        if not rad_match:
            rad_match = re.search(r'\bno\s+(neck\s+or\s+shoulder|shoulder\s+or\s+neck|neck|shoulder)\s+(?:radiation|spread|pain|stiffness)\b', text_lower)
        if rad_match:
            target_rad = rad_match.group(1).strip()
            negative_symptoms.append(f"no radiation to {target_rad}")

        # 2. Aggravating factor denials (screen time, stress, exertion)
        agg_denial_pat = r'\b(?:not\s+worse|no\s+worse|doesn\'t\s+get\s+worse|does\s+not\s+worsen|not\s+aggravated|not\s+triggered)\s+(?:with|by|after)\s+(screen\s+time|screens?|stress|light|exertion|movement|food)\b'
        agg_match = re.search(agg_denial_pat, text_lower)
        if not agg_match:
            agg_match = re.search(r'\bno\s+(screen\s+time|stress|exertion)\s+(?:aggravation|trigger|worsening)\b', text_lower)
        if agg_match:
            factor = agg_match.group(1).strip()
            negative_symptoms.append(f"not worse with {factor}")

        # 3. Sensory denials (vision changes, sound sensitivity)
        if re.search(r'\b(?:no|without|denies|negative\s+for)\s+(?:any\s+)?(?:visual\s+changes?|vision\s+changes?|blurred\s+vision|blurry\s+vision|double\s+vision|flashes|spots)\b', text_lower) or \
           re.search(r'\b(?:visual\s+changes?|vision\s+changes?)\s+(?:nahi|absent|denied|none)\b', text_lower) or \
           re.search(r'\b(?:no|without)\s+[a-z\s,]+?(?:,\s*|\bor\s+)(?:any\s+)?(?:visual\s+changes?|vision\s+changes?)\b', text_lower):
            if not any("vision changes" in ns for ns in negative_symptoms):
                negative_symptoms.append("no vision changes")

        if re.search(r'\b(?:no|without|denies)\s+(?:sound\s+sensitivity|sensitivity\s+to\s+sound|phonophobia)\b', text_lower) or \
           re.search(r'\b(?:sound\s+sensitivity)\s+(?:nahi|absent|denied|none)\b', text_lower):
            if not any("sound sensitivity" in ns for ns in negative_symptoms):
                negative_symptoms.append("no sound sensitivity")

        # 4. Gastrointestinal / Nausea denials
        if re.search(r'\b(?:no|without|denies)\s+nausea\b', text_lower) or re.search(r'\bnausea\s+(?:nahi|absent|denied|none)\b', text_lower) or re.search(r'\bneither\s+nausea\b', text_lower):
            if not any("nausea" in ns for ns in negative_symptoms):
                negative_symptoms.append("no nausea")
            found_symptoms = [s for s in found_symptoms if s != "nausea"]

        # Extract location details (compound phrases prioritized first)
        locations = [
            "forehead and both sides", "forehead and temples", "temples and forehead",
            "both sides of head", "both sides and forehead", "both sides",
            "right side", "left side", "bilateral", "frontal",
            "back of head", "forehead", "behind eyes", "temple", "temples",
            "neck and shoulders", "chest", "abdomen", "stomach",
            "lower right", "upper right", "lower left", "upper left",
            "throat", "neck", "lower back", "knee", "shoulder", "shoulders"
        ]
        extracted_location = None
        for loc in locations:
            if loc in text_lower:
                # Never assign a location that is only mentioned in a radiation, spreading, or denied context
                if self.is_term_negated(loc, text_lower):
                    continue
                # Radiation or spreading negation check
                rad_neg = bool(
                    re.search(r'\b(?:does\s+not|doesn\'t|did\s+not|not|no)\s+(?:radiat(?:e|ing)?|spread(?:ing)?|travel(?:ing)?)\s+(?:down\s+)?(?:to\s+|into\s+)?(?:the\s+)?' + re.escape(loc), text_lower)
                    or re.search(r'\bno\s+(?:radiation|spread|spreading)\s+(?:to\s+|into\s+)?(?:the\s+)?' + re.escape(loc), text_lower)
                    or re.search(r'\bno\s+(?:[a-z\s]+?\s+or\s+)?' + re.escape(loc) + r'\s+(?:radiation|spread|pain|stiffness)\b', text_lower)
                )
                if rad_neg:
                    continue
                if loc in ["chest"] and any("chest" in ns for ns in negative_symptoms) and not any("chest" in s for s in found_symptoms):
                    continue
                if loc in ["head", "temple", "forehead", "forehead and both sides"] and any("headache" in ns for ns in negative_symptoms) and not any("headache" in s for s in found_symptoms):
                    continue
                if loc == "throat" and "sore throat" in text_lower:
                    continue
                extracted_location = loc
                break
        
        # Extract severity (including standalone corrections, negations, progression)
        extracted_severity = self._extract_severity_from_text(text_lower)
        
        # Extract character (must NOT be negated, e.g. "no chest tightness")
        characters = ["throbbing", "pounding", "sharp", "stabbing", "dull", "aching", "burning", "pressure", "tightness", "cramping"]
        extracted_character = None
        for ch in characters:
            if re.search(r'\b' + re.escape(ch) + r'\b', text_lower):
                if self.is_term_negated(ch, text_lower):
                    continue
                if ch == "tightness" and (self.is_term_negated("chest tightness", text_lower) or any("tightness" in ns for ns in negative_symptoms)):
                    continue
                extracted_character = ch
                break
        
        # Deterministic per-symptom timeline & context extraction across clauses
        per_symptom_details = {}
        clauses = re.split(r'[\.\;\!\n]', text_lower)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            clause_dur = None
            clause_onset = None
            m_dur = re.search(r'\b(\d+)\s*(days?|weeks?|months?|hours?|din|hafte|mahine)\b', clause)
            if m_dur:
                n_val = int(m_dur.group(1))
                u_map = {"din": "days", "hafte": "weeks", "mahine": "months"}
                u_eng = u_map.get(m_dur.group(2), m_dur.group(2))
                if not u_eng.endswith('s') and n_val != 1:
                    u_eng += 's'
                clause_dur = f"{n_val} {u_eng}"
            if re.search(r'\b(?:kal\s+se|since\s+yesterday)\b', clause):
                clause_onset = "Since yesterday"
            elif re.search(r'\b(?:aaj\s+se|since\s+today|since\s+this\s+morning|subah\s+se)\b', clause):
                clause_onset = "Since today"
            clause_agg = None
            if re.search(r'\b(?:at\s+night|raat\s+ko|worse\s+at\s+night)\b', clause):
                clause_agg = "Worse at night"
            if clause_dur or clause_onset or clause_agg:
                for fs in found_symptoms:
                    fs_l = fs.lower()
                    if fs_l in clause or (fs_l == "chills" and any(w in clause for w in ["thand", "kapkapi", "chills", "shivering"])) or (fs_l == "dry cough" and "cough" in clause):
                        entry = per_symptom_details.setdefault(fs_l, {"status": "reported"})
                        if clause_dur:
                            entry["duration"] = clause_dur
                        if clause_onset:
                            entry["onset"] = clause_onset
                        if clause_agg:
                            entry["aggravating_factors"] = clause_agg
        
        # Extract progression
        progression_words = ["worsening", "getting worse", "constant", "intermittent", "comes and goes", "improving"]
        extracted_progression = None
        for pr in progression_words:
            if pr in text_lower:
                extracted_progression = pr
                break
        
        # Extract onset
        extracted_onset = None
        if "sudden" in text_lower:
            extracted_onset = "sudden"
        elif "gradual" in text_lower or "slowly" in text_lower:
            extracted_onset = "gradual"

        # Extract time-of-day timing (morning, afternoon, evening, night)
        extracted_timing = None
        timing_patterns = [
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?evenings?\b|\bshaam\s+(?:ko|mein)\b', "evening"),
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?mornings?\b|\bsubah\s+(?:ko|mein)\b', "morning"),
            (r'\b(?:in\s+the\s+|at\s+|every\s+|during\s+(?:the\s+)?)?afternoons?\b|\bdopahar\s+(?:ko|mein)\b', "afternoon"),
            (r'\b(?:at\s+|every\s+|during\s+(?:the\s+)?)?nights?\b|\braat\s+(?:ko|mein)\b', "night"),
        ]
        for t_pat, t_val in timing_patterns:
            if re.search(t_pat, text_lower) and not re.search(r'\b(?:not|no|never)\s+(?:in\s+the\s+)?' + t_val + r'\b', text_lower):
                extracted_timing = t_val
                break

        # Extract frequency (recurring, daily, occasional, intermittent, constant)
        extracted_frequency = None
        for freq_kw in ["recurring", "recurrent", "every day", "daily", "intermittent", "constant", "occasional", "frequently", "often"]:
            if re.search(r'\b' + re.escape(freq_kw) + r'\b', text_lower):
                extracted_frequency = "recurring" if freq_kw == "recurrent" else freq_kw
                break
        
        # Extract severity modifiers if present
        severity_pattern = r'(mild|moderate|severe|sharp|dull|constant|intermittent|persistent|acute|chronic|sudden|gradual|worse|better)\s+(.*?(?:pain|ache|symptom|discomfort))'
        severity_matches = re.findall(severity_pattern, text_lower)
        
        # Extract aggravating/alleviating factors
        extracted_aggravating = None
        aggravating_keywords = ["worse when", "worse with", "worsens when", "aggravated by", "triggered by"]
        for keyword in aggravating_keywords:
            if keyword in text_lower:
                pattern_str = keyword + r'\s+(.+?)(?:\.|,|;|$)'
                match = re.search(pattern_str, text_lower)
                if match:
                    extracted_aggravating = match.group(1).strip()
                    break
        
        extracted_alleviating = None
        alleviating_keywords = ["better when", "better with", "improves when", "relieved by", "helped by"]
        for keyword in alleviating_keywords:
            if keyword in text_lower:
                pattern_str = keyword + r'\s+(.+?)(?:\.|,|;|$)'
                match = re.search(pattern_str, text_lower)
                if match:
                    extracted_alleviating = match.group(1).strip()
                    break
        
        has_symptoms = bool(found_symptoms or negative_symptoms or historical_symptoms or uncertain_symptoms or family_symptoms)
        has_details = bool(extracted_location or extracted_severity or extracted_character or extracted_progression or extracted_onset or extracted_timing or extracted_frequency or extracted_aggravating or extracted_alleviating)
        
        if has_symptoms or has_details:
            result["symptoms_found"] = found_symptoms
            result["negative_symptoms"] = negative_symptoms
            result["historical_symptoms"] = historical_symptoms
            result["uncertain_symptoms"] = uncertain_symptoms
            if family_symptoms:
                result["family_symptoms"] = family_symptoms
            result["symptom_count"] = len(found_symptoms)
            result["raw_text"] = text[:300]
            
            if extracted_location:
                result["location"] = extracted_location
            if extracted_severity:
                result["severity"] = extracted_severity
            if extracted_character:
                result["character"] = extracted_character
            if extracted_progression:
                result["progression"] = extracted_progression
            if extracted_onset:
                result["onset"] = extracted_onset
            if extracted_timing:
                result["timing"] = extracted_timing
            if extracted_frequency:
                result["frequency"] = extracted_frequency
            if severity_matches:
                result["severity_info"] = severity_matches
            if extracted_aggravating:
                result["aggravating_factor"] = extracted_aggravating
            if extracted_alleviating:
                result["alleviating_factor"] = extracted_alleviating
            if per_symptom_details:
                result["per_symptom_details"] = per_symptom_details
        
        return result if result else {}
    
    def _extract_severity_from_text(self, text_lower: str) -> Optional[str]:
        """
        Extract symptom severity rating or description, accurately handling
        corrections, negations, numerical scales, and progression.
        
        Examples:
        - "Actually, it has become severe" -> "severe"
        - "I meant moderate, not severe" -> "moderate"
        - "It is mild, not moderate" -> "mild"
        - "8/10", "8 out of 10" -> "8/10"
        - "Was severe yesterday, now mild" -> "mild"
        - "Getting worse, now excruciating" -> "severe"
        """
        # 1. Numerical rating: e.g. "8/10", "8 out of 10"
        sev_num = re.search(r'\b([1-9]|10)\s*(?:/|\s*out of\s*)\s*10\b', text_lower)
        if sev_num:
            return f"{sev_num.group(1)}/10"
        
        # 2. Check for explicit correction / intended replacement patterns:
        # e.g., "meant moderate, not severe", "intended mild, not severe"
        meant_match = re.search(r'\b(?:meant|mean|intended)\s+(mild|moderate|severe|excruciating)\b', text_lower)
        if meant_match:
            val = meant_match.group(1)
            return "severe" if val == "excruciating" else val
        
        # e.g., "(mild|moderate|severe), not (mild|moderate|severe)"
        not_after_match = re.search(r'\b(mild|moderate|severe|excruciating)\s*,?\s+not\s+(?:mild|moderate|severe|excruciating)\b', text_lower)
        if not_after_match:
            val = not_after_match.group(1)
            return "severe" if val == "excruciating" else val
        
        # e.g., "not (mild|moderate|severe), (?:it's|it is|but)?\s*(mild|moderate|severe)"
        not_before_match = re.search(r'\bnot\s+(?:mild|moderate|severe|excruciating)\s*,?\s*(?:it\'s\s+|it\s+is\s+|but\s+)?(mild|moderate|severe|excruciating)\b', text_lower)
        if not_before_match:
            val = not_before_match.group(1)
            return "severe" if val == "excruciating" else val
        
        # 3. Check for current/now/become progression:
        # e.g., "now severe", "has become severe", "is actually severe"
        now_match = re.search(r'\b(?:now|become|became|actually|currently|getting)\s+(?:very\s+)?(mild|moderate|severe|excruciating)\b', text_lower)
        if now_match:
            val = now_match.group(1)
            return "severe" if val == "excruciating" else val
        
        # 4. Standard mentions respecting direct negation ("not severe", "not mild")
        # Check severe / excruciating first (if not negated)
        if ("severe" in text_lower or "excruciating" in text_lower):
            if not re.search(r'\b(?:not|isn\'t|is\s+not|never|no\s+longer)\s+(?:very\s+)?(?:severe|excruciating)\b', text_lower):
                return "severe"
        
        # Check moderate (if not negated)
        if "moderate" in text_lower:
            if not re.search(r'\b(?:not|isn\'t|is\s+not|never|no\s+longer)\s+moderate\b', text_lower):
                return "moderate"
        
        # Check mild (if not negated)
        if "mild" in text_lower:
            if not re.search(r'\b(?:not|isn\'t|is\s+not|never|no\s+longer)\s+mild\b', text_lower):
                return "mild"
        
        return None
    
    def _extract_medical_history(self, text: str, text_lower: str) -> Dict[str, Any]:
        """
        Extract medical history information with improved phrase matching and negative answer detection.
        
        Correctly handles:
        - "No I don't have chronic conditions"
        - "No previous surgeries"
        - "I have no medical history"
        - "Never had any operations"
        """
        result = {}
        
        # Multi-word condition phrases (most specific - check first)
        condition_phrases = [
            "high blood pressure", "hypertension",
            "type 1 diabetes", "type 2 diabetes", "diabetes",
            "asthma", "chronic asthma",
            "heart disease", "heart attack", "heart condition", "cardiac",
            "coronary artery disease", "stroke", "mini stroke",
            "arthritis", "rheumatoid arthritis", "osteoarthritis",
            "thyroid disease", "hypothyroidism", "hyperthyroidism",
            "depression", "anxiety disorder", "anxiety",
            "migraine", "chronic migraine",
            "kidney disease", "kidney failure", "renal failure",
            "liver disease", "cirrhosis",
            "peptic ulcer", "gastric ulcer", "ulcer",
            "gastroesophageal reflux", "gerd", "acid reflux",
            "cancer", "tumor", "malignancy",
            "chronic obstructive pulmonary disease", "copd",
            "sleep apnea", "obstructive sleep apnea"
        ]
        
        # FIRST: Check for explicit negative statements about medical history
        # This must be done BEFORE looking for positive conditions
        negative_patterns = [
            r'\b(?:no|never|don\'t|do not|haven\'t|have not)\s+(?:chronic|medical|condition|disease|history|surgery|operation|past)\b',
            r'\b(?:no|never)\s+(?:surgeries|operations|procedures)\b',
            r'\b(?:no)\s+(?:i\s+)?don\'t\s+have\b',
            r'\b(?:i\s+)?have\s+no\s+(?:chronic|medical|history)\b',
            r'\bnever\s+(?:had|undergone|had)\s+(?:surgery|operation|procedure)\b',
            r'\b(?:no)\s+(?:previous|past)\s+(?:surgeries|operations|medical)\b',
            r'\b(?:pehle\s+se\s+)?(?:koi\s+)?bimari\s+nahi\b',
            r'\bpehle\s+se\s+koi\s+(?:rog|bimari|takleef)\s+nahi\b',
            r'\bkoi\s+(?:purani\s+)?bimari\s+nahi\s+hai\b',
            r'\bkuch\s+nahi\s+hai\b',
            r'\bkoi\s+history\s+nahi\b',
            r'\bkoi\s+bimari\s+nahi\b',
            r'\bbimari\s+nahi\s+hai\b',
        ]
        
        # First try regex patterns
        for pattern in negative_patterns:
            if re.search(pattern, text_lower):
                result["explicitly_none"] = True
                result["conditions_denied"] = True
                self.logger.debug(f"Detected explicit negative medical history (regex): {pattern}")
                return result
        
        # Fallback: simple substring matching for common natural language patterns
        negative_substrings = [
            "no i don",  # "no i don't have"
            "no, i don",  # "No, I don't have"
            "never had",  # "Never had any surgeries"
            "not had",  # "not had any surgery"
            "don't have any",  # "don't have any medical"
            "do not have any",  # "do not have any"
            "have no ",  # "I have no medical history"
            "have not",  # "have not had"
            "no previous",  # "No previous surgeries"
            "no past",  # "no past surgeries"
            "no chronic",  # "no chronic conditions"
            "no medical",  # "no medical history"
            "no surgery",  # "no surgeries"
            "no operation",  # "no operations"
            "bimari nahi",  # Hindi: no illness
            "koi bimari",
            "pehle se koi",
            "kuch nahi hai"
        ]
        
        if any(substring in text_lower for substring in negative_substrings):
            result["explicitly_none"] = True
            result["conditions_denied"] = True
            self.logger.debug(f"Detected explicit negative medical history (substring match)")
            return result
        
        # SECOND: Look for positive vs negated specific conditions
        found_conditions = []
        denied_conditions = []
        
        # Extract multi-word phrases first (more specific)
        for phrase in condition_phrases:
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                if self.is_term_negated(phrase, text_lower):
                    if phrase not in denied_conditions:
                        denied_conditions.append(phrase)
                else:
                    if phrase not in found_conditions:
                        found_conditions.append(phrase)
        
        if found_conditions:
            result["conditions_mentioned"] = found_conditions
        if denied_conditions:
            result["denied_conditions"] = denied_conditions
            if not found_conditions:
                result["conditions_denied"] = True
        
        # Look for surgery/hospitalization keywords
        has_surgery_mention = False
        if any(word in text_lower for word in ["surgery", "operation", "surgical", "hospital", "admitted", "hospitalized", "procedure"]):
            # Check if preceded by negative words
            text_words = text_lower.split()
            surgery_keywords = ["surgery", "operation", "surgical", "hospital", "admitted", "hospitalized", "procedure"]
            
            for i, word in enumerate(text_words):
                if any(surgery_keyword in word for surgery_keyword in surgery_keywords):
                    # Look back for negation in previous few words
                    context_words = text_words[max(0, i-3):i]
                    if any(neg in context_word for neg in ["never", "no", "not", "don't", "didn't", "haven't", "do not", "did not"] 
                           for context_word in context_words):
                        # Negative context - skip
                        continue
                    else:
                        has_surgery_mention = True
                    break
            
            if has_surgery_mention:
                result["history_mentioned"] = True
                result["raw_text"] = text[:300]
        
        # Look for family history keywords
        if any(word in text_lower for word in ["family", "mother", "father", "sibling", "relative", "brother", "sister", "parents", "grandmother", "grandfather", "grandparent"]):
            result["family_history_mentioned"] = True
            family_clauses = []
            clauses = re.split(r'[\.\;\!\n]|\b(?:but|however|although|whereas)\b', text)
            for c in clauses:
                c_clean = c.strip()
                if any(w in c_clean.lower() for w in ["family", "mother", "father", "sibling", "relative", "brother", "sister", "parents", "grandmother", "grandfather", "grandparent"]):
                    family_clauses.append(c_clean)
            if family_clauses:
                result["family_history_items"] = family_clauses
        
        return result if result else {}
    
    def _extract_medications(self, text: str, text_lower: str) -> Dict[str, Any]:
        """
        Extract medication information with improved negation detection.
        
        Correctly handles:
        - "No, I'm not taking any medications"
        - "I don't take any medications"
        - "No current medications"
        """
        result = {}
        
        # FIRST: Check for explicit negative responses about medications (English + Hindi/Hinglish)
        medication_negative_patterns = [
            r'\b(?:no|not|don\'t|do not|haven\'t|have not)\s+(?:taking|on|taking any|current|active)\b',
            r'\b(?:not|no)\s+(?:medications?|medicines?|meds?|pills?|drugs?|prescriptions?)\b',
            r'\bnone\b',
            r'\b(?:none|no)\s+(?:at all|currently)\b',
            r'\b(?:koi|kisi)\s+(?:bhi\s+)?(?:dawai|dava|medicine|medication|tablet|goli|capsule)\s+(?:nahi|na)\b',
            r'\b(?:dawai|dava|medicine|medication|tablet|goli)\s+(?:nahi\s+le|nahi\s+kha|nahi\s+chal)\b',
            r'\b(?:abhi\s+)?(?:koi\s+)?(?:dawai|dava|medicine)\s+nahi\b',
        ]
        
        for pattern in medication_negative_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result["explicitly_none"] = True
                self.logger.debug(f"Detected explicit medication denial: {pattern}")
                return result
        
        # SECOND: Common medication keywords (partial list)
        # FIX 1.7.4: Removed "medication" and "medicine" (too generic, cause false positives in negation contexts)
        medication_keywords = [
            "aspirin", "ibuprofen", "paracetamol", "metformin", "lisinopril",
            "atorvastatin", "amoxicillin", "penicillin", "insulin", "metoprolol",
            "levothyroxine", "omeprazole", "antibiotic", "steroid",
            "albuterol", "inhaler", "sumatriptan", "ventolin", "tylenol", "advil",
            "prednisone", "gabapentin", "sertraline", "losartan", "amlodipine",
            "cetirizine", "loratadine", "antihistamine", "bronchodilator",
            "tablet", "pill", "injection", "dose"
        ]
        
        found_meds = []
        for med in medication_keywords:
            if med in ["tablet", "pill", "injection", "dose"]:
                continue
            pattern = r'\b' + re.escape(med) + r'\b'
            if re.search(pattern, text_lower):
                post_dose = re.search(r'\b' + re.escape(med) + r'\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml))\b', text, re.IGNORECASE)
                pre_dose = re.search(r'\b(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml))\s+(?:of\s+)?' + re.escape(med) + r'\b', text, re.IGNORECASE)
                if post_dose:
                    dosage_str = re.sub(r'(\d+)\s+(mg|g|mcg|ml)', r'\1\2', post_dose.group(1).strip())
                    formatted_med = f"{med.title()} {dosage_str}"
                elif pre_dose:
                    dosage_str = re.sub(r'(\d+)\s+(mg|g|mcg|ml)', r'\1\2', pre_dose.group(1).strip())
                    formatted_med = f"{med.title()} {dosage_str}"
                else:
                    formatted_med = med.title()
                if formatted_med not in found_meds:
                    found_meds.append(formatted_med)
        
        if found_meds:
            result["medications_mentioned"] = found_meds
        
        # Look for dosage indicators
        if any(word in text_lower for word in ["mg", "tablet", "dose", "times", "daily", "weekly"]):
            result["dosage_mentioned"] = True
        
        # THIRD: Fallback simple negation patterns (if regex didn't match)
        no_med_substrings = [
            "no medication", "not taking", "no pills", "don't take", "do not take",
            "not currently taking", "am not taking", "are not taking", "no current",
            "none at all", "none currently", "not on any", "no active", "not on",
            "koi dawai nahi", "dawai nahi le raha", "dawai nahi le rahi", "dava nahi",
            "koi medicine nahi", "medicine nahi le"
        ]
        if any(substring in text_lower for substring in no_med_substrings):
            result["explicitly_none"] = True
        
        return result if result else {}
    
    def _extract_allergies(self, text: str, text_lower: str) -> Dict[str, Any]:
        """
        Extract allergy information, including detection of symptom-vs-allergy confusion.
        
        TASK 1.9.5 FIXES:
        - Detect symptoms (runny nose, cough, fever) mentioned in allergy context
        - Recognize negative responses ("No, I don't have allergies", Hindi/Hinglish denials)
        - Distinguish allergy from symptom for clarification question
        """
        result = {}
        
        # FIRST: Check for explicit negative responses about allergies (English + Hindi/Hinglish)
        allergy_negative_patterns = [
            r'\b(?:no|nope|nah|never|don\'t|do not|don\'t have|don\'t have any|do not have any|haven\'t|have not)\b.*\b(?:allerg|reaction|drug|food|environmental)',
            r'\bnone\b',
            r'\bnot allergic\b',
            r'\bno (?:known )?(?:drug|food|environmental|other)?\s*(?:or\s+(?:drug|food|environmental|other)\s+)*allerg(?:y|ies)\b',
            r'\bdeny.*allerg',
            r'\b(?:koi|kisi)\s+(?:bhi\s+)?(?:known\s+)?allerg(?:y|ies)\s+(?:nahi|na)\b',
            r'\ballerg(?:y|ies)\s+(?:nahi|na)\s*(?:hai|hain)?\b',
            r'\b(?:kisi|koi)\s+(?:dawai|dava|khane|food|cheez|drug).*?\ballerg(?:y|ies)\s+nahi\b',
        ]
        
        for pattern in allergy_negative_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result["explicitly_none"] = True
                self.logger.debug(f"Detected explicit allergy denial: {pattern}")
                return result
        
        # SECOND: Allergy keywords
        allergy_keywords = [
            "penicillin", "aspirin", "ibuprofen", "peanut", "nut", "seafood",
            "shellfish", "dairy", "lactose", "gluten", "egg", "pollen",
            "dust", "mold", "latex", "sulfa"
        ]
        
        found_allergies = [allergy for allergy in allergy_keywords if allergy in text_lower]
        
        if found_allergies:
            result["allergies_mentioned"] = found_allergies
        
        # THIRD: Look for allergy reactions
        reaction_keywords = ["rash", "swelling", "itching", "breathing", "throat", "anaphylaxis"]
        reactions = [r for r in reaction_keywords if r in text_lower]
        if reactions:
            result["reactions_mentioned"] = reactions
        
        # FOURTH: TASK 1.9.5 FIX - Detect symptoms mistakenly described as allergies
        # Only when patient is answering in an allergy context
        allergy_context = any(w in text_lower for w in ["allerg", "allergic", "reaction"])
        if allergy_context:
            symptom_keywords = [
                "runny nose", "stuffy nose", "congestion", "sneeze", "cough", "fever",
                "chills", "sweat", "fatigue", "tired", "weakness", "ache", "pain",
                "muscle pain", "joint pain", "headache", "throat pain", "chest pain",
                "shortness of breath", "difficulty breathing", "nausea", "vomiting",
                "diarrhea", "constipation", "bloating", "stomach pain"
            ]
            
            found_symptoms = []
            for symptom in symptom_keywords:
                if symptom in text_lower:
                    found_symptoms.append(symptom)
            
            if found_symptoms and not found_allergies and not reactions:
                # Patient described symptoms in response to allergy question
                result["symptom_not_allergy"] = True
                result["symptoms_mentioned"] = found_symptoms
                self.logger.debug(f"Detected symptoms in allergy answer: {found_symptoms}")
                return result
        
        # FIFTH: Simple negation detection as fallback (if regex patterns didn't match)
        simple_negations = [
            "no allergies", "no allergy", "not allergic", "no reaction", 
            "no drug", "no food", "none", "no environmental",
            "allergy nahi", "koi allergy nahi", "koi known allergy nahi", "kisi cheez se allergy nahi"
        ]
        if any(neg in text_lower for neg in simple_negations):
            result["explicitly_none"] = True
        
        return result if result else {}
    
    def _extract_lifestyle(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract lifestyle information."""
        result = {}
        
        # Smoking status
        if "smoke" in text_lower or "smoker" in text_lower:
            result["smoking_mentioned"] = True
            if "never" in text_lower:
                result["smoking_status"] = "never"
            elif "current" in text_lower or "currently" in text_lower:
                result["smoking_status"] = "current"
            elif "former" in text_lower or "quit" in text_lower:
                result["smoking_status"] = "former"
        
        # Alcohol use
        if "alcohol" in text_lower or "drink" in text_lower:
            result["alcohol_mentioned"] = True
            if "not" in text_lower or "no" in text_lower or "never" in text_lower:
                result["alcohol_status"] = "none"
            elif "occasional" in text_lower or "sometimes" in text_lower:
                result["alcohol_status"] = "occasional"
            else:
                result["alcohol_status"] = "regular"
        
        # Exercise
        if any(word in text_lower for word in ["exercise", "gym", "walk", "run", "sport", "activity"]):
            result["exercise_mentioned"] = True
        
        # Diet
        if any(word in text_lower for word in ["diet", "food", "eat", "vegetarian", "vegan"]):
            result["diet_mentioned"] = True
        
        # Recent travel
        if any(word in text_lower for word in ["travel", "traveled", "travelled", "trip", "flight", "abroad"]):
            if self.is_term_negated("travel", text_lower) or re.search(r'\bno\s+(?:recent\s+)?(?:[\w\s]+?\s+or\s+)?travel\b', text_lower):
                result["recent_travel"] = "patient_denied"
                result["recent_travel_status"] = "patient_denied"
            else:
                result["recent_travel"] = text.strip()
                result["recent_travel_status"] = "provided"
        
        # Sick contacts
        if any(word in text_lower for word in ["sick contact", "sick contacts", "contact with someone", "exposed to", "someone sick", "anyone sick"]):
            if self.is_term_negated("sick contact", text_lower) or re.search(r'\bno\s+(?:recent\s+)?sick\s+contacts?\b', text_lower):
                result["sick_contacts"] = "patient_denied"
                result["sick_contacts_status"] = "patient_denied"
            else:
                result["sick_contacts"] = text.strip()
                result["sick_contacts_status"] = "provided"
        
        return result if result else {}
    
    def _extract_vitals(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract vital signs if mentioned, or detect if patient hasn't measured them."""
        result = {}
        
        # Check for explicit denial of vitals measurement (must be specific to vitals/measurements)
        denial_patterns = [
            r'\b(?:haven\'t|have\s+not|didn\'t|did\s+not|never)\s+(?:measured?|checked?|taken|tested)\s*(?:any\s+)?(?:vitals?|temp(?:erature)?|bp|blood\s+pressure|heart\s+rate|pulse)?\b',
            r'\b(?:not|no)\s+(?:measured|checked|taken)\b',
            r'\b(?:don\'t|do\s+not)\s+(?:know|have)\s+(?:my\s+)?(?:temp(?:erature)?|vitals?|bp|thermometer)\b',
            r'\b(?:check\s+nahi\s+kiya|nahi\s+check\s+kiya|measure\s+nahi\s+kiya|naapa\s+nahi|nahi\s+naapa|nahi\s+kiya\s+check)\b',
            r'\b(?:nahi\s+dekha|thermometer\s+nahi\s+hai|check\s+nahi\s+kara)\b',
            r'\b(?:no|none|haven\'t)\s+(?:vitals?|thermometer|reading)\b'
        ]
        
        has_explicit_denial = any(re.search(pattern, text_lower) for pattern in denial_patterns)
        
        # Blood pressure pattern: e.g., "120/80"
        bp_pattern = r'(\d{2,3})/(\d{2,3})\s*(?:mmHg)?'
        bp_matches = re.findall(bp_pattern, text)
        if bp_matches:
            result["blood_pressure"] = f"{bp_matches[0][0]}/{bp_matches[0][1]} mmHg"
        
        # Temperature pattern: Match numbers that are likely temperatures
        # Look for: "temperature 102", "temp 102", "102 F", "102 degree", "102°F", "fever 102", etc.
        # Requires either: vitals keyword before number, or F/C/degree after
        temp_pattern = r'(?:temperature|temp|fever|have|is)\s+(\d{2,3}(?:\.\d+)?)\s*(?:degrees?)?(?:\s*[fcFC])?|(\d{2,3}(?:\.\d+)?)\s*(?:°\s*[fcFC]|degrees?\s*[fcFC]?|[fcFC]\b)'
        temp_matches = re.findall(temp_pattern, text_lower)
        
        if temp_matches:
            for match_pair in temp_matches:
                # match_pair is a tuple because of the regex groups
                match = match_pair[0] if match_pair[0] else match_pair[1]
                if not match:
                    continue
                try:
                    temp_value = float(match)
                    # Basic validation: reasonable temperature range
                    # Fahrenheit: ~95-106°F for live patients, Celsius: ~35-41°C
                    if (95 <= temp_value <= 106) or (35 <= temp_value <= 41):
                        # Store as string, preserve original format (int if no decimal, float if decimal)
                        if '.' in match:
                            result["temperature"] = match  # Preserve decimal format
                        else:
                            result["temperature"] = match  # Preserve integer format
                        result["temperature_unit"] = "celsius" if temp_value < 50 else "fahrenheit"
                        break
                except (ValueError, IndexError):
                    continue
        
        # Heart rate pattern: e.g., "72 bpm"
        if any(word in text_lower for word in ["heart rate", "bpm", "beats"]):
            hr_pattern = r'(\d{2,3})\s*(bpm|beats)'
            hr_matches = re.findall(hr_pattern, text_lower)
            if hr_matches:
                result["heart_rate"] = f"{hr_matches[0][0]} bpm"
        
        # SpO2 pattern: e.g., "98%"
        if "oxygen" in text_lower or "spo2" in text_lower or "o2" in text_lower:
            o2_pattern = r'(\d{2,3})%'
            o2_matches = re.findall(o2_pattern, text)
            if o2_matches:
                result["oxygen_saturation"] = f"{o2_matches[0]}%"
        
        if has_explicit_denial and not (result.get("blood_pressure") or result.get("temperature") or result.get("heart_rate") or result.get("oxygen_saturation")):
            result["explicitly_not_measured"] = True
        
        return result if result else {}
    
    def _extract_demographics(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Extract demographic information if explicitly provided."""
        result = {}
        
        # Age pattern: e.g., "42 years old", "I'm 45", "age 60", "28 yo"
        age_patterns = [
            r'(\d{1,3})\s*(?:years?\s*old|yo|y\.o\.|years?\s*of\s*age|saal|varsh)\b',
            r'\b(?:i\s+am|i\'m)\s+(\d{1,2})\s*(?:years?\s*old)?\b',
            r'\bage(?:\s+is|\s*:\s*|\s+of\s+)?\s*(\d{1,3})\b',
            r'(?:age|year|old)\s*(?:of\s)?(\d{1,3})'
        ]
        for pat in age_patterns:
            m = re.search(pat, text_lower)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 < val < 130:
                        result["age"] = val
                        break
                except (ValueError, IndexError):
                    continue
        
        return result if result else {}
    
    def _update_patient_data(self, extracted: Dict[str, Any]) -> None:
        """
        Update patient data with newly extracted information.
        Preserves existing information if not mentioned in new extraction.
        
        TASK 1.9.7 FIX: Handle contradictions appropriately
        - When patient explicitly denies previous information, mark for clarification
        - Clear lists when explicit denial follows prior positive answer
        - Track contradictions without silently overwriting
        
        Args:
            extracted: Dictionary of newly extracted information
        """
        if "chief_complaint" in extracted and extracted["chief_complaint"]:
            cc_data = extracted["chief_complaint"]
            reason = (
                cc_data.get("reason") or
                cc_data.get("primary_phrase") or
                cc_data.get("text")
            )
            if reason:
                if self.patient_data["chief_complaint"]["reason"] is None or cc_data.get("is_correction"):
                    self.patient_data["chief_complaint"]["reason"] = reason
                    self.patient_data["chief_complaint"]["status"] = "provided"
            if "duration_mentioned" in cc_data or cc_data.get("duration_value"):
                duration_value = cc_data.get("duration_value") or cc_data.get("duration")
                if duration_value:
                    self.patient_data["chief_complaint"]["duration_mentioned"] = True
                    self.patient_data["chief_complaint"]["duration"] = duration_value
                    cc_key = (self.patient_data["chief_complaint"].get("reason") or "").strip().lower()
                    if cc_key and cc_key not in ("not provided", "unknown", "none"):
                        psd_cc = self.patient_data["symptom_details"].setdefault("per_symptom_details", {}).setdefault(cc_key, {"status": "reported"})
                        if not psd_cc.get("duration"):
                            psd_cc["duration"] = duration_value
        
        if "symptoms" in extracted and extracted["symptoms"]:
            s_data = extracted["symptoms"]
            if s_data.get("symptoms_found"):
                # Accumulate symptoms and normalize/deduplicate
                new_symptoms = s_data["symptoms_found"]
                self.patient_data["symptoms"]["list"] = normalize_symptoms(
                    self.patient_data["symptoms"]["list"] + new_symptoms
                )
                self.patient_data["symptoms"]["status"] = "provided"
            if s_data.get("negative_symptoms"):
                new_negs = s_data["negative_symptoms"]
                self.patient_data["symptom_details"]["negative_symptoms"] = list(dict.fromkeys(
                    self.patient_data["symptom_details"]["negative_symptoms"] + new_negs
                ))
                # Check for contradiction: did patient previously report this symptom as positive?
                for neg in new_negs:
                    clean_neg = neg.replace("no ", "").strip()
                    if clean_neg in self.patient_data["symptoms"]["list"] or clean_neg.title() in self.patient_data["symptoms"]["list"]:
                        contradiction = {
                            "category": "symptom_denied",
                            "field": "symptoms",
                            "item": clean_neg,
                            "note": f"Patient previously reported '{clean_neg}' but now explicitly denies it."
                        }
                        if contradiction not in self.patient_data.setdefault("contradictions", []):
                            self.patient_data["contradictions"].append(contradiction)
                            self.logger.warning(f"Contradiction detected: symptom '{clean_neg}' previously reported, now denied")
            
            # Ensure negated symptoms are never kept in positive symptoms list
            neg_clean = set(s.replace("no ", "").strip().lower() for s in self.patient_data["symptom_details"]["negative_symptoms"])
            expanded_negs = set(neg_clean)
            for nc in neg_clean:
                canon = CANONICAL_SYMPTOMS.get(nc)
                if canon:
                    expanded_negs.add(canon.lower())
                for syn, c in CANONICAL_SYMPTOMS.items():
                    if c.lower() == nc or syn == nc:
                        expanded_negs.add(syn.lower())
                        expanded_negs.add(c.lower())

            self.patient_data["symptoms"]["list"] = [
                s for s in self.patient_data["symptoms"]["list"]
                if not any(nc == s.lower() or nc in s.lower() or s.lower() in nc for nc in expanded_negs)
            ]
            self.patient_data["symptoms"]["list"] = normalize_symptoms(self.patient_data["symptoms"]["list"])

            # If current chief complaint was negated, clear it
            current_cc = (self.patient_data["chief_complaint"].get("reason") or "").lower()
            if current_cc and any(nc == current_cc or nc in current_cc or current_cc in nc for nc in expanded_negs):
                self.logger.info(f"Chief complaint '{current_cc}' was negated by patient. Clearing chief complaint.")
                self.patient_data["chief_complaint"]["reason"] = None
                self.patient_data["chief_complaint"]["status"] = "unknown"

            # Fallback chief complaint if missing but positive symptoms exist
            if not self.patient_data["chief_complaint"].get("reason") and self.patient_data["symptoms"]["list"]:
                self.patient_data["chief_complaint"]["reason"] = self.patient_data["symptoms"]["list"][0].title()
                self.patient_data["chief_complaint"]["status"] = "provided"

            # Ensure positive chief complaint reason is recorded in positive symptoms list
            cc_reason = (self.patient_data["chief_complaint"].get("reason") or "").strip().lower()
            if cc_reason and cc_reason not in ["not provided", "unknown", "none"]:
                canon_cc = CANONICAL_SYMPTOMS.get(cc_reason, cc_reason)
                if not any(nc == canon_cc or nc in canon_cc for nc in expanded_negs):
                    if canon_cc not in self.patient_data["symptoms"]["list"]:
                        self.patient_data["symptoms"]["list"].append(canon_cc)
            self.patient_data["symptoms"]["list"] = normalize_symptoms(self.patient_data["symptoms"]["list"])

            # Track historical symptoms
            if s_data.get("historical_symptoms"):
                self.patient_data["symptom_details"]["historical_symptoms"] = list(set(
                    self.patient_data["symptom_details"].get("historical_symptoms", []) + s_data["historical_symptoms"]
                ))
            
            # Track uncertain symptoms
            if s_data.get("uncertain_symptoms"):
                self.patient_data["symptom_details"]["uncertain_symptoms"] = list(set(
                    self.patient_data["symptom_details"].get("uncertain_symptoms", []) + s_data["uncertain_symptoms"]
                ))
            
            # Track family symptoms
            if s_data.get("family_symptoms"):
                self.patient_data["medical_history"]["family_history"] = list(set(
                    self.patient_data["medical_history"].get("family_history", []) + s_data["family_symptoms"]
                ))
                self.patient_data["medical_history"]["status"] = "provided"

            # Preserve location if already established and not explicitly corrected
            if s_data.get("location"):
                new_loc = s_data["location"]
                old_loc = self.patient_data["symptom_details"].get("location")
                if not old_loc or old_loc == "Not specified":
                    self.patient_data["symptom_details"]["location"] = new_loc
                else:
                    # Check for explicit correction
                    raw_text = s_data.get("raw_text", "").lower()
                    if any(w in raw_text for w in ["actually", "correction", "mistake", "not in", "instead of", "changed to"]):
                        self.patient_data["symptom_details"]["location"] = new_loc
                        self.logger.info(f"Location updated by explicit patient correction: '{old_loc}' -> '{new_loc}'")
                    else:
                        self.logger.info(f"Preserving established location '{old_loc}', ignoring secondary mention '{new_loc}'")

            for detail_field in ["character", "onset", "progression", "timing", "frequency"]:
                if s_data.get(detail_field):
                    old_df = self.patient_data["symptom_details"].get(detail_field)
                    new_df = s_data[detail_field]
                    if old_df and old_df != new_df and detail_field == "timing":
                        self.patient_data.setdefault("correction_history", []).append({
                            "field": "timing",
                            "previous": old_df,
                            "new": new_df,
                            "note": f"Symptom timing updated from '{old_df}' to '{new_df}'."
                        })
                    self.patient_data["symptom_details"][detail_field] = new_df
            
            # Merge per-symptom timelines and attributes
            if s_data.get("per_symptom_details"):
                psd_target = self.patient_data["symptom_details"].setdefault("per_symptom_details", {})
                for sym_k, sym_v in s_data["per_symptom_details"].items():
                    existing = psd_target.setdefault(sym_k, {})
                    for k, v in sym_v.items():
                        if v:
                            existing[k] = v
            
            # Independent & Context-Aware Severity Handling
            if s_data.get("severity"):
                new_val = s_data["severity"]
                old_val = self.patient_data["symptom_details"].get("severity")
                active_symps = self.patient_data["symptoms"]["list"]
                
                # Check for pain / ache symptoms among active symptoms
                pain_symps = [s for s in active_symps if any(p in s.lower() for p in ["headache", "pain", "ache", "cramp", "migraine"])]
                
                if len(active_symps) <= 1:
                    # Unambiguous symptom context: single symptom or chief complaint
                    s_name = active_symps[0] if active_symps else (self.patient_data["chief_complaint"].get("reason") or "symptom")
                    if old_val and old_val != new_val:
                        contradiction = {
                            "category": "severity_change",
                            "field": "severity",
                            "symptom": s_name,
                            "previous": old_val,
                            "new": new_val,
                            "note": f"Symptom severity for '{s_name}' changed from '{old_val}' to '{new_val}'."
                        }
                        if not any(c.get("category") == "severity_change" and c.get("new") == new_val for c in self.patient_data.setdefault("contradictions", [])):
                            self.patient_data["contradictions"].append(contradiction)
                            self.logger.info(f"Contradiction noted: severity '{old_val}' -> '{new_val}' for '{s_name}'")
                    self.patient_data["symptom_details"]["severity"] = new_val
                    # Clear any prior ambiguous_severity clarifications
                    self.patient_data["unresolved_clarifications"] = [
                        u for u in self.patient_data.get("unresolved_clarifications", [])
                        if u.get("category") != "ambiguous_severity"
                    ]
                else:
                    # Multiple active symptoms exist
                    raw_text = s_data.get("raw_text", "").lower()
                    mentioned = [s for s in active_symps if s in s_data.get("symptoms_found", []) or s in raw_text]
                    if mentioned:
                        s_name = mentioned[0]
                        if old_val and old_val != new_val:
                            contradiction = {
                                "category": "severity_change",
                                "field": "severity",
                                "symptom": s_name,
                                "previous": old_val,
                                "new": new_val,
                                "note": f"Symptom severity for '{s_name}' changed from '{old_val}' to '{new_val}'."
                            }
                            if not any(c.get("category") == "severity_change" and c.get("new") == new_val for c in self.patient_data.setdefault("contradictions", [])):
                                self.patient_data["contradictions"].append(contradiction)
                                self.logger.info(f"Contradiction noted: severity '{old_val}' -> '{new_val}' for '{s_name}'")
                        self.patient_data["symptom_details"]["severity"] = new_val
                        self.patient_data["unresolved_clarifications"] = [
                            u for u in self.patient_data.get("unresolved_clarifications", [])
                            if u.get("category") != "ambiguous_severity"
                        ]
                    elif len(pain_symps) == 1:
                        # Unambiguously applies to the single active pain symptom (e.g. headache while fever is systemic)
                        s_name = pain_symps[0]
                        if old_val and old_val != new_val:
                            contradiction = {
                                "category": "severity_change",
                                "field": "severity",
                                "symptom": s_name,
                                "previous": old_val,
                                "new": new_val,
                                "note": f"Symptom severity for '{s_name}' changed from '{old_val}' to '{new_val}'."
                            }
                            if not any(c.get("category") == "severity_change" and c.get("new") == new_val for c in self.patient_data.setdefault("contradictions", [])):
                                self.patient_data["contradictions"].append(contradiction)
                                self.logger.info(f"Contradiction noted: severity '{old_val}' -> '{new_val}' for '{s_name}'")
                        self.patient_data["symptom_details"]["severity"] = new_val
                        self.patient_data["unresolved_clarifications"] = [
                            u for u in self.patient_data.get("unresolved_clarifications", [])
                            if u.get("category") != "ambiguous_severity"
                        ]
                    else:
                        # Always set severity if none was recorded yet, rather than leaving it empty
                        self.patient_data["symptom_details"]["severity"] = new_val
                        self.patient_data["unresolved_clarifications"] = [
                            u for u in self.patient_data.get("unresolved_clarifications", [])
                            if u.get("category") != "ambiguous_severity"
                        ]
        
        # TASK 1.9.5 FIX: Handle medical history including explicit denials and empty dicts
        if "medical_history" in extracted:
            # Handle empty dicts too (not just truthy dicts)
            history_data = extracted["medical_history"]
            
            if history_data.get("explicitly_none") or history_data.get("conditions_denied"):
                # Patient explicitly denied having medical history
                # TASK 1.9.7: If previously stated they had history, mark as needs_clarification
                if self.patient_data["medical_history"]["status"] == "provided":
                    self.patient_data["medical_history"]["status"] = "needs_clarification"
                    self.logger.warning("Medical history: contradiction detected - previous info denied")
                else:
                    self.patient_data["medical_history"]["status"] = "patient_denied"
                    self.patient_data["medical_history"]["conditions_status"] = "patient_denied"
                    self.logger.debug("Medical history: explicitly denied")
            
            elif history_data.get("conditions_mentioned"):
                self.patient_data["medical_history"]["conditions"] = list(set(
                    self.patient_data["medical_history"]["conditions"] + 
                    history_data["conditions_mentioned"]
                ))
                self.patient_data["medical_history"]["status"] = "provided"
                self.logger.debug(f"Medical conditions found: {history_data['conditions_mentioned']}")
            
            elif history_data.get("history_mentioned"):
                # Mark as provided if surgery/hospital/admit is mentioned
                self.patient_data["medical_history"]["status"] = "provided"
                self.logger.debug("Medical history: procedures/surgeries mentioned")
            
            if history_data.get("denied_conditions"):
                existing_denied = self.patient_data["medical_history"].setdefault("denied_conditions", [])
                for dc in history_data["denied_conditions"]:
                    if dc not in existing_denied:
                        existing_denied.append(dc)
                # Scrub any denied condition from positive conditions
                self.patient_data["medical_history"]["conditions"] = [
                    c for c in self.patient_data["medical_history"].get("conditions", [])
                    if c.lower() not in [d.lower() for d in existing_denied]
                ]
                if not self.patient_data["medical_history"]["conditions"] and self.patient_data["medical_history"]["status"] == "not_asked":
                    self.patient_data["medical_history"]["status"] = "patient_denied"
                    self.patient_data["medical_history"]["conditions_status"] = "patient_denied"

            if history_data.get("family_history_items"):
                self.patient_data["medical_history"]["family_history"] = list(set(
                    self.patient_data["medical_history"]["family_history"] + history_data["family_history_items"]
                ))
                self.patient_data["medical_history"]["status"] = "provided"
        
        # TASK 1.9.5 FIX: Handle medications including explicit denials and empty dicts
        if "medications" in extracted:
            # Handle empty dicts too (not just truthy dicts)
            meds_data = extracted["medications"]
            
            if meds_data.get("explicitly_none"):
                # Patient explicitly denied taking medications
                # TASK 1.9.7: If previously listed medications, mark as needs_clarification
                if self.patient_data["medications"]["list"]:
                    self.patient_data["medications"]["status"] = "needs_clarification"
                    self.logger.warning("Medications: contradiction detected - listed meds but now denies")
                    # Keep the list for review but mark as needing clarification
                else:
                    self.patient_data["medications"]["status"] = "none_reported"
                    self.logger.debug("Medications: explicitly denied")
            
            elif meds_data.get("medications_mentioned"):
                # New or corrected medications mentioned
                new_meds = meds_data["medications_mentioned"]
                updated_meds = list(self.patient_data["medications"]["list"])
                for nm in new_meds:
                    base_nm = re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', nm, flags=re.IGNORECASE).strip().lower()
                    if base_nm and len(base_nm) > 2:
                        updated_meds = [m for m in updated_meds if re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', m, flags=re.IGNORECASE).strip().lower() != base_nm]
                    updated_meds.append(nm)
                self.patient_data["medications"]["list"] = list(dict.fromkeys(updated_meds))
                self.patient_data["medications"]["status"] = "provided"
                self.logger.debug(f"Medications updated: {self.patient_data['medications']['list']}")
        
        # TASK 1.9.5 FIX: Handle allergies including symptom-vs-allergy disambiguation
        if "allergies" in extracted:
            # Handle empty dicts too (not just truthy dicts)
            allergies_data = extracted["allergies"]
            
            if allergies_data.get("explicitly_none"):
                # Patient explicitly denied having allergies
                # TASK 1.9.7: If previously listed allergies, mark as needs_clarification
                if self.patient_data["allergies"]["drug_allergies"] or \
                   self.patient_data["allergies"]["food_allergies"] or \
                   self.patient_data["allergies"]["environmental_allergies"]:
                    self.patient_data["allergies"]["status"] = "needs_clarification"
                    self.logger.warning("Allergies: contradiction detected - listed allergies but now denies")
                    # Keep the list for review but mark as needing clarification
                else:
                    self.patient_data["allergies"]["status"] = "explicitly_denied"
                    self.patient_data["allergies"]["drug_status"] = "explicitly_denied"
                    self.patient_data["allergies"]["food_status"] = "explicitly_denied"
                    self.patient_data["allergies"]["environmental_status"] = "explicitly_denied"
                    self.logger.debug("Allergies: explicitly denied")
            
            elif allergies_data.get("symptom_not_allergy"):
                # Patient described symptoms in response to allergy question
                # Mark allergies as "needs_clarification" and add symptoms separately
                self.patient_data["allergies"]["status"] = "needs_clarification"
                
                # Store symptoms in symptoms list instead of allergies
                if allergies_data.get("symptoms_mentioned"):
                    self.patient_data["symptoms"]["list"] = list(set(
                        self.patient_data["symptoms"]["list"] + allergies_data["symptoms_mentioned"]
                    ))
                    self.patient_data["symptoms"]["status"] = "provided"
                
                self.logger.debug(f"Symptoms detected in allergy answer: {allergies_data.get('symptoms_mentioned')}")
            
            elif allergies_data.get("allergies_mentioned"):
                # Patient described actual allergies
                self.patient_data["allergies"]["drug_allergies"] = list(set(
                    self.patient_data["allergies"]["drug_allergies"] + allergies_data["allergies_mentioned"]
                ))
                self.patient_data["allergies"]["status"] = "provided"
                self.logger.debug(f"Allergies found: {allergies_data.get('allergies_mentioned')}")
            
            elif allergies_data.get("reactions_mentioned"):
                # Patient described allergic reactions
                self.patient_data["allergies"]["status"] = "provided"
                self.logger.debug(f"Allergy reactions found: {allergies_data.get('reactions_mentioned')}")
        
        if "vitals" in extracted:
            # Handle explicit denial: patient says they haven't measured vitals
            if extracted["vitals"].get("explicitly_not_measured"):
                self.patient_data["vitals"]["status"] = "not_provided"
                self.patient_data["vitals"]["temperature_status"] = "not_measured"
                self.logger.debug("Vitals: patient explicitly stated they have not measured vital signs")
            elif extracted["vitals"]:
                # Patient provided actual vital sign measurements
                if extracted["vitals"].get("blood_pressure"):
                    self.patient_data["vitals"]["blood_pressure"] = extracted["vitals"]["blood_pressure"]
                if extracted["vitals"].get("heart_rate"):
                    self.patient_data["vitals"]["heart_rate"] = extracted["vitals"]["heart_rate"]
                if extracted["vitals"].get("temperature"):
                    self.patient_data["vitals"]["temperature"] = extracted["vitals"]["temperature"]
                if extracted["vitals"].get("oxygen_saturation"):
                    self.patient_data["vitals"]["oxygen_saturation"] = extracted["vitals"]["oxygen_saturation"]
                
                if any([extracted["vitals"].get("blood_pressure"),
                        extracted["vitals"].get("heart_rate"),
                        extracted["vitals"].get("temperature"),
                        extracted["vitals"].get("oxygen_saturation")]):
                    self.patient_data["vitals"]["status"] = "provided"
        
        if "lifestyle" in extracted and extracted["lifestyle"]:
            ls_ext = extracted["lifestyle"]
            ls_target = self.patient_data["lifestyle"]
            if ls_ext.get("recent_travel"):
                ls_target["recent_travel"] = ls_ext["recent_travel"]
                ls_target["recent_travel_status"] = ls_ext.get("recent_travel_status", "provided")
            if ls_ext.get("sick_contacts"):
                ls_target["sick_contacts"] = ls_ext["sick_contacts"]
                ls_target["sick_contacts_status"] = ls_ext.get("sick_contacts_status", "provided")
            if ls_ext.get("smoking_status"):
                ls_target["smoking"] = ls_ext["smoking_status"]
                ls_target["smoking_status"] = "provided"
            if ls_ext.get("alcohol_status"):
                ls_target["alcohol"] = ls_ext["alcohol_status"]
                ls_target["alcohol_status"] = "provided"

        if "demographics" in extracted and extracted["demographics"]:
            if extracted["demographics"].get("age"):
                self.patient_data["demographics"]["age"] = extracted["demographics"]["age"]
                self.patient_data["demographics"]["age_status"] = "provided"
        
        # Update metadata
        self.patient_data["collection_metadata"]["last_updated"] = datetime.now().isoformat()
        self.patient_data["collection_metadata"]["extraction_count"] += 1
    
    def get_patient_data(self) -> Dict[str, Any]:
        """
        Return the current structured patient information.
        
        Returns:
            Dictionary with all collected patient information
        """
        return self.patient_data
    
    def get_missing_information(self) -> Dict[str, bool]:
        """
        Return a summary of missing or unknown information categories.
        
        Returns:
            Dictionary indicating which categories have unknown status
        """
        missing = {}
        
        if self.patient_data["chief_complaint"]["status"] in ["unknown", "not_asked"]:
            missing["chief_complaint"] = True
        
        if self.patient_data["symptoms"]["status"] in ["unknown", "not_asked"]:
            missing["symptoms"] = True
        
        if self.patient_data["medical_history"]["status"] in ["unknown", "not_asked"]:
            missing["medical_history"] = True
        
        if self.patient_data["medications"]["status"] in ["unknown", "not_asked"]:
            missing["medications"] = True
        
        if self.patient_data["allergies"]["status"] in ["unknown", "not_asked"]:
            missing["allergies"] = True
        
        if self.patient_data["vitals"]["status"] in ["unknown", "not_asked"]:
            missing["vitals"] = True
        
        return missing
    
    def merge_structured_updates(self, updates: Dict[str, Any]) -> None:
        """
        Safely merge structured updates into patient data without wiping existing confirmed information.
        """
        if not updates or not isinstance(updates, dict):
            return
        
        # Chief complaint
        if "chief_complaint" in updates:
            if isinstance(updates["chief_complaint"], dict):
                cc = updates["chief_complaint"]
                if cc.get("reason"):
                    self.patient_data["chief_complaint"]["reason"] = cc["reason"]
                    self.patient_data["chief_complaint"]["status"] = "provided"
                if cc.get("duration"):
                    self.patient_data["chief_complaint"]["duration"] = cc["duration"]
                    self.patient_data["chief_complaint"]["duration_mentioned"] = True
            elif isinstance(updates["chief_complaint"], str) and updates["chief_complaint"]:
                self.patient_data["chief_complaint"]["reason"] = updates["chief_complaint"]
                self.patient_data["chief_complaint"]["status"] = "provided"
        
        if updates.get("duration"):
            self.patient_data["chief_complaint"]["duration"] = updates["duration"]
            self.patient_data["chief_complaint"]["duration_mentioned"] = True
        
        # Symptoms
        if "symptoms" in updates and isinstance(updates["symptoms"], list):
            # Never add symptoms that are already in negative_symptoms or negated in this update
            curr_negs = set(
                s.replace("no ", "").strip().lower() 
                for s in (self.patient_data["symptom_details"].get("negative_symptoms", []) + updates.get("negative_symptoms", []))
            )
            new_symps = [
                s for s in updates["symptoms"] 
                if s and isinstance(s, str) and not any(nc == s.strip().lower() or nc in s.strip().lower() for nc in curr_negs)
            ]
            if new_symps:
                self.patient_data["symptoms"]["list"] = normalize_symptoms(
                    self.patient_data["symptoms"]["list"] + new_symps
                )
                self.patient_data["symptoms"]["status"] = "provided"
        
        # Explicit symptom removal / corrections
        if "remove_symptoms" in updates and isinstance(updates["remove_symptoms"], list):
            rem_symps = [s.strip().lower() for s in updates["remove_symptoms"] if s and isinstance(s, str)]
            expanded_rems = set(rem_symps)
            for rs in rem_symps:
                canon = CANONICAL_SYMPTOMS.get(rs)
                if canon:
                    expanded_rems.add(canon.lower())
                for syn, c in CANONICAL_SYMPTOMS.items():
                    if c.lower() == rs or syn == rs:
                        expanded_rems.add(syn.lower())
                        expanded_rems.add(c.lower())

            self.patient_data["symptoms"]["list"] = [
                s for s in self.patient_data["symptoms"]["list"]
                if not any(rs == s.strip().lower() or rs in s.strip().lower() or s.strip().lower() in rs for rs in expanded_rems)
            ]
            self.patient_data["symptoms"]["list"] = normalize_symptoms(self.patient_data["symptoms"]["list"])
            for rs in rem_symps:
                neg_str = f"no {rs}"
                if neg_str not in self.patient_data["symptom_details"]["negative_symptoms"]:
                    self.patient_data["symptom_details"]["negative_symptoms"].append(neg_str)
            
            # If current chief complaint is in removed symptoms, clear it
            current_cc = (self.patient_data["chief_complaint"].get("reason") or "").lower()
            if current_cc and any(rs == current_cc or rs in current_cc or current_cc in rs for rs in expanded_rems):
                self.patient_data["chief_complaint"]["reason"] = None
                self.patient_data["chief_complaint"]["status"] = "unknown"
                if self.patient_data["symptoms"]["list"]:
                    self.patient_data["chief_complaint"]["reason"] = self.patient_data["symptoms"]["list"][0]
                    self.patient_data["chief_complaint"]["status"] = "provided"
        
        # Negative symptoms
        if "negative_symptoms" in updates and isinstance(updates["negative_symptoms"], list):
            neg_symps = [s for s in updates["negative_symptoms"] if s and isinstance(s, str)]
            if neg_symps:
                self.patient_data["symptom_details"]["negative_symptoms"] = list(dict.fromkeys(
                    self.patient_data["symptom_details"]["negative_symptoms"] + neg_symps
                ))
        
        # Filter positive symptoms list to never retain negated symptoms
        neg_clean = set(s.replace("no ", "").strip().lower() for s in self.patient_data["symptom_details"]["negative_symptoms"])
        expanded_negs = set(neg_clean)
        for nc in neg_clean:
            canon = CANONICAL_SYMPTOMS.get(nc)
            if canon:
                expanded_negs.add(canon.lower())
            for syn, c in CANONICAL_SYMPTOMS.items():
                if c.lower() == nc or syn == nc:
                    expanded_negs.add(syn.lower())
                    expanded_negs.add(c.lower())

        self.patient_data["symptoms"]["list"] = [
            s for s in self.patient_data["symptoms"]["list"]
            if not any(nc == s.lower() or nc in s.lower() or s.lower() in nc for nc in expanded_negs)
        ]
        
        # Historical & uncertain symptoms
        if "historical_symptoms" in updates and isinstance(updates["historical_symptoms"], list):
            hist_symps = [s for s in updates["historical_symptoms"] if s and isinstance(s, str)]
            if hist_symps:
                self.patient_data["symptom_details"]["historical_symptoms"] = list(dict.fromkeys(
                    self.patient_data["symptom_details"].get("historical_symptoms", []) + hist_symps
                ))
        if "uncertain_symptoms" in updates and isinstance(updates["uncertain_symptoms"], list):
            uncert_symps = [s for s in updates["uncertain_symptoms"] if s and isinstance(s, str)]
            if uncert_symps:
                self.patient_data["symptom_details"]["uncertain_symptoms"] = list(dict.fromkeys(
                    self.patient_data["symptom_details"].get("uncertain_symptoms", []) + uncert_symps
                ))
        if "contradictions" in updates and isinstance(updates["contradictions"], list):
            for c in updates["contradictions"]:
                if c not in self.patient_data.setdefault("contradictions", []):
                    self.patient_data["contradictions"].append(c)
        
        # Symptom details
        if "symptom_details" in updates and isinstance(updates["symptom_details"], dict):
            sd = updates["symptom_details"]
            if sd.get("location"):
                # Preserve established location unless explicit correction
                old_loc = self.patient_data["symptom_details"].get("location")
                if not old_loc or old_loc == "Not specified":
                    self.patient_data["symptom_details"]["location"] = sd["location"]
            for field in ["severity", "character", "onset", "progression", "timing", "frequency"]:
                if sd.get(field):
                    self.patient_data["symptom_details"][field] = sd[field]
        
        # Medical history
        if "medical_history" in updates:
            mh = updates["medical_history"]
            if isinstance(mh, dict):
                if mh.get("denied_conditions"):
                    denied = [c.strip().lower() for c in mh["denied_conditions"] if c and isinstance(c, str)]
                    existing_denied = self.patient_data["medical_history"].setdefault("denied_conditions", [])
                    for dc in denied:
                        if dc not in existing_denied:
                            existing_denied.append(dc)
                    self.patient_data["medical_history"]["conditions"] = [
                        c for c in self.patient_data["medical_history"].get("conditions", [])
                        if c.lower() not in existing_denied
                    ]
                    if not self.patient_data["medical_history"]["conditions"]:
                        self.patient_data["medical_history"]["conditions_status"] = "patient_denied"
                        if self.patient_data["medical_history"]["status"] == "not_asked":
                            self.patient_data["medical_history"]["status"] = "patient_denied"
                if mh.get("conditions"):
                    denied_set = {d.lower() for d in self.patient_data["medical_history"].get("denied_conditions", [])}
                    valid_conds = [c for c in mh["conditions"] if isinstance(c, str) and c.strip().lower() not in denied_set]
                    if valid_conds:
                        self.patient_data["medical_history"]["conditions"] = list(dict.fromkeys(
                            self.patient_data["medical_history"]["conditions"] + valid_conds
                        ))
                        self.patient_data["medical_history"]["status"] = "provided"
                        self.patient_data["medical_history"]["conditions_status"] = "provided"
                if mh.get("surgeries"):
                    self.patient_data["medical_history"]["surgeries"] = list(dict.fromkeys(
                        self.patient_data["medical_history"]["surgeries"] + mh["surgeries"]
                    ))
                    self.patient_data["medical_history"]["status"] = "provided"
                if mh.get("hospitalizations"):
                    self.patient_data["medical_history"]["hospitalizations"] = list(dict.fromkeys(
                        self.patient_data["medical_history"]["hospitalizations"] + mh["hospitalizations"]
                    ))
                    self.patient_data["medical_history"]["status"] = "provided"
                if mh.get("family_history"):
                    self.patient_data["medical_history"]["family_history"] = list(dict.fromkeys(
                        self.patient_data["medical_history"]["family_history"] + mh["family_history"]
                    ))
                    self.patient_data["medical_history"]["status"] = "provided"
                if mh.get("status") in ["none_reported", "provided", "needs_clarification", "patient_denied"]:
                    self.patient_data["medical_history"]["status"] = mh["status"]
            elif isinstance(mh, list):
                denied_set = {d.lower() for d in self.patient_data["medical_history"].get("denied_conditions", [])}
                valid_conds = [c for c in mh if isinstance(c, str) and c.strip() and c.strip().lower() not in denied_set]
                if valid_conds:
                    self.patient_data["medical_history"]["conditions"] = list(dict.fromkeys(
                        self.patient_data["medical_history"]["conditions"] + valid_conds
                    ))
                    self.patient_data["medical_history"]["status"] = "provided"
        
        # Per-symptom details & symptom detail removals (for corrections)
        if "per_symptom_details" in updates and isinstance(updates["per_symptom_details"], dict):
            psd_target = self.patient_data["symptom_details"].setdefault("per_symptom_details", {})
            for sym_k, sym_v in updates["per_symptom_details"].items():
                if not isinstance(sym_v, dict):
                    continue
                canon_k = CANONICAL_SYMPTOMS.get(sym_k.strip().lower(), sym_k.strip().lower())
                existing = psd_target.setdefault(canon_k, {})
                for k, v in sym_v.items():
                    if k == "is_correction":
                        continue
                    if v:
                        if k in ("timing", "frequency", "location", "character", "severity", "onset", "duration") and existing.get(k) and existing.get(k) != v:
                            self.patient_data.setdefault("correction_history", []).append({
                                "symptom": canon_k,
                                "field": k,
                                "previous": existing[k],
                                "new": v,
                                "note": f"{canon_k} {k} updated from '{existing[k]}' to '{v}'."
                            })
                        existing[k] = v
                        if k in ("timing", "frequency"):
                            self.patient_data["symptom_details"][k] = v
                if sym_v.get("status") == "denied" and canon_k in self.patient_data["symptoms"]["list"]:
                    self.patient_data["symptoms"]["list"] = [s for s in self.patient_data["symptoms"]["list"] if s.lower() != canon_k]
                    neg_str = f"no {canon_k}"
                    if neg_str not in self.patient_data["symptom_details"]["negative_symptoms"]:
                        self.patient_data["symptom_details"]["negative_symptoms"].append(neg_str)

        if updates.get("remove_character"):
            rem_chars = [c.strip().lower() for c in updates["remove_character"] if isinstance(c, str)]
            cur_char = (self.patient_data["symptom_details"].get("character") or "").lower()
            if cur_char and any(rc in cur_char or cur_char in rc for rc in rem_chars):
                self.patient_data["symptom_details"]["character"] = None

        if updates.get("remove_location"):
            rem_locs = [l.strip().lower() for l in updates["remove_location"] if isinstance(l, str)]
            cur_loc = (self.patient_data["symptom_details"].get("location") or "").lower()
            if cur_loc and any(rl in cur_loc or cur_loc in rl for rl in rem_locs):
                self.patient_data["symptom_details"]["location"] = None

        # Lifestyle & exposures
        if "lifestyle" in updates and isinstance(updates["lifestyle"], dict):
            ls = updates["lifestyle"]
            ls_target = self.patient_data["lifestyle"]
            for lk in ["smoking", "alcohol", "exercise", "diet", "recent_travel", "recent_travel_status", "sick_contacts", "sick_contacts_status", "exposures", "exposures_status"]:
                if ls.get(lk):
                    ls_target[lk] = ls[lk]
                    if lk in ["recent_travel", "sick_contacts", "exposures"] and not ls.get(f"{lk}_status"):
                        ls_target[f"{lk}_status"] = "patient_denied" if ls[lk] == "patient_denied" else "provided"

        # Demographics
        if "demographics" in updates and isinstance(updates["demographics"], dict):
            if updates["demographics"].get("age"):
                self.patient_data["demographics"]["age"] = updates["demographics"]["age"]
                self.patient_data["demographics"]["age_status"] = "provided"
        
        # Medications
        if "medications" in updates:
            meds = updates["medications"]
            if isinstance(meds, list):
                valid_meds = [nm.strip() for nm in meds if isinstance(nm, str) and nm.strip()]
                if valid_meds:
                    updated_meds = list(self.patient_data["medications"]["list"])
                    for nm in valid_meds:
                        base_nm = re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', nm, flags=re.IGNORECASE).strip().lower()
                        if base_nm and len(base_nm) > 2:
                            updated_meds = [m for m in updated_meds if re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', m, flags=re.IGNORECASE).strip().lower() != base_nm]
                        updated_meds.append(nm)
                    self.patient_data["medications"]["list"] = list(dict.fromkeys(updated_meds))
                    self.patient_data["medications"]["status"] = "provided"
            elif isinstance(meds, dict):
                if meds.get("list"):
                    updated_meds = list(self.patient_data["medications"]["list"])
                    for nm in meds["list"]:
                        if not nm or not isinstance(nm, str):
                            continue
                        base_nm = re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', nm, flags=re.IGNORECASE).strip().lower()
                        if base_nm and len(base_nm) > 2:
                            updated_meds = [m for m in updated_meds if re.sub(r'\s*\d+\s*(?:mg|g|mcg|ml)\b.*', '', m, flags=re.IGNORECASE).strip().lower() != base_nm]
                        updated_meds.append(nm)
                    self.patient_data["medications"]["list"] = list(dict.fromkeys(updated_meds))
                    self.patient_data["medications"]["status"] = "provided"
                if meds.get("status"):
                    self.patient_data["medications"]["status"] = meds["status"]

        if updates.get("medications_status") in ["patient_denied", "none_reported", "explicitly_denied", "not_provided"]:
            self.patient_data["medications"]["status"] = updates["medications_status"]
            if updates["medications_status"] in ["patient_denied", "none_reported", "explicitly_denied"]:
                self.patient_data["medications"]["list"] = []
        
        # Explicit medication removal
        if "medications_to_remove" in updates and isinstance(updates["medications_to_remove"], list):
            rem_meds = [m.strip().lower() for m in updates["medications_to_remove"] if m and isinstance(m, str)]
            self.patient_data["medications"]["list"] = [
                m for m in self.patient_data["medications"]["list"]
                if not any(rm in m.lower() or m.lower() in rm for rm in rem_meds)
            ]
        
        # Allergies
        if "allergies" in updates:
            alg = updates["allergies"]
            if isinstance(alg, dict):
                for k in ["drug_allergies", "food_allergies", "environmental_allergies"]:
                    if alg.get(k):
                        self.patient_data["allergies"][k] = list(dict.fromkeys(
                            self.patient_data["allergies"][k] + alg[k]
                        ))
                        self.patient_data["allergies"]["status"] = "provided"
                if alg.get("status"):
                    self.patient_data["allergies"]["status"] = alg["status"]
                    if alg["status"] in ["explicitly_denied", "patient_denied", "none_reported"]:
                        if not self.patient_data["allergies"]["drug_allergies"]:
                            self.patient_data["allergies"]["drug_status"] = alg["status"]
                        if not self.patient_data["allergies"]["food_allergies"]:
                            self.patient_data["allergies"]["food_status"] = alg["status"]
                        if not self.patient_data["allergies"]["environmental_allergies"]:
                            self.patient_data["allergies"]["environmental_status"] = alg["status"]
        
        # Vitals
        if "vitals" in updates and isinstance(updates["vitals"], dict):
            if updates["vitals"].get("temperature_status") == "not_measured":
                self.patient_data["vitals"]["temperature_status"] = "not_measured"
                if self.patient_data["vitals"]["status"] == "not_asked":
                    self.patient_data["vitals"]["status"] = "not_provided"
            for v_key in ["blood_pressure", "heart_rate", "temperature", "oxygen_saturation"]:
                if updates["vitals"].get(v_key):
                    self.patient_data["vitals"][v_key] = updates["vitals"][v_key]
                    self.patient_data["vitals"]["status"] = "provided"
                    if v_key == "temperature":
                        self.patient_data["vitals"]["temperature_status"] = "provided"
        
        self.patient_data["collection_metadata"]["last_updated"] = datetime.now().isoformat()

    @staticmethod
    def detect_early_completion_request(text: str) -> bool:
        """Detect if patient explicitly requests to finish or get their report now."""
        if not text:
            return False
        text_lower = text.lower().strip()
        patterns = [
            r'^\s*(?:done|finish|finished|complete|completed)\s*[.!]*$',
            r'(?:^|[.!?]\s*)(?:done|finish|finished)\s*[.!]*$',
            r'\b(?:give|get|show|provide|generate|send|want)\s+(?:me\s+)?(?:the|my|a)?\s*(?:report|summary|intake)\b',
            r'\b(?:that\'s|that\s+is)\s+(?:all|it|everything)\b',
            r'\b(?:i\'m|i\s+am)\s+done\b',
            r'\b(?:stop|finish|end)\s+(?:the\s+)?(?:interview|intake|questions|questioning)\b',
            r'\bno\s+more\s+questions\b',
            r'\bcan\s+i\s+have\s+(?:the|my)?\s*report\b',
            r'\bready\s+for\s+(?:the|my)?\s*report\b'
        ]
        return any(re.search(p, text_lower) for p in patterns)

    def clear_patient_data(self) -> None:
        """Clear all collected patient data and reinitialize."""
        self.logger.info("Clearing patient data")
        self.patient_data = self._create_empty_patient_structure()


# Export the class for easy importing
__all__ = ['PatientInfoCollector']
