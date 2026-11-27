import logging
import re
from typing import Dict, List, Optional, Any, Tuple


class RedFlagDetector:
    """
    Detects emergency warning signs from patient-reported symptoms and vital signs.
    
    Uses rule-based safety layer as primary detection mechanism.
    LLM is optional for interpreting ambiguous descriptions but cannot override rules.
    
    Responsibilities:
    - Detect serious warning signs (chest pain, difficulty breathing, stroke signs, etc.)
    - Evaluate vital signs against emergency thresholds
    - Assess overall severity and escalation need
    - Provide clear emergency guidance
    - Never diagnose or recommend treatment
    - Never reassure without basis
    
    Emergency Thresholds Used:
    - Systolic BP: >180 mmHg (hypertensive emergency threshold per AHA guidelines)
    - Diastolic BP: >120 mmHg (hypertensive emergency threshold per AHA guidelines)
    - Heart Rate: >120 bpm or <40 bpm (tachycardia/bradycardia thresholds)
    - Temperature: >103°F (39.4°C) or <95°F (35°C) (severe fever/hypothermia per CDC)
    - SpO2: <90% (hypoxia threshold per clinical standards)
    - Respiratory Rate: >30 breaths/min or <8 breaths/min (extreme ranges)
    """
    
    # Severity levels
    SEVERITY_CRITICAL = "critical"      # Immediate life threat
    SEVERITY_HIGH = "high"              # Serious, needs urgent evaluation
    SEVERITY_MODERATE = "moderate"      # Notable but not immediately life-threatening
    SEVERITY_LOW = "low"                # Minor concern
    SEVERITY_NONE = "none"              # No flags detected
    
    def __init__(self, config=None, llm=None):
        """
        Initialize the RedFlagDetector.
        
        Args:
            config: Configuration object (optional)
            llm: LLM instance for optional ambiguity resolution (optional)
        """
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing RedFlagDetector")
        
        self.config = config
        self.llm = llm
        
        # Emergency keywords and phrases for rule-based detection
        self.chest_pain_keywords = [
            "chest pain", "chest discomfort", "chest pressure", "chest tightness",
            "cardiac chest pain"
        ]
        
        self.breathing_difficulty_keywords = [
            "difficulty breathing", "shortness of breath", "can't breathe", "struggling to breathe",
            "struggling for breath", "hard to breathe", "breathing difficulty", "dyspnea"
        ]
        
        self.stroke_warning_keywords = [
            "facial drooping", "facial weakness", "mouth drooping",
            "arm weakness", "weak arm", "unable to lift arm",
            "speech difficulty", "slurred speech", "unable to speak",
            "sudden confusion", "sudden dizziness", "sudden severe headache",
            "time to call 108", "time to call 112", "act fast"  # FAST acronym components
        ]
        
        self.severe_bleeding_keywords = [
            "severe bleeding", "severe hemorrhage", "uncontrolled bleeding",
            "bleeding won't stop", "profuse bleeding", "gushing blood"
        ]
        
        self.loss_of_consciousness_keywords = [
            "loss of consciousness", "unconscious", "passed out", "fainted", "fainting",
            "blacked out", "lost consciousness", "collapsed"
        ]
        
        self.seizure_keywords = [
            "seizure", "seizures", "convulsion", "convulsions", "convulsing",
            "shaking uncontrollably", "jerking movements"
        ]
        
        self.severe_allergic_reaction_keywords = [
            "anaphylaxis", "anaphylactic", "severe allergy reaction",
            "swelling throat", "unable to swallow", "throat closing", "lips swollen"
        ]
        
        self.logger.info("RedFlagDetector initialized successfully")
    
    def _is_negated_flag(self, keyword: str, text: str, negative_symptoms: List[str]) -> bool:
        """Check if an emergency sign is explicitly denied, negated, or attributed to family members."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Check explicit negative symptoms list
        for neg in negative_symptoms:
            if keyword_lower in neg.lower() or neg.lower().replace("no ", "").strip() in keyword_lower:
                return True
        
        # Specific breathing Hindi/Hinglish checks
        if any(b in keyword_lower for b in ["breathing", "breath", "shortness of breath", "difficulty breathing"]):
            if re.search(r'\bsaans\s+(?:lene\s+(?:mein|me)\s+)?(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue)\s+nahi\b', text_lower):
                return True
            if re.search(r'\bsaans\s+(?:phool\s+nahi|theek\s+hai|normal\s+hai)\b', text_lower):
                return True

        # Check Hindi / Hinglish post-position negations
        escaped_kw = re.escape(keyword_lower)
        hindi_post_patterns = [
            escaped_kw + r'\s+(?:bhi\s+|toh\s+|to\s+)?(?:nahi\s+hai|nahi\s+ho\s+raha|nahi\s+tha|nahi\s+hoga|nahi|na\s+hai)\b',
            escaped_kw + r'\s+(?:mein\s+|me\s+)?(?:koi\s+|kuch\s+)?(?:dikkat|takleef|problem|issue|dard)\s+nahi\b',
            escaped_kw + r'\s+bilkul\s+(?:bhi\s+)?nahi\b',
            escaped_kw + r'\s*[:\-]\s*(?:nahi|na)\b',
            r'\b(?:koi|kuch)\s+' + escaped_kw + r'\s+(?:bhi\s+)?nahi\b',
            r'\b(?:nahi\s+hai|nahi)\s+(?:koi\s+|kuch\s+)?' + escaped_kw + r'\b'
        ]
        if any(re.search(pat, text_lower) for pat in hindi_post_patterns):
            return True

        # English post-negations
        eng_post_patterns = [
            escaped_kw + r'\s+(?:is\s+)?(?:absent|denied|ruled\s+out|none|zero|0|negative)\b',
            escaped_kw + r'\s*[:\-]\s*(?:no|none|negative|denied|nil)\b'
        ]
        if any(re.search(pat, text_lower) for pat in eng_post_patterns):
            return True

        # Check negation patterns including auxiliary verbs
        adj = r'(?:(?:sudden|severe|mild|moderate|acute|chronic|constant|sharp|dull|throbbing|excruciating)\s+)*'
        neg_patterns = [
            r'\b(?:do\s+not|don\'t|does\s+not|doesn\'t|did\s+not|didn\'t|have\s+not|haven\'t|has\s+not|hasn\'t|am\s+not)\s+(?:have|had|experience|experienced|feel|felt|notice|noticed)?\s*(?:a\s+|an\s+|the\s+|any\s+)?(?:signs?\s+of\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\b(?:have|has|had)\s+never\s+(?:had|experienced|felt|noticed)\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\b(?:no|not|never|without|denies|deny|zero)\s+(?:a\s+|an\s+|the\s+|any\s+)?(?:signs?\s+of\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\bno\s+history\s+of\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\bnegative\s+for\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?\band\s+(?:no\s+)?(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?,\s*(?:no\s+)?(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
            r'\b(?:no|without)\s+[a-z\s,]+?\bor\s+(?:a\s+|an\s+|the\s+|any\s+)?' + adj + re.escape(keyword_lower) + r'\b',
        ]
        if any(re.search(pat, text_lower) for pat in neg_patterns):
            return True
        
        # Check if keyword in text is only in a family attribution clause
        clauses = re.split(r'[\.\;\!\n]|\b(?:but|however|although|whereas)\b', text_lower)
        keyword_clauses = [c for c in clauses if keyword_lower in c]
        if keyword_clauses:
            fam_pattern = r'\b(?:father|mother|mom|dad|brother|sister|parent|parents|family|uncle|aunt|grand(?:father|mother|parent|ma|pa)|cousin|relative)\b'
            if all(re.search(fam_pattern, c) for c in keyword_clauses):
                return True
        
        # Check if keyword in text is solely in a historical/resolved context
        if keyword_clauses:
            hist_pattern = r'\b(?:yesterday|past|previously|prior|last\s+(?:week|month|year|night)|had|resolved|ago)\b.*?(?:but\s+not\s+now|no\s+longer|resolved|gone|subsided|improved|not\s+anymore|not\s+today)\b'
            if all(re.search(hist_pattern, c) for c in keyword_clauses):
                return True
                
        return False

    def detect_emergencies(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect emergency warning signs from patient data.
        
        Args:
            patient_data: Structured patient information from PatientInfoCollector
        
        Returns:
            Dictionary with detection results:
            - has_red_flags (bool)
            - severity (str)
            - flags (list)
            - should_escalate (bool)
            - guidance (str)
            - requires_immediate_attention (bool)
        
        Raises:
            ValueError: If patient_data is invalid
        """
        if not patient_data or not isinstance(patient_data, dict):
            raise ValueError("patient_data must be a non-empty dictionary")
        
        self.logger.info("Detecting emergency warning signs")
        
        flags = []
        negative_symptoms = [ns.lower() for ns in patient_data.get("symptom_details", {}).get("negative_symptoms", [])]
        
        # Rule-based detection from patient symptoms and history
        flags.extend(self._check_chest_pain_symptoms(patient_data, negative_symptoms))
        flags.extend(self._check_breathing_difficulty(patient_data, negative_symptoms))
        flags.extend(self._check_stroke_warnings(patient_data, negative_symptoms))
        flags.extend(self._check_severe_bleeding(patient_data, negative_symptoms))
        flags.extend(self._check_loss_of_consciousness(patient_data, negative_symptoms))
        flags.extend(self._check_seizures(patient_data, negative_symptoms))
        flags.extend(self._check_severe_allergic_reaction(patient_data, negative_symptoms))
        
        # Evaluate vital signs against emergency thresholds
        flags.extend(self._check_vital_signs_emergency(patient_data))
        
        # Assess overall severity
        severity = self._assess_severity(flags, patient_data)
        should_escalate = self._should_escalate_to_emergency(flags, severity)
        requires_immediate = should_escalate
        
        # Generate appropriate guidance
        guidance = self._generate_urgent_care_guidance(flags, severity, should_escalate)
        
        has_flags = len(flags) > 0
        
        result = {
            "has_red_flags": has_flags,
            "severity": severity,
            "flags": flags,
            "should_escalate": should_escalate,
            "guidance": guidance,
            "requires_immediate_attention": requires_immediate,
            "flag_count": len(flags)
        }
        
        self.logger.info(f"Emergency detection complete: {len(flags)} flags, severity={severity}")
        return result
    
    def _check_chest_pain_symptoms(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Check for chest pain and related symptoms, distinguishing resolved historical
        episodes from active emergencies and respecting explicit negations.
        """
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        details = patient_data.get("symptom_details", {})
        historical_symptoms = [h.lower() for h in details.get("historical_symptoms", [])]
        all_text = chief_complaint + " " + " ".join(symptoms)
        
        # 1. Check if chest pain is explicitly negated
        is_negated = (
            any("chest pain" in neg or "chest discomfort" in neg or "chest pressure" in neg for neg in negs) or
            any(self._is_negated_flag(k, all_text, negs) for k in self.chest_pain_keywords)
        )
        if is_negated:
            return flags
        
        # Check mentions of chest pain keywords
        has_cp_mention = (
            any(k in chief_complaint for k in self.chest_pain_keywords) or
            any(k in s for s in symptoms for k in self.chest_pain_keywords) or
            any("chest pain" in h or "chest discomfort" in h or "chest pressure" in h for h in historical_symptoms)
        )
        
        if not has_cp_mention:
            return flags
        
        # 2. Check if chest pain is historical and resolved
        hist_patterns = [
            r'\b(?:yesterday|past|previously|prior|last\s+(?:week|month|year|night)|had|ago)\b.*?\b(?:but\s+not\s+now|no\s+longer|resolved|gone|subsided|improved|disappeared|stopped|not\s+anymore|not\s+now|not\s+today|none\s+today|completely\s+resolved|fully\s+resolved|went\s+away)\b',
            r'\bchest\s+pain\b.*?\b(?:has\s+resolved|is\s+resolved|resolved|went\s+away|gone|subsided|stopped|not\s+having\s+it\s+now|no\s+longer\s+hurts|not\s+hurting\s+today|completely\s+resolved|disappeared)\b'
        ]
        in_historical_list = any("chest pain" in h or "chest discomfort" in h or "chest pressure" in h for h in historical_symptoms)
        text_shows_resolved = any(re.search(hp, all_text) for hp in hist_patterns)
        
        # Check recurrence / persistence markers
        recurrent_pattern = r'\b(?:came\s+back|returned|recurring|recurrent|still\s+hurts|still\s+having|started\s+again|hurting\s+again|happening\s+again|worse\s+today|persisting|continues?)\b'
        is_recurrent = bool(re.search(recurrent_pattern, all_text))
        
        # Active chest pain is in active symptoms list (and not marked past/resolved)
        active_in_symptoms = any(k in s for s in symptoms for k in self.chest_pain_keywords if "past" not in s and "resolved" not in s)
        
        if (in_historical_list or text_shows_resolved) and not is_recurrent and not active_in_symptoms:
            # Resolved historical chest pain without active recurrence
            flags.append({
                "type": "historical_chest_pain",
                "severity": "moderate",
                "description": "Historical chest pain reported (previously resolved)",
                "reason": "Prior chest pain requires medical evaluation to determine etiology, but is not currently reported as an acute active event",
                "source": "Historical symptoms or narrative",
                "should_escalate": False,
                "requires_clarification": True
            })
            return flags
        
        # 3. Active or recurrent chest pain -> Acute emergency escalation
        flags.append({
            "type": "chest_pain",
            "severity": "high",
            "description": "Chest pain reported (active or recurrent)",
            "reason": "Active chest pain can be a sign of acute coronary syndrome or pulmonary emergency",
            "source": "Chief complaint or symptoms",
            "should_escalate": True
        })
        
        # Check for aggravating factors (exercise triggers are particularly concerning)
        chief_text = patient_data.get("chief_complaint", {}).get("reason", "").lower()
        if any(word in chief_text for word in ["exercise", "exertion", "physical activity", "exerted"]):
            flags.append({
                "type": "chest_pain_with_exertion",
                "severity": "critical",
                "description": "Chest pain triggered by exercise or exertion",
                "reason": "Exercise-induced chest pain suggests possible acute cardiac event",
                "source": "Chief complaint narrative",
                "should_escalate": True
            })
        
        # Check for accompanying symptoms that elevate urgency
        concerning_symptoms = ["shortness of breath", "difficulty breathing", "sweating", 
                             "faintness", "dizziness", "nausea", "heart racing", "palpitations"]
        accompanying = [s for s in symptoms if any(concern in s and not self._is_negated_flag(concern, s, negs) for concern in concerning_symptoms)]
        
        if accompanying:
            flags.append({
                "type": "chest_pain_with_symptoms",
                "severity": "critical",
                "description": f"Chest pain with: {', '.join(accompanying[:3])}",
                "reason": "Combination suggests possible acute coronary syndrome",
                "source": "Symptoms list",
                "should_escalate": True
            })
        
        return flags
    
    def _is_sudden_severe_headache(self, patient_data: Dict[str, Any], negative_symptoms: List[str]) -> bool:
        """
        Detect sudden-onset severe or thunderclap headache descriptions with intervening words,
        while strictly preserving separation from historical, negated, or family mentions.
        """
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        all_text = (chief_complaint + " " + " ".join(symptoms)).lower()
        
        details = patient_data.get("symptom_details", {})
        has_headache_symp = any("headache" in s or "head pain" in s or "migraine" in s for s in symptoms)
        onset_sudden = details.get("onset") == "sudden"
        sev_severe = details.get("severity") in ["severe", "10/10", "9/10", "8/10", "excruciating"]
        
        # Check explicit negative symptoms list
        if any("headache" in ns and ("sudden" in ns or "severe" in ns) for ns in negative_symptoms):
            return False
        
        # Thunderclap / worst headache explicit patterns
        thunderclap_patterns = [
            r'\bthunderclap\s+(?:severe\s+|excruciating\s+)?headache\b',
            r'\bworst\s+headache\s+(?:of\s+my\s+life|ever|i(?:\'ve|\s+have)\s+ever\s+had)\b',
            r'\bheadache\b.*?\bworst\s+(?:of\s+my\s+life|ever)\b',
        ]
        
        # Sudden + severe + headache flexible regexes (allowing intervening words/adjectives)
        sudden_words = r'(?:sudden|suddenly|abrupt|abruptly|instantly|all\s+at\s+once|out\s+of\s+nowhere|without\s+warning|thunderclap)'
        severe_words = r'(?:severe|excruciating|worst|extreme|extremely\s+painful|intense|unbearable|10/10|9/10|8/10|splitting)'
        headache_words = r'(?:headache|head\s+pain|head\s+ache|migraine|pain\s+in\s+(?:my\s+|the\s+)?head)'
        
        flexible_patterns = [
            # "sudden excruciating severe headache", "sudden, severe headache"
            rf'\b{sudden_words}\b(?:\s+[\w,-]+){{0,5}}\s+\b{severe_words}\b(?:\s+[\w,-]+){{0,3}}\s+\b{headache_words}\b',
            # "severe headache, starting suddenly", "worst headache of my life, starting suddenly"
            rf'\b{severe_words}\b(?:\s+[\w,-]+){{0,3}}\s+\b{headache_words}\b(?:\s+[\w,-]+){{0,6}}\s+\b{sudden_words}\b',
            # "headache started suddenly and is severe", "headache came on all at once and feels excruciating"
            rf'\b{headache_words}\b(?:\s+[\w,-]+){{0,6}}\s+\b{sudden_words}\b(?:\s+[\w,-]+){{0,6}}\s+\b{severe_words}\b',
            # "sudden onset severe headache", "sudden onset of headache which is severe"
            rf'\b{sudden_words}\b(?:\s+[\w,-]+){{0,4}}\s+\b{headache_words}\b(?:\s+[\w,-]+){{0,6}}\s+\b{severe_words}\b',
        ]
        
        candidate_matches = []
        for pat in thunderclap_patterns + flexible_patterns:
            for match in re.finditer(pat, all_text):
                candidate_matches.append(match.group(0))
        
        if not candidate_matches and has_headache_symp and onset_sudden and sev_severe:
            candidate_matches.append(f"sudden severe headache (structured: {details.get('severity')})")
        
        if not candidate_matches:
            return False
        
        # Verify negation, family attribution, and historical resolution
        for match_str in candidate_matches:
            if self._is_negated_flag("headache", all_text, negative_symptoms) or self._is_negated_flag("sudden severe headache", all_text, negative_symptoms):
                continue
            
            clauses = re.split(r'[\.\;\!\n]|\b(?:but|however|although|whereas)\b', all_text)
            match_clauses = [c for c in clauses if any(w in c for w in ["headache", "head pain", "migraine"])]
            
            # Check direct clause-level negation
            neg_clause_pattern = r'\b(?:do\s+not|don\'t|does\s+not|doesn\'t|did\s+not|didn\'t|have\s+not|haven\'t|has\s+not|hasn\'t|am\s+not|is\s+not|isn\'t|no|not|never|without|denies|deny)\b'
            if match_clauses and all(re.search(neg_clause_pattern, c) for c in match_clauses):
                continue
            
            # Check family attribution
            fam_pattern = r'\b(?:father|mother|mom|dad|brother|sister|parent|parents|family|uncle|aunt|grand(?:father|mother|parent|ma|pa)|cousin|relative)\b'
            if match_clauses and all(re.search(fam_pattern, c) for c in match_clauses):
                continue
            
            # Check historical resolution
            hist_pattern = r'\b(?:yesterday|past|previously|prior|last\s+(?:week|month|year|night)|had|resolved|ago)\b.*?(?:but\s+not\s+now|no\s+longer|resolved|gone|subsided|improved|not\s+anymore|not\s+today)\b'
            if match_clauses and all(re.search(hist_pattern, c) for c in match_clauses):
                continue
            
            return True
            
        return False
    
    def _check_breathing_difficulty(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for difficulty breathing, respecting explicit negations."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        all_text = chief_complaint + " " + " ".join(symptoms)
        
        # Check if breathing difficulty is explicitly negated
        is_breathing_negated = (
            any("breathing" in neg or "shortness of breath" in neg or "dyspnea" in neg for neg in negs) or
            any(self._is_negated_flag(k, all_text, negs) for k in self.breathing_difficulty_keywords)
        )
        if is_breathing_negated:
            return flags
        
        has_breathing_difficulty = (
            any(keyword in chief_complaint and not self._is_negated_flag(keyword, chief_complaint, negs) for keyword in self.breathing_difficulty_keywords) or
            any(keyword in s and not self._is_negated_flag(keyword, s, negs) for s in symptoms for keyword in self.breathing_difficulty_keywords)
        )
        
        if has_breathing_difficulty:
            flags.append({
                "type": "difficulty_breathing",
                "severity": "critical",
                "description": "Difficulty breathing or shortness of breath",
                "reason": "Respiratory distress is a medical emergency requiring immediate evaluation",
                "source": "Chief complaint or symptoms"
            })
        
        return flags
    
    def _check_stroke_warnings(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for stroke warning signs (FAST criteria) and sudden-onset severe / thunderclap headaches."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        
        # Check in both chief complaint and symptoms, plus full text
        all_text = chief_complaint + " " + " ".join(symptoms)
        
        # Check for FAST stroke signs
        has_fast_signs = any(keyword in all_text and not self._is_negated_flag(keyword, all_text, negs) for keyword in self.stroke_warning_keywords)
        
        # Check for sudden-onset severe / thunderclap headache
        has_sudden_severe_headache = self._is_sudden_severe_headache(patient_data, negs)
        
        if has_fast_signs or has_sudden_severe_headache:
            note_text = "FAST criteria / Thunderclap headache emergency" if has_sudden_severe_headache else "FAST criteria: Face drooping, Arm weakness, Speech difficulty, Time to call 108 or 112"
            desc_text = "Sudden-onset severe headache (possible intracranial emergency / thunderclap)" if has_sudden_severe_headache and not has_fast_signs else "Sudden neurological symptoms (possible stroke warning signs)"
            flags.append({
                "type": "stroke_warning_signs",
                "severity": "critical",
                "description": desc_text,
                "reason": "Sudden severe neurological deficit or thunderclap headache requires immediate emergency treatment",
                "source": "Chief complaint or symptoms",
                "note": note_text
            })
        
        return flags
    
    def _check_severe_bleeding(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for severe or uncontrolled bleeding."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        
        has_severe_bleeding = any(keyword in chief_complaint and not self._is_negated_flag(keyword, chief_complaint, negs) for keyword in self.severe_bleeding_keywords) or \
                             any(keyword in " ".join(symptoms) and not self._is_negated_flag(keyword, " ".join(symptoms), negs) for keyword in self.severe_bleeding_keywords)
        
        if has_severe_bleeding:
            flags.append({
                "type": "severe_bleeding",
                "severity": "critical",
                "description": "Severe or uncontrolled bleeding",
                "reason": "Uncontrolled bleeding is a life-threatening emergency requiring immediate care",
                "source": "Chief complaint or symptoms"
            })
        
        return flags
    
    def _check_loss_of_consciousness(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for loss of consciousness or fainting."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        
        all_text = chief_complaint + " " + " ".join(symptoms)
        has_loc = any(keyword in all_text and not self._is_negated_flag(keyword, all_text, negs) for keyword in self.loss_of_consciousness_keywords)
        
        if has_loc:
            flags.append({
                "type": "loss_of_consciousness",
                "severity": "critical",
                "description": "Loss of consciousness or fainting episode",
                "reason": "Loss of consciousness requires immediate emergency evaluation",
                "source": "Chief complaint or symptoms"
            })
        
        return flags
    
    def _check_seizures(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for seizures or convulsions."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        
        has_seizure = any(keyword in chief_complaint and not self._is_negated_flag(keyword, chief_complaint, negs) for keyword in self.seizure_keywords) or \
                     any(keyword in " ".join(symptoms) and not self._is_negated_flag(keyword, " ".join(symptoms), negs) for keyword in self.seizure_keywords)
        
        if has_seizure:
            flags.append({
                "type": "seizure",
                "severity": "critical",
                "description": "Seizure or convulsions",
                "reason": "Active or ongoing seizures require immediate emergency care",
                "source": "Chief complaint or symptoms"
            })
        
        return flags
    
    def _check_severe_allergic_reaction(self, patient_data: Dict[str, Any], negative_symptoms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for severe allergic reactions (anaphylaxis)."""
        flags = []
        negs = negative_symptoms or []
        
        chief_complaint = (patient_data.get("chief_complaint", {}).get("reason") or "").lower()
        symptoms = [s.lower() if s else "" for s in patient_data.get("symptoms", {}).get("list", [])]
        
        has_anaphylaxis = any(keyword in chief_complaint and not self._is_negated_flag(keyword, chief_complaint, negs) for keyword in self.severe_allergic_reaction_keywords) or \
                         any(keyword in " ".join(symptoms) and not self._is_negated_flag(keyword, " ".join(symptoms), negs) for keyword in self.severe_allergic_reaction_keywords)
        
        if has_anaphylaxis:
            flags.append({
                "type": "anaphylaxis",
                "severity": "critical",
                "description": "Severe allergic reaction (possible anaphylaxis)",
                "reason": "Anaphylaxis is a life-threatening emergency requiring immediate treatment",
                "source": "Chief complaint or symptoms"
            })
        
        return flags
    
    def _check_vital_signs_emergency(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check vital signs against emergency thresholds."""
        flags = []
        vitals = patient_data.get("vitals", {})
        
        # Blood Pressure - Emergency threshold: SBP >180 or DBP >120 (per AHA guidelines)
        bp = vitals.get("blood_pressure")
        if bp:
            try:
                # Parse "120/80 mmHg" format
                bp_str = bp.replace("mmHg", "").strip()
                parts = bp_str.split("/")
                if len(parts) == 2:
                    sbp = int(parts[0].strip())
                    dbp = int(parts[1].strip())
                    
                    if sbp > 180 or dbp > 120:
                        flags.append({
                            "type": "hypertensive_emergency",
                            "severity": "high",
                            "description": f"Critically elevated blood pressure: {bp}",
                            "reason": "BP >180/120 indicates hypertensive emergency risk",
                            "source": "Vital signs",
                            "threshold": "SBP >180 or DBP >120 mmHg"
                        })
            except (ValueError, AttributeError):
                pass  # Invalid BP format, skip
        
        # Heart Rate - Emergency thresholds: <40 or >120 bpm
        hr = vitals.get("heart_rate")
        if hr:
            try:
                hr_str = hr.replace("bpm", "").strip()
                hr_val = int(hr_str)
                
                if hr_val > 120:
                    flags.append({
                        "type": "tachycardia",
                        "severity": "moderate",
                        "description": f"Elevated heart rate: {hr}",
                        "reason": "Heart rate >120 bpm may indicate cardiac stress or other emergency",
                        "source": "Vital signs",
                        "threshold": ">120 bpm"
                    })
                elif hr_val < 40:
                    flags.append({
                        "type": "bradycardia",
                        "severity": "high",
                        "description": f"Severely low heart rate: {hr}",
                        "reason": "Heart rate <40 bpm may indicate cardiac emergency",
                        "source": "Vital signs",
                        "threshold": "<40 bpm"
                    })
            except (ValueError, AttributeError):
                pass
        
        # Temperature - Emergency thresholds: >103°F (39.4°C) or <95°F (35°C)
        temp = vitals.get("temperature")
        if temp:
            try:
                temp_val = float(temp)
                
                # Assume Celsius if <50 (as per patient_info_collector logic)
                if temp_val < 50:  # Celsius
                    if temp_val > 39.4:  # >103°F
                        flags.append({
                            "type": "severe_fever",
                            "severity": "moderate",
                            "description": f"Severe fever: {temp}°C",
                            "reason": "Temperature >39.4°C (103°F) indicates significant fever",
                            "source": "Vital signs",
                            "threshold": ">39.4°C (103°F)"
                        })
                    elif temp_val < 35:  # <95°F equivalent
                        flags.append({
                            "type": "hypothermia",
                            "severity": "high",
                            "description": f"Hypothermia: {temp}°C",
                            "reason": "Temperature <35°C (95°F) indicates dangerous hypothermia",
                            "source": "Vital signs",
                            "threshold": "<35°C (95°F)"
                        })
                else:  # Fahrenheit
                    if temp_val > 103:
                        flags.append({
                            "type": "severe_fever",
                            "severity": "moderate",
                            "description": f"Severe fever: {temp}°F",
                            "reason": "Temperature >103°F indicates significant fever",
                            "source": "Vital signs",
                            "threshold": ">103°F"
                        })
                    elif temp_val < 95:
                        flags.append({
                            "type": "hypothermia",
                            "severity": "high",
                            "description": f"Hypothermia: {temp}°F",
                            "reason": "Temperature <95°F indicates dangerous hypothermia",
                            "source": "Vital signs",
                            "threshold": "<95°F"
                        })
            except (ValueError, TypeError):
                pass
        
        # Oxygen Saturation - Emergency threshold: <90%
        o2 = vitals.get("oxygen_saturation")
        if o2:
            try:
                o2_str = o2.replace("%", "").strip()
                o2_val = int(o2_str)
                
                if o2_val < 90:
                    flags.append({
                        "type": "hypoxia",
                        "severity": "critical",
                        "description": f"Critical oxygen saturation: {o2}",
                        "reason": "SpO2 <90% indicates dangerous hypoxia requiring immediate intervention",
                        "source": "Vital signs",
                        "threshold": "<90%"
                    })
            except (ValueError, AttributeError):
                pass
        
        # Respiratory Rate - Emergency thresholds: <8 or >30 breaths/min
        rr = vitals.get("respiratory_rate")
        if rr:
            try:
                rr_str = rr.replace("breaths/min", "").replace("/min", "").strip()
                rr_val = int(rr_str)
                
                if rr_val > 30:
                    flags.append({
                        "type": "tachypnea",
                        "severity": "high",
                        "description": f"Rapid breathing: {rr}",
                        "reason": "Respiratory rate >30 breaths/min indicates respiratory distress",
                        "source": "Vital signs",
                        "threshold": ">30 breaths/min"
                    })
                elif rr_val < 8:
                    flags.append({
                        "type": "bradypnea",
                        "severity": "critical",
                        "description": f"Critically slow breathing: {rr}",
                        "reason": "Respiratory rate <8 breaths/min is a critical emergency",
                        "source": "Vital signs",
                        "threshold": "<8 breaths/min"
                    })
            except (ValueError, AttributeError):
                pass
        
        return flags
    
    def _assess_severity(self, flags: List[Dict[str, Any]], patient_data: Dict[str, Any]) -> str:
        """
        Assess overall severity level from detected flags.
        
        Args:
            flags: List of detected flags
            patient_data: Patient information
        
        Returns:
            Severity level string
        """
        if not flags:
            return self.SEVERITY_NONE
        
        # Check for critical flags
        critical_types = [
            "chest_pain_with_exertion", "chest_pain_with_symptoms",
            "difficulty_breathing", "stroke_warning_signs", "severe_bleeding",
            "loss_of_consciousness", "seizure", "anaphylaxis",
            "hypoxia", "bradypnea", "bradycardia"
        ]
        
        for flag in flags:
            if flag.get("type") in critical_types:
                return self.SEVERITY_CRITICAL
        
        # Check for high-severity flags
        high_types = [
            "chest_pain", "hypertensive_emergency", "hypothermia", "tachypnea"
        ]
        
        has_high = any(flag.get("type") in high_types for flag in flags)
        if has_high:
            return self.SEVERITY_HIGH
        
        # Multiple moderate flags
        if len(flags) >= 3:
            return self.SEVERITY_HIGH
        
        if len(flags) >= 2:
            return self.SEVERITY_MODERATE
        
        return self.SEVERITY_LOW
    
    def _should_escalate_to_emergency(self, flags: List[Dict[str, Any]], severity: str) -> bool:
        """
        Determine if situation requires emergency escalation.
        
        Args:
            flags: Detected flags
            severity: Assessed severity level
        
        Returns:
            Whether to escalate to emergency
        """
        # Escalate if critical severity
        if severity == self.SEVERITY_CRITICAL:
            return True
        
        # Escalate if multiple critical-type flags or active chest pain
        critical_flag_types = [
            "chest_pain", "chest_pain_with_exertion", "chest_pain_with_symptoms",
            "difficulty_breathing", "stroke_warning_signs", "severe_bleeding",
            "loss_of_consciousness", "seizure", "anaphylaxis", "hypoxia"
        ]
        
        critical_count = sum(1 for flag in flags if flag.get("type") in critical_flag_types)
        if critical_count >= 1:
            return True
        
        return False
    
    def _generate_urgent_care_guidance(self, flags: List[Dict[str, Any]], 
                                       severity: str, should_escalate: bool) -> str:
        """
        Generate appropriate guidance based on detected flags.
        
        Args:
            flags: Detected flags
            severity: Assessed severity level
            should_escalate: Whether escalation is needed
        
        Returns:
            Guidance string
        """
        if not flags:
            return "No emergency warning signs detected. Continue with routine intake."
        
        if should_escalate:
            return (
                "[EMERGENCY WARNING SIGNS DETECTED]\n\n"
                "Based on your reported symptoms, you require immediate emergency medical attention.\n\n"
                "EMERGENCY: Call 108 for an ambulance or 112 for integrated emergency assistance immediately. "
                "Do not delay seeking emergency medical care.\n\n"
                "Do NOT wait for an appointment. Do NOT drive yourself if symptoms are active.\n\n"
                "If you are unable to call, ask someone nearby to call for you immediately.\n\n"
                "Reported warning signs requiring emergency care:\n" +
                "\n".join(f"* {flag['description']}" for flag in flags[:5]) +
                ("\n* Additional concerns..." if len(flags) > 5 else "")
            )
        
        # Historical chest pain specific guidance
        if any(f.get("type") == "historical_chest_pain" for f in flags):
            return (
                "Prior chest pain noted. While you report this episode has resolved, you should discuss it with a physician. "
                "If chest pain recurs, radiates, or is accompanied by shortness of breath or dizziness, seek emergency medical care immediately."
            )
        
        elif severity == self.SEVERITY_HIGH:
            return (
                "[URGENT CARE NEEDED]\n\n"
                "Your symptoms indicate a need for urgent medical evaluation.\n\n"
                "Please seek care at an urgent care facility or emergency room as soon as possible.\n"
                "Do not delay - seek evaluation within the next 1-2 hours.\n\n"
                "If symptoms worsen or change, call 108 for an ambulance or 112 for integrated emergency assistance immediately."
            )
        
        elif severity == self.SEVERITY_MODERATE:
            return (
                "[MEDICAL EVALUATION RECOMMENDED]\n\n"
                "Your symptoms warrant medical evaluation.\n\n"
                "Please contact your primary care physician or seek urgent care evaluation today.\n"
                "Do not wait for a routine appointment."
            )
        
        else:
            return (
                "Your symptoms have been reviewed. "
                "Routine follow-up questions will continue to gather necessary information. "
                "If your condition changes or worsens, seek immediate medical attention."
            )
    
    def has_critical_flags(self, detection_result: Dict[str, Any]) -> bool:
        """
        Check if detection result contains critical flags.
        
        Args:
            detection_result: Result from detect_emergencies()
        
        Returns:
            Whether critical flags were detected
        """
        return detection_result.get("severity") == self.SEVERITY_CRITICAL
    
    def get_red_flag_summary(self, detection_result: Dict[str, Any]) -> str:
        """
        Get a brief summary of detected red flags.
        
        Args:
            detection_result: Result from detect_emergencies()
        
        Returns:
            Summary string
        """
        flags = detection_result.get("flags", [])
        if not flags:
            return "No red flags detected."
        
        flag_types = [flag.get("type", "unknown") for flag in flags]
        unique_types = list(set(flag_types))
        
        return f"Red flags detected ({len(flags)}): {', '.join(unique_types)}"


# Export the class for easy importing
__all__ = ['RedFlagDetector']
