import os
import sys
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from agents.agent_decision import config as global_config, get_clinical_agent, reset_clinical_agent_session
from agents.clinical_pre_consultation_agent.patient_info_collector import PatientInfoCollector
from agents.clinical_pre_consultation_agent.report_generator import PreConsultationReportGenerator, clean_text_entities


class TestBrowserFeverScenarioRegression(unittest.TestCase):
    """
    Regression test suite for the 5 real-browser Fever scenario failures:
    1. Retaining symptom-specific severity (5/10) and avoiding repeated severity questions.
    2. Preserving patient-reported location ("forehead and both sides") without overwriting with "neck".
    3. Correct positive vs negative symptom classification (denied nausea strictly absent from reported symptoms).
    4. Complete elimination of HTML entities (&#x53;, &#x6E;, &#x4E;, etc.) across all text pipelines.
    5. Clear capture and formatting of all pertinent negatives (neck/shoulder radiation, screen-time, nausea, light/sound, vision).
    """

    def setUp(self):
        self.orig_llm = global_config.clinical_consultation.llm
        global_config.clinical_consultation.llm = None  # deterministic validation
        reset_clinical_agent_session("*")

    def tearDown(self):
        reset_clinical_agent_session("*")
        global_config.clinical_consultation.llm = self.orig_llm

    def test_failure_1_severity_retention_and_no_re_query(self):
        """
        Failure 1: Patient reports headache severity as 5/10.
        System must retain severity 5/10 and NOT ask for severity again.
        """
        agent = get_clinical_agent(session_id="test_fever_sev")

        # Turn 1: Initial complaint
        r1 = agent.process_patient_info("Mujhe 2 din se fever ho raha hai.")
        self.assertEqual(r1["patient_data"]["chief_complaint"]["reason"], "Fever")

        # Turn 2: Patient reports headache with severity 5/10 and location
        r2 = agent.process_patient_info("I also have a headache. Severity is 5/10, located on forehead and both sides.")
        p2 = r2["patient_data"]
        
        # Verify severity 5/10 is recorded
        self.assertEqual(p2["symptom_details"]["severity"], "5/10")

        # Verify next question does NOT repeat the severity question
        next_q = r2.get("next_question", {}).get("question", "")
        self.assertNotIn("scale of 1 to 10", next_q.lower())
        self.assertNotIn("how severe", next_q.lower())

    def test_failure_2_location_preservation_not_overwritten_by_neck(self):
        """
        Failure 2: Location was reported as 'forehead and both sides'.
        In turn 3, patient denies radiation to neck.
        Report must preserve 'forehead and both sides', NOT overwrite with 'Neck'.
        """
        agent = get_clinical_agent(session_id="test_fever_loc")

        # Turn 1
        agent.process_patient_info("Mujhe 2 din se fever ho raha hai.")
        
        # Turn 2: Forehead and both sides
        r2 = agent.process_patient_info("I also have a headache. Severity is 5/10, located on forehead and both sides.")
        self.assertEqual(r2["patient_data"]["symptom_details"]["location"], "forehead and both sides")

        # Turn 3: Denies radiation to neck and shoulders
        r3 = agent.process_patient_info("Does not radiate to neck or shoulders. Not worse with screen time. No nausea, light sensitivity, or vision changes.")
        p3 = r3["patient_data"]

        # CRITICAL ASSERTION: Location must remain "forehead and both sides" and NEVER become "neck"
        self.assertEqual(p3["symptom_details"]["location"], "forehead and both sides")
        self.assertNotEqual(p3["symptom_details"]["location"].lower(), "neck")

        # In final generated report
        rep = agent.report_generator.generate_report(p3, r3.get("red_flag_result"), agent.conversation_history)
        self.assertEqual(rep["symptoms"]["location"], "forehead and both sides")

    def test_failure_3_denied_nausea_strictly_purged_from_reported_symptoms(self):
        """
        Failure 3: Patient explicitly denies nausea.
        Nausea must NEVER appear under reported symptoms, and MUST appear under negative symptoms.
        """
        agent = get_clinical_agent(session_id="test_fever_nausea")

        # Turn 1 & 2
        agent.process_patient_info("Mujhe 2 din se fever ho raha hai.")
        agent.process_patient_info("I also have a headache. Severity is 5/10, located on forehead and both sides.")

        # Turn 3: Explicit denial of nausea
        r3 = agent.process_patient_info("Does not radiate to neck or shoulders. Not worse with screen time. No nausea, light sensitivity, or vision changes.")
        p3 = r3["patient_data"]

        # Positive symptoms list must NOT contain nausea
        reported_symps = [s.lower() for s in p3["symptoms"]["list"]]
        self.assertNotIn("nausea", reported_symps)

        # Negative symptoms must contain nausea denial
        neg_symps = [s.lower() for s in p3["symptom_details"]["negative_symptoms"]]
        self.assertTrue(any("nausea" in ns for ns in neg_symps))

        # In final formatted report
        rep = agent.report_generator.generate_report(p3, r3.get("red_flag_result"), agent.conversation_history)
        self.assertNotIn("nausea", [s.lower() for s in rep["symptoms"]["reported_symptoms"]])
        self.assertTrue(any("nausea" in ns.lower() for ns in rep["symptoms"]["negative_symptoms"]))

    def test_failure_4_html_entity_elimination(self):
        """
        Failure 4: Hex HTML entities (e.g. &#x53;ore throat, &#x6E;eck, &#x4E;ot asked).
        Must be cleanly decoded to readable characters in all cases (nested, upper/lowercase x).
        """
        # Test cleaning function directly with single-letter hex entities
        self.assertEqual(clean_text_entities("&#x53;ore throat"), "Sore throat")
        self.assertEqual(clean_text_entities("&#x6E;eck"), "neck")
        self.assertEqual(clean_text_entities("&#x4E;ot asked"), "Not asked")
        self.assertEqual(clean_text_entities("&#X53;ore throat"), "Sore throat")  # uppercase X
        self.assertEqual(clean_text_entities("&amp;#x53;ore throat"), "Sore throat")  # double escaped

        # Test within patient collector
        collector = PatientInfoCollector()
        extracted = collector.extract_from_patient_text("&#x53;ore throat and &#x6E;eck discomfort")
        symps = extracted.get("symptoms", {}).get("symptoms_found", [])
        for s in symps:
            self.assertNotIn("&#x", s)
            self.assertNotIn("&#X", s)

        # Test within formatted report text
        generator = PreConsultationReportGenerator()
        sample_data = {
            "chief_complaint": {"reason": "&#x53;ore throat", "duration": "2 days", "status": "provided"},
            "symptoms": {"list": ["&#x53;ore throat"], "status": "provided"},
            "symptom_details": {"location": "&#x6E;eck", "negative_symptoms": ["no cough"]},
            "vitals": {"status": "not_asked"},
            "medical_history": {"status": "not_asked"},
            "medications": {"status": "not_asked"},
            "allergies": {"status": "not_asked"},
            "lifestyle": {"status": "not_asked"},
            "demographics": {"age": None, "age_status": "unknown"}
        }
        rep = generator.generate_report(sample_data, None, [])
        formatted = generator.format_report_text(rep)
        self.assertNotIn("&#x", formatted)
        self.assertNotIn("&#X", formatted)
        self.assertNotIn("&#53;", formatted)
        self.assertIn("Sore throat", formatted)
        self.assertIn("neck", formatted)

    def test_failure_5_all_pertinent_negatives_captured_and_displayed(self):
        """
        Failure 5: Patient provides explicit negative answers for:
        - neck/shoulder radiation
        - screen-time aggravation
        - nausea
        - light/sound sensitivity
        - vision changes
        Report must explicitly display all 5 pertinent negatives.
        """
        agent = get_clinical_agent(session_id="test_fever_pert_negs")

        agent.process_patient_info("Mujhe 2 din se fever ho raha hai.")
        agent.process_patient_info("I also have a headache. Severity is 5/10, located on forehead and both sides.")
        r3 = agent.process_patient_info("Does not radiate to neck or shoulders. Not worse with screen time. No nausea, light sensitivity, or vision changes.")
        p3 = r3["patient_data"]

        neg_symps = [s.lower() for s in p3["symptom_details"]["negative_symptoms"]]
        
        # Check all 5 categories are recorded in structured negative symptoms
        self.assertTrue(any("neck" in s or "shoulder" in s or "radiation" in s for s in neg_symps), "Radiation negative missing")
        self.assertTrue(any("screen" in s for s in neg_symps), "Screen-time aggravation negative missing")
        self.assertTrue(any("nausea" in s for s in neg_symps), "Nausea negative missing")
        self.assertTrue(any("light" in s or "sound" in s for s in neg_symps), "Light/sound sensitivity negative missing")
        self.assertTrue(any("vision" in s or "visual" in s for s in neg_symps), "Vision changes negative missing")

        # Verify in generated report text
        rep = agent.report_generator.generate_report(p3, r3.get("red_flag_result"), agent.conversation_history)
        report_text = agent.report_generator.format_report_text(rep)

        self.assertIn("Explicitly Denied (Pertinent Negatives):", report_text)
        report_text_lower = report_text.lower()
        self.assertTrue("neck" in report_text_lower or "radiation" in report_text_lower)
        self.assertTrue("screen" in report_text_lower)
        self.assertTrue("nausea" in report_text_lower)
        self.assertTrue("light" in report_text_lower)
        self.assertTrue("vision" in report_text_lower)

        # Verify in physician clinical summary
        clin_summary = rep.get("clinical_summary", "").lower()
        self.assertIn("pertinent negatives", clin_summary)
        self.assertTrue("nausea" in clin_summary)


if __name__ == "__main__":
    unittest.main()
