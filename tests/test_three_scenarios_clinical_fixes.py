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

from config import Config
from agents.clinical_pre_consultation_agent import (
    ClinicalPreConsultationAgent,
    PatientInfoCollector,
    RedFlagDetector,
    FollowupQuestionGenerator,
    PreConsultationReportGenerator
)
from agents.agent_decision import get_clinical_agent, reset_clinical_agent_session


class TestThreeScenariosClinicalFixes(unittest.TestCase):
    """
    Comprehensive regression tests for the 9 clinical issues across
    three clinical scenarios: Fever, Abdominal Pain, and Headache.
    """

    def setUp(self):
        from agents.agent_decision import config as global_config
        self.orig_llm = global_config.clinical_consultation.llm
        global_config.clinical_consultation.llm = None  # deterministic validation
        self.config = global_config
        reset_clinical_agent_session("*")  # Clean all agent cache

    def tearDown(self):
        reset_clinical_agent_session("*")
        from agents.agent_decision import config as global_config
        global_config.clinical_consultation.llm = self.orig_llm

    # =========================================================================
    # SCENARIO 1: FEVER
    # =========================================================================
    def test_scenario_1_fever_chief_complaint_and_duration(self):
        """Issue 1 & Issue 5: Chief complaint and duration correctly extracted for Fever."""
        collector = PatientInfoCollector()
        collector.extract_from_patient_text("Mujhe 2 din se tez bukhar ho raha hai.")
        p = collector.get_patient_data()
        
        self.assertEqual(p["chief_complaint"]["reason"], "Fever")
        self.assertEqual(p["chief_complaint"]["duration"], "2 days")
        self.assertIn("fever", p["symptoms"]["list"])

    def test_scenario_1_fever_negations_and_resolved_symptoms(self):
        """Issue 3 & Issue 6: Denied symptoms vs resolved symptoms for Fever."""
        collector = PatientInfoCollector()
        # Patient reports fever, headache resolved, but denies cough and vomiting
        collector.extract_from_patient_text(
            "I have fever for 3 days. My headache has resolved yesterday, but no cough and no vomiting."
        )
        p = collector.get_patient_data()
        symps = p["symptoms"]["list"]
        negs = p["symptom_details"]["negative_symptoms"]
        hist = p["symptom_details"]["historical_symptoms"]

        self.assertIn("fever", symps)
        # Denied symptoms must not be positive
        self.assertNotIn("cough", symps)
        self.assertNotIn("vomiting", symps)
        self.assertTrue(any("cough" in n for n in negs))
        self.assertTrue(any("vomiting" in n for n in negs))
        # Resolved symptom must be in historical, NOT in negative
        self.assertTrue(any("headache" in h for h in hist))
        self.assertFalse(any("no headache" in n for n in negs))

    def test_scenario_1_fever_html_encoding_and_recommendations(self):
        """Issue 7 & Issue 8: HTML unescaping and condition-specific recommendations for Fever."""
        collector = PatientInfoCollector()
        collector.extract_from_patient_text("I&#x27;m having high fever &amp; body pain for 2 days.")
        p = collector.get_patient_data()
        
        gen = PreConsultationReportGenerator()
        rep = gen.generate_report(p)
        text_rep = gen.format_report_text(rep)

        # Issue 7: No HTML encoding in reports
        self.assertNotIn("&#x27;", text_rep)
        self.assertNotIn("&amp;", text_rep)
        self.assertNotIn("&quot;", text_rep)

        # Issue 8: Fever-specific recommendations
        recs = rep.get("recommendations", [])
        self.assertTrue(len(recs) > 0, "Recommendations list should not be empty")
        rec_text = " ".join(recs).lower()
        self.assertTrue(
            "temperature" in rec_text or "fever" in rec_text or "hydration" in rec_text,
            f"Fever recommendations should mention temperature/hydration. Got: {rec_text}"
        )
        self.assertIn("RECOMMENDATIONS & SUGGESTED NEXT STEPS", text_rep)

    def test_scenario_1_fever_patient_corrections(self):
        """Issue 9: Patient corrections properly update fever report."""
        agent = ClinicalPreConsultationAgent(self.config)
        r1 = agent.process_patient_info("I have fever for 1 day.")
        # Trigger early completion -> mandatory final confirmation -> patient confirms
        agent.process_patient_info("That's all the symptoms, generate report please.")
        r2 = agent.process_patient_info("No, nothing else.")
        self.assertIsNotNone(agent._latest_report)
        self.assertEqual(agent._latest_report["chief_complaint"]["duration"], "1 day")

        # Now patient provides a correction: duration is actually 4 days, no fever but headache
        r3 = agent.process_patient_info("Actually the duration is 4 days, and I don't have fever, chief complaint is headache.")
        p = agent.patient_info_collector.get_patient_data()

        self.assertEqual(p["chief_complaint"]["duration"], "4 days")
        self.assertEqual(agent._latest_report["chief_complaint"]["duration"], "4 days")
        self.assertNotIn("fever", p["symptoms"]["list"])
        self.assertIn("headache", p["symptoms"]["list"])

    # =========================================================================
    # SCENARIO 2: ABDOMINAL PAIN
    # =========================================================================
    def test_scenario_2_abdominal_pain_chief_complaint_and_deduplication(self):
        """Issue 1 & Issue 4: Chief complaint and deduplication of synonymous symptoms."""
        collector = PatientInfoCollector()
        # Multi-word variants and generic pain
        collector.extract_from_patient_text("I have severe stomach pain, abdominal cramps, and generic pain.")
        p = collector.get_patient_data()
        
        self.assertEqual(p["chief_complaint"]["reason"], "Abdominal pain")
        symps = p["symptoms"]["list"]
        # Issue 4: Both stomach pain and abdominal cramps normalize to 'abdominal pain' without duplicates
        abdominal_entries = [s for s in symps if s.lower() == "abdominal pain"]
        self.assertEqual(len(abdominal_entries), 1, f"Expected 1 canonical abdominal pain, got: {symps}")
        # Bare generic 'pain' dropped when specific pain present
        self.assertNotIn("pain", [s.lower() for s in symps])

    def test_scenario_2_abdominal_pain_no_invented_duration(self):
        """Issue 5: Agent does NOT invent a duration when none provided."""
        collector = PatientInfoCollector()
        # Patient says 'today' but specifies NO duration value
        collector.extract_from_patient_text("Doctor, today I have severe abdominal pain.")
        p = collector.get_patient_data()
        
        gen = PreConsultationReportGenerator()
        rep = gen.generate_report(p)
        dur = rep["chief_complaint"]["duration"]

        self.assertEqual(dur, "Not provided", f"Expected 'Not provided', got invented duration: '{dur}'")

    def test_scenario_2_abdominal_pain_recommendations(self):
        """Issue 8: Condition-specific recommendations for Abdominal Pain."""
        collector = PatientInfoCollector()
        collector.extract_from_patient_text("I have had sharp stomach cramps for 2 days.")
        p = collector.get_patient_data()
        
        gen = PreConsultationReportGenerator()
        rep = gen.generate_report(p)
        recs = rep.get("recommendations", [])
        rec_text = " ".join(recs).lower()

        self.assertTrue(
            "abdominal" in rec_text or "quadrant" in rec_text or "npo" in rec_text or "diet" in rec_text or "ultrasound" in rec_text,
            f"Expected abdominal examination or diet recommendations. Got: {rec_text}"
        )

    # =========================================================================
    # SCENARIO 3: HEADACHE
    # =========================================================================
    def test_scenario_3_headache_chief_complaint_and_negations(self):
        """Issue 1 & Issue 3: Headache identified, denied symptoms excluded."""
        collector = PatientInfoCollector()
        collector.extract_from_patient_text("My head hurts badly for 3 days, but no fever and no nausea.")
        p = collector.get_patient_data()
        
        self.assertEqual(p["chief_complaint"]["reason"], "Headache")
        self.assertEqual(p["chief_complaint"]["duration"], "3 days")
        symps = p["symptoms"]["list"]
        self.assertIn("headache", symps)
        self.assertNotIn("fever", symps)
        self.assertNotIn("nausea", symps)

    def test_scenario_3_headache_recommendations(self):
        """Issue 8: Condition-specific recommendations for Headache."""
        collector = PatientInfoCollector()
        collector.extract_from_patient_text("Severe throbbing headache for 2 days.")
        p = collector.get_patient_data()
        
        gen = PreConsultationReportGenerator()
        rep = gen.generate_report(p)
        recs = rep.get("recommendations", [])
        rec_text = " ".join(recs).lower()

        self.assertTrue(
            "neurological" in rec_text or "headache" in rec_text or "snoop" in rec_text or "neuroimaging" in rec_text,
            f"Expected neurological or headache recommendations. Got: {rec_text}"
        )

    # =========================================================================
    # CROSS-SCENARIO: SESSION ISOLATION (ISSUE 2)
    # =========================================================================
    def test_session_isolation_across_scenarios(self):
        """Issue 2: Information from previous sessions NEVER appears in new sessions."""
        session_a = "clin_aaaaaaaa-0000-0000-0000-000000000001"
        session_b = "clin_bbbbbbbb-0000-0000-0000-000000000002"

        # Session A: Fever patient
        agent_a = get_clinical_agent(session_a)
        agent_a.process_patient_info("I have fever for 3 days and chills.")
        data_a = agent_a.patient_info_collector.get_patient_data()
        self.assertIn("fever", data_a["symptoms"]["list"])

        # Session B: Brand new Headache patient
        agent_b = get_clinical_agent(session_b)
        agent_b.process_patient_info("I have severe headache since yesterday.")
        data_b = agent_b.patient_info_collector.get_patient_data()

        # Session B MUST NOT have fever or chills from Session A!
        self.assertIn("headache", data_b["symptoms"]["list"])
        self.assertNotIn("fever", data_b["symptoms"]["list"], "Session B leaked 'fever' from Session A!")
        self.assertNotIn("chills", data_b["symptoms"]["list"], "Session B leaked 'chills' from Session A!")
        self.assertEqual(data_b["chief_complaint"]["reason"], "Headache")


if __name__ == "__main__":
    unittest.main()
