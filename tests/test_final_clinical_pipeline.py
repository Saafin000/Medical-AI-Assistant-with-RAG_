import os
import sys
import unittest
from unittest.mock import MagicMock

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from agents.agent_decision import config as global_config, get_clinical_agent, reset_clinical_agent_session


class TestFinalClinicalPipelineDeterministic(unittest.TestCase):
    """
    Deterministic & Mocked-LLM tests for the Clinical Pre-Consultation pipeline:
    A. Observed Browser Fever Scenario (Hinglish + English, timelines, negations, no throat tightness, no duplicate cough)
    B. Headache Scenario (no invented duration, preserved denials)
    C. Abdominal Pain Scenario (positive/negative GI separation, no leakage)
    D. Patient Correction Scenario (LLM-mocked + deterministic correction updates structured data & regenerates report)
    E. Session Isolation (cross-session isolation during intake and corrections)
    """

    def setUp(self):
        self.orig_llm = global_config.clinical_consultation.llm
        global_config.clinical_consultation.llm = None
        reset_clinical_agent_session("*")

    def tearDown(self):
        reset_clinical_agent_session("*")
        global_config.clinical_consultation.llm = self.orig_llm

    def test_a_observed_browser_fever_scenario(self):
        agent = get_clinical_agent(session_id="test_final_fever")

        # Turn 1: Initial Hinglish complaint
        r1 = agent.process_patient_info(
            "Mujhe pichhle 3 din se fever ho raha hai. Kal se headache aur body pain bhi hai. "
            "Raat ko thand lagkar kapkapi hoti hai aur aaj se sore throat aur halka cough bhi shuru ho gaya hai."
        )
        p1 = r1["patient_data"]
        self.assertEqual(p1["chief_complaint"]["reason"], "Fever")
        self.assertEqual(p1["chief_complaint"]["duration"], "3 days")
        symps1 = [s.lower() for s in p1["symptoms"]["list"]]
        for expected in ["fever", "headache", "body pain", "chills", "sore throat", "cough"]:
            self.assertIn(expected, symps1)

        # Per-symptom timelines
        psd = p1["symptom_details"]["per_symptom_details"]
        self.assertEqual(psd.get("fever", {}).get("duration"), "3 days")
        self.assertEqual(psd.get("headache", {}).get("onset"), "Since yesterday")
        self.assertEqual(psd.get("body pain", {}).get("onset"), "Since yesterday")
        self.assertEqual(psd.get("sore throat", {}).get("onset"), "Since today")

        # Turn 2: Temperature not measured, worse at night
        agent.process_patient_info("Temperature not measured; fever feels worse at night and chills occur then.")
        # Turn 3: No breathing difficulty, chest pain or chest tightness (must NOT create throat tightness)
        agent.process_patient_info("No breathing difficulty, chest pain or chest tightness.")
        # Turn 4: Dry cough; no nasal congestion or sneezing
        agent.process_patient_info("Dry cough; no nasal congestion or sneezing.")
        # Turn 5: No urinary difficulty, abdominal pain or vomiting
        agent.process_patient_info("No urinary difficulty, normal urine colour, no abdominal pain or vomiting.")
        # Turn 6: No diabetes, asthma or heart disease
        agent.process_patient_info("No diabetes, asthma or heart disease.")
        # Turn 7: No recent sick contact or travel -> triggers final confirmation, then patient confirms
        agent.process_patient_info("No recent sick contact or travel. Give me the report.")
        r_last = agent.process_patient_info("No, that's all.")

        p_final = r_last["patient_data"]
        symps_final = [s.lower() for s in p_final["symptoms"]["list"]]
        negs_final = " ".join(p_final["symptom_details"]["negative_symptoms"]).lower()

        # Dry cough present without bare cough duplication or "no cough"
        self.assertIn("dry cough", symps_final)
        self.assertNotIn("cough", symps_final)
        self.assertNotIn("no cough", negs_final)

        # No false throat tightness
        self.assertNotEqual(p_final["symptom_details"].get("character"), "tightness")
        self.assertNotEqual(p_final["symptom_details"].get("location"), "throat")

        # Denied chronic conditions NEVER in positive conditions
        self.assertEqual(p_final["medical_history"]["conditions"], [])
        denied_conds = p_final["medical_history"].get("denied_conditions", [])
        for cond in ["diabetes", "asthma", "heart disease"]:
            self.assertIn(cond, denied_conds)

        # Temperature remains not measured
        self.assertIsNone(p_final["vitals"].get("temperature"))
        self.assertEqual(p_final["vitals"].get("temperature_status"), "not_measured")

        # Sick contact & travel denied
        self.assertEqual(p_final["lifestyle"].get("recent_travel"), "patient_denied")
        self.assertEqual(p_final["lifestyle"].get("sick_contacts"), "patient_denied")

        # Formatted report checks
        formatted = r_last["formatted_report"]
        self.assertNotIn("&#x", formatted)
        self.assertIn("HISTORY OF PRESENT ILLNESS", formatted)
        self.assertIn("Recent Travel: Patient denied", formatted)
        self.assertIn("Sick Contacts: Patient denied", formatted)
        self.assertIn("Patient denied (diabetes, asthma, heart disease)", formatted)
        self.assertNotIn("throat tightness", formatted.lower())

    def test_b_headache_scenario_no_invented_duration(self):
        agent = get_clinical_agent(session_id="test_final_headache")
        r1 = agent.process_patient_info("I have a throbbing headache on my forehead. No nausea and no fever.")
        agent.process_patient_info("Give me the report.")
        agent.process_patient_info("No.")
        p = agent.get_latest_report() if hasattr(agent, "get_latest_report") else r1["patient_data"]
        rep = agent._latest_report
        self.assertEqual(rep["chief_complaint"]["reason"], "Headache")
        self.assertEqual(rep["chief_complaint"]["duration"], "Not provided")
        self.assertEqual(rep["symptoms"]["location"], "forehead")
        self.assertEqual(rep["symptoms"]["character"], "throbbing")
        negs = " ".join(rep["symptoms"]["negative_symptoms"]).lower()
        self.assertIn("nausea", negs)
        self.assertIn("fever", negs)

    def test_c_abdominal_pain_scenario(self):
        agent = get_clinical_agent(session_id="test_final_abd")
        agent.process_patient_info("I have lower right abdominal pain since yesterday. No vomiting, no diarrhea.")
        agent.process_patient_info("Give me the report.")
        r2 = agent.process_patient_info("No.")
        rep = r2["report"]
        self.assertEqual(rep["chief_complaint"]["reason"], "Abdominal pain")
        self.assertEqual(rep["symptoms"]["location"], "lower right")
        symps = [s.lower() for s in rep["symptoms"]["reported_symptoms"]]
        self.assertIn("abdominal pain", symps)
        self.assertNotIn("vomiting", symps)
        self.assertNotIn("diarrhea", symps)
        negs = " ".join(rep["symptoms"]["negative_symptoms"]).lower()
        self.assertIn("vomiting", negs)
        self.assertIn("diarrhea", negs)

    def test_d_patient_correction_updates_structured_data_and_regenerates_report(self):
        agent = get_clinical_agent(session_id="test_final_correction")
        agent.process_patient_info("I have fever for 2 days and headache.")
        # Simulate an incorrect character/condition before correction
        agent.patient_info_collector.patient_data["symptom_details"]["character"] = "tightness"
        agent.patient_info_collector.patient_data["medical_history"]["conditions"] = ["diabetes"]
        agent.process_patient_info("Give me the report.")
        r_init = agent.process_patient_info("No.")
        self.assertIn("diabetes", r_init["formatted_report"].lower())

        # Patient submits correction
        r_corr = agent.process_patient_info(
            "Correction: I do not have diabetes or asthma, remove throat tightness, and fever has been 3 days. I also take Paracetamol 500mg."
        )
        p_corr = r_corr["patient_data"]
        self.assertEqual(p_corr["chief_complaint"]["duration"], "3 days")
        self.assertNotIn("diabetes", [c.lower() for c in p_corr["medical_history"]["conditions"]])
        self.assertIn("diabetes", p_corr["medical_history"].get("denied_conditions", []))
        self.assertIsNone(p_corr["symptom_details"].get("character"))
        self.assertIn("Paracetamol 500mg", p_corr["medications"]["list"])
        self.assertEqual(p_corr["patient_review_status"], "corrected_pending_confirmation")
        self.assertGreaterEqual(len(p_corr["correction_history"]), 1)
        # Regenerated report reflects corrections and is NOT just raw appended text
        self.assertIn("3 days", r_corr["formatted_report"])
        self.assertIn("Paracetamol 500mg", r_corr["formatted_report"])
        self.assertNotIn("tightness", r_corr["formatted_report"].lower())

    def test_d2_mocked_llm_correction_and_extraction(self):
        def smart_mock_llm(prompt):
            p_str = str(prompt)
            if "clinical intake extraction engine" in p_str.lower():
                return MagicMock(content='{"chief_complaint":{"reason":"Fever","duration":"3 days"},"symptoms":["fever","dry cough"],"negative_symptoms":["no chest tightness"],"per_symptom_details":{"fever":{"status":"reported","duration":"3 days"},"dry cough":{"status":"reported","onset":"Today"}},"vitals":{"temperature":null,"temperature_status":"not_measured"},"medical_history":{"conditions":[],"denied_conditions":["diabetes"]},"medications":[],"allergies":{},"lifestyle":{"recent_travel":"patient_denied","sick_contacts":"patient_denied"},"demographics":{}}')
            if "history of present illness" in p_str.lower():
                return MagicMock(content="Patient presents with a 3-day history of fever and dry cough starting today, with temperature not measured. Patient denies chest tightness, diabetes, recent travel, and sick contacts.")
            if "physician-facing clinical summary" in p_str.lower():
                return MagicMock(content="3-day fever with dry cough today; temperature unmeasured; denies chest tightness, diabetes, travel, and sick contacts.")
            if "points for physician review" in p_str.lower():
                return MagicMock(content='["Measure baseline temperature and vital signs.", "Perform focused respiratory and ENT exam.", "Verify hydration status."]')
            return MagicMock(content='{"sufficient_info": true, "question": null, "category": "complete", "extracted_updates": {}}')

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = smart_mock_llm
        global_config.clinical_consultation.llm = mock_llm
        agent = get_clinical_agent(session_id="test_final_mock_llm")
        agent.process_patient_info("3 din se fever aur aaj se dry cough hai. No chest tightness, no diabetes, no travel or sick contact.")
        agent.process_patient_info("Give me the report.")
        r = agent.process_patient_info("No.")
        self.assertIn("3-day history of fever", r["report"]["hpi"])
        self.assertIn("Measure baseline temperature", " ".join(r["report"]["recommendations"]))

    def test_e_session_isolation(self):
        a1 = get_clinical_agent(session_id="clin_iso_1")
        a2 = get_clinical_agent(session_id="clin_iso_2")
        a1.process_patient_info("I have severe abdominal pain and take Ibuprofen 400mg. Give me the report.")
        a1.process_patient_info("No.")
        a2.process_patient_info("I have mild headache. Give me the report.")
        r2 = a2.process_patient_info("No.")
        self.assertEqual(r2["patient_data"]["chief_complaint"]["reason"], "Headache")
        self.assertNotIn("Ibuprofen 400mg", r2["patient_data"]["medications"]["list"])
        self.assertNotIn("abdominal pain", [s.lower() for s in r2["patient_data"]["symptoms"]["list"]])

        # Correction on session 1 does not affect session 2
        a1.process_patient_info("Correction: I also have diabetes.")
        self.assertNotIn("diabetes", a2.patient_info_collector.get_patient_data()["medical_history"]["conditions"])

    def test_f_medication_and_allergy_denials_multilingual(self):
        from agents.clinical_pre_consultation_agent.patient_info_collector import PatientInfoCollector
        # Hinglish exact browser inputs
        c1 = PatientInfoCollector()
        c1.extract_from_patient_text("Abhi main fever ke liye koi dawai nahi le raha hoon.")
        self.assertEqual(c1.get_patient_data()["medications"]["status"], "none_reported")

        c2 = PatientInfoCollector()
        c2.extract_from_patient_text("Mujhe kisi dawai, khane ya kisi aur cheez se koi known allergy nahi hai.")
        p2 = c2.get_patient_data()
        self.assertEqual(p2["allergies"]["status"], "explicitly_denied")
        self.assertEqual(p2["allergies"]["drug_status"], "explicitly_denied")
        self.assertEqual(p2["allergies"]["environmental_status"], "explicitly_denied")

        # English explicit denials
        c3 = PatientInfoCollector()
        c3.extract_from_patient_text("I am not taking any medicine. I have no known drug allergies and no food or other allergies.")
        p3 = c3.get_patient_data()
        self.assertEqual(p3["medications"]["status"], "none_reported")
        self.assertEqual(p3["allergies"]["status"], "explicitly_denied")

    def test_g_correction_updates_meds_and_allergies_and_survives_reload(self):
        from storage.session_db import session_db
        sid = "clin_corr_reload_test"
        session_db.create_session(session_id=sid, workspace_type="clinical_pre_consultation", title="Correction Reload Test")
        try:
            agent = get_clinical_agent(session_id=sid)
            agent.process_patient_info("Mujhe 3 din se fever hai. Give me the report.")
            agent.process_patient_info("No.")
            # Apply Hinglish medication + allergy denial correction
            r_corr = agent.process_patient_info(
                "Abhi main fever ke liye koi dawai nahi le raha hoon. "
                "Mujhe kisi dawai, khane ya kisi aur cheez se koi known allergy nahi hai."
            )
            self.assertEqual(r_corr["patient_data"]["medications"]["status"], "patient_denied")
            self.assertEqual(r_corr["patient_data"]["allergies"]["status"], "explicitly_denied")
            self.assertIn("Current: Patient denied", r_corr["formatted_report"])
            self.assertIn("Drug: Patient denied", r_corr["formatted_report"])
            self.assertIn("Environmental: Patient denied", r_corr["formatted_report"])

            # Persist to DB and simulate server/session reload
            session_db.save_clinical_state(
                session_id=sid,
                patient_data=r_corr["patient_data"],
                current_stage="completed",
                latest_report_text=r_corr["formatted_report"],
                formatted_report=r_corr["formatted_report"],
                is_confirmed=False
            )
            reset_clinical_agent_session(sid)
            reloaded_agent = get_clinical_agent(session_id=sid)
            reloaded_data = reloaded_agent.patient_info_collector.get_patient_data()
            self.assertEqual(reloaded_data["medications"]["status"], "patient_denied")
            self.assertEqual(reloaded_data["allergies"]["status"], "explicitly_denied")
        finally:
            session_db.delete_session(sid)

    def test_h_hpi_summary_consistency_and_no_blanket_onset(self):
        agent = get_clinical_agent(session_id="test_hpi_onset")
        agent.patient_info_collector.patient_data["symptom_details"]["onset"] = "3 days ago"
        agent.process_patient_info(
            "Mujhe pichhle 3 din se fever ho raha hai. Kal se headache aur body pain bhi hai. "
            "Aaj se sore throat aur halka cough bhi shuru ho gaya hai. Give me the report."
        )
        agent.process_patient_info("No.")
        rep = agent._latest_report
        # Global onset must NOT be misleadingly "3 days ago" when per-symptom timelines exist
        self.assertEqual(rep["symptoms"]["onset"], "Not specified")
        self.assertIn("sore throat (since today)", rep["hpi"].lower())
        self.assertIn("headache (since yesterday)", rep["hpi"].lower())

    def test_i_entity_decoding_backend_and_frontend_pdf(self):
        from agents.clinical_pre_consultation_agent.report_generator import clean_text_entities, PreConsultationReportGenerator
        sample = "&#x73;ore throat, &#x66;orehead, &#x73;harp, &#x4E;ot asked, &#x4E;o difficulty breathing, &#x50;atient denied"
        cleaned = clean_text_entities(sample)
        self.assertEqual(cleaned, "sore throat, forehead, sharp, Not asked, No difficulty breathing, Patient denied")
        self.assertNotIn("&#x", cleaned)

        # Verify full report dict & formatted text scrub nested entities
        from agents.clinical_pre_consultation_agent.patient_info_collector import PatientInfoCollector
        c = PatientInfoCollector()
        c.patient_data["symptoms"]["list"] = ["&#x73;ore throat"]
        c.patient_data["symptoms"]["status"] = "provided"
        c.patient_data["symptom_details"]["location"] = "&#x66;orehead"
        c.patient_data["symptom_details"]["character"] = "&#x73;harp"
        gen = PreConsultationReportGenerator()
        rep = gen.generate_report(c.get_patient_data())
        fmt = gen.format_report_text(rep)
        self.assertEqual(rep["symptoms"]["reported_symptoms"], ["sore throat"])
        self.assertEqual(rep["symptoms"]["location"], "forehead")
        self.assertEqual(rep["symptoms"]["character"], "sharp")
        self.assertNotIn("&#x", fmt)

    def test_j_next_steps_never_flags_progressive_timeline_as_discrepancy(self):
        def mock_discrepancy_llm(prompt):
            p_str = str(prompt).lower()
            if "points for physician review" in p_str:
                return MagicMock(content='["Clarify timeline discrepancy between 3-day fever and sore throat starting today.", "Perform focused ENT and respiratory exam.", "Measure baseline vital signs including temperature."]')
            if "history of present illness" in p_str:
                return MagicMock(content="Patient has fever for 3 days, headache and body pain since yesterday, and sore throat and dry cough starting today.")
            if "physician-facing clinical summary" in p_str:
                return MagicMock(content="Fever for 3 days, headache/body pain since yesterday, sore throat/dry cough today.")
            return MagicMock(content='{}')

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_discrepancy_llm
        global_config.clinical_consultation.llm = mock_llm
        agent = get_clinical_agent(session_id="test_next_steps_timeline")
        agent.process_patient_info(
            "Mujhe pichhle 3 din se fever ho raha hai. Kal se headache aur body pain bhi hai. "
            "Aaj se sore throat aur dry cough shuru ho gaya hai. Give me the report."
        )
        r = agent.process_patient_info("No.")
        recs_text = " ".join(r["report"]["recommendations"]).lower()
        self.assertNotIn("discrepancy", recs_text)
        self.assertIn("ent and respiratory exam", recs_text)


class TestFinalClinicalRealGroqIntegration(unittest.TestCase):
    """
    Real Groq LLM integration test (uses configured Groq provider from Config/env).
    """

    def setUp(self):
        reset_clinical_agent_session("*")

    def tearDown(self):
        reset_clinical_agent_session("*")

    def test_real_groq_fever_extraction_report_and_correction(self):
        if not global_config.clinical_consultation.llm:
            self.skipTest("Groq LLM not configured in environment")

        agent = get_clinical_agent(session_id="test_real_groq_clin")
        agent.process_patient_info(
            "Mujhe pichhle 3 din se fever ho raha hai. Kal se headache bhi hai. "
            "Temperature check nahi kiya. No chest pain, no diabetes or asthma. No recent travel or sick contact. Give me the report."
        )
        r1 = agent.process_patient_info("No, nothing else.")
        p1 = r1["patient_data"]
        self.assertEqual(p1["chief_complaint"]["reason"], "Fever")
        self.assertEqual(p1["chief_complaint"]["duration"], "3 days")
        self.assertEqual(p1["medical_history"]["conditions"], [])
        self.assertIsNone(p1["vitals"].get("temperature"))
        self.assertTrue(len(r1["report"].get("hpi", "")) > 20)
        self.assertNotIn("&#x", r1["formatted_report"])

        # Real Groq correction turn
        r2 = agent.process_patient_info("Correction: I also started taking Paracetamol 500mg today.")
        self.assertIn("Paracetamol", " ".join(r2["patient_data"]["medications"]["list"]))
        self.assertIn("Paracetamol", r2["formatted_report"])


if __name__ == "__main__":
    unittest.main()
