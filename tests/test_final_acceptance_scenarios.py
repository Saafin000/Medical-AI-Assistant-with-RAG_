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
from agents.clinical_pre_consultation_agent.report_generator import clean_text_entities, PreConsultationReportGenerator
from agents.clinical_pre_consultation_agent.patient_info_collector import PatientInfoCollector


class TestFinalAcceptanceScenarios(unittest.TestCase):
    """
    Verification suite for the Final Clinical Pre-Consultation Implementation (Section 10):
    - Scenario 1: Headache (morning -> evening correction, recurring throbbing, light sensitivity, final confirmation)
    - Scenario 2: Fever (3-day fever, yesterday headache/body pain, today sore throat/dry cough, final confirmation)
    - Scenario 3: Additional Information after final confirmation (reopens interview, asks final confirmation again)
    - Scenario 4: Patient Correction + Cases D & E (Done before confirmation, incomplete session, post-report correction)
    - Scenario 5: Safety / Emergency priority + incomplete screening status
    - HTML Entity Cleaning: &#x68;eadache, &#x74;hrobbing, &#x73;tarts, &#x4E;ot asked
    """

    def setUp(self):
        self.orig_llm = global_config.clinical_consultation.llm
        global_config.clinical_consultation.llm = None
        reset_clinical_agent_session("*")

    def tearDown(self):
        reset_clinical_agent_session("*")
        global_config.clinical_consultation.llm = self.orig_llm

    def test_scenario_1_headache_correction_and_final_confirmation(self):
        agent = get_clinical_agent(session_id="clin_accept_1")
        r1 = agent.process_patient_info("I have recurring throbbing morning headaches with light sensitivity.")
        self.assertEqual(r1["patient_data"]["symptom_details"]["timing"], "morning")
        self.assertEqual(r1["patient_data"]["symptom_details"]["frequency"], "recurring")
        self.assertEqual(r1["patient_data"]["symptom_details"]["character"], "throbbing")

        # Mid-interview correction: morning -> evening
        r2 = agent.process_patient_info("Actually, my headache happens in the evening.")
        self.assertEqual(r2["patient_data"]["symptom_details"]["timing"], "evening")
        self.assertNotEqual(r2["patient_data"]["symptom_details"]["timing"], "morning")

        # Patient says "Done" before final confirmation -> Case D: must ask final confirmation first!
        r_done = agent.process_patient_info("Done")
        self.assertEqual(r_done["status"], "active")
        self.assertIsNone(r_done["report"])
        self.assertEqual(r_done["next_question"]["category"], "final_confirmation")
        self.assertIn("anything else", r_done["next_question"]["question"].lower())

        # Case A: Patient confirms "No, that's everything." -> generates final report
        r_final = agent.process_patient_info("No, that's everything.")
        self.assertEqual(r_final["status"], "completed")
        self.assertIsNotNone(r_final["report"])
        self.assertEqual(r_final["patient_data"]["final_confirmation_status"], "confirmed")
        self.assertEqual(r_final["report"]["symptoms"]["timing"], "evening")
        self.assertIn("Timing: evening", r_final["formatted_report"])
        self.assertNotIn("morning", r_final["report"]["symptoms"]["timing"])

    def test_scenario_2_fever_timelines_and_final_confirmation(self):
        agent = get_clinical_agent(session_id="clin_accept_2")
        agent.process_patient_info(
            "Mujhe pichhle 3 din se fever ho raha hai. Kal se headache aur body pain bhi hai. "
            "Aaj se sore throat aur dry cough shuru ho gaya hai."
        )
        agent.process_patient_info("Temperature not measured. No breathing difficulty or chest pain.")
        agent.process_patient_info("No diabetes, asthma, or heart disease. No medications and no known allergies.")
        r_done = agent.process_patient_info("That's all.")
        self.assertEqual(r_done["next_question"]["category"], "final_confirmation")

        r_final = agent.process_patient_info("Nothing else.")
        self.assertEqual(r_final["status"], "completed")
        rep = r_final["report"]
        self.assertEqual(rep["chief_complaint"]["duration"], "3 days")
        self.assertEqual(rep["symptoms"]["per_symptom_details"]["headache"]["onset"], "Since yesterday")
        self.assertEqual(rep["symptoms"]["per_symptom_details"]["sore throat"]["onset"], "Since today")
        self.assertIn(rep["medications"]["medications"][0], ["None reported", "Patient denied"])
        self.assertEqual(rep["allergies"]["drug_allergies"], ["Patient denied"])
        self.assertNotIn("discrepancy", " ".join(rep["recommendations"]).lower())

    def test_scenario_3_additional_info_after_final_confirmation_reopens_interview(self):
        agent = get_clinical_agent(session_id="clin_accept_3")
        agent.process_patient_info("I have a mild headache for 2 days.")
        r_fc1 = agent.process_patient_info("Give me the report.")
        self.assertEqual(r_fc1["next_question"]["category"], "final_confirmation")
        self.assertIsNone(r_fc1["report"])

        # Case B: Patient adds new symptom at final confirmation!
        r_add = agent.process_patient_info("Yes, I also have dizziness.")
        self.assertEqual(r_add["status"], "active")
        self.assertIsNone(r_add["report"])
        self.assertIn("dizziness", [s.lower() for s in r_add["patient_data"]["symptoms"]["list"]])
        self.assertIsNotNone(r_add["next_question"])

        # Finish follow-up -> final confirmation must be asked AGAIN
        r_fc2 = agent.process_patient_info("Done")
        self.assertEqual(r_fc2["next_question"]["category"], "final_confirmation")
        self.assertIsNone(r_fc2["report"])

        # Now confirm no more info -> report generated with both headache and dizziness
        r_final = agent.process_patient_info("No.")
        self.assertEqual(r_final["status"], "completed")
        self.assertIn("dizziness", [s.lower() for s in r_final["report"]["symptoms"]["reported_symptoms"]])

    def test_scenario_4_post_report_correction_and_incomplete_session(self):
        # Case E: Incomplete session
        a_inc = get_clinical_agent(session_id="clin_accept_inc")
        a_inc.process_patient_info("I have fever for 2 days.")
        r_inc = a_inc.conclude_incomplete_session()
        self.assertEqual(r_inc["status"], "partial")
        self.assertEqual(r_inc["patient_data"]["final_confirmation_status"], "incomplete_unconfirmed")
        self.assertIn("unassessed red flags have not been ruled out", r_inc["formatted_report"].lower())

        # Post-report correction
        a_corr = get_clinical_agent(session_id="clin_accept_corr")
        a_corr.process_patient_info("I have morning headache.")
        a_corr.process_patient_info("Done")
        a_corr.process_patient_info("No.")
        r_upd = a_corr.process_patient_info("Correction: actually headache happens in the evening and I take Ibuprofen 400mg.")
        self.assertEqual(r_upd["patient_data"]["symptom_details"]["timing"], "evening")
        self.assertIn("Ibuprofen 400mg", r_upd["patient_data"]["medications"]["list"])
        self.assertIn("Ibuprofen 400mg", r_upd["formatted_report"])

    def test_scenario_5_safety_emergency_prioritized_over_final_confirmation(self):
        agent = get_clinical_agent(session_id="clin_accept_emerg")
        r = agent.process_patient_info("I have severe crushing chest pain and difficulty breathing.")
        self.assertEqual(r["status"], "emergency_escalation")
        self.assertTrue(r["requires_immediate_attention"])
        self.assertIsNotNone(r["report"])
        self.assertIsNone(r["next_question"])

    def test_html_entity_strings_observed(self):
        raw = "&#x68;eadache, &#x74;hrobbing, &#x73;tarts, &#x4E;ot asked, &#x4E;one reported"
        self.assertEqual(clean_text_entities(raw), "headache, throbbing, starts, Not asked, None reported")

    def test_scenario_6_exact_browser_fever_urine_regression(self):
        agent = get_clinical_agent(session_id="clin_accept_fever_urine")

        r1 = agent.process_patient_info("I have fever")
        self.assertEqual(r1["status"], "active")
        self.assertIsNone(r1["report"])

        r2 = agent.process_patient_info("Its happening from last 2 hours")
        self.assertEqual(r2["patient_data"]["chief_complaint"]["duration"], "2 hours")
        self.assertEqual(r2["status"], "active")
        self.assertIsNone(r2["report"])

        r3 = agent.process_patient_info("No I did not measure any temperature and along with fever I am feeling cold")
        self.assertEqual(r3["patient_data"]["vitals"]["temperature_status"], "not_measured")
        self.assertIsNone(r3["patient_data"]["vitals"].get("temperature"))
        self.assertIn("chills", [s.lower() for s in r3["patient_data"]["symptoms"]["list"]])
        self.assertEqual(r3["status"], "active")

        r4 = agent.process_patient_info("Yes I have also headache")
        self.assertIn("headache", [s.lower() for s in r4["patient_data"]["symptoms"]["list"]])
        self.assertEqual(r4["status"], "active")

        # Simulate assistant asking about neck stiffness and photophobia right before patient says "No"
        agent.conversation_history.append({
            "role": "assistant",
            "content": "Do you have any stiffness in your neck or sensitivity to light (photophobia)?"
        })
        r5 = agent.process_patient_info("No")
        negs5 = " ".join(r5["patient_data"]["symptom_details"]["negative_symptoms"]).lower()
        self.assertIn("neck stiffness", negs5)
        self.assertIn("photophobia", negs5)
        self.assertEqual(r5["status"], "active")

        r6 = agent.process_patient_info("Yes the color of urine becomes yellowish")
        symps6 = [s.lower() for s in r6["patient_data"]["symptoms"]["list"]]
        for expected in ["fever", "chills", "headache", "yellowish urine"]:
            self.assertIn(expected, symps6)
        # Must NOT invent dysuria or urinary frequency
        self.assertNotIn("dysuria", symps6)
        self.assertNotIn("painful urination", symps6)
        self.assertNotIn("urinary frequency", symps6)
        # Must NOT prematurely generate report before final confirmation
        self.assertEqual(r6["status"], "active")
        self.assertIsNone(r6["report"])

        # Complete adaptive questioning & trigger mandatory final confirmation
        r_done = agent.process_patient_info("Done")
        self.assertEqual(r_done["status"], "active")
        self.assertIsNone(r_done["report"])
        self.assertEqual(r_done["next_question"]["category"], "final_confirmation")

        # Patient confirms -> ONLY now generate report
        r_final = agent.process_patient_info("No, nothing else.")
        self.assertEqual(r_final["status"], "completed")
        self.assertIsNotNone(r_final["report"])
        rep = r_final["report"]
        fmt = r_final["formatted_report"]

        # Verify all symptoms, denials, HPI, summary, and zero HTML entities
        rep_symps = [s.lower() for s in rep["symptoms"]["reported_symptoms"]]
        for expected in ["fever", "chills", "headache", "yellowish urine"]:
            self.assertIn(expected, rep_symps)
        self.assertEqual(rep["chief_complaint"]["reason"], "Fever")
        self.assertEqual(rep["chief_complaint"]["duration"], "2 hours")
        self.assertIn("fever", rep["hpi"].lower())
        self.assertIn("yellowish urine", rep["hpi"].lower())
        self.assertIn("fever", rep["clinical_summary"].lower())
        self.assertNotIn("&#x", fmt)
        # Medications were never asked and should NOT falsely become "provided"
        self.assertEqual(r_final["patient_data"]["medications"]["status"], "not_asked")


if __name__ == "__main__":
    unittest.main()
