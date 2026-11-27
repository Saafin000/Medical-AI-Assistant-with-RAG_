import os
import sys
import json
import re

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from agents.clinical_pre_consultation_agent import (
    ClinicalPreConsultationAgent,
    PatientInfoCollector,
    RedFlagDetector,
    FollowupQuestionGenerator,
    PreConsultationReportGenerator
)

def run_all_tests():
    config = Config()
    # Use deterministic question generation for unit validation
    config.clinical_consultation.llm = None
    
    print("=" * 80)
    print("TARGETED IMPROVEMENTS VERIFICATION SUITE — CLINICAL PRE-CONSULTATION AGENT")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # TEST 1: Fever with headache, body pain, chills, sore throat and cough
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Comprehensive Multi-Symptom Intake ---")
    c1 = PatientInfoCollector()
    c1.extract_from_patient_text("I have high fever with severe headache, body pain, chills, sore throat and dry cough for 3 days.")
    p1 = c1.get_patient_data()
    symps1 = p1["symptoms"]["list"]
    assert "fever" in symps1, f"Expected fever in {symps1}"
    assert "headache" in symps1, f"Expected headache in {symps1}"
    assert "body pain" in symps1, f"Expected body pain in {symps1}"
    assert "sore throat" in symps1, f"Expected sore throat in {symps1}"
    assert "cough" in symps1, f"Expected cough in {symps1}"
    assert p1["chief_complaint"]["reason"] == "Fever", f"Expected Fever, got {p1['chief_complaint']['reason']}"
    assert p1["chief_complaint"]["duration"] == "3 days", f"Expected 3 days, got {p1['chief_complaint']['duration']}"
    print("   [PASS] Test 1: All multi-symptoms, chief complaint, and duration correctly extracted.")

    # -------------------------------------------------------------------------
    # TEST 2: CRITICAL TEST — Explicitly denied chest pain & breathing difficulty
    # "Mujhe saans lene mein koi dikkat nahi hai aur chest pain bhi nahi hai."
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: CRITICAL NEGATION TEST (Hindi/Hinglish Denials) ---")
    c2 = PatientInfoCollector()
    c2.extract_from_patient_text("Mujhe saans lene mein koi dikkat nahi hai aur chest pain bhi nahi hai.")
    p2 = c2.get_patient_data()
    
    symps2 = p2["symptoms"]["list"]
    negs2 = p2["symptom_details"]["negative_symptoms"]
    loc2 = p2["symptom_details"].get("location")
    cc2_reason = p2["chief_complaint"].get("reason")
    
    # Assertions for Critical Test:
    assert not any("chest pain" in s for s in symps2), f"Chest pain MUST NOT appear as positive symptom! Found in {symps2}"
    assert not any("breathing" in s for s in symps2), f"Breathing difficulty MUST NOT appear as positive symptom! Found in {symps2}"
    assert any("chest pain" in ns for ns in negs2), f"Expected chest pain in negative symptoms: {negs2}"
    assert any("breathing" in ns for ns in negs2), f"Expected breathing difficulty in negative symptoms: {negs2}"
    assert loc2 != "chest", f"Chest MUST NOT be assigned as symptom location! Got {loc2}"
    assert cc2_reason != "Chest pain" and cc2_reason != "Difficulty breathing", f"Chief complaint cannot be denied emergency! Got {cc2_reason}"
    
    # Check RedFlagDetector: must NOT escalate on explicitly denied symptoms!
    rf2 = RedFlagDetector(config=config, llm=None)
    rf_res2 = rf2.detect_emergencies(p2)
    assert not rf_res2.get("should_escalate"), f"Explicitly denied chest pain/breathing MUST NOT escalate! Result: {rf_res2}"
    assert not rf_res2.get("has_red_flags"), f"Denied symptoms produced false red flags: {rf_res2.get('flags')}"
    print("   [PASS] Test 2: Critical negation test verified! Breathing difficulty & chest pain denied, zero false flags, chest not assigned as location.")

    # -------------------------------------------------------------------------
    # TEST 3: Positive Chest Pain Escalation
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Positive Acute Emergency Detection ---")
    c3 = PatientInfoCollector()
    c3.extract_from_patient_text("I have sudden severe crushing chest pain radiating to my left arm.")
    p3 = c3.get_patient_data()
    rf3 = RedFlagDetector(config=config, llm=None)
    rf_res3 = rf3.detect_emergencies(p3)
    assert rf_res3.get("should_escalate") is True, "Positive acute chest pain MUST escalate immediately!"
    assert ("108" in rf_res3.get("guidance", "") or "112" in rf_res3.get("guidance", "")), "Emergency guidance must include India 108/112 helplines!"
    print("   [PASS] Test 3: Positive acute chest pain correctly escalated with India 108/112 emergency guidance.")

    # -------------------------------------------------------------------------
    # TEST 4: Patient Correcting an Incorrectly Recorded Symptom
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Symptom Removal / Correction ---")
    agent4 = ClinicalPreConsultationAgent(config)
    # Turn 1: Patient mentions headache
    res4_1 = agent4.process_patient_info("I have a fever and headache for 2 days.")
    assert "headache" in agent4.patient_info_collector.get_patient_data()["symptoms"]["list"]
    
    # Simulate completion to test feedback correction
    agent4._latest_report = agent4.report_generator.generate_report(
        agent4.patient_info_collector.get_patient_data(),
        agent4.red_flag_detector.detect_emergencies(agent4.patient_info_collector.get_patient_data()),
        agent4.conversation_history
    )
    agent4._latest_formatted_report = agent4.report_generator.format_report_text(agent4._latest_report)
    agent4._session_active = False
    
    # Patient feedback correction: removes headache
    res4_corr = agent4.process_patient_info("Actually I don't have headache, please remove headache.")
    p4_updated = agent4.patient_info_collector.get_patient_data()
    assert "headache" not in p4_updated["symptoms"]["list"], f"Headache must be removed from positive symptoms! List: {p4_updated['symptoms']['list']}"
    assert any("headache" in ns for ns in p4_updated["symptom_details"]["negative_symptoms"]), "Headache should be recorded in negative symptoms!"
    assert "headache" not in res4_corr["report"]["symptoms"]["reported_symptoms"]
    print("   [PASS] Test 4: Patient successfully removed headache via correction; reflected in updated report.")

    # -------------------------------------------------------------------------
    # TEST 5: Patient Correcting Medication Dosage
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: Medication Dosage Correction ---")
    agent5 = ClinicalPreConsultationAgent(config)
    agent5.process_patient_info("I have fever and take Paracetamol 650mg.")
    assert "Paracetamol 650mg" in agent5.patient_info_collector.get_patient_data()["medications"]["list"]
    
    # Complete intake
    agent5._latest_report = agent5.report_generator.generate_report(
        agent5.patient_info_collector.get_patient_data(),
        agent5.red_flag_detector.detect_emergencies(agent5.patient_info_collector.get_patient_data()),
        agent5.conversation_history
    )
    agent5._session_active = False
    
    # Correction: replace 650mg with 500mg
    agent5.process_patient_info("Correction: I take Paracetamol 500mg, not 650mg.")
    meds5 = agent5.patient_info_collector.get_patient_data()["medications"]["list"]
    assert "Paracetamol 500mg" in meds5, f"Expected Paracetamol 500mg in {meds5}"
    assert "Paracetamol 650mg" not in meds5, f"Old dosage Paracetamol 650mg must be replaced! List: {meds5}"
    print("   [PASS] Test 5: Medication dosage cleanly updated and replaced without duplicate entries.")

    # -------------------------------------------------------------------------
    # TEST 6: Incomplete Patient Information & 4-Way Distinctions
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: Missing vs Negative Distinctions & No Hallucinations ---")
    c6 = PatientInfoCollector()
    c6.extract_from_patient_text("I have had a mild cough for 4 days.")
    p6 = c6.get_patient_data()
    rep_gen = PreConsultationReportGenerator()
    rep6 = rep_gen.generate_report(p6, None, [{"role": "user", "content": "I have had a mild cough for 4 days."}])
    formatted6 = rep_gen.format_report_text(rep6)
    
    assert "Surgeries: Not asked" in formatted6, "Surgeries must be 'Not asked', not None or Denied"
    assert "Hospitalizations: Not asked" in formatted6, "Hospitalizations must be 'Not asked'"
    assert "Current: Not asked" in formatted6, "Medications must be 'Not asked'"
    assert "Drug: Not asked" in formatted6, "Allergies must be 'Not asked'"
    assert "Smoking: Not asked" in formatted6, "Lifestyle must be 'Not asked'"
    assert "Temperature: Not provided" in formatted6, "Unmeasured temperature must be 'Not provided'"
    assert "MISSING INFORMATION" in formatted6, "Report must include Missing Information section"
    print("   [PASS] Test 6: Strict 4-way distinction preserved; unasked items are 'Not asked', zero fabricated claims.")

    # -------------------------------------------------------------------------
    # TEST 7: Multiple Answers in One Turn
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: Dense Multi-Category Single Message ---")
    c7 = PatientInfoCollector()
    c7.extract_from_patient_text("I am 42 years old, have had fever for 3 days, temp 101.5 F, taking Ibuprofen 400mg, no allergies, pehle se koi bimari nahi hai.")
    p7 = c7.get_patient_data()
    
    assert p7["demographics"]["age"] == 42, f"Expected age 42, got {p7['demographics']['age']}"
    assert p7["chief_complaint"]["reason"] == "Fever", f"Expected Fever, got {p7['chief_complaint']['reason']}"
    assert p7["chief_complaint"]["duration"] == "3 days", f"Expected 3 days, got {p7['chief_complaint']['duration']}"
    assert "101.5" in str(p7["vitals"]["temperature"]), f"Expected 101.5 temp, got {p7['vitals']['temperature']}"
    assert any("ibuprofen" in m.lower() for m in p7["medications"]["list"]), f"Expected Ibuprofen in meds: {p7['medications']['list']}"
    assert p7["medical_history"]["status"] == "patient_denied", f"Expected patient_denied for conditions, got {p7['medical_history']['status']}"
    print("   [PASS] Test 7: Extracted demographics, complaint, duration, temperature, medications, and history in a single turn.")

    # -------------------------------------------------------------------------
    # TEST 8: Hindi and Hinglish Responses
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: Natural Hindi/Hinglish Processing ---")
    c8 = PatientInfoCollector()
    c8.extract_from_patient_text("Mujhe 2 din se tez bukhar hai aur badan dard ho raha hai. Gala kharab hai aur sardard bhi hai.")
    p8 = c8.get_patient_data()
    symps8 = p8["symptoms"]["list"]
    assert "fever" in symps8, f"Expected fever in {symps8}"
    assert "body pain" in symps8, f"Expected body pain in {symps8}"
    assert "sore throat" in symps8, f"Expected sore throat in {symps8}"
    assert "headache" in symps8, f"Expected headache in {symps8}"
    assert p8["chief_complaint"]["duration"] == "2 days", f"Expected 2 days, got {p8['chief_complaint']['duration']}"
    print("   [PASS] Test 8: Hindi/Hinglish symptoms and duration accurately standardized.")

    # -------------------------------------------------------------------------
    # TEST 9: Repeated & Contradictory Answers
    # -------------------------------------------------------------------------
    print("\n--- TEST 9: Contradiction Handling & Anti-Repetition ---")
    c9 = PatientInfoCollector()
    c9.extract_from_patient_text("I have a severe headache.")
    assert "headache" in c9.get_patient_data()["symptoms"]["list"]
    # Contradiction: later explicitly denies having headache
    c9.extract_from_patient_text("I do not have headache.")
    p9 = c9.get_patient_data()
    assert "headache" not in p9["symptoms"]["list"], "Negated headache must be removed from positive list"
    assert len(p9["contradictions"]) > 0, "Contradiction must be tracked in patient data"
    print("   [PASS] Test 9: Contradiction accurately tracked and positive symptom list sanitized.")

    # -------------------------------------------------------------------------
    # TEST 10: Appropriate vs Inappropriate Red-Flag Escalation
    # -------------------------------------------------------------------------
    print("\n--- TEST 10: Appropriate vs Inappropriate Red-Flag Verification ---")
    # 10a: Denied chest pain -> NO escalation
    c10a = PatientInfoCollector()
    c10a.extract_from_patient_text("I have a runny nose. No chest pain and no shortness of breath.")
    rf_res10a = RedFlagDetector(config=config, llm=None).detect_emergencies(c10a.get_patient_data())
    assert rf_res10a["should_escalate"] is False, "Denied chest pain must NOT escalate!"
    
    # 10b: Active acute chest pain -> MUST escalate
    c10b = PatientInfoCollector()
    c10b.extract_from_patient_text("I have severe acute chest pain with shortness of breath.")
    rf_res10b = RedFlagDetector(config=config, llm=None).detect_emergencies(c10b.get_patient_data())
    assert rf_res10b["should_escalate"] is True, "Active acute chest pain MUST escalate!"
    print("   [PASS] Test 10: Appropriate escalation on true emergency; zero false escalation on denied symptoms.")

    print("\n" + "=" * 80)
    print("ALL 10 CLINICAL SCENARIOS TESTED & FULLY VERIFIED!")
    print("=" * 80)
    return True

def test_targeted_clinical_improvements():
    assert run_all_tests() is True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
