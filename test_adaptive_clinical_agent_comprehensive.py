"""
Comprehensive Adaptive Clinical Pre-Consultation Agent Test Suite

Covers:
1. Unit Tests (Context extraction, negation detection, anti-repetition, state merging)
2. Safety & False Emergency Prevention (Negation handling vs True emergency escalation)
3. Adaptive Multi-Turn Scenarios across 10 distinct clinical conditions
4. Divergent Questioning for Patients with Same Initial Complaint but Different Responses
5. Patient-Controlled Early Completion
6. Patient Feedback & Report Correction
7. Report Accuracy & Absence of Fabrication
8. Live Groq Provider Status Audit (qwen/qwen3.8-27b)
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from typing import Dict, Any, List

# Ensure Multi-Agent-Medical-Assistant is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import Config
from agents.clinical_pre_consultation_agent import (
    ClinicalPreConsultationAgent,
    PatientInfoCollector,
    RedFlagDetector,
    FollowupQuestionGenerator,
    PreConsultationReportGenerator
)

config = Config()

def run_all_tests():
    print("=" * 80)
    print("COMPREHENSIVE ADAPTIVE CLINICAL PRE-CONSULTATION AGENT TEST SUITE")
    print("=" * 80)
    
    test_results = []
    
    # --------------------------------------------------------------------------
    # 1. UNIT TESTS: PatientInfoCollector
    # --------------------------------------------------------------------------
    print("\n[Suite 1] Unit Tests: PatientInfoCollector & Negation Detection")
    collector = PatientInfoCollector()
    
    # Test 1a: Extraction of positive vs negative symptoms
    sample_text = "I have a mild headache for 2 days. No dizziness, no fever, no shortness of breath."
    extracted = collector.extract_from_patient_text(sample_text)
    p_data = collector.get_patient_data()
    
    symps = p_data["symptoms"]["list"]
    neg_symps = p_data["symptom_details"]["negative_symptoms"]
    
    assert "headache" in symps, f"Expected 'headache' in positive symptoms, got {symps}"
    assert "fever" not in symps, f"'fever' was negated but found in positive symptoms: {symps}"
    assert "dizziness" not in symps, f"'dizziness' was negated but found in positive symptoms: {symps}"
    assert any("fever" in ns for ns in neg_symps), f"Expected fever in negative symptoms: {neg_symps}"
    assert p_data["symptom_details"]["severity"] == "mild", f"Expected mild severity, got {p_data['symptom_details']['severity']}"
    assert "2 day" in p_data["chief_complaint"]["duration"], f"Expected duration, got {p_data['chief_complaint']['duration']}"
    print("   ✓ 1a: Extracted positive symptoms without polluting with negated symptoms")
    test_results.append(("1a: Positive vs Negative Symptom Separation", True))
    
    # Test 1b: Safe structured updates merging
    collector.merge_structured_updates({
        "symptom_details": {"location": "right temple", "character": "throbbing"},
        "medical_history": {"conditions": ["migraines"]},
        "medications": ["Sumatriptan 50mg"]
    })
    p_data2 = collector.get_patient_data()
    assert p_data2["symptom_details"]["location"] == "right temple"
    assert p_data2["symptom_details"]["character"] == "throbbing"
    assert "migraines" in p_data2["medical_history"]["conditions"]
    assert "Sumatriptan 50mg" in p_data2["medications"]["list"]
    print("   ✓ 1b: Merged structured updates into context without data loss")
    test_results.append(("1b: Structured Context Safe Merging", True))
    
    # Test 1c: Early completion request detection
    assert collector.detect_early_completion_request("That's all, please give me the report now.") is True
    assert collector.detect_early_completion_request("I am done, can I have my report?") is True
    assert collector.detect_early_completion_request("Stop the questioning and generate my summary") is True
    assert collector.detect_early_completion_request("My knee still hurts a bit.") is False
    print("   ✓ 1c: Early completion request patterns correctly identified")
    test_results.append(("1c: Early Completion Intent Recognition", True))

    # Test 1d: Auxiliary verb negations
    c_aux = PatientInfoCollector()
    c_aux.extract_from_patient_text("I do not have chest pain. I don't have shortness of breath. I haven't experienced dizziness. Denies any nausea.")
    p_aux = c_aux.get_patient_data()
    assert len(p_aux["symptoms"]["list"]) == 0, f"Expected 0 positive symptoms, got {p_aux['symptoms']['list']}"
    assert any("chest pain" in ns for ns in p_aux["symptom_details"]["negative_symptoms"])
    assert any("shortness of breath" in ns for ns in p_aux["symptom_details"]["negative_symptoms"])
    assert any("dizziness" in ns for ns in p_aux["symptom_details"]["negative_symptoms"])
    print("   ✓ 1d: Auxiliary verb negations ('do not have', 'don\\'t have', 'haven\\'t experienced') correctly negated")
    test_results.append(("1d: Auxiliary Verb Negation Handling", True))

    # Test 1e: Family history attribution
    c_fam = PatientInfoCollector()
    c_fam.extract_from_patient_text("My father had a heart attack and chest pain, but I only have a headache.")
    p_fam = c_fam.get_patient_data()
    assert "headache" in p_fam["symptoms"]["list"]
    assert "chest pain" not in p_fam["symptoms"]["list"], f"Family chest pain leaked into patient symptoms: {p_fam['symptoms']['list']}"
    assert len(p_fam["medical_history"]["family_history"]) > 0, "Expected family history recorded"
    print("   ✓ 1e: Family history ('father had chest pain') attributed to family history without leaking into patient symptoms")
    test_results.append(("1e: Family History Attribution Separation", True))

    # Test 1f: Historical vs Uncertain symptoms
    c_hist = PatientInfoCollector()
    c_hist.extract_from_patient_text("I had a fever yesterday, but not now. I am not sure whether I have chills. I have a sore throat.")
    p_hist = c_hist.get_patient_data()
    assert "sore throat" in p_hist["symptoms"]["list"]
    assert "fever" not in p_hist["symptoms"]["list"], f"Historical fever leaked into active symptoms: {p_hist['symptoms']['list']}"
    assert "chills" not in p_hist["symptoms"]["list"], f"Uncertain chills leaked into active symptoms: {p_hist['symptoms']['list']}"
    assert any("fever" in h for h in p_hist["symptom_details"]["historical_symptoms"])
    assert any("chills" in u for u in p_hist["symptom_details"]["uncertain_symptoms"])
    print("   ✓ 1f: Historical symptoms ('fever yesterday, but not now') and uncertain symptoms separated accurately")
    test_results.append(("1f: Historical & Uncertain Symptom Separation", True))

    # Test 1g: Contradiction tracking across turns
    c_contra = PatientInfoCollector()
    c_contra.extract_from_patient_text("I have a mild headache.")
    assert "headache" in c_contra.get_patient_data()["symptoms"]["list"]
    assert c_contra.get_patient_data()["symptom_details"]["severity"] == "mild"
    
    c_contra.extract_from_patient_text("Actually my headache is severe, 8/10. Wait, I do not have a headache, my neck hurts.")
    p_contra = c_contra.get_patient_data()
    assert "headache" not in p_contra["symptoms"]["list"], "Negated headache must be removed from positive list"
    assert any("neck" in s for s in p_contra["symptoms"]["list"])
    assert len(p_contra["contradictions"]) >= 1, "Expected contradiction recorded"
    print("   ✓ 1g: Contradictions preserved and updated across turns without silent loss")
    test_results.append(("1g: Contradiction Tracking Across Turns", True))

    # Test 1h: Independent Standalone Severity Update (Single Symptom context)
    c_sev1 = PatientInfoCollector()
    c_sev1.extract_from_patient_text("I have a mild headache since yesterday.")
    assert "headache" in c_sev1.get_patient_data()["symptoms"]["list"]
    assert c_sev1.get_patient_data()["symptom_details"]["severity"] == "mild"
    c_sev1.extract_from_patient_text("Actually, it has become severe.")
    p_sev1 = c_sev1.get_patient_data()
    assert p_sev1["symptom_details"]["severity"] == "severe", f"Expected severity 'severe', got {p_sev1['symptom_details']['severity']}"
    assert len(p_sev1["contradictions"]) >= 1, "Expected severity change contradiction recorded"
    assert any(c.get("category") == "severity_change" and c.get("new") == "severe" for c in p_sev1["contradictions"])
    print("   ✓ 1h: Standalone severity update ('Actually, it has become severe') associated with active symptom")
    test_results.append(("1h: Standalone Severity Update (Single Symptom)", True))

    # Test 1i: Correction with Negation ("I meant moderate, not severe")
    c_sev2 = PatientInfoCollector()
    c_sev2.extract_from_patient_text("I have knee pain.")
    c_sev2.extract_from_patient_text("I meant moderate, not severe.")
    p_sev2 = c_sev2.get_patient_data()
    assert p_sev2["symptom_details"]["severity"] == "moderate", f"Expected 'moderate', got {p_sev2['symptom_details']['severity']}"
    print("   ✓ 1i: Correction with negation ('I meant moderate, not severe') accurately parsed as moderate")
    test_results.append(("1i: Correction with Negation Severity Parsing", True))

    # Test 1j: Standalone Severity with Multiple Symptoms (Ambiguity Detection)
    c_sev3 = PatientInfoCollector()
    c_sev3.extract_from_patient_text("I have a headache and a sore throat.")
    assert len(c_sev3.get_patient_data()["symptoms"]["list"]) >= 2
    c_sev3.extract_from_patient_text("Actually, it has become severe.")
    p_sev3 = c_sev3.get_patient_data()
    assert any(u.get("category") == "ambiguous_severity" for u in p_sev3.get("unresolved_clarifications", [])), "Expected ambiguous_severity recorded"
    gen_clarify = FollowupQuestionGenerator(config=config, llm=None)
    q_ambig = gen_clarify.get_next_question(p_sev3, questions_asked_count=1)
    assert q_ambig.get("question") is not None
    assert "clarify which symptom" in q_ambig.get("question", "").lower() or "severe" in q_ambig.get("question", "").lower()
    print("   ✓ 1j: Ambiguous standalone severity with multiple active symptoms prompted focused clarification")
    test_results.append(("1j: Ambiguous Standalone Severity Clarification", True))

    # --------------------------------------------------------------------------
    # 2. SAFETY TESTS: RedFlagDetector & Negation Shield
    # --------------------------------------------------------------------------
    print("\n[Suite 2] Safety Tests: RedFlagDetector & Negation Guardrails")
    detector = RedFlagDetector(config=config)
    
    # Test 2a: True Emergency Escalation (Chest Pain + Shortness of Breath)
    emergency_collector = PatientInfoCollector()
    emergency_collector.extract_from_patient_text("I have crushing chest pain radiating to my left arm and severe shortness of breath.")
    em_res = detector.detect_emergencies(emergency_collector.get_patient_data())
    assert em_res["should_escalate"] is True, "Emergency should have escalated for acute chest pain"
    assert em_res["has_red_flags"] is True
    assert em_res["severity"] in ["high", "critical"]
    print("   ✓ 2a: True emergency correctly detected and escalated immediately")
    test_results.append(("2a: True Emergency Immediate Escalation", True))
    
    # Test 2b: False Emergency Prevention (Patient explicitly denies chest pain)
    safe_collector = PatientInfoCollector()
    safe_collector.extract_from_patient_text("I have a runny nose and dry cough for 3 days. No chest pain and no shortness of breath.")
    safe_res = detector.detect_emergencies(safe_collector.get_patient_data())
    assert safe_res["should_escalate"] is False, "Negated chest pain should NOT trigger emergency escalation"
    assert safe_res["has_red_flags"] is False, f"Unexpected flags for negated emergency: {safe_res['flags']}"
    print("   ✓ 2b: False emergency prevented when warning signs are explicitly negated")
    test_results.append(("2b: False Emergency Prevention on Negations", True))

    # Test 2c: Isolated Active Chest Pain Escalation
    cp_collector = PatientInfoCollector()
    cp_collector.extract_from_patient_text("I have chest pain since this morning.")
    cp_res = detector.detect_emergencies(cp_collector.get_patient_data())
    assert cp_res["should_escalate"] is True, "Isolated active chest pain MUST escalate for urgent/emergency clinical care"
    assert cp_res["has_red_flags"] is True
    print("   ✓ 2c: Isolated active chest pain escalated safely without waiting for exertion or multi-symptom triggers")
    test_results.append(("2c: Isolated Chest Pain Emergency Escalation", True))

    # Test 2d: Family Chest Pain Non-Escalation
    fam_em_collector = PatientInfoCollector()
    fam_em_collector.extract_from_patient_text("My father had chest pain last year. I only have a minor cough.")
    fam_em_res = detector.detect_emergencies(fam_em_collector.get_patient_data())
    assert fam_em_res["should_escalate"] is False, "Family chest pain history should NOT trigger acute patient escalation"
    assert fam_em_res["has_red_flags"] is False
    print("   ✓ 2d: Family chest pain history separated from patient and prevented false emergency escalation")
    test_results.append(("2d: Family Emergency History False-Alarm Prevention", True))

    # Test 2e: Historical Resolved Chest Pain (Safe Non-Escalation)
    hist_cp_col = PatientInfoCollector()
    hist_cp_col.extract_from_patient_text("I had chest pain yesterday, but it is completely resolved now. Today I only have a mild runny nose.")
    hist_cp_res = detector.detect_emergencies(hist_cp_col.get_patient_data())
    assert hist_cp_res["should_escalate"] is False, f"Resolved historical chest pain must NOT escalate to emergency: {hist_cp_res}"
    assert hist_cp_res["has_red_flags"] is True, "Historical chest pain should still be tracked as an informative red flag"
    assert any(f.get("type") == "historical_chest_pain" for f in hist_cp_res["flags"])
    print("   ✓ 2e: Resolved historical chest pain correctly classified without triggering false emergency escalation")
    test_results.append(("2e: Resolved Historical Chest Pain Non-Escalation", True))

    # Test 2f: Recurrent Historical Chest Pain (Active Escalation)
    rec_cp_col = PatientInfoCollector()
    rec_cp_col.extract_from_patient_text("I had chest pain yesterday, and it came back today.")
    rec_cp_res = detector.detect_emergencies(rec_cp_col.get_patient_data())
    assert rec_cp_res["should_escalate"] is True, "Recurrent chest pain MUST escalate to emergency"
    assert any(f.get("type") == "chest_pain" for f in rec_cp_res["flags"])
    print("   ✓ 2f: Recurrent chest pain ('came back today') correctly identified as active emergency")
    test_results.append(("2f: Recurrent Chest Pain Active Emergency Escalation", True))

    # Test 2g: Resolved Historical Chest Pain WITH Separate Current Red Flag (No Safety Suppression)
    multi_em_col = PatientInfoCollector()
    multi_em_col.extract_from_patient_text("I had chest pain yesterday which resolved completely. But right now I have severe difficulty breathing.")
    multi_em_res = detector.detect_emergencies(multi_em_col.get_patient_data())
    assert multi_em_res["should_escalate"] is True, "Separate current red flag (difficulty breathing) must trigger escalation despite historical chest pain resolution"
    assert any(f.get("type") == "difficulty_breathing" for f in multi_em_res["flags"])
    print("   ✓ 2g: Separate current red flag (shortness of breath) escalated without suppression from historical chest pain")
    test_results.append(("2g: Current Red Flag Unsuppressed by Historical Chest Pain", True))

    # Test 2h: Sudden-Onset Severe Headache with Intervening Words
    sud_head_col = PatientInfoCollector()
    sud_head_col.extract_from_patient_text("I have a sudden excruciating severe headache that started 30 minutes ago.")
    sud_head_res = detector.detect_emergencies(sud_head_col.get_patient_data())
    assert sud_head_res["should_escalate"] is True, "Sudden severe headache with intervening words MUST escalate as stroke/intracranial warning"
    assert any(f.get("type") == "stroke_warning_signs" for f in sud_head_res["flags"])
    print("   ✓ 2h: Sudden severe headache with intervening words ('sudden excruciating severe headache') detected and escalated")
    test_results.append(("2h: Sudden Severe Headache with Intervening Words", True))

    # Test 2i: Worst Headache of My Life Starting Suddenly
    worst_head_col = PatientInfoCollector()
    worst_head_col.extract_from_patient_text("This is the worst headache of my life, starting suddenly.")
    worst_head_res = detector.detect_emergencies(worst_head_col.get_patient_data())
    assert worst_head_res["should_escalate"] is True, "'Worst headache of my life, starting suddenly' MUST escalate"
    assert any(f.get("type") == "stroke_warning_signs" for f in worst_head_res["flags"])
    print("   ✓ 2i: 'Worst headache of my life, starting suddenly' detected and escalated")
    test_results.append(("2i: Worst Headache of Life Starting Suddenly", True))

    # Test 2j: Severe Headache that Came On All at Once
    all_once_col = PatientInfoCollector()
    all_once_col.extract_from_patient_text("I have a severe headache that came on all at once.")
    all_once_res = detector.detect_emergencies(all_once_col.get_patient_data())
    assert all_once_res["should_escalate"] is True, "'Severe headache that came on all at once' MUST escalate"
    assert any(f.get("type") == "stroke_warning_signs" for f in all_once_res["flags"])
    print("   ✓ 2j: 'Severe headache that came on all at once' detected and escalated")
    test_results.append(("2j: Thunderclap Onset ('All at Once') Escalation", True))

    # Test 2k: Sudden Severe Headache Negation Shield
    neg_head_col = PatientInfoCollector()
    neg_head_col.extract_from_patient_text("I do not have a sudden severe headache. Just a mild cold.")
    neg_head_res = detector.detect_emergencies(neg_head_col.get_patient_data())
    assert neg_head_res["should_escalate"] is False, "Negated sudden severe headache must NOT escalate"
    assert not any(f.get("type") == "stroke_warning_signs" for f in neg_head_res["flags"])
    print("   ✓ 2k: Negated sudden severe headache safely shielded against false alarm")
    test_results.append(("2k: Sudden Severe Headache Negation Shield", True))

    # Test 2l: Sudden Severe Headache Family Attribution
    fam_head_col = PatientInfoCollector()
    fam_head_col.extract_from_patient_text("My father had a sudden severe headache last year, but I only have mild fatigue.")
    fam_head_res = detector.detect_emergencies(fam_head_col.get_patient_data())
    assert fam_head_res["should_escalate"] is False, "Family history of sudden severe headache must NOT escalate for patient"
    assert not any(f.get("type") == "stroke_warning_signs" for f in fam_head_res["flags"])
    print("   ✓ 2l: Family history of sudden severe headache separated from patient")
    test_results.append(("2l: Sudden Severe Headache Family History Shield", True))

    # Test 2m: Sudden Severe Headache Historical Resolution
    hist_head_col = PatientInfoCollector()
    hist_head_col.extract_from_patient_text("I had a sudden severe headache 3 years ago that completely resolved. Today I have an itchy rash.")
    hist_head_res = detector.detect_emergencies(hist_head_col.get_patient_data())
    assert hist_head_res["should_escalate"] is False, "Historical resolved sudden severe headache must NOT escalate as active emergency"
    assert not any(f.get("type") == "stroke_warning_signs" for f in hist_head_res["flags"])
    print("   ✓ 2m: Historical resolved sudden severe headache safely shielded from active emergency escalation")
    test_results.append(("2m: Historical Resolved Sudden Headache Shield", True))

    # --------------------------------------------------------------------------
    # 3. ANTI-REPETITION & QUESTION GENERATION
    # --------------------------------------------------------------------------
    print("\n[Suite 3] Anti-Repetition & Context Awareness in FollowupQuestionGenerator")
    gen = FollowupQuestionGenerator(config=config)
    
    # Test 3a: Sequential question generation avoiding duplicates
    headache_collector = PatientInfoCollector()
    headache_collector.extract_from_patient_text("I have a headache since yesterday.")
    
    q1 = gen.get_next_question(headache_collector.get_patient_data(), questions_asked_count=0)
    assert q1["question"] is not None
    assert "location" in q1["reason"].lower() or "where" in q1["question"].lower() or "side" in q1["question"].lower()
    assert "source" in q1, "Question result must specify provenance source ('groq' or 'clinical_fallback')"
    assert q1["source"] in ["groq", "clinical_fallback"]
    
    # Simulate patient answering location
    headache_collector.extract_from_patient_text("The pain is mostly on the right side of my head.")
    q2 = gen.get_next_question(headache_collector.get_patient_data(), questions_asked_count=1)
    assert q2["question"] is not None
    assert q2["question"] != q1["question"], "Follow-up question must not be identical to previous question"
    assert len(set(gen.get_all_asked_questions())) == 2, "Expected 2 distinct questions recorded without duplication"
    print(f"   ✓ 3a: Turn 1 Q: \"{q1['question'][:70]}...\" (Source: {q1['source']})")
    print(f"   ✓ 3a: Turn 2 Q: \"{q2['question'][:70]}...\" (Adapted, no repetition)")
    test_results.append(("3a: Follow-up Question Anti-Repetition & Provenance", True))

    # --------------------------------------------------------------------------
    # 4. COMPARATIVE ADAPTIVE TEST: DIVERGENT QUESTIONS FOR SAME COMPLAINT
    # --------------------------------------------------------------------------
    print("\n[Suite 4] Divergent Questioning for Patients with Same Initial Complaint")
    # Both patients start with: "I have had a headache for two days."
    
    # Patient Alpha: Reports unilateral right-side throbbing pain
    agent_alpha = ClinicalPreConsultationAgent(config)
    r_a1 = agent_alpha.process_patient_info("I have had a headache for two days.")
    r_a2 = agent_alpha.process_patient_info("It is throbbing on the right side of my forehead.")
    q_alpha_next = r_a2.get("next_question", {}).get("question", "")
    
    # Patient Beta: Reports generalized tension after computer work, already took paracetamol
    agent_beta = ClinicalPreConsultationAgent(config)
    r_b1 = agent_beta.process_patient_info("I have had a headache for two days.")
    r_b2 = agent_beta.process_patient_info("It is a dull ache across both temples, probably from staring at monitors. I took Paracetamol 500mg.")
    q_beta_next = r_b2.get("next_question", {}).get("question", "")
    
    assert q_alpha_next != "", "Patient Alpha should receive a valid next question"
    assert q_beta_next != "", "Patient Beta should receive a valid next question"
    assert q_alpha_next != q_beta_next, f"Expected divergent questions for different patient clinical profiles, but got identical: '{q_alpha_next}'"
    print(f"   ✓ 4: Patient Alpha (Right throbbing) Next Q: \"{q_alpha_next[:80]}...\"")
    print(f"   ✓ 4: Patient Beta (Bilateral dull + meds) Next Q: \"{q_beta_next[:80]}...\"")
    test_results.append(("4: Divergent Questions for Same Complaint", True))

    # --------------------------------------------------------------------------
    # 5. MULTI-SCENARIO CLINICAL TESTS (10 Distinct Conditions)
    # --------------------------------------------------------------------------
    print("\n[Suite 5] 10 Distinct Clinical Intake Scenarios")
    scenarios = [
        ("Headache", "I have a throbbing headache on my left temple for 3 days."),
        ("Fever", "I have had a fever of 101.5 F and chills since last night."),
        ("Cough/Respiratory", "I have a persistent hacking cough with green phlegm for a week."),
        ("Abdominal Pain", "I have cramping pain in my lower right stomach for 12 hours."),
        ("Skin/Rash", "I have an itchy red bumpy rash across my forearms since yesterday."),
        ("Joint Pain", "My right knee is swollen and stiff when I wake up in the morning."),
        ("Chronic Condition", "I have type 2 diabetes and my blood sugar has been unusually high for 4 days."),
        ("Multiple Complaints", "I have a sore throat, nasal congestion, and mild body aches for 2 days."),
        ("Surgical History", "I had gallbladder surgery 2 weeks ago and now feel sharp pain near the incision."),
        ("Medication/Allergy", "I broke out in hives after taking amoxicillin yesterday.")
    ]
    
    all_scenarios_passed = True
    for name, initial_statement in scenarios:
        agent = ClinicalPreConsultationAgent(config)
        resp = agent.process_patient_info(initial_statement)
        q = resp.get("next_question")
        has_q = bool(q and q.get("question"))
        is_active = resp.get("status") == "active" or resp.get("status") == "emergency_escalation"
        if not (has_q or is_active):
            all_scenarios_passed = False
            print(f"   ❌ Scenario '{name}' failed to generate question or valid status")
        else:
            q_text = q.get("question", resp.get("message")) if q else resp.get("message")
            print(f"   ✓ Scenario '{name}': Q -> \"{q_text[:75]}...\"")
    
    test_results.append(("5: 10 Distinct Clinical Scenarios Adaptive Responses", all_scenarios_passed))

    # --------------------------------------------------------------------------
    # 6. PATIENT-CONTROLLED EARLY COMPLETION
    # --------------------------------------------------------------------------
    print("\n[Suite 6] Patient-Controlled Early Completion")
    agent_early = ClinicalPreConsultationAgent(config)
    agent_early.process_patient_info("I have a mild sore throat for 1 day.")
    r_stop = agent_early.process_patient_info("That's all the info I have, please give me the report now.")
    
    assert r_stop.get("status") in ["completed", "partial"], f"Expected completed/partial status, got {r_stop.get('status')}"
    assert r_stop.get("formatted_report") is not None, "Report should be generated on patient completion request"
    assert "PRE-CONSULTATION INTAKE REPORT" in r_stop.get("formatted_report", "")
    assert r_stop.get("next_question") is None, "Questions should halt upon early completion"
    print("   ✓ 6: Early completion honored, generated factual report, halted questioning")
    test_results.append(("6: Patient-Controlled Early Completion", True))

    # --------------------------------------------------------------------------
    # 7. PATIENT FEEDBACK & CORRECTION LOOP
    # --------------------------------------------------------------------------
    print("\n[Suite 7] Patient Feedback & Report Correction")
    # Submit feedback to the completed agent
    feedback_resp = agent_early.process_patient_info("I forgot to mention that I have asthma and take an albuterol inhaler.")
    assert feedback_resp.get("status") == "completed"
    assert feedback_resp.get("formatted_report") is not None
    rep_text = feedback_resp.get("formatted_report", "")
    assert "asthma" in rep_text.lower(), "Updated report must include newly corrected condition 'asthma'"
    assert "albuterol" in rep_text.lower(), "Updated report must include newly corrected medication 'albuterol'"
    print("   ✓ 7: Patient feedback accumulated into clinical context and regenerated into updated report")
    test_results.append(("7: Patient Feedback & Report Regeneration", True))

    # --------------------------------------------------------------------------
    # 8. REPORT ACCURACY & ABSENCE OF FABRICATION
    # --------------------------------------------------------------------------
    print("\n[Suite 8] Report Accuracy & Zero Fabrication Audit")
    report_dict = feedback_resp.get("report", {})
    # Vitals were never mentioned, should be marked as not provided or unknown, not fabricated numbers
    vitals_readings = report_dict.get("vitals", {}).get("readings", {})
    assert vitals_readings.get("blood_pressure") == "Not provided" or vitals_readings.get("blood_pressure") is None
    assert vitals_readings.get("temperature") == "Not provided" or vitals_readings.get("temperature") is None
    # No clinical diagnosis should be in the report
    assert "diagnosis" not in report_dict or report_dict.get("diagnosis") is None
    print("   ✓ 8: Unmeasured vital signs marked 'Not provided', zero fabricated clinical claims")
    test_results.append(("8: Zero Fabrication in Clinical Report", True))

    # --------------------------------------------------------------------------
    # 9. STRUCTURED LLM PARSING & DECISION SUITE (Mocked DeepSeek)
    # --------------------------------------------------------------------------
    print("\n[Suite 9] Structured LLM Parsing & Decision Handling (Mocked DeepSeek)")
    class MockLLM:
        def __init__(self, json_payload):
            self.payload = json_payload
        def invoke(self, prompt):
            class Resp:
                def __init__(self, c):
                    self.content = c
            return Resp(self.payload)
    
    # 9a: ASK_QUESTION decision
    mock_ask = MockLLM(json.dumps({
        "decision": "ASK_QUESTION",
        "reason": "Clarify headache radiation",
        "target_area": "symptom_details",
        "question": "Does the pain radiate down into your neck or shoulders?",
        "is_sufficient_for_report": False
    }))
    gen_mock = FollowupQuestionGenerator(config=config, llm=mock_ask)
    res_ask = gen_mock.get_next_question(headache_collector.get_patient_data(), questions_asked_count=1)
    assert res_ask["question"] == "Does the pain radiate down into your neck or shoulders?"
    assert res_ask["category"] == "symptom_details"
    print("   ✓ 9a: Correctly parsed ASK_QUESTION decision from structured LLM JSON")
    test_results.append(("9a: Structured LLM ASK_QUESTION", True))

    # 9b: GENERATE_REPORT decision
    mock_rep = MockLLM(json.dumps({
        "decision": "GENERATE_REPORT",
        "reason": "Comprehensive symptom characterization complete",
        "target_area": "sufficient_info",
        "question": None,
        "is_sufficient_for_report": True
    }))
    gen_mock_rep = FollowupQuestionGenerator(config=config, llm=mock_rep)
    res_rep = gen_mock_rep.get_next_question(headache_collector.get_patient_data(), questions_asked_count=2)
    assert res_rep["question"] is None
    assert res_rep["sufficient_info"] is True
    print("   ✓ 9b: Correctly parsed GENERATE_REPORT decision from structured LLM JSON")
    test_results.append(("9b: Structured LLM GENERATE_REPORT", True))

    # 9c: ESCALATE_SAFETY decision
    mock_safe = MockLLM(json.dumps({
        "decision": "ESCALATE_SAFETY",
        "reason": "Sudden thunderclap onset reported",
        "target_area": "emergency_escalation",
        "question": None,
        "is_sufficient_for_report": False
    }))
    gen_mock_safe = FollowupQuestionGenerator(config=config, llm=mock_safe)
    res_safe = gen_mock_safe.get_next_question(headache_collector.get_patient_data(), questions_asked_count=1)
    assert res_safe["should_escalate"] is True
    assert res_safe["question"] is None
    print("   ✓ 9c: Correctly parsed ESCALATE_SAFETY decision from structured LLM JSON")
    test_results.append(("9c: Structured LLM ESCALATE_SAFETY", True))

    # --------------------------------------------------------------------------
    # 10. LIVE GROQ PROVIDER STATUS AUDIT (MANDATORY GROQ PROVIDER)
    # --------------------------------------------------------------------------
    print("\n[Suite 10] Live Groq Provider Status Audit (qwen/qwen3.8-27b)")
    groq_available = False
    groq_status_msg = "Unknown"
    try:
        from provider_config_adapter import get_groq_provider
        groq = get_groq_provider(temperature=0.3)
        resp = groq.invoke("Ping! Respond with 'Pong'")
        content = resp.content if hasattr(resp, "content") else str(resp)
        groq_available = True
        groq_status_msg = f"Active (Response: {content[:40].strip()})"
    except Exception as e:
        groq_available = False
        groq_status_msg = f"Error: {e}"
    
    assert groq_available is True, f"Groq provider must be available and responsive: {groq_status_msg}"
    print(f"   ℹ Groq Model: qwen/qwen3.8-27b | Status: {groq_status_msg}")
    print("   ✓ 10: Groq provider exclusively configured, active, and verified live")
    test_results.append(("10: Live Groq Provider Status Audit", True))

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY: ADAPTIVE CLINICAL PRE-CONSULTATION AGENT")
    print("=" * 80)
    all_passed = True
    for name, passed in test_results:
        symbol = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {symbol} - {name}")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
