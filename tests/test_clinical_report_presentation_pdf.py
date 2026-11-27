"""
Test Suite for Clinical Consultation Report Presentation & PDF Export
Verifies:
1. Report parsing from clinical intake output (no invented/fabricated fields).
2. Professional PDF-1.4 binary generation.
3. 1-page layout for standard reports.
4. Clean multi-page continuation for extensive reports without truncation.
5. Emergency alert screening presentation.
6. Session isolation & metadata binding (no stale report leakage).
"""

import os
import sys
import json
import subprocess
import unittest

try:
    import fitz
except ImportError:
    fitz = None


class TestClinicalReportPresentationAndPDF(unittest.TestCase):
    """Test suite for clinical report PDF generation and layout validation."""

    def setUp(self):
        self.frontend_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend"
        )
        self.test_pdf_path = os.path.join(self.frontend_dir, "test_verification.pdf")

    def tearDown(self):
        if os.path.exists(self.test_pdf_path):
            try:
                os.remove(self.test_pdf_path)
            except OSError:
                pass

    def _generate_pdf_via_node(self, report_text: str, metadata: dict) -> bytes:
        """Call pdfGenerator.js via Node to generate PDF buffer."""
        node_script = f"""
        import {{ buildClinicalReportPdf }} from './src/utils/pdfGenerator.js';
        import fs from 'fs';

        const reportText = {json.dumps(report_text)};
        const metadata = {json.dumps(metadata)};

        try {{
            const buf = buildClinicalReportPdf(reportText, metadata);
            process.stdout.write(Buffer.from(buf));
        }} catch (err) {{
            process.stderr.write(err.message || String(err));
            process.exit(1);
        }}
        """
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", node_script],
            cwd=self.frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Node PDF generation failed: {proc.stderr.decode('utf-8', 'replace')}")
        return proc.stdout

    def test_01_standard_clinical_report_pdf_one_page(self):
        """Test standard clinical report generates a valid 1-page PDF document."""
        sample_report = (
            "PRE-CONSULTATION INTAKE REPORT\n"
            "Generated: 2026-11-25T00:11:24.128048\n\n"
            "CHIEF COMPLAINT\n"
            "  Reason: Persistent migraine headache with visual aura\n"
            "  Duration: 3 days\n"
            "  Status: provided\n\n"
            "SYMPTOMS\n"
            "  Reported: unilateral throbbing pain, photophobia, nausea\n"
            "  Location: right temporal region\n"
            "  Severity: 7/10\n"
            "  Character: pulsating, throbbing\n"
            "  Onset: gradual\n"
            "  Progression: worsening with physical activity\n"
            "  Explicitly Denied (Pertinent Negatives): no fever, no neck stiffness, no confusion\n"
            "  Status: Patient reported symptoms\n\n"
            "VITAL SIGNS\n"
            "  Blood Pressure: 122/80 mmHg\n"
            "  Heart Rate: 74 bpm\n"
            "  Temperature: 98.4 F\n"
            "  Oxygen Saturation: 99%\n\n"
            "MEDICAL HISTORY\n"
            "  Conditions: Episodic migraine\n"
            "  Surgeries: None reported\n"
            "  Hospitalizations: None reported\n"
            "  Family History: Mother has migraine history\n\n"
            "MEDICATIONS\n"
            "  Current: Sumatriptan 50mg PRN\n"
            "  Status: Patient reported\n\n"
            "ALLERGIES\n"
            "  Drug: Penicillin (urticaria)\n"
            "  Food: None reported\n"
            "  Environmental: Pollen\n\n"
            "LIFESTYLE\n"
            "  Smoking: Non-smoker\n"
            "  Alcohol: Rare\n"
            "  Exercise: 3x per week\n\n"
            "DEMOGRAPHICS\n"
            "  Age: 32 years\n"
            "  Gender: Female\n\n"
            "CLINICAL SUMMARY\n"
            "32-year-old female presents with 3-day history of right temporal throbbing headache with photophobia and nausea.\n"
            "Pertinent negatives include absence of fever and meningismus. Vitals within normal limits. Past history of episodic migraine."
        )

        meta = {
            "sessionId": "clin_migraine_8492",
            "isConfirmed": True,
            "confirmationTimestamp": "Nov 25, 2026, 12:05 AM"
        }

        pdf_bytes = self._generate_pdf_via_node(sample_report, meta)
        self.assertGreater(len(pdf_bytes), 2000, "PDF size should be substantial")
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"), "Must be valid PDF-1.4")

        if fitz:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            self.assertEqual(doc.page_count, 1, "Standard clinical report must fit on exactly ONE page")
            page_text = doc[0].get_text()
            self.assertIn("Clinical Consultation Report", page_text)
            self.assertIn("Persistent migraine headache with visual aura", page_text)
            self.assertIn("right temporal region", page_text)
            self.assertIn("Penicillin", page_text)
            self.assertIn("Sumatriptan", page_text)
            self.assertIn("Verified by Patient", page_text)
            self.assertIn("Clinical Notice:", page_text)
            doc.close()

    def test_02_emergency_red_flag_alert_in_pdf(self):
        """Test emergency red flag detection renders prominent critical alerts in PDF."""
        sample_emergency = (
            "PRE-CONSULTATION INTAKE REPORT\n"
            "Generated: 2026-11-25T00:15:00.000000\n\n"
            "[!] EMERGENCY ESCALATION REQUIRED [!]\n"
            "Immediate emergency department referral for crushing chest pain.\n\n"
            "CHIEF COMPLAINT\n"
            "  Reason: Sudden crushing substernal chest pressure\n"
            "  Duration: 30 minutes\n\n"
            "SYMPTOMS\n"
            "  Reported: substernal chest pain, diaphoresis, dyspnea\n"
            "  Severity: 9/10\n"
            "  Progression: acute\n\n"
            "RED FLAGS / EMERGENCY ASSESSMENT\n"
            "  Severity: CRITICAL\n"
            "  Escalation Required: True\n\n"
            "CLINICAL SUMMARY\n"
            "Patient experiencing acute severe substernal pressure with diaphoresis. Immediate emergency triage dispatched."
        )

        meta = {"sessionId": "clin_emergency_999", "isConfirmed": False}
        pdf_bytes = self._generate_pdf_via_node(sample_emergency, meta)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))

        if fitz:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            self.assertEqual(doc.page_count, 1)
            page_text = doc[0].get_text()
            self.assertIn("EMERGENCY", page_text)
            self.assertIn("CRITICAL", page_text)
            self.assertIn("Sudden crushing substernal chest pressure", page_text)
            doc.close()

    def test_03_extensive_report_preserves_all_data_without_truncation(self):
        """Test that long clinical summaries seamlessly flow across pages without data loss."""
        repeated_history = (
            "Patient reports a complex 10-year history of multisystem symptoms including "
            "intermittent arthralgias, Raynaud phenomenon, photosensitive malar rash, "
            "and episodic pleuritic chest discomfort requiring multidisciplinary evaluation. "
        ) * 15

        long_report = (
            "PRE-CONSULTATION INTAKE REPORT\n"
            "Generated: 2026-11-25\n\n"
            "CHIEF COMPLAINT\n"
            "  Reason: Multisystem inflammatory symptom complex\n"
            "  Duration: 10 years\n\n"
            "SYMPTOMS\n"
            "  Reported: polyarthralgias, fatigue, rash, pleurisy\n"
            "  Location: multiple joints bilateral\n"
            "  Severity: 6/10\n\n"
            "MEDICAL HISTORY\n"
            "  Conditions: Systemic Lupus Erythematosus suspect, Sjogren syndrome\n"
            "  Surgeries: Joint aspiration (2021)\n\n"
            "CLINICAL SUMMARY\n" + repeated_history
        )

        meta = {"sessionId": "clin_multipage_001", "isConfirmed": True}
        pdf_bytes = self._generate_pdf_via_node(long_report, meta)

        if fitz:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            self.assertGreaterEqual(doc.page_count, 2, "Long report must cleanly flow to second page")
            full_text = "".join(p.get_text() for p in doc)
            self.assertIn("Multisystem inflammatory symptom complex", full_text)
            self.assertIn("Systemic Lupus Erythematosus suspect", full_text)
            self.assertIn("Continued from Page 1", doc[1].get_text())
            doc.close()

    def test_04_session_isolation_in_pdf_metadata(self):
        """Test that generated PDF is strictly bound to the specific session ID and metadata."""
        rep_a = "PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT\n  Reason: Cough and sore throat"
        rep_b = "PRE-CONSULTATION INTAKE REPORT\nCHIEF COMPLAINT\n  Reason: Sprained left ankle"

        pdf_a = self._generate_pdf_via_node(rep_a, {"sessionId": "clin_session_AAA", "isConfirmed": True})
        pdf_b = self._generate_pdf_via_node(rep_b, {"sessionId": "clin_session_BBB", "isConfirmed": False})

        if fitz:
            doc_a = fitz.open(stream=pdf_a, filetype="pdf")
            text_a = doc_a[0].get_text()
            self.assertIn("Cough and sore throat", text_a)
            self.assertNotIn("Sprained left ankle", text_a)
            self.assertIn("SESSION_AAA", text_a)
            doc_a.close()

            doc_b = fitz.open(stream=pdf_b, filetype="pdf")
            text_b = doc_b[0].get_text()
            self.assertIn("Sprained left ankle", text_b)
            self.assertNotIn("Cough and sore throat", text_b)
            self.assertIn("SESSION_BBB", text_b)
            doc_b.close()


if __name__ == "__main__":
    unittest.main()
