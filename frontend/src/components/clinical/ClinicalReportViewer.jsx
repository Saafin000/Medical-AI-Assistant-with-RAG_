import React, { useState } from 'react';
import { useClinicalPreConsultation } from '../../context/ClinicalPreConsultationContext';
import { parseClinicalReport, downloadClinicalReportPdf, decodeHtmlEntities } from '../../utils/pdfGenerator';

function buildReportSpeechText(report, fallbackText) {
  if (!report || typeof report !== 'object') {
    return fallbackText || '';
  }
  const parts = [];
  if (report.chiefComplaint?.reason) {
    const duration = report.chiefComplaint?.duration
      ? ` for ${decodeHtmlEntities(report.chiefComplaint.duration)}`
      : '';
    parts.push(`Chief complaint: ${decodeHtmlEntities(report.chiefComplaint.reason)}${duration}.`);
  }
  if (report.hpi) {
    parts.push(`History of present illness: ${decodeHtmlEntities(report.hpi)}.`);
  }
  if (report.symptoms?.reported) {
    parts.push(`Reported symptoms: ${decodeHtmlEntities(report.symptoms.reported)}.`);
  }
  if (report.symptoms?.location) {
    parts.push(`Location: ${decodeHtmlEntities(report.symptoms.location)}.`);
  }
  if (report.symptoms?.severity) {
    parts.push(`Severity: ${decodeHtmlEntities(report.symptoms.severity)}.`);
  }
  if (report.clinicalSummary) {
    parts.push(`Clinical summary: ${decodeHtmlEntities(report.clinicalSummary)}.`);
  }
  if (Array.isArray(report.recommendations) && report.recommendations.length > 0) {
    const recs = report.recommendations.map((r) => decodeHtmlEntities(r)).filter(Boolean).join('. ');
    if (recs) {
      parts.push(`Suggested next steps: ${recs}.`);
    }
  }
  return parts.length > 0 ? parts.join(' ') : (fallbackText || '');
}

export default function ClinicalReportViewer({ reportText }) {
  const {
    sessionId,
    isConfirmed,
    confirmationTimestamp,
    confirmReport,
    submitFeedback,
    isFeedbackSubmitting,
    feedbackError,
    playSpeech,
    stopSpeech,
    activeAudioId,
    isAudioPlaying,
    isAudioPaused,
    isAudioLoading,
    audioError,
    clearAudioError
  } = useClinicalPreConsultation();

  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');
  const [updateSuccess, setUpdateSuccess] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState('idle'); // 'idle' | 'loading' | 'success' | 'error'
  const [downloadErrorMessage, setDownloadErrorMessage] = useState('');

  if (!reportText) return null;

  // Parse structured data from report text
  const report = parseClinicalReport(reportText);

  // Handle PDF Download
  const handleDownload = () => {
    try {
      setDownloadStatus('loading');
      setDownloadErrorMessage('');
      const result = downloadClinicalReportPdf(reportText, {
        sessionId: sessionId || 'INTAKE',
        isConfirmed,
        confirmationTimestamp
      });
      setDownloadStatus('success');
      setTimeout(() => setDownloadStatus('idle'), 3500);
    } catch (err) {
      console.error('Report download error:', err);
      setDownloadStatus('error');
      setDownloadErrorMessage(err.message || 'Unable to download report. Please try again.');
    }
  };

  const handleFeedbackSubmit = async (e) => {
    e.preventDefault();
    if (!feedbackText.trim()) return;
    try {
      await submitFeedback(feedbackText);
      setFeedbackText('');
      setShowFeedbackForm(false);
      setUpdateSuccess(true);
      setTimeout(() => setUpdateSuccess(false), 6000);
    } catch (err) {
      console.error('Feedback submit error:', err);
    }
  };

  const dateDisplay = report?.generatedAt
    ? report.generatedAt.slice(0, 10)
    : new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

  return (
    <div className="clinical-report-view-container">
      {/* Top Document Action Bar */}
      <div className="report-action-toolbar">
        <div className="toolbar-left">
          <span className="toolbar-doc-badge">
            <i className="fas fa-file-medical-alt text-primary"></i>
            <strong>Clinical Document Ready</strong>
          </span>
          {isConfirmed && (
            <span className="toolbar-status-badge badge-confirmed">
              <i className="fas fa-check-circle"></i> Patient Verified
            </span>
          )}
        </div>

        <div className="toolbar-actions">
          {/* Download Report Button */}
          <button
            type="button"
            className={`btn-download-report ${downloadStatus === 'loading' ? 'btn-loading' : ''} ${
              downloadStatus === 'success' ? 'btn-success' : ''
            }`}
            onClick={handleDownload}
            disabled={downloadStatus === 'loading'}
            title="Download this clinical consultation report as a professional PDF"
          >
            {downloadStatus === 'loading' ? (
              <>
                <i className="fas fa-spinner fa-spin"></i>
                <span>Generating PDF...</span>
              </>
            ) : downloadStatus === 'success' ? (
              <>
                <i className="fas fa-check"></i>
                <span>PDF Downloaded</span>
              </>
            ) : (
              <>
                <i className="fas fa-file-pdf"></i>
                <span>Download Report</span>
              </>
            )}
          </button>

          {/* Voice Summary Audio Controls */}
          <button
            type="button"
            className={`btn-report-audio ${isAudioPlaying && activeAudioId === 'clinical-report' ? 'btn-audio-active' : ''}`}
            onClick={() => playSpeech('clinical-report', buildReportSpeechText(report, reportText))}
            title="Listen to an audio readout of this pre-consultation summary"
          >
            {isAudioLoading && activeAudioId === 'clinical-report' ? (
              <>
                <i className="fas fa-spinner fa-spin"></i>
                <span>Generating Audio...</span>
              </>
            ) : isAudioPlaying && activeAudioId === 'clinical-report' ? (
              <>
                <i className="fas fa-pause"></i>
                <span>Pause Speech</span>
              </>
            ) : isAudioPaused && activeAudioId === 'clinical-report' ? (
              <>
                <i className="fas fa-play"></i>
                <span>Resume Speech</span>
              </>
            ) : (
              <>
                <i className="fas fa-volume-up"></i>
                <span>Read Aloud</span>
              </>
            )}
          </button>

          {activeAudioId === 'clinical-report' && (isAudioLoading || isAudioPlaying || isAudioPaused) && (
            <button
              type="button"
              className="btn-report-audio"
              onClick={stopSpeech}
              title="Stop report audio readout"
              aria-label="Stop report speech"
            >
              <i className="fas fa-stop"></i>
              <span>Stop</span>
            </button>
          )}
        </div>
      </div>

      {/* Audio Playback Error Banner if any */}
      {audioError && audioError.id === 'clinical-report' && (
        <div className="download-error-toast" role="alert">
          <i className="fas fa-exclamation-circle"></i>
          <span>{audioError.message}</span>
          <button type="button" onClick={clearAudioError} className="btn-close-toast" aria-label="Dismiss audio error">
            &times;
          </button>
        </div>
      )}

      {/* Download Error Banner if any */}
      {downloadStatus === 'error' && (
        <div className="download-error-toast" role="alert">
          <i className="fas fa-exclamation-circle"></i>
          <span>{downloadErrorMessage}</span>
          <button type="button" onClick={() => setDownloadStatus('idle')} className="btn-close-toast">
            &times;
          </button>
        </div>
      )}

      {/* =========================================================================
          PROFESSIONAL CLINICAL DOCUMENT (PHYSICAL PAPER PRESENTATION)
          ========================================================================= */}
      <article className="clinical-document-paper" id="printable-clinical-report">
        {/* Document Header */}
        <header className="clinical-doc-header">
          <div className="doc-header-main">
            <div className="doc-clinic-brand">
              <div className="clinic-crest-icon">
                <i className="fas fa-hospital-user"></i>
              </div>
              <div className="clinic-title-group">
                <h2 className="clinical-doc-title">Clinical Consultation Report</h2>
                <p className="clinical-doc-subtitle">
                  AI-Assisted Pre-Consultation Summary &middot; Outpatient Clinical Intake
                </p>
              </div>
            </div>

            <div className="doc-meta-card">
              <div className="doc-meta-row">
                <span className="meta-label">Encounter Date:</span>
                <span className="meta-value">{dateDisplay}</span>
              </div>
              <div className="doc-meta-row">
                <span className="meta-label">Session ID:</span>
                <span className="meta-value font-mono">
                  {sessionId ? `${sessionId.slice(0, 14)}...` : 'Pre-Consult'}
                </span>
              </div>
              <div className="doc-meta-row">
                <span className="meta-label">Record Type:</span>
                <span className="meta-value badge-pill-reported">Patient-Reported</span>
              </div>
            </div>
          </div>

          <div className="doc-header-rule"></div>
        </header>

        {/* Emergency Alert Banner (if acute red flags detected) */}
        {report?.isEmergency && (
          <section className="doc-emergency-alert" role="alert" aria-label="Emergency warning">
            <div className="alert-icon-wrap">
              <i className="fas fa-exclamation-triangle"></i>
            </div>
            <div className="alert-body">
              <strong>CRITICAL CLINICAL ALERT &mdash; IMMEDIATE MEDICAL EVALUATION RECOMMENDED</strong>
              <p>
                Patient intake responses contain acute red-flag indicators requiring priority healthcare evaluation.
                This summary should be escalated to attending clinical staff immediately.
              </p>
            </div>
          </section>
        )}

        {/* Patient Profile & Demographics Strip */}
        <section className="doc-patient-profile-strip">
          <div className="profile-chip">
            <span className="profile-chip-label">Subject:</span>
            <span className="profile-chip-val">Outpatient Pre-Consultation</span>
          </div>
          <div className="profile-chip">
            <span className="profile-chip-label">Age:</span>
            <span className="profile-chip-val">
              {decodeHtmlEntities(report?.demographics?.age) || 'Adult (Not specified)'}
            </span>
          </div>
          {report?.demographics?.gender && (
            <div className="profile-chip">
              <span className="profile-chip-label">Gender:</span>
              <span className="profile-chip-val">{decodeHtmlEntities(report.demographics.gender)}</span>
            </div>
          )}
          <div className="profile-chip profile-chip-status">
            <span className="profile-chip-label">Patient Review:</span>
            <span className={`profile-chip-val ${isConfirmed ? 'text-success' : 'text-muted'}`}>
              {isConfirmed ? (
                <>
                  <i className="fas fa-check-circle"></i> Verified by Patient
                </>
              ) : (
                'Pending Review'
              )}
            </span>
          </div>
        </section>

        {/* Chief Complaint Card */}
        <section className="doc-section doc-chief-complaint-section">
          <div className="section-header-bar">
            <h3 className="section-heading">
              <i className="fas fa-stethoscope"></i> Chief Complaint
            </h3>
            {report?.chiefComplaint?.duration && (
              <span className="section-meta-tag">
                Duration: {decodeHtmlEntities(report.chiefComplaint.duration)}
              </span>
            )}
          </div>
          <div className="chief-complaint-body">
            <p className="complaint-primary-text">
              {decodeHtmlEntities(report?.chiefComplaint?.reason) || 'Clinical pre-consultation symptom assessment.'}
            </p>
          </div>
        </section>

        {/* History of Present Illness (HPI) */}
        {report?.hpi && (
          <section className="doc-section doc-clinical-summary-box" style={{ marginBottom: '16px' }}>
            <div className="summary-box-header">
              <div className="summary-title-group">
                <i className="fas fa-book-medical text-primary"></i>
                <h4>History of Present Illness (HPI)</h4>
              </div>
              <span className="card-source-tag">Chronological Narrative</span>
            </div>
            <div className="summary-narrative-content">
              <p className="summary-text-block">{decodeHtmlEntities(report.hpi)}</p>
            </div>
          </section>
        )}

        {/* Two-Column Clinical Information Grid */}
        <div className="doc-two-column-grid">
          {/* LEFT COLUMN: Symptoms & Vital Signs */}
          <div className="doc-col">
            {/* Symptoms & Timeline */}
            <section className="doc-card">
              <div className="doc-card-header">
                <h4>
                  <i className="fas fa-notes-medical"></i> Symptoms &amp; Clinical Details
                </h4>
                <span className="card-source-tag">Patient-Reported</span>
              </div>
              <div className="doc-card-body">
                <div className="card-field-row">
                  <span className="field-name">Reported:</span>
                  <span className="field-val font-semibold">
                    {decodeHtmlEntities(report?.symptoms?.reported || 'None reported')}
                  </span>
                </div>
                {report?.symptoms?.location && (
                  <div className="card-field-row">
                    <span className="field-name">Location:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.location)}</span>
                  </div>
                )}
                {report?.symptoms?.severity && (
                  <div className="card-field-row">
                    <span className="field-name">Severity:</span>
                    <span className="field-val severity-indicator">{decodeHtmlEntities(report.symptoms.severity)}</span>
                  </div>
                )}
                {report?.symptoms?.character && (
                  <div className="card-field-row">
                    <span className="field-name">Character:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.character)}</span>
                  </div>
                )}
                {report?.symptoms?.onset && (
                  <div className="card-field-row">
                    <span className="field-name">Onset:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.onset)}</span>
                  </div>
                )}
                {report?.symptoms?.timing && (
                  <div className="card-field-row">
                    <span className="field-name">Timing:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.timing)}</span>
                  </div>
                )}
                {report?.symptoms?.frequency && (
                  <div className="card-field-row">
                    <span className="field-name">Frequency:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.frequency)}</span>
                  </div>
                )}
                {report?.symptoms?.progression && (
                  <div className="card-field-row">
                    <span className="field-name">Progression:</span>
                    <span className="field-val">{decodeHtmlEntities(report.symptoms.progression)}</span>
                  </div>
                )}
                {report?.symptoms?.timelines && report.symptoms.timelines.length > 0 && (
                  <div className="card-field-row">
                    <span className="field-name">Timelines:</span>
                    <span className="field-val">
                      {report.symptoms.timelines.map((t, i) => (
                        <div key={i}>{decodeHtmlEntities(t)}</div>
                      ))}
                    </span>
                  </div>
                )}
                {report?.symptoms?.negatives && (
                  <div className="card-field-row card-field-negatives">
                    <span className="field-name">Pertinent Negatives:</span>
                    <span className="field-val text-muted">{decodeHtmlEntities(report.symptoms.negatives)}</span>
                  </div>
                )}
              </div>
            </section>

            {/* Vital Signs */}
            <section className="doc-card">
              <div className="doc-card-header">
                <h4>
                  <i className="fas fa-heartbeat"></i> Vital Signs
                </h4>
                <span className="card-source-tag">Self-Reported</span>
              </div>
              <div className="doc-card-body doc-vitals-grid">
                <div className="vital-metric">
                  <span className="vital-label">Blood Pressure</span>
                  <span className={`vital-val ${!report?.vitals?.bloodPressure ? 'val-empty' : ''}`}>
                    {decodeHtmlEntities(report?.vitals?.bloodPressure) || 'Not provided'}
                  </span>
                </div>
                <div className="vital-metric">
                  <span className="vital-label">Heart Rate</span>
                  <span className={`vital-val ${!report?.vitals?.heartRate ? 'val-empty' : ''}`}>
                    {decodeHtmlEntities(report?.vitals?.heartRate) || 'Not provided'}
                  </span>
                </div>
                <div className="vital-metric">
                  <span className="vital-label">Temperature</span>
                  <span className={`vital-val ${!report?.vitals?.temperature ? 'val-empty' : ''}`}>
                    {decodeHtmlEntities(report?.vitals?.temperature) || 'Not provided'}
                  </span>
                </div>
                <div className="vital-metric">
                  <span className="vital-label">Oxygen Saturation</span>
                  <span className={`vital-val ${!report?.vitals?.oxygenSaturation ? 'val-empty' : ''}`}>
                    {decodeHtmlEntities(report?.vitals?.oxygenSaturation) || 'Not provided'}
                  </span>
                </div>
              </div>
            </section>
          </div>

          {/* RIGHT COLUMN: Medical History, Medications, Allergies, Lifestyle */}
          <div className="doc-col">
            {/* Medical History */}
            <section className="doc-card">
              <div className="doc-card-header">
                <h4>
                  <i className="fas fa-history"></i> Medical &amp; Surgical History
                </h4>
                <span className="card-source-tag">Past History</span>
              </div>
              <div className="doc-card-body">
                <div className="card-field-row">
                  <span className="field-name">Conditions:</span>
                  <span className="field-val">
                    {decodeHtmlEntities(report?.medicalHistory?.conditions || 'Not asked')}
                  </span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Surgeries:</span>
                  <span className="field-val">
                    {decodeHtmlEntities(report?.medicalHistory?.surgeries || 'Not asked')}
                  </span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Hospitalizations:</span>
                  <span className="field-val">
                    {decodeHtmlEntities(report?.medicalHistory?.hospitalizations || 'Not asked')}
                  </span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Family History:</span>
                  <span className="field-val">
                    {decodeHtmlEntities(report?.medicalHistory?.familyHistory || 'Not asked')}
                  </span>
                </div>
              </div>
            </section>

            {/* Medications & Allergies */}
            <section className="doc-card">
              <div className="doc-card-header">
                <h4>
                  <i className="fas fa-pills"></i> Medications &amp; Allergies
                </h4>
                <span className="card-source-tag">Active List</span>
              </div>
              <div className="doc-card-body">
                <div className="card-field-row">
                  <span className="field-name">Current Meds:</span>
                  <span className="field-val font-semibold">
                    {decodeHtmlEntities(report?.medications?.current || 'Not asked')}
                  </span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Drug Allergies:</span>
                  <span className="field-val text-danger-dark font-semibold">
                    {decodeHtmlEntities(report?.allergies?.drug || 'Not asked')}
                  </span>
                </div>
                {report?.allergies?.environmental && (
                  <div className="card-field-row">
                    <span className="field-name">Environmental:</span>
                    <span className="field-val">{decodeHtmlEntities(report.allergies.environmental)}</span>
                  </div>
                )}
              </div>
            </section>

            {/* Lifestyle & Social History */}
            <section className="doc-card">
              <div className="doc-card-header">
                <h4>
                  <i className="fas fa-walking"></i> Lifestyle &amp; Habits
                </h4>
                <span className="card-source-tag">Social Hx</span>
              </div>
              <div className="doc-card-body">
                <div className="card-field-row">
                  <span className="field-name">Smoking:</span>
                  <span className="field-val">{decodeHtmlEntities(report?.lifestyle?.smoking) || 'Not asked'}</span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Alcohol:</span>
                  <span className="field-val">{decodeHtmlEntities(report?.lifestyle?.alcohol) || 'Not asked'}</span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Recent Travel:</span>
                  <span className="field-val">{decodeHtmlEntities(report?.lifestyle?.recentTravel) || 'Not asked'}</span>
                </div>
                <div className="card-field-row">
                  <span className="field-name">Sick Contacts:</span>
                  <span className="field-val">{decodeHtmlEntities(report?.lifestyle?.sickContacts) || 'Not asked'}</span>
                </div>
                {report?.lifestyle?.exercise && (
                  <div className="card-field-row">
                    <span className="field-name">Exercise:</span>
                    <span className="field-val">{decodeHtmlEntities(report.lifestyle.exercise)}</span>
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>

        {/* AI-Assisted Clinical Summary Box */}
        <section className="doc-section doc-clinical-summary-box">
          <div className="summary-box-header">
            <div className="summary-title-group">
              <i className="fas fa-file-medical text-primary"></i>
              <h4>Clinical Summary for Physician</h4>
            </div>
            <span className="ai-badge-tag">
              <i className="fas fa-robot"></i> AI Pre-Consultation Synthesis
            </span>
          </div>

          <div className="summary-narrative-content">
            <p className="summary-text-block">
              {decodeHtmlEntities(report?.clinicalSummary) ||
                'Patient pre-consultation information gathered through adaptive intake. Attending clinician to conduct physical evaluation, establish formal differential diagnosis, and recommend appropriate diagnostics.'}
            </p>
          </div>
        </section>

        {/* Clinical Recommendations & Suggested Next Steps */}
        <section className="doc-section doc-next-steps-section">
          <div className="next-steps-header">
            <h4>
              <i className="fas fa-clipboard-check"></i> Suggested Clinical Next Steps
            </h4>
            <span className="text-muted text-xs">For Attending Clinician Review</span>
          </div>
          <ul className="next-steps-list">
            {report?.recommendations && report.recommendations.length > 0 ? (
              report.recommendations.map((rec, idx) => (
                <li key={idx}>{decodeHtmlEntities(rec)}</li>
              ))
            ) : (
              <>
                <li>Conduct focused physical examination targeted to reported symptom location and character.</li>
                <li>Review and reconcile patient-reported medications and verified allergy tolerances.</li>
                <li>Evaluate pertinent negatives and assess need for preliminary lab or diagnostic imaging studies.</li>
              </>
            )}
          </ul>
        </section>

        {/* Formal Clinical Disclaimer & Footer */}
        <footer className="doc-legal-footer">
          <div className="legal-divider-line"></div>
          <div className="legal-disclaimer-content">
            <i className="fas fa-shield-alt legal-icon"></i>
            <p>
              <strong>Clinical Practice Notice:</strong> This pre-consultation report is compiled strictly from
              patient-reported symptoms and history to streamline outpatient clinical intake. It does not establish a formal
              physician-patient relationship, make definitive clinical diagnoses, or prescribe pharmacotherapy. All
              observations and proposed next steps must be clinically reviewed and validated by a licensed physician prior
              to clinical intervention. In acute medical emergencies, call 108 (Ambulance) or 112 (Emergency Helpline) or proceed to the nearest emergency facility.
            </p>
          </div>
          <div className="doc-footer-meta-line">
            <span>Clinical Consultation Report &middot; Confidential Health Information</span>
            <span>Document ID: {sessionId ? sessionId.slice(0, 16) : 'PRE-CONSULT'}</span>
            <span>Page 1 of 1</span>
          </div>
        </footer>
      </article>

      {/* =========================================================================
          PATIENT REVIEW & CORRECTION CONTROLS (PRESERVED WORKFLOW)
          ========================================================================= */}
      <section className={`patient-review-control-card ${isConfirmed ? 'is-confirmed-card' : ''}`}>
        {updateSuccess && (
          <div className="alert alert-success d-flex align-items-center mb-3" style={{ padding: '12px 16px', borderRadius: '8px', background: '#e8f5e9', border: '1px solid #c8e6c9', color: '#2e7d32', marginBottom: '16px' }} role="status">
            <i className="fas fa-check-circle" style={{ marginRight: '8px', fontSize: '1.2rem' }}></i>
            <span><strong>Corrections applied!</strong> The clinical report above has been updated with your feedback. Please review the updated report before confirming.</span>
          </div>
        )}
        {!isConfirmed ? (
          <>
            <div className="review-prompt-header">
              <div className="prompt-icon">
                <i className="fas fa-user-check"></i>
              </div>
              <div>
                <h5>Patient Information Review</h5>
                <p>Please review your pre-consultation report above. Is all information accurate and complete?</p>
              </div>
            </div>

            {!showFeedbackForm ? (
              <div className="review-action-row">
                <button
                  type="button"
                  className="btn-confirm-intake"
                  onClick={confirmReport}
                >
                  <i className="fas fa-check-circle"></i>
                  <span>✓ Yes, This Information is Correct</span>
                </button>

                <button
                  type="button"
                  className="btn-edit-intake"
                  onClick={() => setShowFeedbackForm(true)}
                >
                  <i className="fas fa-pen"></i>
                  <span>+ Add or Correct Information</span>
                </button>
              </div>
            ) : (
              <form onSubmit={handleFeedbackSubmit} className="patient-feedback-form">
                <label htmlFor="patient-feedback-textarea" className="feedback-form-label">
                  <strong>Please enter any corrections, missing medications, or additional symptom details:</strong>
                </label>
                <textarea
                  id="patient-feedback-textarea"
                  className="feedback-textarea"
                  rows={4}
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  placeholder="Example: I forgot to mention I also take Lisinopril 10mg daily, and the headache started after working out..."
                  disabled={isFeedbackSubmitting}
                />

                {feedbackError && (
                  <div className="feedback-error-alert" role="alert">
                    <i className="fas fa-exclamation-circle"></i> {feedbackError}
                  </div>
                )}

                <div className="feedback-form-actions">
                  <button
                    type="submit"
                    className="btn-submit-feedback"
                    disabled={isFeedbackSubmitting || !feedbackText.trim()}
                  >
                    {isFeedbackSubmitting ? (
                      <>
                        <i className="fas fa-spinner fa-spin"></i>
                        <span>Updating Report...</span>
                      </>
                    ) : (
                      <>
                        <i className="fas fa-paper-plane"></i>
                        <span>Submit Corrections</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="btn-cancel-feedback"
                    onClick={() => {
                      setShowFeedbackForm(false);
                      setFeedbackText('');
                    }}
                    disabled={isFeedbackSubmitting}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            )}
          </>
        ) : (
          <div className="patient-verified-banner">
            <i className="fas fa-shield-check banner-verified-icon"></i>
            <div>
              <h6>Report Verified by Patient</h6>
              <p>
                Confirmed on {confirmationTimestamp || dateDisplay}. This structured intake summary is ready for your healthcare provider's review.
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
