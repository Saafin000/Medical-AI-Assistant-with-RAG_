import React, { useRef, useState } from 'react';
import { useImageAnalysis } from '../../context/ImageAnalysisContext';
import { useWorkspaceAutoFocus } from '../../hooks/useWorkspaceAutoFocus';
import { renderMarkdown } from '../../utils/markdown';

export default function HumanValidationCard() {
  const {
    validationStatus,
    validationDecision,
    validationComments,
    validationResult,
    sessionId,
    submitValidation,
    playSpeech,
    activeAudioId,
    isAudioPlaying,
    isAudioLoading
  } = useImageAnalysis();

  const [selectedDecision, setSelectedDecision] = useState(null); // 'yes' | 'no'
  const [comments, setComments] = useState('');
  const [formError, setFormError] = useState(null);
  const commentsInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedDecision) {
      setFormError('Please select whether you agree or disagree with the automated finding.');
      return;
    }
    setFormError(null);
    submitValidation(selectedDecision, comments);
  };

  const isCompleted = validationStatus === 'completed';
  const isSubmitting = validationStatus === 'loading';

  useWorkspaceAutoFocus({
    inputRef: commentsInputRef,
    routeMatches: ['/image-analysis'],
    isBusy: isSubmitting,
    isBlocked: isCompleted,
    sessionId
  });

  return (
    <div className={`human-validation-card ${isCompleted ? 'validation-card-completed' : ''}`}>
      <div className="validation-header">
        <div className="validation-title-group">
          <span className="validation-icon-badge">
            <i className={`fas ${isCompleted ? 'fa-user-check' : 'fa-user-shield'}`}></i>
          </span>
          <div>
            <h4 className="validation-title">
              {isCompleted ? 'Clinician Validation Recorded' : 'Human-in-the-Loop Clinical Validation'}
            </h4>
            <p className="validation-subtitle">
              {isCompleted
                ? 'Your verification has been recorded in the session record.'
                : 'Medical safety guardrail: Please review the automated findings and record your clinical confirmation.'}
            </p>
          </div>
        </div>

        {isCompleted && (
          <span className={`validation-status-pill pill-${validationDecision === 'yes' ? 'confirmed' : 'flagged'}`}>
            <i className={`fas ${validationDecision === 'yes' ? 'fa-check-circle' : 'fa-flag'}`}></i>
            {validationDecision === 'yes' ? 'Confirmed by Clinician' : 'Flagged for Review'}
          </span>
        )}
      </div>

      {!isCompleted ? (
        <form onSubmit={handleSubmit} className="validation-form">
          <div className="validation-prompt">
            <strong>Clinical Question:</strong> Do you agree with this automated diagnostic finding?
          </div>

          {/* Decision Buttons */}
          <div className="validation-decision-buttons">
            <button
              type="button"
              className={`btn-choice btn-choice-agree ${selectedDecision === 'yes' ? 'active' : ''}`}
              onClick={() => {
                setSelectedDecision('yes');
                setFormError(null);
              }}
              disabled={isSubmitting}
            >
              <i className="fas fa-check"></i>
              <span>✔ Yes, Agree with Result</span>
            </button>

            <button
              type="button"
              className={`btn-choice btn-choice-disagree ${selectedDecision === 'no' ? 'active' : ''}`}
              onClick={() => {
                setSelectedDecision('no');
                setFormError(null);
              }}
              disabled={isSubmitting}
            >
              <i className="fas fa-times"></i>
              <span>✖ No, Disagree / Flag</span>
            </button>
          </div>

          {/* Optional Comments */}
          <div className="validation-comments-group">
            <label htmlFor="validation-comments-input" className="validation-comments-label">
              Clinical Observations or Review Comments <span className="text-muted">(Optional)</span>:
            </label>
            <textarea
              ref={commentsInputRef}
              id="validation-comments-input"
              className="validation-comments-textarea"
              rows={2}
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Add any specific clinical notes, differential diagnosis, or notes for record..."
              disabled={isSubmitting}
            />
          </div>

          {formError && (
            <div className="validation-error-text">
              <i className="fas fa-info-circle"></i> {formError}
            </div>
          )}

          <div className="validation-actions">
            <button
              type="submit"
              className="btn-submit-validation"
              disabled={isSubmitting || !selectedDecision}
            >
              {isSubmitting ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i>
                  <span>Submitting Clinical Record...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-paper-plane"></i>
                  <span>Submit Validation</span>
                </>
              )}
            </button>
          </div>
        </form>
      ) : (
        /* Completed Validation Results Display */
        <div className="validation-results-display">
          {validationResult?.message && (
            <div className="validation-message-alert">
              <i className="fas fa-clipboard-check"></i>
              <span>{validationResult.message}</span>
            </div>
          )}

          {validationResult?.response && (
            <div className="validation-backend-response">
              <div
                className="markdown-content"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(validationResult.response) }}
              />

              {/* TTS Audio Controls for validation response */}
              <div className="audio-controls mt-2">
                <button
                  type="button"
                  className={`btn-tts-play ${isAudioPlaying && activeAudioId === 'validation-response' ? 'btn-tts-playing' : ''}`}
                  onClick={() => playSpeech('validation-response', validationResult.response)}
                  disabled={isAudioLoading}
                  aria-label="Play validation voice response"
                >
                  {isAudioLoading && activeAudioId === 'validation-response' ? (
                    <>
                      <i className="fas fa-spinner fa-spin"></i>
                      <span>Generating Audio...</span>
                    </>
                  ) : isAudioPlaying && activeAudioId === 'validation-response' ? (
                    <>
                      <i className="fas fa-pause"></i>
                      <span>Pause Speech</span>
                    </>
                  ) : (
                    <>
                      <i className="fas fa-play"></i>
                      <span>Play Voice Response</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {validationComments && (
            <div className="validation-recorded-comments">
              <strong>Recorded Clinician Notes:</strong> {validationComments}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
