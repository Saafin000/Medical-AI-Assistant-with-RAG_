import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useClinicalPreConsultation } from '../../context/ClinicalPreConsultationContext';
import { useWorkspaceAutoFocus } from '../../hooks/useWorkspaceAutoFocus';
import { useVoiceRecorder } from '../../hooks/useVoiceRecorder';
import { focusElementSafely, resizeTextareaSafely } from '../../utils/focusUtils';

export default function ClinicalInputBar() {
  const {
    isLoading,
    isStreaming,
    stopGeneration,
    streamingStatus,
    sendMessage,
    sessionId
  } = useClinicalPreConsultation();

  const [inputText, setInputText] = useState('');
  const textareaRef = useRef(null);
  const voiceBlockedRef = useRef(false);

  const isBusy = isLoading || isStreaming;

  const handleTranscript = useCallback((transcript) => {
    setInputText((prev) => {
      const trimmed = prev.trim();
      return trimmed ? `${trimmed} ${transcript}` : transcript;
    });
  }, []);

  const handleFocusAfterVoice = useCallback(() => {
    if (textareaRef.current && !isBusy) {
      focusElementSafely(textareaRef.current, { allowProtectedButton: true });
    }
  }, [isBusy]);

  const {
    isRecording,
    isTranscribing,
    recordingSeconds,
    speechError,
    setSpeechError,
    isRecordingRef,
    isTranscribingRef,
    toggleMicrophone,
    stopRecordingAndTranscribe,
    cancelRecording,
    abortRecordingOnSubmit
  } = useVoiceRecorder({
    isBlocked: isBusy,
    onTranscript: handleTranscript,
    onFocusAfterVoice: handleFocusAfterVoice
  });

  useEffect(() => {
    voiceBlockedRef.current = isRecordingRef.current || isTranscribingRef.current;
  }, [isRecording, isTranscribing, isRecordingRef, isTranscribingRef]);

  useWorkspaceAutoFocus({
    inputRef: textareaRef,
    routeMatches: ['/clinical-pre-consultation', '/pre-consultation'],
    isBusy,
    isBlocked: isRecording || isTranscribing,
    sessionId,
    blockedRef: voiceBlockedRef
  });

  // Auto-resize textarea safely without collapsing height to 0px
  useEffect(() => {
    resizeTextareaSafely(textareaRef.current, 140, 28);
  }, [inputText]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || isBusy || isTranscribing) return;

    abortRecordingOnSubmit();
    sendMessage(inputText.trim());
    setInputText('');
    setSpeechError(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent?.isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTimer = (totalSec) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="clinical-input-container">
      {/* Speech Error Banner (Dismissible) */}
      {speechError && (
        <div className="speech-error-notice" role="alert">
          <div className="speech-error-content">
            <i className="fas fa-exclamation-circle text-warning"></i>
            <span>{speechError}</span>
          </div>
          <button
            type="button"
            className="btn-speech-error-close"
            onClick={() => setSpeechError(null)}
            title="Dismiss error"
            aria-label="Dismiss speech error"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
      )}

      {/* Voice Recording / Transcribing Indicator Banner */}
      {(isRecording || isTranscribing) && (
        <div className="speech-listening-banner" role="status" aria-live="polite">
          {isRecording ? (
            <>
              <span className="listening-pulse-dot" aria-hidden="true"></span>
              <span className="listening-text">
                Recording ({formatTimer(recordingSeconds)}) — Speak in English, Hindi, or Hinglish. Click mic or Stop when done.
              </span>
              <div className="recording-actions">
                <button
                  type="button"
                  className="btn-speech-stop-pill"
                  onClick={stopRecordingAndTranscribe}
                  title="Stop recording and convert speech to text"
                  aria-label="Stop recording and transcribe"
                >
                  <i className="fas fa-stop"></i> Stop &amp; Transcribe
                </button>
                <button
                  type="button"
                  className="btn-voice-cancel"
                  onClick={cancelRecording}
                  title="Cancel recording"
                  aria-label="Cancel recording"
                >
                  <i className="fas fa-times"></i> Cancel
                </button>
              </div>
            </>
          ) : (
            <>
              <i className="fas fa-circle-notch fa-spin" aria-hidden="true"></i>
              <span className="listening-text">Transcribing your voice response...</span>
            </>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} className={`clinical-form ${isRecording ? 'input-listening-active' : ''}`}>
        {/* Microphone Toggle Button (MediaRecorder -> /transcribe) */}
        <button
          type="button"
          className={`btn-mic-toggle ${isRecording ? 'mic-active-recording' : ''}`}
          onClick={toggleMicrophone}
          disabled={isBusy || isTranscribing}
          title={
            isTranscribing
              ? 'Transcribing recorded audio...'
              : isRecording
                ? 'Stop recording and transcribe'
                : 'Click to record voice (English, Hindi, or Hinglish)'
          }
          aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
        >
          <i className={`fas ${isTranscribing ? 'fa-spinner fa-spin' : isRecording ? 'fa-stop' : 'fa-microphone'}`}></i>
        </button>

        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isRecording
              ? 'Recording voice... Click Stop or microphone when finished to review and edit text.'
              : isTranscribing
                ? 'Converting speech to text...'
                : 'Describe your symptoms, duration, medical history, or answer follow-up questions... (Enter to send, Shift+Enter for line break)'
          }
          rows={1}
          className="clinical-textarea"
          disabled={isBusy}
          aria-label="Clinical information input"
        />

        {isStreaming ? (
          <button
            type="button"
            className="btn-chat-stop"
            onClick={stopGeneration}
            title="Stop intake response"
            aria-label="Stop intake response"
          >
            <i className="fas fa-stop"></i>
          </button>
        ) : (
          <button
            type="submit"
            className="btn-send-clinical"
            disabled={!inputText.trim() || isBusy || isTranscribing}
            title="Send response"
            aria-label="Send response"
          >
            {isBusy ? (
              <i className="fas fa-spinner fa-spin"></i>
            ) : (
              <i className="fas fa-paper-plane"></i>
            )}
          </button>
        )}
      </form>

      {streamingStatus && (
        <div className="streaming-status-pill">
          <i className="fas fa-circle-notch fa-spin"></i> {streamingStatus}
        </div>
      )}

      <div className="clinical-input-hint">
        <span>Press <strong>Enter</strong> to send &bull; <strong>Shift + Enter</strong> for line break</span>
      </div>
    </div>
  );
}
