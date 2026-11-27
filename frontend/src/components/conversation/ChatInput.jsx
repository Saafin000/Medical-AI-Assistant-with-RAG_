import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useConversation } from '../../context/ConversationContext';
import { useWorkspaceAutoFocus } from '../../hooks/useWorkspaceAutoFocus';
import { useVoiceRecorder } from '../../hooks/useVoiceRecorder';
import { focusElementSafely, resizeTextareaSafely } from '../../utils/focusUtils';

export default function ChatInput() {
  const { sendMessage, isLoading, isStreaming, stopGeneration, streamingStatus, sessionId } = useConversation();
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
    routeMatches: ['/conversation'],
    isBusy,
    isBlocked: isRecording || isTranscribing,
    sessionId,
    blockedRef: voiceBlockedRef
  });

  // Auto-resize textarea height safely without collapsing height to 0px
  useEffect(() => {
    resizeTextareaSafely(textareaRef.current, 160, 38);
  }, [inputText]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isBusy || isTranscribing) return;

    abortRecordingOnSubmit();
    sendMessage(trimmed);
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
    <div className="chat-input-wrapper">
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

      {/* Voice Recording / Transcribing Active Indicator */}
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
                  title="Stop recording and transcribe"
                >
                  <i className="fas fa-stop"></i> Stop &amp; Transcribe
                </button>
                <button
                  type="button"
                  className="btn-voice-cancel"
                  onClick={cancelRecording}
                  title="Cancel recording"
                >
                  <i className="fas fa-times"></i> Cancel
                </button>
              </div>
            </>
          ) : (
            <>
              <i className="fas fa-circle-notch fa-spin" aria-hidden="true"></i>
              <span className="listening-text">Transcribing your voice query...</span>
            </>
          )}
        </div>
      )}

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className={`chat-input-container ${isRecording ? 'input-listening-active' : ''}`}>
          {/* Microphone Button (MediaRecorder -> /transcribe) */}
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
                  : 'Click to speak (English, Hindi, or Hinglish)'
            }
            aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
          >
            <i className={`fas ${isTranscribing ? 'fa-spinner fa-spin' : isRecording ? 'fa-stop' : 'fa-microphone'}`}></i>
          </button>

          {/* Message Input Textarea */}
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={
              isRecording
                ? 'Recording voice... Click Stop or microphone when finished to review and edit text.'
                : isTranscribing
                  ? 'Converting speech to text...'
                  : 'Ask a medical question, inquire about symptoms, or consult clinical literature... (Enter to send, Shift+Enter for new line)'
            }
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isBusy}
            rows={1}
            aria-label="Medical query input"
          />

          {/* Send or Stop Message Button */}
          {isStreaming ? (
            <button
              type="button"
              className="btn-chat-stop"
              onClick={stopGeneration}
              title="Stop generating response"
              aria-label="Stop generating response"
            >
              <i className="fas fa-stop"></i>
            </button>
          ) : (
            <button
              type="submit"
              className="btn-chat-send"
              disabled={!inputText.trim() || isBusy || isTranscribing}
              title="Send message"
              aria-label="Send message"
            >
              {isBusy ? (
                <i className="fas fa-spinner fa-spin"></i>
              ) : (
                <i className="fas fa-paper-plane"></i>
              )}
            </button>
          )}
        </div>

        {streamingStatus && (
          <div className="streaming-status-pill">
            <i className="fas fa-circle-notch fa-spin"></i> {streamingStatus}
          </div>
        )}

        <div className="chat-input-hint">
          <span>Press <strong>Enter</strong> to send &bull; <strong>Shift + Enter</strong> for line break</span>
        </div>
      </form>
    </div>
  );
}
