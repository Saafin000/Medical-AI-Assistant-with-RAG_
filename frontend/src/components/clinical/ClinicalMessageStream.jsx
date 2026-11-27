import React, { useEffect, useRef } from 'react';
import { useClinicalPreConsultation } from '../../context/ClinicalPreConsultationContext';
import { renderMarkdown } from '../../utils/markdown';
import AgentBadge from '../conversation/AgentBadge';
import ClinicalReportViewer from './ClinicalReportViewer';

const STARTER_PROMPTS = [
  {
    title: 'Respiratory Symptoms',
    icon: 'fa-lungs',
    text: 'I would like to start a pre-consultation. I have had a persistent dry cough and mild fever for 3 days.'
  },
  {
    title: 'Musculoskeletal Pain',
    icon: 'fa-bone',
    text: 'Starting intake: Experiencing sharp lower back pain radiating down my right leg for the past week.'
  },
  {
    title: 'Neurological / Headache',
    icon: 'fa-brain',
    text: 'I want to begin pre-consultation for recurring throbbing morning headaches with sensitivity to light.'
  },
  {
    title: 'General Consultation Intake',
    icon: 'fa-stethoscope',
    text: 'I want to start a pre-consultation intake assessment before seeing my doctor.'
  }
];

export default function ClinicalMessageStream() {
  const {
    messages,
    isLoading,
    sendMessage,
    playSpeech,
    stopSpeech,
    activeAudioId,
    isAudioPlaying,
    isAudioPaused,
    isAudioLoading,
    audioError,
    clearAudioError
  } = useClinicalPreConsultation();

  const streamEndRef = useRef(null);

  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // If there are no messages, display a helpful structured empty state with starter scenarios
  if (messages.length === 0) {
    return (
      <div className="clinical-empty-state">
        <div className="empty-state-header">
          <div className="empty-state-badge">
            <i className="fas fa-clipboard-user"></i>
          </div>
          <h3>Outpatient Clinical Intake</h3>
          <p>
            Welcome to the AI-assisted pre-consultation service. This 5-stage intake collects your symptoms, history, and context using adaptive questioning, monitors for red-flag symptoms, and produces a doctor-ready summary.
          </p>
        </div>

        <div className="starter-prompts-wrapper">
          <p className="starter-prompts-label">
            <i className="fas fa-lightbulb text-primary"></i> Select a clinical scenario to start immediately, or type your complaint below:
          </p>
          <div className="clinical-starters-grid">
            {STARTER_PROMPTS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                className="starter-chip-card"
                onClick={() => sendMessage(p.text)}
                disabled={isLoading}
              >
                <div className="starter-card-title">
                  <i className={`fas ${p.icon} text-primary`}></i> {p.title}
                </div>
                <div className="starter-card-snippet">{p.text}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="clinical-message-stream" role="log" aria-live="polite">
      {messages.map((msg) => {
        const isUser = msg.sender === 'user';

        if (msg.isReport) {
          return (
            <div key={msg.id} className="clinical-stream-item item-report">
              <div className="item-agent-tag">
                <AgentBadge agent={msg.agent || 'CLINICAL_PRE_CONSULTATION_AGENT'} />
                <span className="item-time">{msg.timestamp}</span>
              </div>
              <ClinicalReportViewer reportText={msg.text} />
            </div>
          );
        }

        return (
          <div
            key={msg.id}
            className={`clinical-message-wrapper ${isUser ? 'wrapper-user' : 'wrapper-assistant'}`}
          >
            {!isUser && (
              <div className="message-meta-header">
                <AgentBadge agent={msg.agent || 'CLINICAL_PRE_CONSULTATION_AGENT'} />
                <span className="item-time">{msg.timestamp}</span>
              </div>
            )}

            <div
              className={`clinical-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'} ${
                msg.isEmergency ? 'bubble-emergency' : ''
              }`}
            >
              {msg.isEmergency && (
                <div className="emergency-callout-header">
                  <i className="fas fa-exclamation-triangle"></i>
                  <span>URGENT MEDICAL NOTICE</span>
                </div>
              )}

              {msg.status === 'streaming' && !msg.text ? (
                <div className="streaming-initial-placeholder">
                  <i className="fas fa-circle-notch fa-spin"></i> Formulating clinical intake response...
                </div>
              ) : (
                <div
                  className="markdown-content"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.text) }}
                />
              )}

              {msg.status === 'streaming' && msg.text && (
                <span className="streaming-cursor-pulse" aria-hidden="true">▍</span>
              )}

              {/* TTS Controls for Assistant Messages */}
              {!isUser && msg.status !== 'streaming' && msg.text && (
                <>
                  <div className="message-audio-controls" style={{ gap: '0.35rem', alignItems: 'center' }}>
                    <button
                      type="button"
                      className={`btn-tts-mini ${
                        isAudioPlaying && activeAudioId === msg.id ? 'btn-tts-mini-playing' : ''
                      }`}
                      onClick={() => playSpeech(msg.id, msg.text)}
                      title={
                        isAudioLoading && activeAudioId === msg.id
                          ? 'Generating audio...'
                          : isAudioPlaying && activeAudioId === msg.id
                            ? 'Pause audio'
                            : isAudioPaused && activeAudioId === msg.id
                              ? 'Resume audio'
                              : 'Read aloud'
                      }
                      aria-label={
                        isAudioPlaying && activeAudioId === msg.id
                          ? 'Pause question audio'
                          : isAudioPaused && activeAudioId === msg.id
                            ? 'Resume question audio'
                            : 'Play question audio'
                      }
                    >
                      {isAudioLoading && activeAudioId === msg.id ? (
                        <i className="fas fa-spinner fa-spin"></i>
                      ) : isAudioPlaying && activeAudioId === msg.id ? (
                        <i className="fas fa-pause"></i>
                      ) : (
                        <i className="fas fa-volume-up"></i>
                      )}
                    </button>

                    {activeAudioId === msg.id && (isAudioLoading || isAudioPlaying || isAudioPaused) && (
                      <button
                        type="button"
                        className="btn-tts-mini"
                        onClick={stopSpeech}
                        title="Stop audio playback"
                        aria-label="Stop question audio"
                      >
                        <i className="fas fa-stop"></i>
                      </button>
                    )}
                  </div>

                  {audioError && audioError.id === msg.id && (
                    <div className="speech-error-notice" role="alert" style={{ marginTop: '0.45rem', marginBottom: 0 }}>
                      <div className="speech-error-content">
                        <i className="fas fa-exclamation-circle"></i>
                        <span>{audioError.message}</span>
                      </div>
                      <button
                        type="button"
                        className="btn-speech-error-close"
                        onClick={clearAudioError}
                        aria-label="Dismiss audio error"
                      >
                        <i className="fas fa-times"></i>
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>

            {isUser && <span className="user-time-stamp">{msg.timestamp}</span>}
          </div>
        );
      })}

      {/* Thinking / Question Generation Indicator */}
      {isLoading && (
        <div className="clinical-thinking-indicator">
          <div className="pulse-dot"></div>
          <div className="pulse-dot"></div>
          <div className="pulse-dot"></div>
          <span>Analyzing symptoms and formulating adaptive follow-up...</span>
        </div>
      )}

      <div ref={streamEndRef} />
    </div>
  );
}
