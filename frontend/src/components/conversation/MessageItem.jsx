import React from 'react';
import AgentBadge from './AgentBadge';
import { renderMarkdown } from '../../utils/markdown';
import { useConversation } from '../../context/ConversationContext';

export default function MessageItem({ message }) {
  const {
    playSpeech,
    stopSpeech,
    activeAudioMessageId,
    isAudioPlaying,
    isAudioLoading,
    audioError,
    retryLast,
    isLoading
  } = useConversation();

  const isUser = message.sender === 'user';
  const isError = message.status === 'error';

  const isCurrentAudio = activeAudioMessageId === message.id;
  const isCurrentlyPlaying = isCurrentAudio && isAudioPlaying;
  const isCurrentlyAudioLoading = isCurrentAudio && isAudioLoading;

  if (isUser) {
    return (
      <div className="message-wrapper message-wrapper-user">
        <div className="message-bubble message-bubble-user">
          <div className="message-header">
            <span className="user-label">
              <i className="fas fa-user"></i> You
            </span>
            <span className="message-timestamp">{message.timestamp}</span>
          </div>
          <div className="message-text">{message.text}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`message-wrapper message-wrapper-assistant ${isError ? 'message-wrapper-error' : ''}`}>
      <div className={`message-bubble message-bubble-assistant ${isError ? 'message-bubble-error' : ''}`}>
        <div className="message-header">
          <AgentBadge agent={message.agent} />
          <span className="message-timestamp">{message.timestamp}</span>
        </div>

        {message.status === 'streaming' && !message.text ? (
          <div className="streaming-initial-placeholder">
            <i className="fas fa-circle-notch fa-spin"></i> Generating response...
          </div>
        ) : (
          <div
            className="message-content markdown-rendered"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) }}
          />
        )}

        {message.status === 'streaming' && message.text && (
          <span className="streaming-cursor-pulse" aria-hidden="true">▍</span>
        )}

        {message.resultImage && (
          <div className="message-image-container">
            <img src={message.resultImage} alt="Analysis Result" className="result-img" />
          </div>
        )}

        <div className="message-footer">
          {!isError && message.status !== 'streaming' && message.text && (
            <div className="tts-action-container">
              <button
                type="button"
                className={`tts-button ${isCurrentlyPlaying ? 'playing' : ''} ${isCurrentlyAudioLoading ? 'loading' : ''}`}
                onClick={() => playSpeech(message.id, message.text)}
                title={isCurrentlyPlaying ? 'Pause audio' : 'Listen to response'}
              >
                {isCurrentlyAudioLoading ? (
                  <>
                    <i className="fas fa-spinner fa-spin"></i> Generating voice...
                  </>
                ) : isCurrentlyPlaying ? (
                  <>
                    <i className="fas fa-pause"></i> Pause Audio
                  </>
                ) : (
                  <>
                    <i className="fas fa-volume-high"></i> Read Aloud
                  </>
                )}
              </button>
              {isCurrentAudio && (isCurrentlyPlaying || isCurrentlyAudioLoading) && (
                <button
                  type="button"
                  className="tts-button"
                  onClick={stopSpeech}
                  title="Stop audio playback"
                  aria-label="Stop audio playback"
                >
                  <i className="fas fa-stop"></i> Stop
                </button>
              )}
              {audioError && audioError.id === message.id && (
                <span className="tts-inline-error" title={audioError.error}>
                  <i className="fas fa-exclamation-circle"></i> {audioError.error}
                </span>
              )}
            </div>
          )}

          {message.canRetry && (
            <button
              type="button"
              className="retry-button"
              onClick={retryLast}
              disabled={isLoading}
              title="Retry last query"
            >
              <i className="fas fa-redo"></i> Retry Query
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
