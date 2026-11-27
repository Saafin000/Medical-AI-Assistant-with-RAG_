import React, { useState, useEffect, useRef } from 'react';

export default function VoiceInputBar({ onTranscribeComplete, onCancel }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  // Start recording on mount
  useEffect(() => {
    startRecording();
    return () => {
      cleanupResources();
    };
  }, []);

  const cleanupResources = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        // Ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const startRecording = async () => {
    setErrorMessage(null);
    audioChunksRef.current = [];
    setSeconds(0);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Audio recording is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Select supported MIME type
      let options = {};
      if (typeof MediaRecorder.isTypeSupported === 'function') {
        if (MediaRecorder.isTypeSupported('audio/webm')) {
          options = { mimeType: 'audio/webm' };
        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
          options = { mimeType: 'audio/ogg' };
        }
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Trigger transcription when stopped by user
        if (audioChunksRef.current.length > 0) {
          await submitAudioForTranscription();
        }
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Start elapsed timer
      timerRef.current = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone error:', err);
      let msg = 'Failed to access microphone.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Microphone permission was denied. Please allow microphone access in your browser.';
      } else if (err.name === 'NotFoundError') {
        msg = 'No microphone device was found.';
      }
      setErrorMessage(msg);
      setIsRecording(false);
    }
  };

  const handleStop = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    setIsTranscribing(true);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const handleCancel = () => {
    cleanupResources();
    onCancel();
  };

  const submitAudioForTranscription = async () => {
    try {
      const audioBlob = new Blob(audioChunksRef.current, {
        type: mediaRecorderRef.current?.mimeType || 'audio/webm'
      });

      if (audioBlob.size === 0) {
        throw new Error('Recorded audio file is empty.');
      }

      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice_recording.webm');

      const response = await fetch('/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Transcription failed with HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.transcript) {
        onTranscribeComplete(data.transcript);
      } else {
        throw new Error('No transcription returned by voice service.');
      }
    } catch (err) {
      console.error('Transcription error:', err);
      setErrorMessage(`Transcription failed: ${err.message}`);
      setIsTranscribing(false);
    }
  };

  const formatTime = (totalSec) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="voice-recorder-bar">
      {errorMessage ? (
        <div className="voice-error-container">
          <div className="voice-error-text">
            <i className="fas fa-exclamation-triangle"></i> {errorMessage}
          </div>
          <button type="button" className="btn-secondary-sm" onClick={handleCancel}>
            Close
          </button>
        </div>
      ) : isTranscribing ? (
        <div className="voice-transcribing-container">
          <i className="fas fa-circle-notch fa-spin"></i>
          <span>Transcribing voice with ElevenLabs Speech-to-Text...</span>
        </div>
      ) : (
        <div className="voice-recording-container">
          <div className="recording-status">
            <span className="recording-pulse-dot"></span>
            <span className="recording-label">Recording Voice</span>
            <span className="recording-timer">{formatTime(seconds)}</span>
          </div>

          <div className="recording-actions">
            <button
              type="button"
              className="btn-voice-stop"
              onClick={handleStop}
              title="Stop and transcribe"
            >
              <i className="fas fa-check"></i> Stop & Transcribe
            </button>
            <button
              type="button"
              className="btn-voice-cancel"
              onClick={handleCancel}
              title="Cancel recording"
            >
              <i className="fas fa-times"></i> Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
