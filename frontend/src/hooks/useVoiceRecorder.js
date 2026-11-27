import { useState, useRef, useEffect, useCallback } from 'react';

function pickSupportedAudioMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return null;
  }
  if (typeof MediaRecorder.isTypeSupported !== 'function') {
    return '';
  }
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4'
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return null;
}

function extensionForMimeType(mimeType) {
  const lower = (mimeType || '').toLowerCase();
  if (lower.includes('ogg')) return 'ogg';
  if (lower.includes('mp4') || lower.includes('m4a')) return 'mp4';
  if (lower.includes('wav')) return 'wav';
  if (lower.includes('mpeg') || lower.includes('mp3')) return 'mp3';
  return 'webm';
}

/**
 * Shared MediaRecorder -> POST /transcribe hook for voice input without auto-submit.
 */
export function useVoiceRecorder({ isBlocked = false, onTranscript, onFocusAfterVoice }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [speechError, setSpeechError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const transcribeAbortRef = useRef(null);
  const isCancelledRef = useRef(false);
  const isUnmountedRef = useRef(false);

  const isRecordingRef = useRef(false);
  const isTranscribingRef = useRef(false);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const releaseMediaTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {}
      });
      streamRef.current = null;
    }
  }, []);

  const cleanupRecordingResources = useCallback((abortTranscription = true) => {
    stopTimer();

    if (abortTranscription && transcribeAbortRef.current) {
      try {
        transcribeAbortRef.current.abort();
      } catch (_) {}
      transcribeAbortRef.current = null;
    }

    if (mediaRecorderRef.current) {
      const recorder = mediaRecorderRef.current;
      recorder.ondataavailable = null;
      recorder.onerror = null;
      recorder.onstop = null;
      if (recorder.state !== 'inactive') {
        try {
          recorder.stop();
        } catch (_) {}
      }
      mediaRecorderRef.current = null;
    }

    releaseMediaTracks();
    audioChunksRef.current = [];
    isRecordingRef.current = false;
    isTranscribingRef.current = false;

    if (!isUnmountedRef.current) {
      setIsRecording(false);
      setIsTranscribing(false);
      setRecordingSeconds(0);
    }
  }, [releaseMediaTracks, stopTimer]);

  useEffect(() => {
    isUnmountedRef.current = false;
    return () => {
      isUnmountedRef.current = true;
      isCancelledRef.current = true;
      cleanupRecordingResources(true);
    };
  }, [cleanupRecordingResources]);

  const submitAudioForTranscription = useCallback(async (chunks, recordedMimeType) => {
    if (isCancelledRef.current || isUnmountedRef.current) {
      return;
    }

    if (!chunks || chunks.length === 0) {
      isTranscribingRef.current = false;
      setIsTranscribing(false);
      setSpeechError('No audio was recorded. Please speak clearly while recording and try again.');
      return;
    }

    const mimeType = recordedMimeType || chunks[0]?.type || 'audio/webm';
    const audioBlob = new Blob(chunks, { type: mimeType });

    if (!audioBlob || audioBlob.size === 0) {
      isTranscribingRef.current = false;
      setIsTranscribing(false);
      setSpeechError('Recorded audio is empty. Please speak before stopping recording.');
      return;
    }

    isTranscribingRef.current = true;
    setIsTranscribing(true);
    setSpeechError(null);

    const abortController = new AbortController();
    transcribeAbortRef.current = abortController;

    try {
      const ext = extensionForMimeType(mimeType);
      const formData = new FormData();
      formData.append('audio', audioBlob, `voice_recording.${ext}`);

      const response = await fetch('/transcribe', {
        method: 'POST',
        body: formData,
        signal: abortController.signal
      });

      if (isCancelledRef.current || isUnmountedRef.current) {
        return;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || errorData.detail || `Transcription failed (HTTP ${response.status}).`);
      }

      const data = await response.json();
      const transcript = (data?.transcript || '').trim();
      if (!transcript) {
        throw new Error('No speech could be transcribed from the recording. Please try speaking again.');
      }

      if (onTranscript) {
        onTranscript(transcript);
      }
      setSpeechError(null);
      isTranscribingRef.current = false;
      setIsTranscribing(false);
      if (onFocusAfterVoice) {
        onFocusAfterVoice();
      }
    } catch (err) {
      if (err?.name === 'AbortError' || isCancelledRef.current || isUnmountedRef.current) {
        return;
      }
      console.error('STT error:', err);
      const isNetworkErr =
        err instanceof TypeError ||
        (typeof navigator !== 'undefined' && navigator.onLine === false);
      setSpeechError(
        isNetworkErr
          ? 'Network interruption while transcribing audio. Please check your connection and try again.'
          : (err.message || 'Unable to transcribe your recording. Please try again.')
      );
    } finally {
      if (transcribeAbortRef.current === abortController) {
        transcribeAbortRef.current = null;
      }
      isTranscribingRef.current = false;
      if (!isUnmountedRef.current) {
        setIsTranscribing(false);
      }
    }
  }, [onTranscript, onFocusAfterVoice]);

  const stopRecordingAndTranscribe = useCallback(() => {
    stopTimer();
    isCancelledRef.current = false;
    isRecordingRef.current = false;
    setIsRecording(false);

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      isTranscribingRef.current = true;
      setIsTranscribing(true);
      try {
        recorder.stop();
      } catch (err) {
        console.error('Failed to stop MediaRecorder:', err);
        cleanupRecordingResources(false);
        setSpeechError('Could not stop audio recording cleanly. Please try again.');
      }
    } else {
      releaseMediaTracks();
    }
  }, [cleanupRecordingResources, releaseMediaTracks, stopTimer]);

  const cancelRecording = useCallback(() => {
    isCancelledRef.current = true;
    cleanupRecordingResources(true);
    setSpeechError(null);
    if (onFocusAfterVoice) {
      onFocusAfterVoice();
    }
  }, [cleanupRecordingResources, onFocusAfterVoice]);

  const startRecording = useCallback(async () => {
    if (isBlocked || isTranscribingRef.current) return;
    setSpeechError(null);
    isCancelledRef.current = false;

    if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setSpeechError('Microphone recording is unavailable in this browser or context.');
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      setSpeechError('Voice recording is not supported in this browser.');
      return;
    }

    const supportedMime = pickSupportedAudioMimeType();
    if (supportedMime === null) {
      setSpeechError('No supported audio recording format is available in this browser.');
      return;
    }

    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (isUnmountedRef.current || isCancelledRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      const audioTracks = stream.getAudioTracks();
      if (!audioTracks || audioTracks.length === 0) {
        stream.getTracks().forEach((t) => t.stop());
        setSpeechError('No microphone input track was found.');
        return;
      }

      streamRef.current = stream;
      audioChunksRef.current = [];
      setRecordingSeconds(0);

      const recorderOptions = supportedMime ? { mimeType: supportedMime } : undefined;
      const recorder = recorderOptions ? new MediaRecorder(stream, recorderOptions) : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      audioTracks.forEach((track) => {
        track.onended = () => {
          if (isRecordingRef.current && recorder.state === 'recording') {
            stopRecordingAndTranscribe();
          }
        };
      });

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = (event) => {
        console.error('MediaRecorder runtime error:', event?.error || event);
        cleanupRecordingResources(false);
        setSpeechError('Audio recording encountered an error. Please try again.');
      };

      recorder.onstop = async () => {
        const capturedChunks = [...audioChunksRef.current];
        const recorderMime = recorder.mimeType || supportedMime || 'audio/webm';
        audioChunksRef.current = [];
        mediaRecorderRef.current = null;
        releaseMediaTracks();

        if (!isCancelledRef.current && !isUnmountedRef.current) {
          await submitAudioForTranscription(capturedChunks, recorderMime);
        }
      };

      recorder.start(250);
      isRecordingRef.current = true;
      setIsRecording(true);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      if (stream) {
        stream.getTracks().forEach((t) => {
          try { t.stop(); } catch (_) {}
        });
      }
      releaseMediaTracks();
      isRecordingRef.current = false;
      setIsRecording(false);

      console.error('Microphone start error:', err);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setSpeechError('Microphone permission was denied. Please allow microphone access in your browser settings.');
      } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        setSpeechError('No microphone device was detected. Please connect a microphone and try again.');
      } else if (err?.name === 'NotReadableError' || err?.name === 'TrackStartError') {
        setSpeechError('Microphone is busy or unavailable. Close other apps using it and try again.');
      } else if (err?.name === 'NotSupportedError') {
        setSpeechError('The audio recording format is not supported by this browser.');
      } else {
        setSpeechError(err?.message || 'Could not access microphone. Please check your device settings.');
      }
    }
  }, [cleanupRecordingResources, isBlocked, releaseMediaTracks, stopRecordingAndTranscribe, submitAudioForTranscription]);

  const toggleMicrophone = useCallback(() => {
    if (isRecordingRef.current) {
      stopRecordingAndTranscribe();
    } else if (!isTranscribingRef.current) {
      startRecording();
    }
  }, [startRecording, stopRecordingAndTranscribe]);

  const abortRecordingOnSubmit = useCallback(() => {
    if (isRecordingRef.current) {
      isCancelledRef.current = true;
      cleanupRecordingResources(true);
    }
  }, [cleanupRecordingResources]);

  return {
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
  };
}
