import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import { cleanTextForSpeech, DEFAULT_VOICE_ID } from '../utils/speech';
import { streamChat } from '../utils/streamClient';

const ClinicalPreConsultationContext = createContext(null);

export function ClinicalPreConsultationProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState(null);
  const [error, setError] = useState(null);

  // Workflow tracking
  const [intakeStatus, setIntakeStatus] = useState('idle'); // 'idle' | 'active' | 'completed' | 'partial' | 'emergency'
  const [isEmergency, setIsEmergency] = useState(false);
  const [latestReport, setLatestReport] = useState(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [confirmationTimestamp, setConfirmationTimestamp] = useState(null);
  const [isFeedbackSubmitting, setIsFeedbackSubmitting] = useState(false);
  const [feedbackError, setFeedbackError] = useState(null);

  // Phase 6 Session Management State
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('active_session_clinical_pre_consultation') || null;
  });
  const [sessions, setSessions] = useState([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);

  // Stale async response tracking (Test E)
  // Ensures delayed async responses from previous intakes cannot overwrite current intake
  const activeSessionIdRef = useRef(sessionId);
  useEffect(() => {
    activeSessionIdRef.current = sessionId;
  }, [sessionId]);

  // Audio / TTS state
  const [activeAudioId, setActiveAudioId] = useState(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isAudioPaused, setIsAudioPaused] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null); // { id, message } | null

  // Audio refs
  const currentAudioRef = useRef(null);
  const currentAudioUrlRef = useRef(null);
  const audioAbortControllerRef = useRef(null);
  const audioRequestIdRef = useRef(0);

  const clearAudioError = useCallback(() => {
    setAudioError(null);
  }, []);

  /**
   * Stop currently playing or loading audio, abort in-flight TTS requests, and release resources
   */
  const stopSpeech = useCallback(() => {
    audioRequestIdRef.current += 1;

    if (audioAbortControllerRef.current) {
      try {
        audioAbortControllerRef.current.abort();
      } catch (_) {}
      audioAbortControllerRef.current = null;
    }

    if (currentAudioRef.current) {
      const audio = currentAudioRef.current;
      audio.onplay = null;
      audio.onpause = null;
      audio.onended = null;
      audio.onerror = null;
      try {
        audio.pause();
        audio.removeAttribute('src');
        audio.load();
      } catch (_) {}
      currentAudioRef.current = null;
    }

    if (currentAudioUrlRef.current) {
      URL.revokeObjectURL(currentAudioUrlRef.current);
      currentAudioUrlRef.current = null;
    }

    setActiveAudioId(null);
    setIsAudioPlaying(false);
    setIsAudioPaused(false);
    setIsAudioLoading(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSpeech();
    };
  }, [stopSpeech]);

  /**
   * Reset local in-memory intake view
   */
  const clearLocalIntakeView = useCallback(() => {
    stopSpeech();
    setMessages([]);
    setIsLoading(false);
    setError(null);
    setIntakeStatus('idle');
    setIsEmergency(false);
    setLatestReport(null);
    setIsConfirmed(false);
    setConfirmationTimestamp(null);
    setFeedbackError(null);
  }, [stopSpeech]);

  /**
   * Load session list for clinical_pre_consultation workspace
   */
  const loadSessions = useCallback(async () => {
    try {
      setIsLoadingSessions(true);
      const res = await fetch('/sessions?workspace_type=clinical_pre_consultation&include_archived=false', {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const raw = data.sessions || [];
        // Strict prefix isolation
        const filtered = raw.filter((s) => s.session_id && s.session_id.startsWith('clin_'));
        setSessions(filtered);
      }
    } catch (err) {
      console.error('Failed to load clinical sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  /**
   * Load state and messages for a specific clinical session
   */
  const loadSessionData = useCallback(async (sid) => {
    if (!sid) {
      clearLocalIntakeView();
      return;
    }

    // Set active session ref immediately
    activeSessionIdRef.current = sid;

    // Clear previous view state immediately to prevent stale report flash
    clearLocalIntakeView();

    try {
      setIsLoading(true);
      const [stateRes, msgRes] = await Promise.all([
        fetch(`/sessions/${encodeURIComponent(sid)}/clinical-state`, { credentials: 'include' }),
        fetch(`/sessions/${encodeURIComponent(sid)}/messages`, { credentials: 'include' })
      ]);

      // Guard: if user switched sessions while loading, abort
      if (activeSessionIdRef.current !== sid) {
        return;
      }

      let loadedReport = null;
      let loadedStage = 'idle';
      let loadedConfirmed = false;

      if (stateRes.ok) {
        const sData = await stateRes.json();
        const cs = sData.clinical_state;
        if (cs) {
          loadedStage = cs.current_stage || 'active';
          let rep = cs.latest_report_text || cs.formatted_report || null;
          if (typeof rep === 'string' && rep.trim().startsWith('{')) {
            try {
              const parsed = JSON.parse(rep);
              rep = parsed.formatted_report || parsed.report || parsed.latest_report_text || rep;
            } catch (e) {
              // Not JSON
            }
          }
          loadedReport = rep;
          loadedConfirmed = Boolean(cs.is_confirmed);
        }
      }

      if (msgRes.ok) {
        const mData = await msgRes.json();
        const rawMsgs = mData.messages || [];
        const mappedAll = rawMsgs
          .filter((m) => !(m.metadata && m.metadata.type === 'clinical_feedback'))
          .map((m) => ({
            id: m.message_id || `clinical-msg-${m.sequence_num}`,
            sender: m.sender,
            text: m.content,
            agent: m.agent_name || (m.sender === 'user' ? 'Patient' : 'Clinical Pre-Consultation Agent'),
            timestamp: m.created_at
              ? new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : '',
            status: 'complete',
            isReport: Boolean(m.is_report),
            isEmergency: Boolean(m.is_emergency)
          }));

        // Keep only the latest report message if multiple exist from prior corrections
        let lastReportIdx = -1;
        mappedAll.forEach((m, idx) => {
          if (m.isReport) lastReportIdx = idx;
        });
        const mapped = mappedAll.filter((m, idx) => !m.isReport || idx === lastReportIdx);

        if (activeSessionIdRef.current === sid) {
          setMessages(mapped);
          // If clinical state had no report, check message stream
          if (!loadedReport) {
            const reportMsg = mapped.find((m) => m.isReport);
            if (reportMsg) {
              loadedReport = reportMsg.text;
              loadedStage = 'completed';
            }
          }
        }
      }

      if (activeSessionIdRef.current === sid) {
        setLatestReport(loadedReport);
        setIntakeStatus(loadedStage);
        setIsConfirmed(loadedConfirmed);
        const hasEmergency = loadedStage === 'emergency' || (loadedReport && loadedReport.toLowerCase().includes('emergency'));
        setIsEmergency(Boolean(hasEmergency));
      }
    } catch (err) {
      if (activeSessionIdRef.current === sid) {
        console.error(`Failed to load clinical session ${sid}:`, err);
      }
    } finally {
      if (activeSessionIdRef.current === sid) {
        setIsLoading(false);
      }
    }
  }, [clearLocalIntakeView]);

  /**
   * Create a genuinely new clinical pre-consultation session (clin_<uuid>)
   * Resets local state immediately and registers session on backend
   */
  const newSession = useCallback(async () => {
    // 1. Immediately reset all clinical view state
    clearLocalIntakeView();

    // 2. Clear old session ID so no pending interactions reuse previous session
    setSessionId(null);
    activeSessionIdRef.current = null;
    localStorage.removeItem('active_session_clinical_pre_consultation');

    // 3. Register fresh clinical session on backend
    try {
      const res = await fetch('/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_type: 'clinical_pre_consultation' }),
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const newId = data.session_id || data.session?.session_id;
        if (newId) {
          setSessionId(newId);
          activeSessionIdRef.current = newId;
          localStorage.setItem('active_session_clinical_pre_consultation', newId);
          await loadSessions();
          return newId;
        }
      }
    } catch (err) {
      console.error('Failed to create new clinical session on backend:', err);
    }

    // 4. Client-side fallback if server creation fails
    const fallbackId = `clin_${crypto.randomUUID()}`;
    setSessionId(fallbackId);
    activeSessionIdRef.current = fallbackId;
    localStorage.setItem('active_session_clinical_pre_consultation', fallbackId);
    return fallbackId;
  }, [clearLocalIntakeView, loadSessions]);

  /**
   * Reset intake - delegates to newSession to guarantee complete session isolation
   */
  const resetIntake = useCallback(async () => {
    return await newSession();
  }, [newSession]);

  // On mount: load sessions and restore active session if present
  useEffect(() => {
    loadSessions();
    const storedId = localStorage.getItem('active_session_clinical_pre_consultation');
    if (storedId && storedId.startsWith('clin_')) {
      setSessionId(storedId);
      activeSessionIdRef.current = storedId;
      loadSessionData(storedId);
    } else if (storedId) {
      localStorage.removeItem('active_session_clinical_pre_consultation');
    }
  }, [loadSessions, loadSessionData]);

  /**
   * Switch active session
   */
  const selectSession = useCallback((sid) => {
    stopSpeech();
    setSessionId(sid);
    activeSessionIdRef.current = sid;
    localStorage.setItem('active_session_clinical_pre_consultation', sid);
    loadSessionData(sid);
  }, [stopSpeech, loadSessionData]);

  /**
   * Delete session and reset if active
   */
  const deleteSession = useCallback(async (sid) => {
    try {
      const res = await fetch(`/sessions/${encodeURIComponent(sid)}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      if (res.ok) {
        if (sessionId === sid) {
          localStorage.removeItem('active_session_clinical_pre_consultation');
          await newSession();
        } else {
          await loadSessions();
        }
      }
    } catch (err) {
      console.error(`Failed to delete session ${sid}:`, err);
    }
  }, [sessionId, newSession, loadSessions]);

  /**
   * Helper: check if a text is an actual formatted clinical report (never match conversational questions like final confirmation)
   */
  const isClinicalReportText = (text, agent) => {
    if (!text) return false;
    const isClinicalAgent = (agent || '').toUpperCase().includes('CLINICAL');
    if (!isClinicalAgent) return false;
    const upper = text.toUpperCase();
    if (
      upper.includes('PRE-CONSULTATION INTAKE REPORT') ||
      upper.includes('CLINICAL PRE-CONSULTATION INTAKE REPORT') ||
      upper.includes('PATIENT INTAKE SUMMARY REPORT')
    ) {
      return true;
    }
    const lower = text.toLowerCase();
    // Never classify the mandatory final confirmation question or short follow-up questions as a report
    if (
      lower.includes('before i prepare your pre-consultation report') ||
      lower.includes('is there anything else about your symptoms')
    ) {
      return false;
    }
    return (
      lower.includes('chief complaint') &&
      lower.includes('symptoms') &&
      (lower.includes('medical history') || lower.includes('vital signs') || lower.includes('clinical summary'))
    );
  };

  const abortControllerRef = useRef(null);

  /**
   * Stop an active clinical streaming request
   */
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setIsLoading(false);
    setStreamingStatus(null);
  }, []);

  /**
   * Send a patient message with progressive token streaming over /chat/stream
   * Strict session isolation & stale response discard
   */
  const sendMessage = useCallback(async (queryText) => {
    const trimmed = (queryText || '').trim();
    if (!trimmed || isLoading || isStreaming) return;

    setError(null);
    setFeedbackError(null);

    // CRITICAL: Ensure an active clinical session exists before sending
    let targetSessionId = sessionId || activeSessionIdRef.current;
    if (!targetSessionId || !targetSessionId.startsWith('clin_')) {
      targetSessionId = await newSession();
    }

    const userMessageId = `clinical-user-${Date.now()}`;
    const userMessage = {
      id: userMessageId,
      sender: 'user',
      text: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'complete'
    };

    const assistantMessageId = `clinical-assistant-${Date.now() + 1}`;
    const initialAssistantMessage = {
      id: assistantMessageId,
      sender: 'assistant',
      text: '',
      agent: 'CLINICAL_PRE_CONSULTATION_AGENT',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'streaming',
      isReport: false,
      isEmergency: false
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingStatus(null);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      await streamChat({
        query: trimmed,
        workspace: 'clinical_pre_consultation',
        sessionId: targetSessionId,
        signal: abortController.signal,
        onStart: (data) => {
          if (activeSessionIdRef.current !== targetSessionId) return;
          if (data.session_id && data.session_id !== sessionId) {
            setSessionId(data.session_id);
            activeSessionIdRef.current = data.session_id;
            localStorage.setItem('active_session_clinical_pre_consultation', data.session_id);
          }
          if (data.agent) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? { ...msg, agent: data.agent } : msg
              )
            );
          }
        },
        onStatus: (data) => {
          setStreamingStatus(data.message || null);
        },
        onToken: (chunk) => {
          if (activeSessionIdRef.current !== targetSessionId) return;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, text: msg.text + chunk }
                : msg
            )
          );
        },
        onComplete: (data) => {
          if (activeSessionIdRef.current !== targetSessionId) return;
          const rawResponse = data.response || '';
          const agentName = data.agent || 'CLINICAL_PRE_CONSULTATION_AGENT';
          const isReport = isClinicalReportText(rawResponse, agentName);

          const lower = rawResponse.toLowerCase();
          const emergencyDetected = lower.includes('emergency') && (lower.includes('escalation') || lower.includes('immediate medical attention'));

          if (emergencyDetected) {
            setIsEmergency(true);
            setIntakeStatus('emergency');
          } else if (isReport) {
            setIntakeStatus('completed');
            setLatestReport(rawResponse);
          } else {
            setIntakeStatus('active');
          }

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    text: rawResponse || msg.text,
                    agent: agentName,
                    status: 'complete',
                    isReport,
                    isEmergency: emergencyDetected
                  }
                : msg
            )
          );
          loadSessions();
        },
        onError: (err) => {
          if (activeSessionIdRef.current !== targetSessionId) return;
          console.error('[Clinical] Streaming error:', err);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    text: msg.text
                      ? `${msg.text}\n\n⚠️ *Generation interrupted: ${err.message}*`
                      : `⚠️ **Error:** ${err.message || 'Failed to process clinical information.'}`,
                    status: 'error'
                  }
                : msg
            )
          );
          setError(err.message);
        }
      });
    } catch (err) {
      if (abortController.signal.aborted) {
        console.log('[Clinical] Generation stopped by user');
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  text: msg.text ? `${msg.text} *(Stopped)*` : '*(Generation stopped)*',
                  status: 'complete'
                }
              : msg
          )
        );
      }
    } finally {
      if (activeSessionIdRef.current === targetSessionId) {
        setIsLoading(false);
        setIsStreaming(false);
        setStreamingStatus(null);
        abortControllerRef.current = null;
      }
    }
  }, [isLoading, isStreaming, sessionId, newSession, loadSessions]);

  /**
   * Submit patient feedback/corrections to POST /clinical-patient-feedback
   */
  const submitFeedback = useCallback(async (feedbackText) => {
    const trimmed = (feedbackText || '').trim();
    if (!trimmed) {
      setFeedbackError('Please enter your corrections or additional information.');
      return;
    }

    const targetSessionId = sessionId || activeSessionIdRef.current;
    setIsFeedbackSubmitting(true);
    setFeedbackError(null);
    stopSpeech();

    const formData = new FormData();
    formData.append('patient_feedback', trimmed);

    try {
      const headers = {};
      if (targetSessionId) {
        headers['X-Session-ID'] = targetSessionId;
      }

      const response = await fetch('/clinical-patient-feedback', {
        method: 'POST',
        headers,
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `Server returned ${response.status}`);
      }

      const data = await response.json();

      // Stale async response guard
      if (activeSessionIdRef.current !== targetSessionId) {
        return;
      }

      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        activeSessionIdRef.current = data.session_id;
        localStorage.setItem('active_session_clinical_pre_consultation', data.session_id);
      }

      const updatedFormattedReport = data.formatted_report || (typeof data.report === 'string' ? data.report : '') || '';

      if (updatedFormattedReport) {
        setLatestReport(updatedFormattedReport);
        setIsConfirmed(false); // Reset confirmation after edit
        setConfirmationTimestamp(null);
      }

      // Update existing report card in-place without appending raw correction text below the report
      if (updatedFormattedReport) {
        setMessages((prev) => {
          const hasReport = prev.some((m) => m.isReport);
          if (hasReport) {
            return prev.map((m) => (m.isReport ? { ...m, text: updatedFormattedReport } : m));
          }
          return [
            ...prev,
            {
              id: `feedback-report-${Date.now()}`,
              sender: 'assistant',
              text: updatedFormattedReport,
              agent: data.agent || 'CLINICAL_PRE_CONSULTATION_AGENT',
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              status: 'complete',
              isReport: true,
              isEmergency: Boolean(data.requires_immediate_attention || isEmergency)
            }
          ];
        });
        setIntakeStatus('completed');
      } else {
        throw new Error('Server did not return an updated clinical report.');
      }

      loadSessions();
    } catch (err) {
      if (activeSessionIdRef.current !== targetSessionId) {
        return;
      }
      console.error('Patient feedback error:', err);
      setFeedbackError(err.message || 'Failed to submit feedback.');
      throw err;
    } finally {
      if (activeSessionIdRef.current === targetSessionId) {
        setIsFeedbackSubmitting(false);
      }
    }
  }, [isEmergency, stopSpeech, sessionId, loadSessions]);

  /**
   * Confirm that the clinical report is accurate via backend /clinical-confirm
   */
  const confirmReport = useCallback(async () => {
    const targetSessionId = sessionId || activeSessionIdRef.current;
    try {
      const headers = {};
      if (targetSessionId) {
        headers['X-Session-ID'] = targetSessionId;
      }
      const response = await fetch('/clinical-confirm', {
        method: 'POST',
        headers,
        credentials: 'include'
      });
      if (response.ok) {
        const data = await response.json().catch(() => ({}));
        setIsConfirmed(true);
        const ts = data.confirmed_at
          ? new Date(data.confirmed_at).toLocaleString()
          : new Date().toLocaleString();
        setConfirmationTimestamp(ts);
        loadSessions();
      } else {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || err.message || 'Failed to record confirmation.');
      }
    } catch (err) {
      console.error('Confirmation error:', err);
      // Fallback local confirmation if offline
      setIsConfirmed(true);
      setConfirmationTimestamp(new Date().toLocaleString());
    }
  }, [sessionId, loadSessions]);

  // Listen for sidebar clear view event
  useEffect(() => {
    const handleClearEvent = () => {
      newSession();
    };
    window.addEventListener('medical-assistant:clear-active-view', handleClearEvent);
    return () => {
      window.removeEventListener('medical-assistant:clear-active-view', handleClearEvent);
    };
  }, [newSession]);

  /**
   * Play or pause voice audio for assistant text using /generate-speech
   */
  const playSpeech = useCallback(async (messageId, rawText) => {
    // Toggle pause/resume if clicking the currently loaded clip
    if (activeAudioId === messageId && currentAudioRef.current && !isAudioLoading) {
      if (isAudioPlaying) {
        currentAudioRef.current.pause();
        setIsAudioPlaying(false);
        setIsAudioPaused(true);
      } else {
        try {
          setAudioError(null);
          await currentAudioRef.current.play();
          setIsAudioPlaying(true);
          setIsAudioPaused(false);
        } catch (err) {
          console.error('Audio resume error:', err);
          setAudioError({
            id: messageId,
            message: err?.name === 'NotAllowedError'
              ? 'Playback was blocked by browser autoplay rules. Please click Play again.'
              : 'Unable to resume audio playback.'
          });
          stopSpeech();
        }
      }
      return;
    }

    // Stop any previous audio or in-flight TTS request to guarantee single-stream playback
    stopSpeech();
    setAudioError(null);

    const textToSpeak = cleanTextForSpeech(rawText);
    if (!textToSpeak) {
      setAudioError({
        id: messageId,
        message: 'No readable clinical text available for speech playback.'
      });
      return;
    }

    if (typeof Audio === 'undefined') {
      setAudioError({
        id: messageId,
        message: 'Audio playback is not supported in this browser.'
      });
      return;
    }

    const requestId = audioRequestIdRef.current + 1;
    audioRequestIdRef.current = requestId;
    const abortController = new AbortController();
    audioAbortControllerRef.current = abortController;

    setIsAudioLoading(true);
    setIsAudioPlaying(false);
    setIsAudioPaused(false);
    setActiveAudioId(messageId);

    let createdAudioUrl = null;
    try {
      const response = await fetch('/generate-speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToSpeak,
          voice_id: DEFAULT_VOICE_ID
        }),
        signal: abortController.signal
      });

      if (audioRequestIdRef.current !== requestId) {
        return;
      }

      if (!response.ok) {
        let errMessage = `Speech service error (${response.status})`;
        try {
          const errData = await response.json();
          if (errData && (errData.error || errData.detail || errData.message)) {
            errMessage = errData.error || errData.detail || errData.message;
          }
        } catch (_) {}
        throw new Error(errMessage);
      }

      const contentType = (response.headers.get('content-type') || '').toLowerCase();
      if (contentType.includes('application/json')) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || 'Speech service did not return audio data.');
      }

      const audioBlob = await response.blob();
      if (audioRequestIdRef.current !== requestId) {
        return;
      }

      if (!audioBlob || audioBlob.size < 50) {
        throw new Error('Speech service returned an empty audio response.');
      }

      const probeAudio = new Audio();
      const mimeToCheck = audioBlob.type || contentType || 'audio/mpeg';
      const canPlay = typeof probeAudio.canPlayType === 'function'
        ? (probeAudio.canPlayType('audio/mpeg') || probeAudio.canPlayType(mimeToCheck))
        : 'maybe';
      if (canPlay === '') {
        throw new Error('The returned audio format is not supported by this browser.');
      }

      createdAudioUrl = URL.createObjectURL(audioBlob);
      currentAudioUrlRef.current = createdAudioUrl;

      const audio = new Audio(createdAudioUrl);
      currentAudioRef.current = audio;

      audio.onplay = () => {
        if (audioRequestIdRef.current !== requestId) return;
        setIsAudioPlaying(true);
        setIsAudioPaused(false);
        setIsAudioLoading(false);
      };

      audio.onpause = () => {
        if (audioRequestIdRef.current !== requestId) return;
        setIsAudioPlaying(false);
        if (!audio.ended && currentAudioRef.current === audio) {
          setIsAudioPaused(true);
        }
      };

      audio.onended = () => {
        if (audioRequestIdRef.current !== requestId) return;
        stopSpeech();
      };

      audio.onerror = () => {
        if (audioRequestIdRef.current !== requestId) return;
        stopSpeech();
        setAudioError({
          id: messageId,
          message: 'Unable to play the audio stream. The format may be unsupported or corrupted.'
        });
      };

      await audio.play();
      if (audioRequestIdRef.current !== requestId) {
        stopSpeech();
      }
    } catch (err) {
      if (err?.name === 'AbortError' || audioRequestIdRef.current !== requestId) {
        if (createdAudioUrl && currentAudioUrlRef.current !== createdAudioUrl) {
          URL.revokeObjectURL(createdAudioUrl);
        }
        return;
      }
      console.error('TTS request/playback failed:', err);
      stopSpeech();
      let friendlyMsg = err?.message || 'Failed to generate or play speech audio.';
      if (err?.name === 'NotAllowedError') {
        friendlyMsg = 'Audio playback was blocked by browser autoplay policy. Please click Play again.';
      } else if (err?.name === 'NotSupportedError') {
        friendlyMsg = 'This audio format is not supported by your browser.';
      } else if (err instanceof TypeError && !err.message.includes('TTS')) {
        friendlyMsg = 'Network error while reaching the speech service. Please try again.';
      }
      setAudioError({
        id: messageId,
        message: friendlyMsg
      });
    } finally {
      if (audioAbortControllerRef.current === abortController) {
        audioAbortControllerRef.current = null;
      }
    }
  }, [activeAudioId, isAudioPlaying, isAudioLoading, stopSpeech]);

  const value = {
    messages,
    isLoading,
    isStreaming,
    streamingStatus,
    stopGeneration,
    error,
    intakeStatus,
    isEmergency,
    latestReport,
    isConfirmed,
    confirmationTimestamp,
    isFeedbackSubmitting,
    feedbackError,
    sessionId,
    sessions,
    isLoadingSessions,
    newSession,
    selectSession,
    deleteSession,
    loadSessions,
    activeAudioId,
    isAudioPlaying,
    isAudioPaused,
    isAudioLoading,
    audioError,
    clearAudioError,
    sendMessage,
    submitFeedback,
    confirmReport,
    resetIntake,
    playSpeech,
    stopSpeech
  };

  return (
    <ClinicalPreConsultationContext.Provider value={value}>
      {children}
    </ClinicalPreConsultationContext.Provider>
  );
}

export function useClinicalPreConsultation() {
  const context = useContext(ClinicalPreConsultationContext);
  if (!context) {
    throw new Error('useClinicalPreConsultation must be used within a ClinicalPreConsultationProvider');
  }
  return context;
}
