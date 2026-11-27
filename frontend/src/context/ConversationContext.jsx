import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import { cleanTextForSpeech, DEFAULT_VOICE_ID } from '../utils/speech';
import { streamChat } from '../utils/streamClient';

const ConversationContext = createContext(null);

export function ConversationProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState(null);
  const [lastQuery, setLastQuery] = useState('');
  const [activeAudioMessageId, setActiveAudioMessageId] = useState(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  // Phase 6 Session Management State
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('active_session_conversation') || null;
  });
  const [sessions, setSessions] = useState([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);

  // Store active Audio object, request token, and abort controller for cleanup
  const currentAudioRef = useRef(null);
  const currentAudioUrlRef = useRef(null);
  const audioAbortControllerRef = useRef(null);
  const audioRequestIdRef = useRef(0);
  const abortControllerRef = useRef(null);
  const isSendingRef = useRef(false);

  /**
   * Stop currently playing or pending audio and revoke object URL
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
    setActiveAudioMessageId(null);
    setIsAudioPlaying(false);
    setIsAudioLoading(false);
  }, []);

  // Cleanup audio resources on unmount
  useEffect(() => {
    return () => {
      stopSpeech();
    };
  }, [stopSpeech]);

  /**
   * Play or pause text-to-speech for an assistant message
   */
  const playSpeech = useCallback(async (messageId, rawText) => {
    if (activeAudioMessageId === messageId && currentAudioRef.current && !isAudioLoading) {
      if (isAudioPlaying) {
        currentAudioRef.current.pause();
        setIsAudioPlaying(false);
      } else {
        try {
          setAudioError(null);
          await currentAudioRef.current.play();
          setIsAudioPlaying(true);
        } catch (err) {
          console.error('Audio resume error:', err);
          setAudioError({ id: messageId, error: 'Unable to resume audio playback.' });
          stopSpeech();
        }
      }
      return;
    }

    stopSpeech();
    setAudioError(null);

    const textToSpeak = cleanTextForSpeech(rawText);
    if (!textToSpeak) return;

    const requestId = audioRequestIdRef.current + 1;
    audioRequestIdRef.current = requestId;
    const abortController = new AbortController();
    audioAbortControllerRef.current = abortController;

    setIsAudioLoading(true);
    setActiveAudioMessageId(messageId);

    try {
      const response = await fetch('/generate-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: textToSpeak,
          voice_id: DEFAULT_VOICE_ID
        }),
        signal: abortController.signal
      });

      if (audioRequestIdRef.current !== requestId) return;

      if (!response.ok) {
        let errMessage = `TTS server error (${response.status})`;
        try {
          const errData = await response.json();
          if (errData && (errData.error || errData.detail || errData.message)) {
            errMessage = errData.error || errData.detail || errData.message;
          }
        } catch (_) {}
        throw new Error(errMessage);
      }

      const audioBlob = await response.blob();
      if (audioRequestIdRef.current !== requestId) return;
      if (!audioBlob || audioBlob.size < 50) {
        throw new Error('Speech service returned an empty audio response.');
      }

      const audioUrl = URL.createObjectURL(audioBlob);
      currentAudioUrlRef.current = audioUrl;

      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.onplay = () => {
        if (audioRequestIdRef.current !== requestId) return;
        setIsAudioPlaying(true);
        setIsAudioLoading(false);
      };

      audio.onpause = () => {
        if (audioRequestIdRef.current !== requestId) return;
        setIsAudioPlaying(false);
      };

      audio.onended = () => {
        if (audioRequestIdRef.current !== requestId) return;
        stopSpeech();
      };

      audio.onerror = () => {
        if (audioRequestIdRef.current !== requestId) return;
        stopSpeech();
        setAudioError({ id: messageId, error: 'Unable to play the generated audio stream.' });
      };

      await audio.play();
    } catch (err) {
      if (err?.name === 'AbortError' || audioRequestIdRef.current !== requestId) return;
      const errMsg = err?.message || 'Text-to-speech failed';
      console.error('TTS request failed:', errMsg);
      stopSpeech();
      setAudioError({ id: messageId, error: errMsg });
    } finally {
      if (audioAbortControllerRef.current === abortController) {
        audioAbortControllerRef.current = null;
      }
    }
  }, [activeAudioMessageId, isAudioPlaying, isAudioLoading, stopSpeech]);

  /**
   * Load session list for conversation workspace
   */
  const loadSessions = useCallback(async () => {
    try {
      setIsLoadingSessions(true);
      const res = await fetch('/sessions?workspace_type=conversation&include_archived=false', {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const raw = data.sessions || [];
        // Strict prefix isolation
        const filtered = raw.filter((s) => s.session_id && s.session_id.startsWith('conv_'));
        setSessions(filtered);
      }
    } catch (err) {
      console.error('Failed to load conversation sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  /**
   * Load messages for a given session ID without locking the chat composer input
   */
  const loadMessages = useCallback(async (sid) => {
    if (!sid) {
      setMessages([]);
      return;
    }
    try {
      const res = await fetch(`/sessions/${encodeURIComponent(sid)}/messages`, {
        credentials: 'include'
      });
      if (res.status === 404) {
        if (localStorage.getItem('active_session_conversation') === sid) {
          localStorage.removeItem('active_session_conversation');
        }
        setSessionId((prev) => (prev === sid ? null : prev));
        setMessages([]);
        return;
      }
      if (res.ok) {
        const data = await res.json();
        const mapped = (data.messages || []).map((m) => ({
          id: m.message_id || `msg-${m.sequence_num}`,
          sender: m.sender,
          text: m.content,
          agent: m.agent_name || (m.sender === 'user' ? 'User' : 'Medical Conversation Agent'),
          timestamp: m.created_at
            ? new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '',
          status: 'complete',
          resultImage: m.metadata?.result_image || null,
          isReport: Boolean(m.is_report),
          isEmergency: Boolean(m.is_emergency)
        }));
        setMessages(mapped);
      }
    } catch (err) {
      console.error(`Failed to load messages for session ${sid}:`, err);
    }
  }, []);

  // On mount: load sessions and active session messages
  useEffect(() => {
    loadSessions();
    const storedId = localStorage.getItem('active_session_conversation');
    if (storedId && storedId.startsWith('conv_')) {
      setSessionId(storedId);
      loadMessages(storedId);
    } else if (storedId) {
      localStorage.removeItem('active_session_conversation');
    }
  }, [loadSessions, loadMessages]);

  /**
   * Switch active session
   */
  const selectSession = useCallback((sid) => {
    stopSpeech();
    setSessionId(sid);
    localStorage.setItem('active_session_conversation', sid);
    loadMessages(sid);
  }, [stopSpeech, loadMessages]);

  /**
   * Create a new conversation session
   */
  const newSession = useCallback(async () => {
    stopSpeech();
    try {
      const res = await fetch('/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_type: 'conversation' }),
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const newId = data.session_id || data.session?.session_id;
        if (newId) {
          setSessionId(newId);
          localStorage.setItem('active_session_conversation', newId);
          setMessages([]);
          setLastQuery('');
          await loadSessions();
          return newId;
        }
      }
    } catch (err) {
      console.error('Failed to create new conversation session:', err);
    }

    const fallbackId = `conv_${crypto.randomUUID()}`;
    setSessionId(fallbackId);
    localStorage.setItem('active_session_conversation', fallbackId);
    setMessages([]);
    setLastQuery('');
    return fallbackId;
  }, [stopSpeech, loadSessions]);

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
          localStorage.removeItem('active_session_conversation');
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
   * Stop an active generation stream
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
   * Send user message with progressive token streaming over /chat/stream
   */
  const sendMessage = useCallback(async (queryText) => {
    const trimmed = (queryText || '').trim();
    if (!trimmed || isLoading || isStreaming || isSendingRef.current) return;

    isSendingRef.current = true;
    setLastQuery(trimmed);
    setIsLoading(true);

    let targetSessionId = sessionId;
    if (!targetSessionId || !targetSessionId.startsWith('conv_')) {
      targetSessionId = await newSession();
    }

    const userMessageId = `user-${Date.now()}`;
    const userMessage = {
      id: userMessageId,
      sender: 'user',
      text: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'complete'
    };

    const assistantMessageId = `assistant-${Date.now() + 1}`;
    const initialAssistantMessage = {
      id: assistantMessageId,
      sender: 'assistant',
      text: '',
      agent: 'Medical Assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'streaming',
      resultImage: null
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsStreaming(true);
    setStreamingStatus(null);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      await streamChat({
        query: trimmed,
        workspace: 'conversation',
        sessionId: targetSessionId,
        signal: abortController.signal,
        onStart: (data) => {
          if (data.session_id && data.session_id !== targetSessionId) {
            setSessionId(data.session_id);
            localStorage.setItem('active_session_conversation', data.session_id);
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
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, text: msg.text + chunk }
                : msg
            )
          );
        },
        onComplete: (data) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    text: data.response || msg.text,
                    agent: data.agent || msg.agent,
                    status: 'complete',
                    resultImage: data.result_image || null
                  }
                : msg
            )
          );
          loadSessions();
        },
        onError: (err) => {
          console.error('[Conversation] Streaming error:', err);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    text: msg.text
                      ? `${msg.text}\n\n⚠️ *Generation interrupted: ${err.message}*`
                      : `⚠️ **Error:** ${err.message || 'Failed to process request.'}`,
                    status: 'error',
                    canRetry: true
                  }
                : msg
            )
          );
        }
      });
    } catch (err) {
      if (abortController.signal.aborted) {
        console.log('[Conversation] Generation stopped by user');
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
      isSendingRef.current = false;
      setIsLoading(false);
      setIsStreaming(false);
      setStreamingStatus(null);
      abortControllerRef.current = null;
    }
  }, [isLoading, isStreaming, sessionId, newSession, loadSessions]);

  /**
   * Retry sending the last user message
   */
  const retryLast = useCallback(() => {
    if (lastQuery && !isLoading) {
      sendMessage(lastQuery);
    }
  }, [lastQuery, isLoading, sendMessage]);

  /**
   * Clear all messages in the conversation stream
   */
  const clearConversation = useCallback(() => {
    stopSpeech();
    setMessages([]);
    setLastQuery('');
  }, [stopSpeech]);

  const value = {
    messages,
    isLoading,
    isStreaming,
    streamingStatus,
    stopGeneration,
    lastQuery,
    sessionId,
    sessions,
    isLoadingSessions,
    sendMessage,
    retryLast,
    clearConversation,
    newSession,
    selectSession,
    deleteSession,
    loadSessions,
    playSpeech,
    stopSpeech,
    activeAudioMessageId,
    isAudioPlaying,
    isAudioLoading,
    audioError
  };

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversation() {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
}
