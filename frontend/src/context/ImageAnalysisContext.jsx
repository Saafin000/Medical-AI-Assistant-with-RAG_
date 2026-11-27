import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import { cleanTextForSpeech, DEFAULT_VOICE_ID } from '../utils/speech';

const ImageAnalysisContext = createContext(null);

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB matching config.api.max_image_upload_size
const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg'];
const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

export function ImageAnalysisProvider({ children }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Analysis result state
  const [analysisResult, setAnalysisResult] = useState(null);

  // Human validation state
  const [validationStatus, setValidationStatus] = useState('idle'); // 'idle' | 'loading' | 'completed' | 'error'
  const [validationDecision, setValidationDecision] = useState(null); // 'yes' | 'no'
  const [validationComments, setValidationComments] = useState('');
  const [validationResult, setValidationResult] = useState(null);

  // Phase 6 Session Management State
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('active_session_image_analysis') || null;
  });
  const [sessions, setSessions] = useState([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);

  // Audio / TTS state
  const [activeAudioId, setActiveAudioId] = useState(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);

  // Refs for audio and object URL cleanup
  const currentAudioRef = useRef(null);
  const currentAudioUrlRef = useRef(null);
  const activePreviewUrlRef = useRef(null);

  // Keep ref synchronized with previewUrl state for cleanup
  useEffect(() => {
    activePreviewUrlRef.current = previewUrl;
  }, [previewUrl]);

  /**
   * Stop active audio playback and revoke blob URL
   */
  const stopSpeech = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    if (currentAudioUrlRef.current) {
      URL.revokeObjectURL(currentAudioUrlRef.current);
      currentAudioUrlRef.current = null;
    }
    setActiveAudioId(null);
    setIsAudioPlaying(false);
    setIsAudioLoading(false);
  }, []);

  /**
   * Cleanup resources on component unmount
   */
  useEffect(() => {
    return () => {
      stopSpeech();
      if (activePreviewUrlRef.current && activePreviewUrlRef.current.startsWith('blob:')) {
        URL.revokeObjectURL(activePreviewUrlRef.current);
      }
    };
  }, [stopSpeech]);

  /**
   * Clear entire workspace view
   */
  const clearAnalysis = useCallback(() => {
    stopSpeech();
    if (activePreviewUrlRef.current && activePreviewUrlRef.current.startsWith('blob:')) {
      URL.revokeObjectURL(activePreviewUrlRef.current);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setClinicalNotes('');
    setIsLoading(false);
    setError(null);
    setAnalysisResult(null);
    setValidationStatus('idle');
    setValidationDecision(null);
    setValidationComments('');
    setValidationResult(null);
  }, [stopSpeech]);

  /**
   * Load session list for image_analysis workspace
   */
  const loadSessions = useCallback(async () => {
    try {
      setIsLoadingSessions(true);
      const res = await fetch('/sessions?workspace_type=image_analysis&include_archived=false', {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const raw = data.sessions || [];
        // Strict prefix isolation
        const filtered = raw.filter((s) => s.session_id && s.session_id.startsWith('img_'));
        setSessions(filtered);
      }
    } catch (err) {
      console.error('Failed to load image analysis sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  /**
   * Load artifacts and messages for an active session
   */
  const loadSessionData = useCallback(async (sid) => {
    if (!sid) {
      clearAnalysis();
      return;
    }
    try {
      setIsLoading(true);
      const [artRes, msgRes] = await Promise.all([
        fetch(`/sessions/${encodeURIComponent(sid)}/artifacts`, { credentials: 'include' }),
        fetch(`/sessions/${encodeURIComponent(sid)}/messages`, { credentials: 'include' })
      ]);

      let restoredResultImage = null;

      if (artRes.ok) {
        const artData = await artRes.json();
        const artifacts = artData.artifacts || [];
        const inputArt = artifacts.find(a => a.artifact_type === 'input_image');
        if (inputArt) {
          // Normalize file path for static serving on all OS platforms (including Windows backslashes)
          const normPath = inputArt.file_path.replace(/\\/g, '/');
          const path = normPath.startsWith('/') ? normPath : `/${normPath}`;
          setPreviewUrl(path);
        }
        const resultArt = artifacts.find(a => a.artifact_type === 'result_image');
        if (resultArt) {
          const normResult = resultArt.file_path.replace(/\\/g, '/');
          restoredResultImage = normResult.startsWith('/') ? normResult : `/${normResult}`;
        }
      }

      if (msgRes.ok) {
        const msgData = await msgRes.json();
        const messages = msgData.messages || [];
        const assistantMsg = messages.find(m => m.sender === 'assistant' && m.agent_name !== 'HUMAN_VALIDATION');
        if (assistantMsg) {
          let msgImg = assistantMsg.metadata?.result_image || restoredResultImage;
          if (msgImg) {
            msgImg = msgImg.replace(/\\/g, '/');
            if (!msgImg.startsWith('/')) msgImg = `/${msgImg}`;
          }
          setAnalysisResult({
            agent: assistantMsg.agent_name || 'Medical Vision Agent',
            response: assistantMsg.content,
            resultImage: msgImg,
            timestamp: assistantMsg.created_at
              ? new Date(assistantMsg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : ''
          });
        }

        // Restore human validation status if previously confirmed
        const valMsg = messages.find(m => m.agent_name === 'HUMAN_VALIDATION');
        if (valMsg) {
          setValidationStatus('completed');
          setValidationResult({
            status: valMsg.metadata?.status || 'completed',
            message: valMsg.content,
            response: valMsg.content
          });
        }
      }
    } catch (err) {
      console.error(`Failed to load image session data for ${sid}:`, err);
    } finally {
      setIsLoading(false);
    }
  }, [clearAnalysis]);

  // On mount: load sessions and active session
  useEffect(() => {
    loadSessions();
    const storedId = localStorage.getItem('active_session_image_analysis');
    if (storedId && storedId.startsWith('img_')) {
      setSessionId(storedId);
      loadSessionData(storedId);
    } else if (storedId) {
      localStorage.removeItem('active_session_image_analysis');
    }
  }, [loadSessions, loadSessionData]);

  /**
   * Switch active session
   */
  const selectSession = useCallback((sid) => {
    clearAnalysis();
    setSessionId(sid);
    localStorage.setItem('active_session_image_analysis', sid);
    loadSessionData(sid);
  }, [clearAnalysis, loadSessionData]);

  /**
   * Create new image analysis session
   */
  const newSession = useCallback(async () => {
    clearAnalysis();
    try {
      const res = await fetch('/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_type: 'image_analysis' }),
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        const newId = data.session_id;
        setSessionId(newId);
        localStorage.setItem('active_session_image_analysis', newId);
        await loadSessions();
        return newId;
      }
    } catch (err) {
      console.error('Failed to create new image analysis session:', err);
    }
  }, [clearAnalysis, loadSessions]);

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
          localStorage.removeItem('active_session_image_analysis');
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
   * Validate and select an image file
   */
  const selectFile = useCallback((file) => {
    if (!file) return;

    setError(null);

    // Validate extension and MIME type
    const extension = file.name.split('.').pop().toLowerCase();
    const isExtensionAllowed = ALLOWED_EXTENSIONS.includes(extension);
    const isMimeAllowed = ALLOWED_MIME_TYPES.includes(file.type);

    if (!isExtensionAllowed && !isMimeAllowed) {
      setError(`Unsupported file format. Please upload a PNG, JPG, or JPEG image.`);
      return;
    }

    // Validate size limit (10MB)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setError(`File size (${sizeMB}MB) exceeds the maximum allowed limit of 10MB.`);
      return;
    }

    // Revoke previous object URL if any
    if (activePreviewUrlRef.current && activePreviewUrlRef.current.startsWith('blob:')) {
      URL.revokeObjectURL(activePreviewUrlRef.current);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
    setError(null);
    setAnalysisResult(null);
    setValidationStatus('idle');
    setValidationDecision(null);
    setValidationComments('');
    setValidationResult(null);
    stopSpeech();
  }, [stopSpeech]);

  /**
   * Remove currently selected file
   */
  const removeFile = useCallback(() => {
    if (activePreviewUrlRef.current && activePreviewUrlRef.current.startsWith('blob:')) {
      URL.revokeObjectURL(activePreviewUrlRef.current);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  }, []);

  // Listen for global clear view event from sidebar
  useEffect(() => {
    const handleClearEvent = () => {
      clearAnalysis();
    };
    window.addEventListener('medical-assistant:clear-active-view', handleClearEvent);
    return () => {
      window.removeEventListener('medical-assistant:clear-active-view', handleClearEvent);
    };
  }, [clearAnalysis]);

  /**
   * Send image to POST /upload with active X-Session-ID header
   */
  const analyzeImage = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select or drop a medical image to analyze.');
      return;
    }

    // Ensure active image analysis session exists before uploading
    let targetSessionId = sessionId;
    if (!targetSessionId || !targetSessionId.startsWith('img_')) {
      targetSessionId = await newSession();
    }

    setIsLoading(true);
    setError(null);
    stopSpeech();

    const formData = new FormData();
    formData.append('image', selectedFile);
    if (clinicalNotes.trim()) {
      formData.append('text', clinicalNotes.trim());
    }

    try {
      const headers = {};
      if (targetSessionId) {
        headers['X-Session-ID'] = targetSessionId;
      }

      const response = await fetch('/upload', {
        method: 'POST',
        headers,
        body: formData,
        credentials: 'include' // Preserves session_id cookie
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || errorData.message || `Server returned error (${response.status})`;
        throw new Error(errorMessage);
      }

      const data = await response.json();

      // Ensure active sessionId is synchronized
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem('active_session_image_analysis', data.session_id);
      }

      setAnalysisResult({
        agent: data.agent || 'Computer Vision Agent',
        response: data.response || 'No analysis details returned.',
        resultImage: data.result_image || null,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      });

      // Reset validation states for the new result
      setValidationStatus('idle');
      setValidationDecision(null);
      setValidationComments('');
      setValidationResult(null);

      // Refresh sessions
      loadSessions();
    } catch (err) {
      console.error('Image analysis error:', err);
      setError(err.message || 'Failed to analyze medical image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, clinicalNotes, stopSpeech, sessionId, loadSessions]);

  /**
   * Submit Human-in-the-Loop validation to POST /validate
   */
  const submitValidation = useCallback(async (decision, comments = '') => {
    if (!decision || (decision !== 'yes' && decision !== 'no')) {
      setError('Please select Agree or Disagree for human validation.');
      return;
    }

    setValidationStatus('loading');
    setError(null);

    const formData = new FormData();
    formData.append('validation_result', decision);
    formData.append('comments', comments.trim());

    try {
      let targetSessionId = sessionId;
      if (!targetSessionId || !targetSessionId.startsWith('img_')) {
        targetSessionId = await newSession();
      }

      const headers = {};
      if (targetSessionId) {
        headers['X-Session-ID'] = targetSessionId;
      }

      const response = await fetch('/validate', {
        method: 'POST',
        headers,
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || errorData.message || `Validation error (${response.status})`;
        throw new Error(errorMessage);
      }

      const data = await response.json();

      setValidationDecision(decision);
      setValidationComments(comments);
      setValidationResult({
        status: data.status,
        message: data.message || (decision === 'yes' ? 'Analysis confirmed by clinician.' : 'Analysis flagged for review.'),
        response: data.response || ''
      });
      setValidationStatus('completed');
      loadSessions();
    } catch (err) {
      console.error('Validation submission error:', err);
      setError(err.message || 'Failed to submit human validation. Please retry.');
      setValidationStatus('error');
    }
  }, [sessionId, loadSessions]);

  /**
   * Play or pause voice audio for assistant text using /generate-speech
   */
  const playSpeech = useCallback(async (messageId, rawText) => {
    if (activeAudioId === messageId && currentAudioRef.current) {
      if (isAudioPlaying) {
        currentAudioRef.current.pause();
        setIsAudioPlaying(false);
      } else {
        currentAudioRef.current.play().catch(err => console.error('Audio play error:', err));
        setIsAudioPlaying(true);
      }
      return;
    }

    stopSpeech();

    const textToSpeak = cleanTextForSpeech(rawText);
    if (!textToSpeak) return;

    setIsAudioLoading(true);
    setActiveAudioId(messageId);

    try {
      const response = await fetch('/generate-speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToSpeak,
          voice_id: DEFAULT_VOICE_ID
        })
      });

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
      const audioUrl = URL.createObjectURL(audioBlob);
      currentAudioUrlRef.current = audioUrl;

      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.onplay = () => {
        setIsAudioPlaying(true);
        setIsAudioLoading(false);
      };

      audio.onpause = () => {
        setIsAudioPlaying(false);
      };

      audio.onended = () => {
        setIsAudioPlaying(false);
        setActiveAudioId(null);
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
          currentAudioUrlRef.current = null;
        }
      };

      audio.onerror = (e) => {
        console.error('Audio playback error:', e);
        setIsAudioPlaying(false);
        setIsAudioLoading(false);
        setActiveAudioId(null);
      };

      await audio.play();
    } catch (err) {
      console.error('TTS request failed:', err);
      setIsAudioLoading(false);
      setActiveAudioId(null);
      setIsAudioPlaying(false);
    }
  }, [activeAudioId, isAudioPlaying, stopSpeech]);

  const value = {
    selectedFile,
    previewUrl,
    clinicalNotes,
    isLoading,
    error,
    analysisResult,
    validationStatus,
    validationDecision,
    validationComments,
    validationResult,
    sessionId,
    sessions,
    isLoadingSessions,
    newSession,
    selectSession,
    deleteSession,
    loadSessions,
    activeAudioId,
    isAudioPlaying,
    isAudioLoading,
    selectFile,
    removeFile,
    setClinicalNotes,
    analyzeImage,
    submitValidation,
    clearAnalysis,
    playSpeech,
    stopSpeech,
    setError
  };

  return (
    <ImageAnalysisContext.Provider value={value}>
      {children}
    </ImageAnalysisContext.Provider>
  );
}

export function useImageAnalysis() {
  const context = useContext(ImageAnalysisContext);
  if (!context) {
    throw new Error('useImageAnalysis must be used within an ImageAnalysisProvider');
  }
  return context;
}
