import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useConversation } from '../../context/ConversationContext.jsx';
import { useImageAnalysis } from '../../context/ImageAnalysisContext.jsx';
import { useClinicalPreConsultation } from '../../context/ClinicalPreConsultationContext.jsx';

export default function Sidebar({ onClearChat }) {
  const location = useLocation();
  const convContext = useConversation();
  const imageContext = useImageAnalysis();
  const clinicalContext = useClinicalPreConsultation();

  const isHome = location.pathname === '/';
  const isConversation = location.pathname.startsWith('/conversation');
  const isImageAnalysis = location.pathname.startsWith('/image-analysis');
  const isClinical = location.pathname.startsWith('/clinical-pre-consultation') || location.pathname.startsWith('/pre-consultation');

  // Refresh active workspace session list whenever entering a workspace
  useEffect(() => {
    if (isConversation && convContext.loadSessions) {
      convContext.loadSessions();
    } else if (isImageAnalysis && imageContext.loadSessions) {
      imageContext.loadSessions();
    } else if (isClinical && clinicalContext.loadSessions) {
      clinicalContext.loadSessions();
    }
  }, [location.pathname, isConversation, isImageAnalysis, isClinical]);

  // Determine active workspace configuration for session history
  let panelTitle = '';
  let newBtnLabel = '';
  let emptyLabel = '';
  let emptySubtext = '';
  let emptyIcon = 'fa-inbox';
  let currentSessionId = null;
  let rawSessions = [];
  let isLoadingSessions = false;
  let handleNewSession = () => {};
  let handleSelectSession = () => {};
  let handleDeleteSession = () => {};
  let currentWorkspace = '';

  if (isConversation) {
    currentWorkspace = 'conversation';
    panelTitle = 'Chat History';
    newBtnLabel = 'New Chat';
    emptyLabel = 'No saved conversation sessions';
    emptySubtext = 'Start a new chat to begin';
    emptyIcon = 'fa-comments';
    currentSessionId = convContext.sessionId;
    // Strict workspace prefix isolation: conv_ only
    rawSessions = (convContext.sessions || []).filter(
      (s) => s.session_id && s.session_id.startsWith('conv_')
    );
    isLoadingSessions = Boolean(convContext.isLoadingSessions);
    handleNewSession = convContext.newSession;
    handleSelectSession = convContext.selectSession;
    handleDeleteSession = convContext.deleteSession;
  } else if (isImageAnalysis) {
    currentWorkspace = 'image_analysis';
    panelTitle = 'Image Sessions';
    newBtnLabel = 'New Analysis';
    emptyLabel = 'No saved image sessions';
    emptySubtext = 'Upload an image to start';
    emptyIcon = 'fa-x-ray';
    // Strict workspace prefix isolation: img_ only
    currentSessionId = imageContext.sessionId;
    rawSessions = (imageContext.sessions || []).filter(
      (s) => s.session_id && s.session_id.startsWith('img_')
    );
    isLoadingSessions = Boolean(imageContext.isLoadingSessions);
    handleNewSession = imageContext.newSession;
    handleSelectSession = imageContext.selectSession;
    handleDeleteSession = imageContext.deleteSession;
  } else if (isClinical) {
    currentWorkspace = 'clinical_pre_consultation';
    panelTitle = 'Intake Sessions';
    newBtnLabel = 'New Intake';
    emptyLabel = 'No saved clinical intake sessions';
    emptySubtext = 'Begin a pre-consultation intake';
    emptyIcon = 'fa-clipboard-user';
    // Strict workspace prefix isolation: clin_ only
    currentSessionId = clinicalContext.sessionId;
    rawSessions = (clinicalContext.sessions || []).filter(
      (s) => s.session_id && s.session_id.startsWith('clin_')
    );
    isLoadingSessions = Boolean(clinicalContext.isLoadingSessions);
    handleNewSession = clinicalContext.newSession;
    handleSelectSession = clinicalContext.selectSession;
    handleDeleteSession = clinicalContext.deleteSession;
  }

  // Sort sessions newest first by updated_at / created_at
  const sortedSessions = [...rawSessions].sort((a, b) => {
    const timeA = new Date(a.updated_at || a.created_at || 0).getTime();
    const timeB = new Date(b.updated_at || b.created_at || 0).getTime();
    return timeB - timeA;
  });

  // Meaningful session title with fallback
  const formatSessionTitle = (sess) => {
    if (!sess) return 'Untitled Session';
    const raw = (sess.title || '').trim();
    if (raw && !raw.toLowerCase().startsWith('new ') && raw !== 'Untitled Session') {
      return raw;
    }
    if (sess.metadata) {
      if (sess.metadata.query) return sess.metadata.query.slice(0, 45);
      if (sess.metadata.filename) return `Image: ${sess.metadata.filename}`;
      if (sess.metadata.chief_complaint) return sess.metadata.chief_complaint.slice(0, 45);
    }
    const dateObj = sess.updated_at || sess.created_at ? new Date(sess.updated_at || sess.created_at) : null;
    const dateStr = dateObj ? dateObj.toLocaleDateString([], { month: 'short', day: 'numeric' }) : '';
    const timeStr = dateObj ? dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
    const timePart = dateStr ? (timeStr ? `${dateStr}, ${timeStr}` : dateStr) : '';
    if (currentWorkspace === 'conversation') return timePart ? `Chat • ${timePart}` : 'New Chat';
    if (currentWorkspace === 'image_analysis') return timePart ? `Analysis • ${timePart}` : 'New Analysis';
    if (currentWorkspace === 'clinical_pre_consultation') return timePart ? `Intake • ${timePart}` : 'New Intake';
    return raw || 'Untitled Session';
  };

  return (
    <aside className="sidebar-wrapper" aria-label="Agent Overview Sidebar">
      <div className="sidebar">
        {/* Workspace Brand / Header */}
        <div className="sidebar-brand-header">
          <h5>
            <i className="fas fa-robot text-primary"></i>
            Medical Assistant
          </h5>
          {isHome && <span className="sidebar-subtitle">About & Agent Overview</span>}
        </div>

        {/* ------------------------------------------------------------- */}
        {/* A. HOME PAGE SIDEBAR: Project Information & Agent Details ONLY */}
        {/* ------------------------------------------------------------- */}
        {isHome && (
          <div className="sidebar-home-content">
            <div className="agent-info">
              <h6>
                <i className="fas fa-comments text-primary"></i> Conversational Agents
              </h6>
              <ul>
                <li><i className="fas fa-comment-medical"></i> Medical Conversation Agent</li>
                <li><i className="fas fa-database"></i> Medical RAG Agent</li>
                <li><i className="fas fa-search"></i> Web Search Agent</li>
                <li><i className="fas fa-clipboard-check"></i> Clinical Pre-Consultation Agent</li>
              </ul>
            </div>

            <div className="agent-info">
              <h6>
                <i className="fas fa-eye text-primary"></i> Computer Vision Agents
              </h6>
              <ul>
                <li><i className="fas fa-brain"></i> Brain Tumor Detection</li>
                <li><i className="fas fa-lungs"></i> Chest X-ray Covid-19 Classification</li>
                <li><i className="fas fa-allergies"></i> Skin Lesion Segmentation</li>
              </ul>
            </div>

            <div className="agent-info">
              <h6>
                <i className="fas fa-cogs text-primary"></i> Agent Capabilities
              </h6>
              <ul className="capabilities-list">
                <li>
                  <strong>Medical Conversation Agent:</strong>
                  <ul>
                    <li>General health information and conversational assistance</li>
                    <li>Answers common medical questions with context-aware responses</li>
                  </ul>
                </li>
                <li>
                  <strong>Medical RAG Agent:</strong>
                  <ul>
                    <li>Docling-based document parsing to extract text, tables, and images from PDFs</li>
                    <li>LLM-based semantic chunking with structural boundaries</li>
                    <li>LLM-based query expansion with related medical terms</li>
                    <li>Qdrant hybrid vector database retrieval</li>
                    <li>Input-output guardrails for safe responses</li>
                    <li>Confidence-based handoff to Web Search to prevent hallucinations</li>
                  </ul>
                </li>
                <li>
                  <strong>Web Search Agent:</strong>
                  <ul>
                    <li>External medical information retrieval from reliable sources</li>
                    <li>Supplements medical knowledge when confidence is low</li>
                    <li>Provides current medical research and updates</li>
                  </ul>
                </li>
                <li>
                  <strong>Clinical Pre-Consultation Agent:</strong>
                  <ul>
                    <li>Structured patient information collection</li>
                    <li>Adaptive follow-up questions based on patient responses</li>
                    <li>Red-flag detection for emergency escalation</li>
                    <li>Structured clinical pre-consultation report generation</li>
                    <li>Patient review, correction, and confirmation workflow</li>
                  </ul>
                </li>
                <li>
                  <strong>Computer Vision Agents (Human-in-the-loop):</strong>
                  <ul>
                    <li>Brain Tumor Detection from medical imaging</li>
                    <li>Chest X-ray COVID-19 Classification</li>
                    <li>Skin Lesion Segmentation and analysis</li>
                    <li>Human verification required for all CV results</li>
                  </ul>
                </li>
              </ul>
            </div>

            <div className="agent-info">
              <h6>
                <i className="fas fa-project-diagram text-primary"></i> Multi-Agent Orchestration
              </h6>
              <ul>
                <li><i className="fas fa-project-diagram"></i> Coordinated interaction between agents</li>
                <li><i className="fas fa-tasks"></i> Dynamic task allocation based on capabilities</li>
                <li><i className="fas fa-sync-alt"></i> Seamless integration of agent responses</li>
              </ul>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* B, C, D. WORKSPACE-SPECIFIC SIDEBAR: Isolated Sessions List ONLY   */}
        {/* ------------------------------------------------------------------ */}
        {!isHome && (
          <div className="sidebar-sessions-section">
            <div className="sessions-header">
              <h6>
                <i className="fas fa-history"></i> {panelTitle}
              </h6>
              <button
                type="button"
                className="btn-new-session"
                onClick={handleNewSession}
                title={`Start ${newBtnLabel}`}
              >
                <i className="fas fa-plus"></i> {newBtnLabel}
              </button>
            </div>

            <div className="sessions-list">
              {isLoadingSessions && sortedSessions.length === 0 ? (
                <div className="sessions-loading">
                  <i className="fas fa-spinner fa-spin text-primary"></i>
                  <span>Loading history...</span>
                </div>
              ) : sortedSessions.length === 0 ? (
                <div className="sessions-empty">
                  <i className={`fas ${emptyIcon} text-muted`}></i>
                  <span>{emptyLabel}</span>
                  {emptySubtext && <small className="text-muted">{emptySubtext}</small>}
                </div>
              ) : (
                sortedSessions.map((sess) => {
                  const isActive = sess.session_id === currentSessionId;
                  const formattedDate = sess.updated_at || sess.created_at
                    ? new Date(sess.updated_at || sess.created_at).toLocaleDateString([], { month: 'short', day: 'numeric' })
                    : '';
                  const titleToDisplay = formatSessionTitle(sess);
                  return (
                    <div
                      key={sess.session_id}
                      className={`session-item ${isActive ? 'active' : ''}`}
                      onClick={() => handleSelectSession(sess.session_id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSelectSession(sess.session_id);
                      }}
                    >
                      <div className="session-info">
                        <span className="session-title" title={sess.title || titleToDisplay}>
                          {titleToDisplay}
                        </span>
                        {formattedDate && <span className="session-date">{formattedDate}</span>}
                      </div>
                      <button
                        type="button"
                        className="btn-delete-session"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(sess.session_id);
                        }}
                        title="Delete session"
                        aria-label="Delete session"
                      >
                        <i className="fas fa-trash-alt"></i>
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
