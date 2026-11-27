import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <header className="home-header">
        <h1>
          <i className="fas fa-heartbeat" style={{ color: '#0d6efd' }}></i>
          Multi-Agent Medical Assistant Workspaces
        </h1>
        <p>
          Select an independent medical feature workspace below to begin. Each workspace operates with its own isolated workflow, preserving its state and context as you switch between tasks.
        </p>
      </header>

      <section className="feature-cards-grid" aria-label="Feature Selection">
        {/* Workspace Card 1: Medical AI Conversation */}
        <article className="feature-card">
          <div className="card-icon-wrapper icon-conversation">
            <i className="fas fa-comment-medical"></i>
          </div>
          <h3>Medical AI Conversation</h3>
          <p>
            Engage in general health conversations, retrieve peer-reviewed clinical knowledge via Qdrant/BM25 hybrid RAG, and access real-time web literature when confidence thresholds require external verification.
          </p>
          <div className="agent-badges">
            <span className="agent-badge">Conversation Agent</span>
            <span className="agent-badge">RAG Agent (Qdrant)</span>
            <span className="agent-badge">Web Search Agent</span>
          </div>
          <button 
            type="button" 
            className="btn-open-workspace"
            onClick={() => navigate('/conversation')}
          >
            <span>Open Conversation Workspace</span>
            <i className="fas fa-arrow-right"></i>
          </button>
        </article>

        {/* Workspace Card 2: Medical Image Analysis */}
        <article className="feature-card">
          <div className="card-icon-wrapper icon-image">
            <i className="fas fa-x-ray"></i>
          </div>
          <h3>Medical Image Analysis</h3>
          <p>
            Upload clinical imaging for computer vision analysis: classify frontal chest radiographs for COVID-19, generate semantic segmentation masks for skin lesions, and review brain MRI scans with mandatory human-in-the-loop review.
          </p>
          <div className="agent-badges">
            <span className="agent-badge">Chest X-Ray Agent</span>
            <span className="agent-badge">Skin Lesion UNet</span>
            <span className="agent-badge">Brain Tumor Agent</span>
            <span className="agent-badge">Human Validation</span>
          </div>
          <button 
            type="button" 
            className="btn-open-workspace"
            onClick={() => navigate('/image-analysis')}
          >
            <span>Open Image Analysis Workspace</span>
            <i className="fas fa-arrow-right"></i>
          </button>
        </article>

        {/* Workspace Card 3: Clinical Pre-Consultation */}
        <article className="feature-card">
          <div className="card-icon-wrapper icon-clinical">
            <i className="fas fa-clipboard-user"></i>
          </div>
          <h3>Clinical Pre-Consultation</h3>
          <p>
            Initiate a structured outpatient intake interview. Gathers chief complaints and symptom timelines through adaptive follow-up questions, monitors for acute red-flag emergencies, and generates physician-ready SBAR reports.
          </p>
          <div className="agent-badges">
            <span className="agent-badge">Clinical Intake Agent</span>
            <span className="agent-badge">Red-Flag Detector</span>
            <span className="agent-badge">SBAR Generator</span>
            <span className="agent-badge">Patient Feedback</span>
          </div>
          <button 
            type="button" 
            className="btn-open-workspace"
            onClick={() => navigate('/clinical-pre-consultation')}
          >
            <span>Open Clinical Workspace</span>
            <i className="fas fa-arrow-right"></i>
          </button>
        </article>
      </section>
    </div>
  );
}
