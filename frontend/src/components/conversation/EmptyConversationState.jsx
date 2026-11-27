import React from 'react';
import { useConversation } from '../../context/ConversationContext';

const STARTER_PROMPTS = [
  {
    title: 'Clinical Knowledge & RAG',
    icon: 'fa-book-medical',
    prompt: 'What are the classic diagnostic signs and long-term complications of Type 2 Diabetes?'
  },
  {
    title: 'Hypertension Management',
    icon: 'fa-heart-pulse',
    prompt: 'What are the first-line pharmacologic recommendations for essential hypertension?'
  },
  {
    title: 'Acute Musculoskeletal Care',
    icon: 'fa-bone',
    prompt: 'What is the RICE protocol for an acute ankle sprain and when should an X-ray be ordered?'
  },
  {
    title: 'Medical Web Search Fallback',
    icon: 'fa-globe-americas',
    prompt: 'What are the latest clinical findings regarding long COVID neurological symptoms?'
  }
];

export default function EmptyConversationState() {
  const { sendMessage, isLoading } = useConversation();

  return (
    <div className="empty-conversation-state">
      <div className="empty-hero">
        <div className="hero-icon-wrapper">
          <i className="fas fa-comment-medical hero-icon"></i>
        </div>
        <h2 className="empty-title">Medical AI Conversation Workspace</h2>
        <p className="empty-description">
          Ask clinical questions, review disease protocols, and retrieve evidence-based medical literature.
          Our multi-agent system orchestrates conversational guidance, Hybrid RAG, and live medical web search.
        </p>
      </div>

      <div className="starter-prompts-section">
        <h3 className="starter-prompts-title">
          <i className="fas fa-lightbulb"></i> Suggested Clinical Inquiries
        </h3>
        <div className="starter-prompts-grid">
          {STARTER_PROMPTS.map((item, index) => (
            <button
              key={index}
              type="button"
              className="starter-prompt-card"
              onClick={() => sendMessage(item.prompt)}
              disabled={isLoading}
            >
              <div className="prompt-card-header">
                <i className={`fas ${item.icon} prompt-card-icon`}></i>
                <span className="prompt-card-title">{item.title}</span>
              </div>
              <p className="prompt-card-text">"{item.prompt}"</p>
              <span className="prompt-card-action">
                Ask this question <i className="fas fa-arrow-right"></i>
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="empty-disclaimer">
        <i className="fas fa-shield-halved"></i>
        <span>
          <strong>Medical Notice:</strong> This AI assistant provides informational guidance based on curated literature.
          It does not replace professional clinical judgment or formal medical diagnosis.
        </span>
      </div>
    </div>
  );
}
