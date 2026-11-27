import React from 'react';

/**
 * Normalizes agent names and maps them to human-readable labels, icons, and CSS classes
 */
export default function AgentBadge({ agent }) {
  if (!agent) {
    return (
      <span className="agent-badge agent-badge-default">
        <i className="fas fa-robot"></i> Assistant
      </span>
    );
  }

  const normalized = agent.toUpperCase();

  if (normalized.includes('RAG')) {
    return (
      <span className="agent-badge agent-badge-rag">
        <i className="fas fa-book-medical"></i> Hybrid RAG Agent
      </span>
    );
  }

  if (normalized.includes('WEB_SEARCH') || normalized.includes('SEARCH')) {
    return (
      <span className="agent-badge agent-badge-web">
        <i className="fas fa-globe-americas"></i> Web Search Agent
      </span>
    );
  }

  if (normalized.includes('CONVERSATION')) {
    return (
      <span className="agent-badge agent-badge-convo">
        <i className="fas fa-comment-medical"></i> Conversation Agent
      </span>
    );
  }

  if (normalized.includes('CLINICAL')) {
    return (
      <span className="agent-badge agent-badge-clinical">
        <i className="fas fa-clipboard-user"></i> Clinical Pre-Consultation
      </span>
    );
  }

  if (normalized.includes('BRAIN')) {
    return (
      <span className="agent-badge agent-badge-brain">
        <i className="fas fa-brain"></i> Brain MRI Agent (Placeholder)
      </span>
    );
  }

  if (normalized.includes('CHEST') || normalized.includes('XRAY') || normalized.includes('X-RAY')) {
    return (
      <span className="agent-badge agent-badge-chest">
        <i className="fas fa-lungs"></i> Chest X-ray Agent
      </span>
    );
  }

  if (normalized.includes('SKIN') || normalized.includes('LESION')) {
    return (
      <span className="agent-badge agent-badge-skin">
        <i className="fas fa-allergies"></i> Skin Lesion Agent
      </span>
    );
  }

  if (normalized.includes('HUMAN_VALIDAT')) {
    return (
      <span className="agent-badge agent-badge-validation">
        <i className="fas fa-user-md"></i> Clinician Validated
      </span>
    );
  }

  if (normalized.includes('SYSTEM')) {
    return (
      <span className="agent-badge agent-badge-system">
        <i className="fas fa-info-circle"></i> System
      </span>
    );
  }

  return (
    <span className="agent-badge agent-badge-default">
      <i className="fas fa-microchip"></i> {agent}
    </span>
  );
}
