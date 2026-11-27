import React, { useEffect } from 'react';
import { useConversation } from '../context/ConversationContext';
import MessageStream from '../components/conversation/MessageStream';
import EmptyConversationState from '../components/conversation/EmptyConversationState';
import ChatInput from '../components/conversation/ChatInput';

export default function MedicalConversationPage() {
  const { messages, clearConversation, isLoading, loadSessions } = useConversation();

  useEffect(() => {
    if (loadSessions) {
      loadSessions();
    }
  }, [loadSessions]);

  return (
    <div className="conversation-workspace">
      {/* Workspace Sub-Header */}
      <div className="workspace-header">
        <div className="workspace-header-title">
          <div className="workspace-icon-pill">
            <i className="fas fa-comment-medical"></i>
          </div>
          <div>
            <h1 className="workspace-heading">Medical AI Conversation</h1>
            <p className="workspace-subheading">
              Evidence-based clinical assistance powered by Hybrid RAG, Vector Search & PubMed Fallback
            </p>
          </div>
        </div>

        <div className="workspace-header-actions">
          <div className="engine-status-badge">
            <span className="status-dot-pulse"></span>
            <span>Multi-Agent Active</span>
          </div>

          {messages.length > 0 && (
            <button
              type="button"
              className="btn-clear-chat"
              onClick={clearConversation}
              disabled={isLoading}
              title="Clear current conversation"
            >
              <i className="fas fa-trash-alt"></i> Clear Chat
            </button>
          )}
        </div>
      </div>

      {/* Workspace Body */}
      <div className="workspace-body">
        {messages.length === 0 ? (
          <EmptyConversationState />
        ) : (
          <MessageStream />
        )}
      </div>

      {/* Workspace Chat Input Footer */}
      <div className="workspace-footer">
        <ChatInput />
      </div>
    </div>
  );
}
