import React, { useEffect, useRef } from 'react';
import MessageItem from './MessageItem';
import { useConversation } from '../../context/ConversationContext';

export default function MessageStream() {
  const { messages, isLoading } = useConversation();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="message-stream-container">
      <div className="message-stream">
        {messages.map((message) => (
          <MessageItem key={message.id} message={message} />
        ))}

        {isLoading && (
          <div className="message-wrapper message-wrapper-assistant">
            <div className="message-bubble message-bubble-assistant thinking-bubble">
              <div className="thinking-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="thinking-text">Consulting medical AI agents...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} className="stream-bottom-anchor" />
      </div>
    </div>
  );
}
