import React from 'react';
import { Outlet } from 'react-router-dom';
import TopNavigation from './TopNavigation.jsx';
import Sidebar from './Sidebar.jsx';
import { useConversation } from '../../context/ConversationContext.jsx';

export default function AppLayout() {
  const { clearConversation } = useConversation();

  const handleClearChat = () => {
    clearConversation();
    window.dispatchEvent(new CustomEvent('medical-assistant:clear-active-view'));
  };

  return (
    <div className="app-container">
      <TopNavigation />
      
      <main className="app-main">
        <Sidebar onClearChat={handleClearChat} />
        
        <section className="content-wrapper">
          <Outlet />
        </section>
      </main>
    </div>
  );
}
