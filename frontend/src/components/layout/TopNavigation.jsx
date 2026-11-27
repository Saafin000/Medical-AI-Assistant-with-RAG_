import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';

export default function TopNavigation() {
  const navigate = useNavigate();

  return (
    <header className="top-nav">
      <div 
        className="brand-section" 
        onClick={() => navigate('/')} 
        title="Go to Home"
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter') navigate('/'); }}
      >
        <i className="fas fa-stethoscope"></i>
        <span>Multi-Agent Medical Assistant</span>
      </div>

      <nav aria-label="Main Workspace Navigation">
        <ul className="nav-links">
          <li className="nav-item">
            <NavLink 
              to="/" 
              end
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <i className="fas fa-home"></i>
              <span>Home</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink 
              to="/conversation" 
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <i className="fas fa-comment-medical"></i>
              <span>Medical Conversation</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink 
              to="/image-analysis" 
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <i className="fas fa-x-ray"></i>
              <span>Image Analysis</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink 
              to="/clinical-pre-consultation" 
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <i className="fas fa-clipboard-user"></i>
              <span>Clinical Pre-Consultation</span>
            </NavLink>
          </li>
        </ul>
      </nav>

      <div className="nav-actions">
        <span className="badge-status" title="FastAPI Backend Connected">
          Active
        </span>
      </div>
    </header>
  );
}
