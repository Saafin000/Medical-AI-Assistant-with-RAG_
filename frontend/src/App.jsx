import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConversationProvider } from './context/ConversationContext.jsx';
import { ImageAnalysisProvider } from './context/ImageAnalysisContext.jsx';
import { ClinicalPreConsultationProvider } from './context/ClinicalPreConsultationContext.jsx';
import AppLayout from './components/layout/AppLayout.jsx';
import HomePage from './pages/HomePage.jsx';
import MedicalConversationPage from './pages/MedicalConversationPage.jsx';
import MedicalImageAnalysisPage from './pages/MedicalImageAnalysisPage.jsx';
import ClinicalPreConsultationPage from './pages/ClinicalPreConsultationPage.jsx';
import './assets/styles/main.css';

function App() {
  return (
    <ConversationProvider>
      <ImageAnalysisProvider>
        <ClinicalPreConsultationProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<AppLayout />}>
                {/* Default Home route displaying Workspace Cards */}
                <Route index element={<HomePage />} />
                
                {/* Medical AI Conversation Workspace */}
                <Route path="conversation" element={<MedicalConversationPage />} />
                
                {/* Medical Image Analysis Workspace */}
                <Route path="image-analysis" element={<MedicalImageAnalysisPage />} />
                
                {/* Clinical Pre-Consultation Workspace */}
                <Route path="clinical-pre-consultation" element={<ClinicalPreConsultationPage />} />
                <Route path="pre-consultation" element={<ClinicalPreConsultationPage />} />
                
                {/* Catch-all redirect to Home */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ClinicalPreConsultationProvider>
      </ImageAnalysisProvider>
    </ConversationProvider>
  );
}

export default App;
