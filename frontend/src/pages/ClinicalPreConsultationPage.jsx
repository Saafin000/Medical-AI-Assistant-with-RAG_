import React, { useEffect } from 'react';
import { useClinicalPreConsultation } from '../context/ClinicalPreConsultationContext';
import ClinicalIntakeHeader from '../components/clinical/ClinicalIntakeHeader';
import ClinicalMessageStream from '../components/clinical/ClinicalMessageStream';
import ClinicalInputBar from '../components/clinical/ClinicalInputBar';

export default function ClinicalPreConsultationPage() {
  const { loadSessions } = useClinicalPreConsultation();

  useEffect(() => {
    if (loadSessions) {
      loadSessions();
    }
  }, [loadSessions]);
  return (
    <div className="clinical-workspace-page">
      {/* Accessible Fixed/Sticky Header */}
      <ClinicalIntakeHeader />

      {/* Independently Scrollable Message Stream Body */}
      <div className="clinical-workspace-body">
        <ClinicalMessageStream />
      </div>

      {/* Bottom-Oriented Sticky Composer Bar */}
      <div className="clinical-workspace-footer">
        <ClinicalInputBar />
      </div>
    </div>
  );
}
