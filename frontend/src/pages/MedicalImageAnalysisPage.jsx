import React, { useEffect } from 'react';
import { useImageAnalysis } from '../context/ImageAnalysisContext';
import ImageUploadZone from '../components/imageAnalysis/ImageUploadZone';
import AnalysisResultView from '../components/imageAnalysis/AnalysisResultView';

export default function MedicalImageAnalysisPage() {
  const { analysisResult, loadSessions } = useImageAnalysis();

  useEffect(() => {
    if (loadSessions) {
      loadSessions();
    }
  }, [loadSessions]);

  return (
    <div className="image-analysis-workspace">
      {/* Workspace Header */}
      <div className="workspace-header image-workspace-header">
        <div className="workspace-header-title">
          <div className="workspace-icon-pill icon-image-pill">
            <i className="fas fa-x-ray"></i>
          </div>
          <div className="workspace-title-text">
            <h1 className="workspace-heading">Medical Image Analysis Workspace</h1>
            <p className="workspace-subheading">
              Automated multi-agent diagnostic computer vision with clinical guardrails and Human-in-the-Loop verification
            </p>
          </div>
        </div>
      </div>

      {/* Main Workspace Body */}
      <main className="image-workspace-body">
        {!analysisResult ? (
          <ImageUploadZone />
        ) : (
          <AnalysisResultView />
        )}
      </main>
    </div>
  );
}
