import React from 'react';

export default function ClinicalIntakeHeader() {
  return (
    <div className="workspace-header clinical-workspace-header">
      <div className="workspace-header-title">
        <div className="workspace-icon-pill icon-clinical-pill">
          <i className="fas fa-clipboard-user"></i>
        </div>
        <div className="workspace-title-text">
          <h1 className="workspace-heading">Clinical Pre-Consultation Workspace</h1>
          <p className="workspace-subheading">
            5-stage adaptive clinical intake pipeline with emergency red-flag detection and summary reports
          </p>
        </div>
      </div>
    </div>
  );
}

