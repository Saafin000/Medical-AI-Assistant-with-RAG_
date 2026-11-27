import React from 'react';
import { useImageAnalysis } from '../../context/ImageAnalysisContext';
import { renderMarkdown } from '../../utils/markdown';
import AgentBadge from '../conversation/AgentBadge';
import SideBySideImageViewer from './SideBySideImageViewer';
import HumanValidationCard from './HumanValidationCard';

export default function AnalysisResultView() {
  const {
    previewUrl,
    selectedFile,
    analysisResult,
    clearAnalysis,
    playSpeech,
    activeAudioId,
    isAudioPlaying,
    isAudioLoading
  } = useImageAnalysis();

  if (!analysisResult) return null;

  const agentUpper = (analysisResult.agent || '').toUpperCase();
  const isBrainMRI = agentUpper.includes('BRAIN');
  const isHumanValidationRequired = agentUpper.includes('HUMAN_VALIDATION');
  const hasResultImage = Boolean(analysisResult.resultImage);

  return (
    <div className="analysis-result-view">
      {/* Independently Scrollable Results Content */}
      <div className="analysis-result-content">
        {/* Result Status Header */}
        <div className="result-header-card">
          <div className="result-header-left">
            <AgentBadge agent={analysisResult.agent} />
            {analysisResult.timestamp && (
              <span className="result-timestamp">
                <i className="far fa-clock"></i> {analysisResult.timestamp}
              </span>
            )}
            <span className="analysis-status-pill">
              <i className="fas fa-check-circle text-success"></i> Analysis Completed
            </span>
          </div>

          <button
            type="button"
            className="btn-new-analysis-header"
            onClick={clearAnalysis}
            title="Clear and analyze another image"
          >
            <i className="fas fa-plus-circle"></i>
            <span>New Analysis</span>
          </button>
        </div>

        {/* Brain MRI Placeholder Notice */}
        {isBrainMRI && (
          <div className="placeholder-agent-notice">
            <div className="notice-icon">
              <i className="fas fa-info-circle"></i>
            </div>
            <div className="notice-content">
              <h5>Brain MRI Diagnostic Node (Architectural Placeholder)</h5>
              <p>
                As configured in the multi-agent backend graph, the Brain Tumor detection pipeline is currently operating in placeholder mode. The response below reflects the authentic agent output and mandates human verification.
              </p>
            </div>
          </div>
        )}

        {/* Main Grid: Visual Scans & Diagnostic Report */}
        <div className="result-main-grid">
          {/* Visual Inspection Section */}
          {hasResultImage ? (
            <SideBySideImageViewer
              originalImage={previewUrl}
              resultImage={analysisResult.resultImage}
            />
          ) : (
            <div className="single-image-inspection-card">
              <div className="inspection-card-header">
                <span className="badge-uploaded-scan">
                  <i className="fas fa-x-ray"></i> Analyzed Medical Scan
                </span>
                {selectedFile?.name && (
                  <span className="file-name-meta" title={selectedFile.name}>
                    {selectedFile.name}
                  </span>
                )}
              </div>
              <div className="inspection-card-body">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Analyzed medical scan"
                    className="inspection-image"
                  />
                ) : (
                  <div className="inspection-placeholder">
                    <i className="fas fa-file-image"></i>
                    <span>Scan Preview</span>
                  </div>
                )}
              </div>
              <div className="inspection-card-footer">
                <small>Processed by {analysisResult.agent}</small>
              </div>
            </div>
          )}

          {/* Findings and Diagnostic Report Card */}
          <div className="diagnostic-report-card">
            <div className="report-card-header">
              <h4>
                <i className="fas fa-file-medical-alt text-primary"></i> Diagnostic Findings & Report
              </h4>

              {/* TTS Audio Button */}
              <div className="report-audio-controls">
                <button
                  type="button"
                  className={`btn-tts-play ${isAudioPlaying && activeAudioId === 'analysis-main' ? 'btn-tts-playing' : ''}`}
                  onClick={() => playSpeech('analysis-main', analysisResult.response)}
                  disabled={isAudioLoading}
                  aria-label="Play voice response"
                >
                  {isAudioLoading && activeAudioId === 'analysis-main' ? (
                    <>
                      <i className="fas fa-spinner fa-spin"></i>
                      <span>Generating Audio...</span>
                    </>
                  ) : isAudioPlaying && activeAudioId === 'analysis-main' ? (
                    <>
                      <i className="fas fa-pause"></i>
                      <span>Pause Speech</span>
                    </>
                  ) : (
                    <>
                      <i className="fas fa-play"></i>
                      <span>Play Voice Response</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            <div
              className="markdown-content report-body"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(analysisResult.response) }}
            />
          </div>
        </div>

        {/* Human Validation Card (if required) */}
        {isHumanValidationRequired && (
          <div className="human-validation-wrapper">
            <HumanValidationCard />
          </div>
        )}
      </div>

      {/* Sticky Bottom Actions Bar */}
      <div className="result-footer-bar">
        <button
          type="button"
          className="btn-start-new-analysis"
          onClick={clearAnalysis}
        >
          <i className="fas fa-redo-alt"></i>
          <span>Upload & Analyze Another Medical Image</span>
        </button>
      </div>
    </div>
  );
}
