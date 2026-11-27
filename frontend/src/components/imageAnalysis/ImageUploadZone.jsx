import React, { useRef, useState } from 'react';
import { useImageAnalysis } from '../../context/ImageAnalysisContext';
import { useWorkspaceAutoFocus } from '../../hooks/useWorkspaceAutoFocus';

export default function ImageUploadZone() {
  const {
    selectedFile,
    previewUrl,
    clinicalNotes,
    isLoading,
    error,
    sessionId,
    selectFile,
    removeFile,
    setClinicalNotes,
    analyzeImage
  } = useImageAnalysis();

  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const notesInputRef = useRef(null);

  useWorkspaceAutoFocus({
    inputRef: notesInputRef,
    routeMatches: ['/image-analysis'],
    isBusy: isLoading,
    sessionId,
    extraFocusTrigger: selectedFile
  });

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      selectFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      selectFile(e.target.files[0]);
    }
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <div className="image-upload-card">
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/png, image/jpeg, image/jpg"
        style={{ display: 'none' }}
        id="medical-image-file-input"
        data-testid="file-upload-input"
      />

      {/* Independently Scrollable Upload / Preview Content Region */}
      <div className="upload-zone-content">
        {!selectedFile ? (
          <div
            className={`dropzone-container ${isDragOver ? 'dropzone-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleBrowseClick}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleBrowseClick();
              }
            }}
          >
            <div className="dropzone-icon">
              <i className="fas fa-cloud-upload-alt text-primary"></i>
            </div>
            <h3 className="dropzone-title">Upload Medical Image</h3>
            <p className="dropzone-instructions">
              Drag and drop your medical scan here, or <span className="dropzone-highlight">browse files</span>
            </p>
            <div className="dropzone-meta">
              <span className="meta-pill">
                <i className="fas fa-file-image"></i> PNG, JPG, JPEG
              </span>
              <span className="meta-pill">
                <i className="fas fa-weight-hanging"></i> Max 50MB
              </span>
              <span className="meta-pill">
                <i className="fas fa-microscope"></i> Chest X-ray, Skin Lesion, Brain MRI
              </span>
            </div>
          </div>
        ) : (
          /* Selected File Preview Box */
          <div className="selected-file-preview-card">
            <div className="preview-image-wrapper">
              <img
                src={previewUrl}
                alt="Selected medical upload"
                className="preview-thumbnail"
              />
            </div>
            <div className="preview-file-details">
              <div className="preview-file-header">
                <div className="file-name-group">
                  <i className="fas fa-file-medical text-primary"></i>
                  <span className="file-name" title={selectedFile.name}>
                    {selectedFile.name}
                  </span>
                </div>
                <span className="file-size-badge">
                  {formatFileSize(selectedFile.size)}
                </span>
              </div>

              <p className="preview-file-tip">
                Image loaded and verified. Add diagnostic questions or symptoms below, then click Analyze.
              </p>

              <div className="preview-file-actions">
                <button
                  type="button"
                  className="btn-action-replace"
                  onClick={handleBrowseClick}
                  disabled={isLoading}
                >
                  <i className="fas fa-exchange-alt"></i> Replace Image
                </button>
                <button
                  type="button"
                  className="btn-action-remove"
                  onClick={removeFile}
                  disabled={isLoading}
                >
                  <i className="fas fa-trash-alt"></i> Remove
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sticky Bottom Composer: Clinical Notes & Analysis Submit Button */}
      <div className="upload-zone-footer-composer">
        {error && (
          <div className="analysis-error-banner" role="alert">
            <i className="fas fa-exclamation-triangle"></i>
            <span>{error}</span>
          </div>
        )}

        <div className="clinical-notes-section">
          <label htmlFor="clinical-notes-input" className="clinical-notes-label">
            <i className="fas fa-notes-medical text-primary"></i> Clinical Notes or Diagnostic Question{' '}
            <span className="text-muted">(Optional)</span>
          </label>
          <div className="notes-input-row">
            <textarea
              ref={notesInputRef}
              id="clinical-notes-input"
              className="clinical-notes-textarea"
              rows={2}
              value={clinicalNotes}
              onChange={(e) => setClinicalNotes(e.target.value)}
              placeholder="e.g. 54-year-old patient with persistent dry cough and fever. Evaluate for COVID-19 or pneumonia."
              disabled={isLoading}
            />
            <button
              type="button"
              className="btn-primary-analyze"
              onClick={analyzeImage}
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <i className="fas fa-microscope"></i>
                  <span>Analyze Image</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
