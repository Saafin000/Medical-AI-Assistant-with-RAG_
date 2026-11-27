import React, { useState } from 'react';

export default function SideBySideImageViewer({ originalImage, resultImage }) {
  const [modalImage, setModalImage] = useState(null);
  const [modalTitle, setModalTitle] = useState('');

  const openModal = (src, title) => {
    setModalImage(src);
    setModalTitle(title);
  };

  const closeModal = () => {
    setModalImage(null);
    setModalTitle('');
  };

  return (
    <div className="side-by-side-section">
      <div className="side-by-side-header">
        <h4>
          <i className="fas fa-columns text-primary"></i> Side-by-Side Visual Comparison
        </h4>
        <p className="side-by-side-subtitle">
          Compare the patient's original dermatological lesion against the neural network segmentation mask.
        </p>
      </div>

      <div className="side-by-side-grid">
        {/* Original Image Card */}
        <div className="image-card">
          <div className="image-card-header">
            <span className="image-card-badge original-badge">
              <i className="fas fa-camera"></i> Original Clinical Image
            </span>
            <button
              type="button"
              className="btn-image-zoom"
              onClick={() => openModal(originalImage, 'Original Clinical Image')}
              title="Click to enlarge"
            >
              <i className="fas fa-search-plus"></i> Enlarge
            </button>
          </div>
          <div 
            className="image-card-body"
            onClick={() => openModal(originalImage, 'Original Clinical Image')}
          >
            {originalImage ? (
              <img
                src={originalImage}
                alt="Original dermatological scan"
                className="comparison-img"
              />
            ) : (
              <div className="image-placeholder-box">
                <i className="fas fa-image"></i>
                <span>Original image unavailable</span>
              </div>
            )}
          </div>
          <div className="image-card-footer">
            <small>Input submitted by clinician/patient</small>
          </div>
        </div>

        {/* Model Output Image Card */}
        <div className="image-card">
          <div className="image-card-header">
            <span className="image-card-badge result-badge">
              <i className="fas fa-draw-polygon"></i> Segmentation Mask Result
            </span>
            <button
              type="button"
              className="btn-image-zoom"
              onClick={() => openModal(resultImage, 'Model Segmentation Mask Result')}
              title="Click to enlarge"
            >
              <i className="fas fa-search-plus"></i> Enlarge
            </button>
          </div>
          <div 
            className="image-card-body"
            onClick={() => openModal(resultImage, 'Model Segmentation Mask Result')}
          >
            {resultImage ? (
              <img
                src={resultImage}
                alt="Model segmentation mask plot"
                className="comparison-img"
              />
            ) : (
              <div className="image-placeholder-box">
                <i className="fas fa-spinner fa-spin"></i>
                <span>Generating segmentation plot...</span>
              </div>
            )}
          </div>
          <div className="image-card-footer">
            <small>PyTorch UNet semantic boundary extraction</small>
          </div>
        </div>
      </div>

      {/* Lightbox / Enlarged View Modal */}
      {modalImage && (
        <div className="image-modal-backdrop" onClick={closeModal}>
          <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="image-modal-header">
              <h5>{modalTitle}</h5>
              <button
                type="button"
                className="btn-modal-close"
                onClick={closeModal}
                aria-label="Close"
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="image-modal-body">
              <img src={modalImage} alt={modalTitle} className="modal-enlarged-img" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
