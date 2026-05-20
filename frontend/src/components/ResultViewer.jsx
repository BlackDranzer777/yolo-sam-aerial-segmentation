/**
 * ResultViewer.jsx — Displays pipeline results and output images.
 *
 * Shows:
 *   - Run Detection (YOLO only) and Run Segmentation (YOLO+SAM) buttons
 *   - Layer toggle to switch between boxes / masks / combined views
 *   - Detection count and class labels
 *   - Output image for the selected layer
 *
 * Props:
 *   imageId — the uploaded image ID from Flask
 */

import { useState, useEffect } from 'react'
import { detectObjects, segmentImage, imageUrl } from '../services/api'
import LayerToggle from './LayerToggle'
import './ResultViewer.css'

const MODELS = [
  { key: 'visdrone', label: 'VisDrone', desc: 'vehicles · aerial pedestrians' },
  { key: 'coco',     label: 'COCO',     desc: 'close-range persons' },
]

export default function ResultViewer({ imageId, onSegmentComplete }) {
  const [results, setResults]           = useState(null)
  const [activeLayer, setActiveLayer]   = useState('combined')
  const [loading, setLoading]           = useState(false)
  const [loadingMsg, setLoadingMsg]     = useState('')
  const [error, setError]               = useState(null)
  const [confidence, setConfidence]     = useState(0.50)
  const [activeModels, setActiveModels] = useState(['visdrone', 'coco'])
  const [resultTs, setResultTs]         = useState(null)

  function toggleModel(key) {
    setActiveModels(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    )
  }

  // Reset all state whenever a new image is uploaded
  useEffect(() => {
    setResults(null)
    setActiveLayer('combined')
    setError(null)
  }, [imageId])

  async function handleDetect() {
    setLoading(true)
    setLoadingMsg('Running YOLO detection…')
    setError(null)
    try {
      const data = await detectObjects(imageId, confidence)
      setResults(data)
      setResultTs(Date.now())
      setActiveLayer('boxes')
    } catch (err) {
      setError(err.response?.data?.error || 'Detection failed.')
    } finally {
      setLoading(false)
    }
  }

  async function handleSegment() {
    setLoading(true)
    setLoadingMsg('Running YOLO → SAM pipeline… (this may take ~30s)')
    setError(null)
    try {
      const data = await segmentImage(imageId, confidence, activeModels)
      const ts = Date.now()
      setResults(data)
      setResultTs(ts)
      setActiveLayer('combined')
      if (onSegmentComplete) onSegmentComplete({ ...data, _ts: ts })
    } catch (err) {
      setError(err.response?.data?.error || 'Segmentation failed.')
    } finally {
      setLoading(false)
    }
  }

  const currentImage = results?.output_paths?.[activeLayer]

  return (
    <div className="card">
      <h2>Results</h2>

      {/* Confidence threshold slider */}
      <div className="confidence-control">
        <label htmlFor="conf-slider">
          Detection Confidence Threshold: <strong>{confidence.toFixed(2)}</strong>
        </label>
        <input
          id="conf-slider"
          type="range"
          min="0.10"
          max="0.90"
          step="0.05"
          value={confidence}
          onChange={e => setConfidence(parseFloat(e.target.value))}
          disabled={loading}
        />
        <div className="conf-labels">
          <span>0.10 (more detections)</span>
          <span>0.90 (fewer, more certain)</span>
        </div>
      </div>

      {/* Model selector */}
      <div className="model-selector">
        <span className="model-selector-label">Detection Models</span>
        <div className="model-checkboxes">
          {MODELS.map(m => (
            <label key={m.key} className={`model-checkbox ${activeModels.includes(m.key) ? 'active' : ''}`}>
              <input
                type="checkbox"
                checked={activeModels.includes(m.key)}
                onChange={() => toggleModel(m.key)}
                disabled={loading}
              />
              <span className="model-checkbox-name">{m.label}</span>
              <span className="model-checkbox-desc">{m.desc}</span>
            </label>
          ))}
        </div>
        {activeModels.length === 0 && (
          <p className="error" style={{ marginTop: 8 }}>Select at least one model.</p>
        )}
      </div>

      {/* Action buttons */}
      <div className="result-actions">
        <button
          className="btn btn-secondary"
          onClick={handleDetect}
          disabled={loading}
        >
          Run Detection (YOLO only)
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSegment}
          disabled={loading || activeModels.length === 0}
        >
          Run Segmentation (YOLO + SAM)
        </button>
      </div>

      {/* Loading state */}
      {loading && <p className="status">{loadingMsg}</p>}

      {/* Error */}
      {error && <p className="error">{error}</p>}

      {/* Results */}
      {results && !loading && (
        <>
          {/* Detection summary */}
          <div className="detection-summary">
            <span className="detection-count">
              {results.detections.length} object(s) detected
            </span>
            <div className="detection-tags">
              {results.detections.map((det, i) => (
                <span key={i} className="tag">
                  {det.class_name} {det.confidence.toFixed(2)}
                </span>
              ))}
            </div>
          </div>

          {/* Layer toggle — disable masks/combined until segmentation is run */}
          <LayerToggle
            activeLayer={activeLayer}
            onChange={setActiveLayer}
            availableLayers={Object.keys(results.output_paths || {})}
          />

          {/* Output image */}
          {currentImage ? (
            <img
              src={`${imageUrl(currentImage)}?t=${resultTs}`}
              alt={`${activeLayer} output`}
              className="result-img"
            />
          ) : (
            <p className="error">
              Run segmentation to see masks and combined view.
            </p>
          )}
        </>
      )}
    </div>
  )
}