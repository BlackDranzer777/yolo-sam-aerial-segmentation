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

export default function ResultViewer({ imageId, onSegmentComplete }) {
  const [results, setResults]         = useState(null)
  const [activeLayer, setActiveLayer] = useState('combined')
  const [loading, setLoading]         = useState(false)
  const [loadingMsg, setLoadingMsg]   = useState('')
  const [error, setError]             = useState(null)

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
      const data = await detectObjects(imageId)
      setResults(data)
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
      const data = await segmentImage(imageId)
      setResults(data)
      setActiveLayer('combined')
      if (onSegmentComplete) onSegmentComplete(data)
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
          disabled={loading}
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
              src={imageUrl(currentImage)}
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