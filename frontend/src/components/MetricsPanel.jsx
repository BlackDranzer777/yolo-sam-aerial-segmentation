/**
 * MetricsPanel.jsx — Displays IoU / Precision / Recall evaluation metrics.
 *
 * Shows a button to trigger evaluation against the ICG ground truth,
 * then renders per-object and overall metrics in a table.
 *
 * Props:
 *   imageId — the image_id used for segmentation (must match ICG dataset filename)
 */

import { useState, useEffect } from 'react'
import { evaluateResults } from '../services/api'
import './MetricsPanel.css'

export default function MetricsPanel({ imageId, datasetId: datasetIdProp }) {
  const [metrics, setMetrics]     = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [datasetId, setDatasetId] = useState(datasetIdProp || '')

  // When component mounts (after segmentation), auto-run if we already know the dataset ID
  useEffect(() => {
    if (datasetIdProp) {
      setDatasetId(datasetIdProp)
      runEvaluation(datasetIdProp)
    }
  }, [])

  async function runEvaluation(icgId) {
    setLoading(true)
    setError(null)
    setMetrics(null)
    try {
      const data = await evaluateResults(imageId, icgId)
      setMetrics(data)
    } catch (err) {
      setError(err.response?.data?.error || 'Evaluation failed.')
    } finally {
      setLoading(false)
    }
  }

  async function handleEvaluate() {
    const icgId = datasetId.trim().padStart(3, '0')
    if (!icgId || isNaN(Number(icgId))) {
      setError('Enter a valid ICG dataset ID (e.g. 001).')
      return
    }
    runEvaluation(icgId)
  }

  return (
    <div className="card metrics-panel">
      <h2>Evaluation Metrics</h2>
      <p className="metrics-note">
        Compares SAM masks against ICG Semantic Drone Dataset ground truth.
        {datasetIdProp
          ? <> Detected image <code>{datasetIdProp}</code> — evaluation runs automatically.</>
          : <> Enter the ICG image number that matches the image you uploaded (e.g.&nbsp;<code>001</code>).</>
        }
      </p>

      <div className="metrics-input-row">
        <input
          className="metrics-id-input"
          type="text"
          placeholder="ICG Image ID (e.g. 001)"
          value={datasetId}
          onChange={e => setDatasetId(e.target.value)}
          maxLength={3}
          disabled={loading}
        />
        <button
          className="btn btn-secondary"
          onClick={handleEvaluate}
          disabled={loading || !datasetId.trim()}
        >
          {loading ? 'Evaluating…' : 'Evaluate Against Ground Truth'}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {metrics && (
        <>
          {/* Overall summary */}
          <div className="metrics-overall">
            <div className="metric-box">
              <span className="metric-label">Mean IoU</span>
              <span className="metric-value">{metrics.overall.mean_iou.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Mean Precision</span>
              <span className="metric-value">{metrics.overall.mean_precision.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Mean Recall</span>
              <span className="metric-value">{metrics.overall.mean_recall.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Masks Evaluated</span>
              <span className="metric-value">{metrics.overall.evaluated_count}</span>
            </div>
          </div>

          {/* Per-object table */}
          <h3>Per-object results</h3>
          <div className="metrics-table-wrap">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Class</th>
                  <th>Conf</th>
                  <th>IoU</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {metrics.per_object.map((obj, i) => (
                  <tr key={i} className={obj.iou === null ? 'skipped' : ''}>
                    <td>{i + 1}</td>
                    <td>{obj.class_name}</td>
                    <td>{obj.confidence != null ? obj.confidence.toFixed(2) : '—'}</td>
                    <td>{obj.iou != null ? obj.iou.toFixed(4) : '—'}</td>
                    <td>{obj.precision != null ? obj.precision.toFixed(4) : '—'}</td>
                    <td>{obj.recall != null ? obj.recall.toFixed(4) : '—'}</td>
                    <td className="note-cell">{obj.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-class summary */}
          {Object.keys(metrics.per_class).length > 0 && (
            <>
              <h3>Per-class summary</h3>
              <div className="metrics-table-wrap">
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Count</th>
                      <th>Mean IoU</th>
                      <th>Mean Precision</th>
                      <th>Mean Recall</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.per_class).map(([cls, m]) => (
                      <tr key={cls}>
                        <td>{cls}</td>
                        <td>{m.count}</td>
                        <td>{m.mean_iou.toFixed(4)}</td>
                        <td>{m.mean_precision.toFixed(4)}</td>
                        <td>{m.mean_recall.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}