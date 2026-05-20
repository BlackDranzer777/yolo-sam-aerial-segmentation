/**
 * ComparisonView.jsx — Side-by-side YOLO-only vs YOLO+SAM comparison.
 *
 * Directly supports Research Question 2:
 * "What is the quantitative improvement of YOLO+SAM over YOLO detection alone?"
 *
 * Props:
 *   results — pipeline result object from Flask /api/segment
 */

import { imageUrl } from '../services/api'
import './ComparisonView.css'

export default function ComparisonView({ results }) {
  if (!results) return null

  const { detections, seg_results, output_paths } = results

  // Count unique classes
  const classCounts = detections.reduce((acc, det) => {
    acc[det.class_name] = (acc[det.class_name] || 0) + 1
    return acc
  }, {})

  return (
    <div className="card">
      <h2>Comparison View</h2>
      <p className="comparison-subtitle">
        YOLO detection only &nbsp;vs&nbsp; YOLO + SAM segmentation
      </p>

      <div className="comparison-grid">

        {/* Left — YOLO only */}
        <div className="comparison-panel">
          <div className="panel-label yolo-label">YOLO Detection</div>
          <img
            src={`${imageUrl(output_paths.boxes)}?t=${results._ts || 0}`}
            alt="YOLO boxes"
            className="comparison-img"
          />
          <div className="panel-stats">
            <span className="stat-count">{detections.length} objects detected</span>
            <div className="class-breakdown">
              {Object.entries(classCounts).map(([cls, count]) => (
                <span key={cls} className="class-chip">
                  {cls}: {count}
                </span>
              ))}
            </div>
            <p className="panel-note">
              Output: bounding boxes only
            </p>
          </div>
        </div>

        {/* Right — YOLO + SAM */}
        <div className="comparison-panel">
          <div className="panel-label sam-label">YOLO + SAM</div>
          <img
            src={`${imageUrl(output_paths.combined)}?t=${results._ts || 0}`}
            alt="YOLO + SAM combined"
            className="comparison-img"
          />
          <div className="panel-stats">
            <span className="stat-count">{seg_results?.length ?? 0} masks generated</span>
            <div className="class-breakdown">
              {Object.entries(classCounts).map(([cls, count]) => (
                <span key={cls} className="class-chip sam-chip">
                  {cls}: {count}
                </span>
              ))}
            </div>
            <p className="panel-note">
              Output: pixel-level segmentation masks
            </p>
          </div>
        </div>

      </div>
    </div>
  )
}