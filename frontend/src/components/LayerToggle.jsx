/**
 * LayerToggle.jsx — Toggle buttons to switch between output views.
 *
 * Props:
 *   activeLayer  — "boxes" | "masks" | "combined"
 *   onChange(layer) — called when user selects a different layer
 */

import './LayerToggle.css'

const LAYERS = [
  { id: 'boxes',    label: 'YOLO Boxes' },
  { id: 'masks',    label: 'SAM Masks' },
  { id: 'combined', label: 'Combined' },
]

/**
 * Props:
 *   activeLayer   — "boxes" | "masks" | "combined"
 *   onChange(layer) — called when user selects a layer
 *   availableLayers — array of layer ids that have output images
 */
export default function LayerToggle({ activeLayer, onChange, availableLayers }) {
  return (
    <div className="layer-toggle">
      {LAYERS.map(layer => {
        const available = !availableLayers || availableLayers.includes(layer.id)
        return (
          <button
            key={layer.id}
            className={`toggle-btn ${activeLayer === layer.id ? 'active' : ''} ${!available ? 'disabled' : ''}`}
            onClick={() => available && onChange(layer.id)}
            title={!available ? 'Run segmentation to enable this view' : ''}
          >
            {layer.label}
          </button>
        )
      })}
    </div>
  )
}