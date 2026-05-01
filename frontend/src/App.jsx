import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import ResultViewer from './components/ResultViewer'
import ComparisonView from './components/ComparisonView'
import MetricsPanel from './components/MetricsPanel'
import './App.css'

function App() {
  const [imageId, setImageId]         = useState(null)
  const [segResults, setSegResults]   = useState(null)

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-eyebrow">MSc Thesis · Warsaw University of Technology</div>
        <h1>Aerial Image Segmentation</h1>
        <p>YOLOv8 + Segment Anything Model · ICG Semantic Drone Dataset</p>
      </header>

      <main className="app-main">
        <ImageUploader onUploadSuccess={(id) => {
          setImageId(id)
          setSegResults(null)   // reset results on new upload
        }} />

        {imageId && (
          <ResultViewer
            imageId={imageId}
            onSegmentComplete={setSegResults}
          />
        )}

        {segResults && <ComparisonView results={segResults} />}

        {segResults && <MetricsPanel imageId={imageId} />}
      </main>
    </div>
  )
}

export default App