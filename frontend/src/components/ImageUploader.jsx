/**
 * ImageUploader.jsx — File selection and upload component.
 *
 * Lets the user pick an aerial image, shows upload progress,
 * and calls onUploadSuccess(imageId) when the backend confirms the upload.
 */

import { useState, useRef } from 'react'
import { uploadImage } from '../services/api'
import './ImageUploader.css'

export default function ImageUploader({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview]           = useState(null)
  const [uploading, setUploading]       = useState(false)
  const [progress, setProgress]         = useState(0)
  const [error, setError]               = useState(null)
  const [uploaded, setUploaded]         = useState(false)
  const inputRef = useRef()

  function handleFileChange(e) {
    const file = e.target.files[0]
    if (!file) return

    setSelectedFile(file)
    setError(null)
    setProgress(0)
    setUploaded(false)

    // Show local preview immediately
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(file)
  }

  async function handleUpload() {
    if (!selectedFile) return

    setUploading(true)
    setError(null)

    try {
      const data = await uploadImage(selectedFile, setProgress)
      setUploaded(true)
      onUploadSuccess(data.image_id, selectedFile.name)
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed. Is the Flask server running?')
    } finally {
      setUploading(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setUploaded(false)
      const reader = new FileReader()
      reader.onload = ev => setPreview(ev.target.result)
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="card">
      <h2>Upload Image</h2>

      {/* Drop zone */}
      <div
        className={`drop-zone ${preview ? 'has-preview' : ''}`}
        onClick={() => inputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
      >
        {preview ? (
          <img src={preview} alt="Selected" className="preview-img" />
        ) : (
          <div className="drop-placeholder">
            <span className="drop-icon">+</span>
            <p>Click or drag an aerial image here</p>
            <p className="drop-hint">JPG, PNG, TIFF supported</p>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.tif,.tiff"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {/* File name */}
      {selectedFile && (
        <p className="file-name">{selectedFile.name}</p>
      )}

      {/* Progress bar */}
      {uploading && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Error */}
      {error && <p className="error">{error}</p>}

      {/* Success */}
      {uploaded && !uploading && (
        <p className="upload-success">Image uploaded successfully. Run detection or segmentation below.</p>
      )}

      {/* Upload button */}
      <button
        className="btn btn-primary"
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        style={{ marginTop: '16px' }}
      >
        {uploading ? `Uploading… ${progress}%` : uploaded ? 'Upload Again' : 'Upload Image'}
      </button>
    </div>
  )
}