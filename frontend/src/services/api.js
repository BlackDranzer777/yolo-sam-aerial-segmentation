/**
 * services/api.js — All Axios calls to the Flask backend.
 *
 * Every function returns the response data directly so components
 * don't need to know about Axios internals.
 */

import axios from 'axios'

const BASE = '/api'

/**
 * Upload an image file.
 * @param {File} file
 * @param {function} onProgress  — optional callback(percent)
 * @returns {{ image_id, filename }}
 */
export async function uploadImage(file, onProgress) {
  const formData = new FormData()
  formData.append('image', file)

  const res = await axios.post(`${BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress) onProgress(Math.round((e.loaded * 100) / e.total))
    }
  })
  return res.data
}

/**
 * Run YOLO detection only (no SAM).
 * @param {string} imageId
 * @returns {{ image_id, detections, output_paths }}
 */
export async function detectObjects(imageId) {
  const res = await axios.post(`${BASE}/detect/${imageId}`)
  return res.data
}

/**
 * Run full YOLO → SAM pipeline.
 * @param {string} imageId
 * @returns {{ image_id, detections, seg_results, output_paths }}
 */
export async function segmentImage(imageId) {
  const res = await axios.post(`${BASE}/segment/${imageId}`)
  return res.data
}

/**
 * Get previously computed results for an image.
 * @param {string} imageId
 */
export async function getResults(imageId) {
  const res = await axios.get(`${BASE}/results/${imageId}`)
  return res.data
}

/**
 * Evaluate SAM masks against ICG ground truth for a dataset image.
 * Only works when image_id matches an ICG dataset filename (e.g. "001").
 * @param {string} imageId
 * @returns {{ per_object, per_class, overall }}
 */
export async function evaluateResults(imageId, datasetId) {
  const res = await axios.post(`${BASE}/evaluate/${imageId}`, { dataset_id: datasetId })
  return res.data
}

/**
 * Build the full URL for a saved output image filename.
 * @param {string} filename  e.g. "abc123_combined.jpg"
 * @returns {string}
 */
export function imageUrl(filename) {
  return `${BASE}/image/${filename}`
}