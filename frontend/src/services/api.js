/**
 * API service for communicating with Flask backend
 * Handles all HTTP requests and error handling
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
})

/**
 * Make prediction request
 * @param {Array} rows - Array of feature objects
 * @param {string} mode - 'single' or 'batch'
 * @returns {Promise} Prediction results
 */
export const predict = async (rows, mode = 'batch') => {
  try {
    const response = await api.post('/predict', {
      rows,
      mode,
    })
    return response.data
  } catch (error) {
    console.error('Prediction error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to get predictions. Please try again.'
    )
  }
}

/**
 * Upload CSV file for batch processing
 * @param {File} file - CSV file
 * @returns {Promise} Upload response
 */
export const uploadFile = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    console.error('Upload error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to upload file. Please try again.'
    )
  }
}

/**
 * Get model statistics and training metrics
 * @returns {Promise} Model stats including accuracy, confusion matrix, etc.
 */
export const getModelStats = async () => {
  try {
    const response = await api.get('/model-stats')
    return response.data
  } catch (error) {
    console.error('Model stats error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to load model statistics.'
    )
  }
}

/**
 * Get detailed model metrics
 * @returns {Promise} Metrics including accuracy, precision, recall, F1, ROC-AUC
 */
export const getMetrics = async () => {
  try {
    const response = await api.get('/metrics')
    return response.data
  } catch (error) {
    console.error('Metrics error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to load metrics.'
    )
  }
}

/**
 * Get sample training data
 * @returns {Promise} Sample data (raw and processed)
 */
export const getSampleData = async () => {
  try {
    const response = await api.get('/sample-data')
    return response.data
  } catch (error) {
    console.error('Sample data error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to load sample data.'
    )
  }
}

/**
 * Get visualization data based on user selections
 * @param {Object} options - Visualization options
 * @returns {Promise} Chart-ready data
 */
export const getVisualizationData = async (options) => {
  try {
    const response = await api.post('/visualize', options)
    return response.data
  } catch (error) {
    console.error('Visualization error:', error)
    throw new Error(
      error.response?.data?.error || 'Failed to generate visualization.'
    )
  }
}

export default api

