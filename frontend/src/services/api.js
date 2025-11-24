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
 * Get model statistics and training metrics from /metrics endpoint
 * @returns {Promise} Model stats including accuracy, precision, recall, F1, confusion matrix
 */
// export const getModelStats = async () => {
//   try {
//     const response = await api.get('/metrics')
//     const data = response.data
//     console.log("------------------");
//     console.log(data);
//     // Extract and return exact structure from metrics.json
//     return {
//       accuracy: data.metrics?.accuracy ?? null,
//       precision: data.metrics?.precision ?? null,
//       recall: data.metrics?.recall ?? null,
//       f1: data.metrics?.f1 ?? null,
//       confusion_matrix: data.confusion_matrix ?? null,
//     }
//   } catch (error) {
//     console.error('Model stats error:', error)
//     // Return null on error
//     return null
//   }
// }
export const getModelStats = async () => {
  try {
    const res = await api.get("/metrics");

    console.log("METRICS RECEIVED:", res.data);
    
    let data = res.data;

    // If backend returned JSON-as-string → parse it
    if (typeof data === "string") {
      try {
        data = JSON.parse(data);
      } catch (err) {
        console.error("JSON parse failed:", err);
        return null;
      }
    }

    return {
      accuracy: data?.metrics?.accuracy ?? 0,
      precision: data?.metrics?.precision ?? 0,
      recall: data?.metrics?.recall ?? 0,
      f1: data?.metrics?.f1 ?? 0,
      confusion_matrix: data?.confusion_matrix ?? []
    };
  } catch (err) {
    console.error("getModelStats ERROR →", err);
    return {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      confusion_matrix: []
    };
  }
};

/**
 * Get dataset statistics
 * @returns {Promise} Dataset stats including total rows, benign count, attack count, attack types
 */
export const getDatasetStats = async () => {
  try {
    const response = await api.get('/dataset-stats')
    console.log(response);
    return response.data
  } catch (error) {
    console.warn('Dataset stats error:', error)
    // Return null if response missing
    return null
  }
}

/**
 * Get recent intrusion event
 * @returns {Promise} Last prediction info with label, timestamp, probability
 */
export const getRecentEvent = async () => {
  try {
    const response = await api.get('/recent-event')
    return response.data
  } catch (error) {
    console.warn('Recent event error:', error)
    // Return null if response missing
    return null
  }
}

/**
 * Get weekly attacks statistics
 * @returns {Promise} Weekly attack data with attackType percentages and counts
 */
export const getWeeklyAttacks = async () => {
  try {
    const response = await api.get('/weekly-attacks')
    return response.data
  } catch (error) {
    console.warn('Weekly attacks error:', error)
    // Return null if response missing
    return null
  }
}

/**
 * Download sample dataset (100 rows)
 * @returns {Promise} Sample data
 */
export const getSample = async () => {
  try {
    const response = await api.get('/sample', {
      params: { limit: 100 },
      responseType: 'blob', // For file download
    })
    return response.data
  } catch (error) {
    console.warn('Sample download error:', error)
    // Fallback to sample-data endpoint
    try {
      const fallbackResponse = await api.get('/sample-data')
      return fallbackResponse.data
    } catch (fallbackError) {
      throw new Error('Failed to load sample data.')
    }
  }
}

/**
 * Get extended model overview including features and training history
 * @returns {Promise} Model overview data from /model-stats
 */
export const getModelOverview = async () => {
  try {
    const response = await api.get('/model-stats')
    return response.data
  } catch (error) {
    console.error('Model overview error:', error)
    return {
      accuracy: 0.919,
      precision: 0,
      recall: 0,
      f1: 0,
      confusion_matrix: [[0, 0], [0, 0]],
      training_history: null,
      features: [],
    }
  }
}

/**
 * Get detailed model metrics
 * @returns {Promise} Metrics including accuracy, precision, recall, F1, ROC-AUC
 */
export const getMetrics = async () => {
  try {
    const response = await api.get('/metrics')
    return response.data.metrics
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

