/**
 * IDS page - Main interaction page for CSV upload and predictions
 * Includes file upload, preview, prediction results, and model stats
 */

import { useState, useEffect } from 'react'
import FileUploader from '../components/FileUploader'
import PreviewTable from '../components/PreviewTable'
import ChartCard from '../components/ChartCard'
import MetricsViewer from '../components/MetricsViewer'
import SampleDataViewer from '../components/SampleDataViewer'
import { useToast } from '../utils/toastContext'
import { predict, getModelStats } from '../services/api'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement
)

const IDS = () => {
  const [activeTab, setActiveTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [csvData, setCsvData] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [modelStats, setModelStats] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [requiredFeatures, setRequiredFeatures] = useState([])
  const { addToast } = useToast()

  useEffect(() => {
    // Load required features and model stats
    loadModelInfo()
  }, [])

  const loadModelInfo = async () => {
    try {
      const stats = await getModelStats()
      setModelStats(stats)
      if (stats.features) {
        setRequiredFeatures(stats.features)
      }
    } catch (error) {
      console.warn('Could not load model info:', error)
      // Use default features if API fails
      setRequiredFeatures([])
    }
  }

  const handleFileLoaded = (uploadedFile, previewData) => {
    setFile(uploadedFile)
    setCsvData(previewData)
    setPredictions(null)
  }

  const handlePredict = async () => {
    if (!csvData || csvData.length === 0) {
      addToast('Please upload a CSV file first', 'warning')
      return
    }

    setIsLoading(true)
    try {
      // Convert CSV data to rows format
      const rows = csvData.map((row) => {
        const rowObj = {}
        requiredFeatures.forEach((feat) => {
          rowObj[feat] = parseFloat(row[feat]) || 0
        })
        return rowObj
      })

      const result = await predict(rows, 'batch')
      setPredictions(result.predictions || [])
      addToast(
        `Predictions complete: ${result.summary?.attacks || 0} attacks detected`,
        'success'
      )
    } catch (error) {
      addToast(error.message || 'Prediction failed', 'error')
      console.error('Prediction error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Prepare confusion matrix data
  const confusionMatrixData = modelStats?.confusion_matrix
    ? {
        labels: ['Normal', 'Attack'],
        datasets: [
          {
            label: 'True Negative',
            data: [modelStats.confusion_matrix[0]?.[0] || 0, 0],
            backgroundColor: '#10b981',
          },
          {
            label: 'False Positive',
            data: [modelStats.confusion_matrix[0]?.[1] || 0, 0],
            backgroundColor: '#f59e0b',
          },
          {
            label: 'False Negative',
            data: [0, modelStats.confusion_matrix[1]?.[0] || 0],
            backgroundColor: '#ef4444',
          },
          {
            label: 'True Positive',
            data: [0, modelStats.confusion_matrix[1]?.[1] || 0],
            backgroundColor: '#3b82f6',
          },
        ],
      }
    : null

  // Training accuracy chart
  const accuracyData = modelStats?.training_history
    ? {
        labels: modelStats.training_history.epochs || [],
        datasets: [
          {
            label: 'Training Accuracy',
            data: modelStats.training_history.train_accuracy || [],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
          },
          {
            label: 'Validation Accuracy',
            data: modelStats.training_history.val_accuracy || [],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
          },
        ],
      }
    : null

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12" data-aos="fade-up">
        <h1 className="text-4xl font-bold mb-4">AI-Based Intrusion Detection System</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Upload your network flow data to detect potential intrusions and security threats
        </p>
      </div>

      {/* Tabs for different views */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Upload & Predict
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'metrics'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Model Metrics
            </button>
            <button
              onClick={() => setActiveTab('sample')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'sample'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Sample Data
            </button>
          </nav>
        </div>
      </div>

      {activeTab === 'upload' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Upload Section */}
          <div className="lg:col-span-2 space-y-6">
          {/* File Upload */}
          <div data-aos="fade-up">
            <h2 className="text-2xl font-semibold mb-4">Upload CSV File</h2>
            <FileUploader
              onFileLoaded={handleFileLoaded}
              requiredFeatures={requiredFeatures}
            />
          </div>

          {/* Preview Table */}
          {csvData && (
            <div data-aos="fade-up" data-aos-delay="100">
              <PreviewTable
                data={csvData}
                predictions={predictions}
                onDownload={() => {
                  // Download handled in PreviewTable component
                }}
              />
            </div>
          )}

          {/* Predict Button */}
          {csvData && (
            <div className="flex justify-center" data-aos="fade-up" data-aos-delay="200">
              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="btn-primary text-lg px-8 py-4 relative"
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  '🔍 Run Predictions'
                )}
              </button>
            </div>
          )}

          {/* Results Summary */}
          {predictions && (
            <div className="card" data-aos="fade-up">
              <h3 className="text-xl font-semibold mb-4">Prediction Summary</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Normal</p>
                  <p className="text-2xl font-bold text-green-700">
                    {predictions.filter((p) => p.label === 0 || p.label === 'normal').length}
                  </p>
                </div>
                <div className="bg-red-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Attacks</p>
                  <p className="text-2xl font-bold text-red-700">
                    {predictions.filter((p) => p.label === 1 || p.label === 'attack' || p.label === -1).length}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Model Stats Sidebar */}
        <div className="space-y-6">
          <div data-aos="fade-up">
            <h2 className="text-2xl font-semibold mb-4">Model Statistics</h2>
            
            {modelStats ? (
              <div className="space-y-4">
                <div className="card">
                  <h3 className="font-semibold mb-2">Model Accuracy</h3>
                  <p className="text-3xl font-bold text-primary-600">
                    {((modelStats.accuracy || 0) * 100).toFixed(2)}%
                  </p>
                </div>

                {confusionMatrixData && (
                  <ChartCard title="Confusion Matrix" size="sm">
                    <Bar
                      data={confusionMatrixData}
                      options={{
                        maintainAspectRatio: false,
                        scales: {
                          x: { stacked: true },
                          y: { stacked: true },
                        },
                      }}
                    />
                  </ChartCard>
                )}

                {accuracyData && (
                  <ChartCard title="Training History" size="sm">
                    <Line
                      data={accuracyData}
                      options={{
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: true,
                            max: 1,
                          },
                        },
                      }}
                    />
                  </ChartCard>
                )}
              </div>
            ) : (
              <div className="card text-center text-gray-500">
                Loading model statistics...
              </div>
            )}
          </div>
        </div>
      </div>
      )}

      {activeTab === 'metrics' && (
        <div>
          <MetricsViewer />
        </div>
      )}

      {activeTab === 'sample' && (
        <div>
          <SampleDataViewer />
        </div>
      )}
    </div>
  )
}

export default IDS

