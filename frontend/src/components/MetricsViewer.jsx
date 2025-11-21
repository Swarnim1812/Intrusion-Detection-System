/**
 * Metrics Viewer component
 * Displays model metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
 */

import { useState, useEffect } from 'react'
import { getMetrics, getModelStats } from '../services/api'
import { useToast } from '../utils/toastContext'
import ChartCard from './ChartCard'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js'
import { Bar, Pie, Doughnut } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
)

const MetricsViewer = () => {
  const [metrics, setMetrics] = useState(null)
  const [modelStats, setModelStats] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const { addToast } = useToast()

  useEffect(() => {
    loadMetrics()
  }, [])

  const loadMetrics = async () => {
    setIsLoading(true)
    try {
      const [metricsData, statsData] = await Promise.all([
        getMetrics(),
        getModelStats(),
      ])
      setMetrics(metricsData)
      setModelStats(statsData)
    } catch (error) {
      addToast('Failed to load metrics', 'error')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="card text-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading metrics...</p>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="card text-center py-8 text-gray-500">
        No metrics available
      </div>
    )
  }

  // Metrics bar chart
  const metricsChartData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [
      {
        label: 'Score',
        data: [
          (metrics.accuracy || 0) * 100,
          (metrics.precision || 0) * 100,
          (metrics.recall || 0) * 100,
          (metrics.f1 || 0) * 100,
        ],
        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
      },
    ],
  }

  // Confusion matrix data
  const confusionMatrix = modelStats?.confusion_matrix || [[0, 0], [0, 0]]
  const cmData = {
    labels: ['Normal', 'Attack'],
    datasets: [
      {
        label: 'True Negative',
        data: [confusionMatrix[0]?.[0] || 0, 0],
        backgroundColor: '#10b981',
      },
      {
        label: 'False Positive',
        data: [confusionMatrix[0]?.[1] || 0, 0],
        backgroundColor: '#f59e0b',
      },
      {
        label: 'False Negative',
        data: [0, confusionMatrix[1]?.[0] || 0],
        backgroundColor: '#ef4444',
      },
      {
        label: 'True Positive',
        data: [0, confusionMatrix[1]?.[1] || 0],
        backgroundColor: '#3b82f6',
      },
    ],
  }

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-2">Accuracy</p>
          <p className="text-3xl font-bold text-primary-600">
            {((metrics.accuracy || 0) * 100).toFixed(2)}%
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-2">Precision</p>
          <p className="text-3xl font-bold text-green-600">
            {((metrics.precision || 0) * 100).toFixed(2)}%
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-2">Recall</p>
          <p className="text-3xl font-bold text-yellow-600">
            {((metrics.recall || 0) * 100).toFixed(2)}%
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-2">F1 Score</p>
          <p className="text-3xl font-bold text-red-600">
            {((metrics.f1 || 0) * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {metrics.roc_auc && (
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-2">ROC-AUC Score</p>
          <p className="text-3xl font-bold text-purple-600">
            {(metrics.roc_auc * 100).toFixed(2)}%
          </p>
        </div>
      )}

      {/* Metrics Bar Chart */}
      <ChartCard
        title="Model Performance Metrics"
        description="Comparison of accuracy, precision, recall, and F1 score"
      >
        <Bar
          data={metricsChartData}
          options={{
            maintainAspectRatio: false,
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                  callback: function (value) {
                    return value + '%'
                  },
                },
              },
            },
          }}
        />
      </ChartCard>

      {/* Confusion Matrix */}
      <ChartCard
        title="Confusion Matrix"
        description="True/False Positive/Negative predictions"
      >
        <Bar
          data={cmData}
          options={{
            maintainAspectRatio: false,
            responsive: true,
            scales: {
              x: { stacked: true },
              y: { stacked: true },
            },
          }}
        />
      </ChartCard>
    </div>
  )
}

export default MetricsViewer

