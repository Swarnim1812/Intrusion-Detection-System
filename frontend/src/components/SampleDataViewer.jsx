/**
 * Sample Data Viewer component
 * Shows top 10 rows of training dataset with toggle between raw and processed data
 */

import { useState, useEffect } from 'react'
import { getSampleData } from '../services/api'
import { useToast } from '../utils/toastContext'

const SampleDataViewer = () => {
  const [data, setData] = useState(null)
  const [isRaw, setIsRaw] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const { addToast } = useToast()

  useEffect(() => {
    loadSampleData()
  }, [])

  const loadSampleData = async () => {
    setIsLoading(true)
    try {
      const result = await getSampleData()
      setData(result)
    } catch (error) {
      addToast('Failed to load sample data', 'error')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="card text-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading sample data...</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="card text-center py-8 text-gray-500">
        No sample data available
      </div>
    )
  }

  const displayData = isRaw ? data.raw : data.processed
  const columns = data.feature_names || Object.keys(displayData[0] || {})

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">View Sample Data</h3>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">Data View:</span>
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setIsRaw(true)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                isRaw
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Raw
            </button>
            <button
              onClick={() => setIsRaw(false)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                !isRaw
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Processed
            </button>
          </div>
        </div>
      </div>

      <div className="mb-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          {isRaw
            ? '📊 Raw data: Original feature values before preprocessing (scaling, encoding, etc.)'
            : '⚙️ Processed data: Features after scaling and encoding, ready for model input'}
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Row
              </th>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {displayData.slice(0, 10).map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-3 text-sm font-medium text-gray-900">
                  {idx + 1}
                </td>
                {columns.map((col) => (
                  <td key={col} className="px-4 py-3 text-sm text-gray-900">
                    {typeof row[col] === 'number'
                      ? row[col].toFixed(4)
                      : String(row[col] || '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-sm text-gray-500 mt-4">
        Showing top 10 rows of training dataset ({displayData.length} total rows available)
      </p>
    </div>
  )
}

export default SampleDataViewer

