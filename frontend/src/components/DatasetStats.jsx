/**
 * CICIDS2017 Dataset Visualization component
 * Shows dataset statistics and download button
 */

import { getSample } from '../services/api'

const DatasetStats = ({ datasetStats, isLoading }) => {
  const handleDownload = async () => {
    try {
      const data = await getSample()
      
      // If it's a blob (file download)
      if (data instanceof Blob) {
        const url = window.URL.createObjectURL(data)
        const a = document.createElement('a')
        a.href = url
        a.download = 'cicids2017_sample_100_rows.csv'
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      } else {
        // If it's JSON data, convert to CSV
        const csv = convertToCSV(data)
        const blob = new Blob([csv], { type: 'text/csv' })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'cicids2017_sample_100_rows.csv'
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Download error:', error)
      alert('Failed to download sample. Please try again.')
    }
  }

  const convertToCSV = (data) => {
    if (!data || !data.raw) return ''
    
    const rows = data.raw.slice(0, 100)
    if (rows.length === 0) return ''
    
    const headers = Object.keys(rows[0])
    const csvRows = [
      headers.join(','),
      ...rows.map(row => headers.map(header => JSON.stringify(row[header] || '')).join(','))
    ]
    
    return csvRows.join('\n')
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
        <h3 className="text-xl font-bold text-gray-800 mb-4">CICIDS2017 Dataset</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  const stats = datasetStats || {}
  const totalRows = stats.total_rows || 0
  const benignCount = stats.benign_count || 0
  const attackCount = stats.attack_count || 0
  const attackTypes = stats.attack_types || []

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
      <h3 className="text-xl font-bold text-gray-800 mb-4">CICIDS2017 Dataset</h3>
      
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">Total Rows</p>
            <p className="text-2xl font-bold text-blue-600">
              {totalRows.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">Benign Count</p>
            <p className="text-2xl font-bold text-green-600">
              {benignCount.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-red-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">Attack Count</p>
            <p className="text-2xl font-bold text-red-600">
              {attackCount.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">Attack Types</p>
            <p className="text-2xl font-bold text-purple-600">
              {attackTypes.length || 0}
            </p>
          </div>
        </div>

        {attackTypes.length > 0 && (
          <div>
            <p className="text-sm text-gray-600 mb-2">Attack Types:</p>
            <div className="flex flex-wrap gap-2">
              {attackTypes.slice(0, 5).map((type, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"
                >
                  {type}
                </span>
              ))}
              {attackTypes.length > 5 && (
                <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                  +{attackTypes.length - 5} more
                </span>
              )}
            </div>
          </div>
        )}

        <button
          onClick={handleDownload}
          className="w-full bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors duration-200 flex items-center justify-center gap-2"
        >
          <span>📥</span>
          <span>Download Sample (10 rows)</span>
        </button>
      </div>
    </div>
  )
}

export default DatasetStats

