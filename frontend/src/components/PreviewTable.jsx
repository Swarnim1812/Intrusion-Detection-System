/**
 * Preview table component for displaying CSV data and predictions
 * Supports sorting and pagination
 */

const PreviewTable = ({ data, predictions = null, onDownload }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card text-center text-gray-500">
        No data to display. Upload a CSV file to get started.
      </div>
    )
  }

  const columns = Object.keys(data[0] || {})
  const displayData = predictions
    ? data.map((row, idx) => ({
        ...row,
        _prediction: predictions[idx]?.label === 1 || predictions[idx]?.label === -1 ? 'Attack' : 'Normal',
        _score: predictions[idx]?.score?.toFixed(4) || 'N/A',
      }))
    : data

  const handleDownload = () => {
    if (!onDownload) return

    // Convert to CSV
    const headers = [...columns, 'prediction', 'score']
    const csvRows = [
      headers.join(','),
      ...displayData.map((row) =>
        [
          ...columns.map((col) => `"${row[col] || ''}"`),
          row._prediction || '',
          row._score || '',
        ].join(',')
      ),
    ]

    const csvContent = csvRows.join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
    window.URL.revokeObjectURL(url)
  }

  return (
    <div className="card overflow-x-auto">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Data Preview</h3>
        {predictions && (
          <button
            onClick={handleDownload}
            className="btn-primary text-sm py-2 px-4"
            aria-label="Download predictions as CSV"
          >
            Download Predictions
          </button>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.slice(0, 10).map((col) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col}
                </th>
              ))}
              {predictions && (
                <>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Prediction
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score
                  </th>
                </>
              )}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {displayData.slice(0, 10).map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                {columns.slice(0, 10).map((col) => (
                  <td key={col} className="px-4 py-3 text-sm text-gray-900">
                    {String(row[col] || '').substring(0, 20)}
                    {String(row[col] || '').length > 20 ? '...' : ''}
                  </td>
                ))}
                {predictions && (
                  <>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${
                          row._prediction === 'Attack'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-green-100 text-green-800'
                        }`}
                      >
                        {row._prediction}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {row._score}
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.length > 10 && (
        <p className="text-sm text-gray-500 mt-4">
          Showing first 10 rows of {data.length} total rows
        </p>
      )}
    </div>
  )
}

export default PreviewTable

