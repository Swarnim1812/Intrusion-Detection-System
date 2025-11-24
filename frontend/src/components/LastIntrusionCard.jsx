/**
 * Last Intrusion Detected card component
 * Shows attack type, time, and confidence
 */

const LastIntrusionCard = ({ recentEvent, isLoading }) => {
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A'
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    } catch {
      return timestamp
    }
  }

  const getAttackTypeLabel = (label) => {
    if (label === 1 || label === 'attack' || label === 'Attack') {
      return 'Attack Detected'
    }
    if (label === 0 || label === 'normal' || label === 'Normal') {
      return 'Normal Traffic'
    }
    return label || 'Unknown'
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Last Intrusion Detected</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!recentEvent || !recentEvent.label || recentEvent.label === 0 || recentEvent.label === 'normal') {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Last Intrusion Detected</h3>
        <div className="flex items-center justify-center py-8">
          <div className="text-center">
            <div className="text-5xl mb-3">✅</div>
            <p className="text-gray-600 font-medium">No recent intrusions detected</p>
            <p className="text-sm text-gray-500 mt-2">System is secure</p>
          </div>
        </div>
      </div>
    )
  }

  const confidence = recentEvent.probability
    ? (recentEvent.probability * 100).toFixed(1)
    : 'N/A'

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300 border-l-4 border-red-500">
      <h3 className="text-xl font-bold text-gray-800 mb-4">Last Intrusion Detected</h3>
      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-600 mb-1">Attack Type</p>
          <p className="text-lg font-semibold text-red-600">
            {recentEvent.attack_type || getAttackTypeLabel(recentEvent.label)}
          </p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600 mb-1">Time Detected</p>
          <p className="text-lg font-semibold text-gray-800">
            {formatTime(recentEvent.timestamp)}
          </p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600 mb-1">Confidence</p>
          <div className="flex items-center gap-2">
            <p className="text-lg font-semibold text-gray-800">{confidence}%</p>
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, parseFloat(confidence) || 0)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LastIntrusionCard

