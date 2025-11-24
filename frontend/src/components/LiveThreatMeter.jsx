/**
 * Live Threat Meter component with circular gauge
 * Shows threat level: LOW / MEDIUM / HIGH
 */

const LiveThreatMeter = ({ threatLevel, threatPercentage }) => {
  // Determine threat level and color
  const getThreatInfo = () => {
    if (threatPercentage >= 50) {
      return { level: 'HIGH', color: 'text-red-600', bgColor: 'bg-red-100', gaugeColor: '#ef4444' }
    } else if (threatPercentage >= 20) {
      return { level: 'MEDIUM', color: 'text-yellow-600', bgColor: 'bg-yellow-100', gaugeColor: '#f59e0b' }
    } else {
      return { level: 'LOW', color: 'text-green-600', bgColor: 'bg-green-100', gaugeColor: '#10b981' }
    }
  }

  const threatInfo = getThreatInfo()
  const percentage = Math.min(100, Math.max(0, threatPercentage || 0))
  
  // Calculate SVG arc for circular gauge
  const radius = 60
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
      <h3 className="text-xl font-bold text-gray-800 mb-4">Live Threat Meter</h3>
      <div className="flex flex-col items-center">
        {/* Circular Gauge */}
        <div className="relative w-40 h-40 mb-4">
          <svg className="transform -rotate-90 w-40 h-40">
            {/* Background circle */}
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke="#e5e7eb"
              strokeWidth="12"
              fill="none"
            />
            {/* Progress circle */}
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke={threatInfo.gaugeColor}
              strokeWidth="12"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              className="transition-all duration-500"
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-3xl font-bold ${threatInfo.color}`}>
              {percentage.toFixed(1)}%
            </span>
            <span className="text-sm text-gray-500 mt-1">Threat Level</span>
          </div>
        </div>
        
        {/* Threat Level Badge */}
        <div className={`px-4 py-2 rounded-full ${threatInfo.bgColor} ${threatInfo.color} font-semibold`}>
          {threatInfo.level}
        </div>
        
        <p className="text-sm text-gray-600 mt-3 text-center">
          Based on last 24 hours
        </p>
      </div>
    </div>
  )
}

export default LiveThreatMeter

