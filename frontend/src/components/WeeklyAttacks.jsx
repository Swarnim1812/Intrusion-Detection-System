/**
 * Top 5 Attacks of the Week component
 * Displays weekly attack statistics visually
 */

const WeeklyAttacks = ({ weeklyAttacks, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Top 5 Attacks of the Week</h3>
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  // Convert weeklyAttacks object to array and sort by count/percentage
  const attacksArray = weeklyAttacks
    ? Object.entries(weeklyAttacks)
        .map(([attackType, data]) => ({
          attackType,
          percentage: typeof data === 'number' ? data : data.percentage || data.count || 0,
          count: typeof data === 'object' ? data.count : null,
        }))
        .sort((a, b) => b.percentage - a.percentage)
        .slice(0, 5)
    : []

  if (attacksArray.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Top 5 Attacks of the Week</h3>
        <div className="text-center py-8 text-gray-500">
          <p>No attack data available</p>
        </div>
      </div>
    )
  }

  const maxPercentage = Math.max(...attacksArray.map(a => a.percentage))

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
      <h3 className="text-xl font-bold text-gray-800 mb-4">Top 5 Attacks of the Week</h3>
      
      <div className="space-y-4">
        {attacksArray.map((attack, idx) => {
          const barWidth = maxPercentage > 0 ? (attack.percentage / maxPercentage) * 100 : 0
          
          return (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">
                  {attack.attackType}
                </span>
                <div className="flex items-center gap-2">
                  {attack.count !== null && (
                    <span className="text-xs text-gray-500">
                      {attack.count} attacks
                    </span>
                  )}
                  <span className="text-sm font-semibold text-gray-900">
                    {attack.percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
              
              {/* Mini bar chart */}
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default WeeklyAttacks

