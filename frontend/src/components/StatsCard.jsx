/**
 * Statistics card component for displaying key metrics
 * Supports icons and animations
 */

const StatsCard = ({ title, value, icon, trend, className = '' }) => {
  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
          {trend && (
            <p className={`text-sm mt-2 ${trend.positive ? 'text-green-600' : 'text-red-600'}`}>
              {trend.positive ? '↑' : '↓'} {trend.value}
            </p>
          )}
        </div>
        {icon && (
          <div className="text-4xl text-primary-500">{icon}</div>
        )}
      </div>
    </div>
  )
}

export default StatsCard

