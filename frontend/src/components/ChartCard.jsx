/**
 * Reusable chart card component wrapper
 * Provides consistent styling for all charts
 */

const ChartCard = ({ title, description, children, className = '' }) => {
  return (
    <div className={`card ${className}`}>
      {title && (
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          {description && (
            <p className="text-sm text-gray-500 mt-1">{description}</p>
          )}
        </div>
      )}
      <div className="h-64">{children}</div>
    </div>
  )
}

export default ChartCard

