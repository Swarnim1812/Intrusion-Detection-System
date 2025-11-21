/**
 * Tooltip component for UI explanations
 * Shows helpful text next to form controls
 */

import { useState } from 'react'

const Tooltip = ({ text, children }) => {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        className="inline-flex items-center"
      >
        {children}
        <button
          type="button"
          className="ml-2 text-gray-400 hover:text-gray-600 focus:outline-none"
          aria-label="Show explanation"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </button>
      </div>
      {isVisible && (
        <div className="absolute z-50 w-64 p-3 mt-2 text-sm text-white bg-gray-800 rounded-lg shadow-lg left-0">
          {text}
          <div className="absolute w-2 h-2 bg-gray-800 transform rotate-45 -top-1 left-4"></div>
        </div>
      )}
    </div>
  )
}

export default Tooltip

