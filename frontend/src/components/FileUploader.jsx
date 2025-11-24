/**
 * File uploader component with CSV validation and preview
 * Supports drag-and-drop and manual file selection
 */

import { useState, useRef } from 'react'
import Papa from 'papaparse'
import { useToast } from '../utils/toastContext'

const FileUploader = ({ onFileLoaded, requiredFeatures = [] }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [missingColumns, setMissingColumns] = useState([])
  const fileInputRef = useRef(null)
  const { addToast } = useToast()

  const handleFile = (uploadedFile) => {
    if (!uploadedFile) return

    // Validate file type
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 'text/plain']
    const isValidType = validTypes.includes(uploadedFile.type) || 
                       uploadedFile.name.endsWith('.csv') ||
                       uploadedFile.name.endsWith('.csv.gz')

    if (!isValidType) {
      addToast('Please upload a CSV file', 'error')
      return
    }

    // Check file size (10MB limit)
    const maxSize = 10 * 1024 * 1024 // 10MB
    if (uploadedFile.size > maxSize) {
      addToast('File size exceeds 10MB limit', 'error')
      return
    }

    setFile(uploadedFile)

    // Parse CSV for preview
    if (uploadedFile.name.endsWith('.gz')) {
      addToast('Gzipped CSV detected. Please decompress before uploading.', 'warning')
      return
    }

    Papa.parse(uploadedFile, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      transformHeader: (header) => header?.trim?.() || header,
      complete: (results) => {
        if (results.errors?.length) {
          console.error('CSV parsing errors:', results.errors)
          addToast('Error parsing CSV file. Please check the format.', 'error')
          return
        }

        // const cleanedRows = (results.data || []).map((row) => {
        //   const cleanedRow = {}
        //   Object.entries(row || {}).forEach(([key, value]) => {
        //     const trimmedKey = key?.trim?.()
        //     if (!trimmedKey) return
        //     cleanedRow[trimmedKey] = value
        //   })
        //   return cleanedRow
        // })

        // SAME normalization used everywhere
        const normalize = (name) =>
          name
            ?.trim()
            ?.replace(/\s+/g, "_")
            ?.replace(/\./g, "_")
            ?.replace(/[^a-zA-Z0-9_]/g, "")
            ?.toLowerCase();

        const cleanedRows = (results.data || []).map((row) => {
          const cleanedRow = {}
          Object.entries(row || {}).forEach(([key, value]) => {
            const normalizedKey = normalize(key)
            if (!normalizedKey) return
            cleanedRow[normalizedKey] = value
          })
          return cleanedRow
        })
        console.log("=== RAW CSV ROWS (first 2) ===", results.data.slice(0,2));
        console.log("=== CLEANED + NORMALIZED ROWS (first 2) ===", cleanedRows.slice(0,2));
        setPreview(cleanedRows.slice(0, 5))

        if (requiredFeatures.length > 0) {
          const csvColumns = Object.keys(cleanedRows[0] || {})
          const normalizedRequired = requiredFeatures.map(f => normalize(f))

          const missing = normalizedRequired.filter(
            (feat) => !csvColumns.includes(feat)
          )
          setMissingColumns(missing)

          if (missing.length > 0) {
            addToast(
              `Missing columnsssss: ${missing.slice(0, 3).join(', ')}${missing.length > 3 ? '...' : ''}`,
              'warning'
            )
          }
        }

        if (onFileLoaded) {
          onFileLoaded(uploadedFile, cleanedRows)
        }
      },
      error: (error) => {
        addToast('Error parsing CSV file', 'error')
        console.error('CSV parsing error:', error)
      },
    })
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFile = e.dataTransfer.files[0]
    handleFile(droppedFile)
  }

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    handleFile(selectedFile)
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.csv.gz"
          onChange={handleFileSelect}
          className="hidden"
          aria-label="Upload CSV file"
        />
        
        <div className="space-y-2">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
            aria-hidden="true"
          >
            <path
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="text-gray-600">
            Drag and drop your CSV file here, or
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="text-primary-600 hover:text-primary-700 font-medium ml-1"
            >
              browse
            </button>
          </p>
          <p className="text-sm text-gray-500">CSV files up to 10MB</p>
        </div>
      </div>

      {/* File Info */}
      {file && (
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">{file.name}</p>
              <p className="text-sm text-gray-500">
                {(file.size / 1024).toFixed(2)} KB
              </p>
            </div>
            <button
              onClick={() => {
                setFile(null)
                setPreview(null)
                setMissingColumns([])
                if (fileInputRef.current) fileInputRef.current.value = ''
              }}
              className="text-red-600 hover:text-red-700"
              aria-label="Remove file"
            >
              Remove
            </button>
          </div>
        </div>
      )}

      {/* Missing Columns Warning */}
      {missingColumns.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <p className="text-sm font-medium text-yellow-800 mb-2">
            ⚠ Missing required columns:
          </p>
          <ul className="text-sm text-yellow-700 list-disc list-inside">
            {missingColumns.map((col) => (
              <li key={col}>{col}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default FileUploader

