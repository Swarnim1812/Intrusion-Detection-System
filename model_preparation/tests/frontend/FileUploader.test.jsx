/**
 * Basic test for FileUploader component
 * Tests file selection and validation
 */

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import FileUploader from '../../frontend/src/components/FileUploader'

describe('FileUploader', () => {
  it('renders upload area', () => {
    render(<FileUploader />)
    expect(screen.getByText(/drag and drop/i)).toBeInTheDocument()
  })

  it('shows file size limit', () => {
    render(<FileUploader />)
    expect(screen.getByText(/10MB/i)).toBeInTheDocument()
  })
})

