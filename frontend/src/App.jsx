/**
 * Main App component with routing configuration
 * Handles navigation between all 5 pages
 */

import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { gsap } from 'gsap'

import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Toast from './components/Toast'
import { ToastProvider, useToast } from './utils/toastContext'

import Home from './pages/Home'
import About from './pages/About'
import Threats from './pages/Threats'
import IDS from './pages/IDS'
import Dashboard from './pages/Dashboard'

function AppContent() {
  const { toasts, removeToast } = useToast()

  useEffect(() => {
    // GSAP initialization for page transitions
    gsap.from('body', { opacity: 1, duration: 0.5 })
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/threats" element={<Threats />} />
          <Route path="/ids" element={<IDS />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </main>
      <Footer />
      
      {/* Toast notifications */}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </div>
  )
}

function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  )
}

export default App

