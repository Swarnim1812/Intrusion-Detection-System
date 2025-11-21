/**
 * Home page with hero section, key benefits, and animated counters
 * Uses GSAP for animations and AOS for scroll animations
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { gsap } from 'gsap'
import { getModelStats } from '../services/api'
import StatsCard from '../components/StatsCard'

const Home = () => {
  const [stats, setStats] = useState({
    totalAttacks: 1250,
    accuracy: 95.2,
    threatsDetected: 342,
  })

  useEffect(() => {
    // GSAP hero animation
    gsap.from('.hero-title', {
      opacity: 1,
      y: -30,
      duration: 1,
      ease: 'power3.out',
    })
    gsap.from('.hero-subtitle', {
      opacity: 0.3,
      y: 20,
      duration: 1,
      delay: 0.3,
      ease: 'power3.out',
    })

    // Load model stats
    getModelStats()
      .then((data) => {
        if (data.accuracy) {
          setStats((prev) => ({
            ...prev,
            accuracy: (data.accuracy * 100).toFixed(1),
          }))
        }
      })
      .catch((err) => console.warn('Could not load model stats:', err))
  }, [])

  return (
    <div>
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center" data-aos="fade-up">
            <h1 className="hero-title text-5xl md:text-6xl font-bold mb-6">
              🛡️ Jharkhand-IDS
            </h1>
            <p className="hero-subtitle text-xl md:text-2xl mb-8 max-w-3xl mx-auto">
              AI-Powered Intrusion Detection System protecting e-governance
              infrastructure with advanced machine learning
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/ids"
                className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Try the IDS
              </Link>
              <Link
                to="/about"
                className="bg-primary-700 text-white px-8 py-3 rounded-lg font-semibold hover:bg-primary-600 transition-colors border border-primary-500"
              >
                Learn More
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatsCard
              title="Total Attacks Detected"
              value={stats.totalAttacks.toLocaleString()}
              icon="🚨"
              data-aos="fade-up"
              data-aos-delay="0"
            />
            <StatsCard
              title="Model Accuracy"
              value={`${stats.accuracy}%`}
              icon="🎯"
              data-aos="fade-up"
              data-aos-delay="100"
            />
            <StatsCard
              title="Threats Detected Today"
              value={stats.threatsDetected}
              icon="⚡"
              data-aos="fade-up"
              data-aos-delay="200"
            />
          </div>
        </div>
      </section>

      {/* Key Benefits */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-up">
            Why Choose Jharkhand-IDS?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'AI-Powered Detection',
                description:
                  'Advanced machine learning algorithms detect intrusions with high accuracy',
                icon: '🤖',
              },
              {
                title: 'Real-Time Monitoring',
                description:
                  'Continuous monitoring of network traffic and system activities',
                icon: '⚡',
              },
              {
                title: 'Comprehensive Analysis',
                description:
                  'Detailed threat analysis and actionable security insights',
                icon: '📊',
              },
            ].map((benefit, idx) => (
              <div
                key={idx}
                className="card text-center"
                data-aos="fade-up"
                data-aos-delay={idx * 100}
              >
                <div className="text-5xl mb-4">{benefit.icon}</div>
                <h3 className="text-xl font-semibold mb-2">{benefit.title}</h3>
                <p className="text-gray-600">{benefit.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-primary-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-4" data-aos="fade-up">
            Ready to Protect Your Infrastructure?
          </h2>
          <p className="text-xl mb-8" data-aos="fade-up" data-aos-delay="100">
            Upload your network flow data and get instant intrusion detection
          </p>
          <Link
            to="/ids"
            className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors inline-block"
            data-aos="fade-up"
            data-aos-delay="200"
          >
            Get Started Now
          </Link>
        </div>
      </section>
    </div>
  )
}

export default Home

