/**
 * Home page with hero section, key benefits, and animated counters
 * Uses GSAP for animations and AOS for scroll animations
 * Now includes real backend-driven cybersecurity metrics and interactive UI elements
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { gsap } from 'gsap'
import {
  getModelStats,
  getDatasetStats,
  getRecentEvent,
  getWeeklyAttacks,
} from '../services/api'
import StatsCard from '../components/StatsCard'
import LiveThreatMeter from '../components/LiveThreatMeter'
import LastIntrusionCard from '../components/LastIntrusionCard'
import DatasetStats from '../components/DatasetStats'
import WeeklyAttacks from '../components/WeeklyAttacks'

const Home = () => {
  const [stats, setStats] = useState({
    totalAttacks: 0,
    accuracy: 0,
    threatsDetected: 0,
  })
  const [metrics, setMetrics] = useState(null)
  const [datasetStats, setDatasetStats] = useState(null)
  const [recentEvent, setRecentEvent] = useState(null)
  const [weeklyAttacks, setWeeklyAttacks] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [traffic, setTraffic] = useState({ normal: 0, attack: 0, total: 0 })

  
  useEffect(() => {
    const loadTraffic = async () => {
      try {
        const res = await fetch('http://localhost:5000/traffic-counts')
        const data = await res.json()
        setTraffic(data)
      } catch (err) {
        console.error("Traffic counts error:", err)
      }
    }
    loadTraffic()
  }, [])
  
  useEffect(() => {
    setStats(prev => ({
      ...prev,
      totalAttacks: traffic.attack
    }))
  }, [traffic])

  useEffect(() => {
    // GSAP hero animation
    gsap.from('.hero-title', {
      opacity: 1,
      y: -30,
      duration: 1,
      ease: 'power3.out',
    })
    gsap.from('.hero-subtitle', {
      opacity: 0.8,
      y: 20,
      duration: 1,
      delay: 0.3,
      ease: 'power3.out',
    })

    // Load all data from backend
    const loadAllData = async () => {
      setIsLoading(true)
      try {
        // Load all data in parallel (using allSettled so failures don't block metrics)
        const results = await Promise.allSettled([
          getModelStats(),
          getDatasetStats(),
          getRecentEvent(),
          getWeeklyAttacks(),
        ])

        const metricsData = results[0].status === 'fulfilled' ? results[0].value : null
        const datasetData = results[1].status === 'fulfilled' ? results[1].value : null
        const recentData = results[2].status === 'fulfilled' ? results[2].value : null
        const weeklyData = results[3].status === 'fulfilled' ? results[3].value : null

        // Set metrics
        setMetrics(metricsData)

        // Calculate stats from metrics
        if (metricsData) {
          const accuracy = metricsData.accuracy ?? 0
          const accuracyPercent = accuracy ? (accuracy * 100).toFixed(1) : 0

          // Calculate total attacks from confusion matrix
          // confusion_matrix format: [[TN, FP], [FN, TP]]
          // Total attacks detected = FN + TP (row 1: false negatives + true positives)
          let totalAttacks = 0
          if (metricsData.confusion_matrix && Array.isArray(metricsData.confusion_matrix) && metricsData.confusion_matrix.length >= 2) {
            const row1 = metricsData.confusion_matrix[1] ?? []
            totalAttacks = (row1[0] ?? 0) + (row1[1] ?? 0)
          }

          setStats(prev => ({
            ...prev,
            accuracy: parseFloat(accuracyPercent),
            threatsDetected: datasetData?.attacks_last_24h ?? 0,
          }))
        }

        // Set dataset stats
        setDatasetStats(datasetData)

        // Set recent event
        setRecentEvent(recentData)

        // Set weekly attacks
        setWeeklyAttacks(weeklyData)
      } catch (error) {
        console.error('Error loading data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadAllData()
  }, [])

  // Calculate threat level percentage
  const calculateThreatPercentage = () => {
    if (!datasetStats) return 0
    const totalFlows = datasetStats.total_flows_last_24h ?? 0
    const attacks = datasetStats.attacks_last_24h ?? 0
    if (totalFlows === 0) return 0
    return (attacks / totalFlows) * 100
  }

  const threatPercentage = calculateThreatPercentage()

  // return (
  //   <div>
  //     {/* Hero Section */}
  //     <section className="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //         <div className="text-center" data-aos="fade-up">
  //           <h1 className="hero-title text-5xl md:text-6xl font-bold mb-6">
  //             🛡️ Jharkhand-IDS
  //           </h1>
  //           <p className="hero-subtitle text-xl md:text-2xl mb-8 max-w-3xl mx-auto">
  //             AI-Powered Intrusion Detection System protecting e-governance
  //             infrastructure with advanced machine learning
  //           </p>
  //           <div className="flex flex-col sm:flex-row gap-4 justify-center">
  //             <Link
  //               to="/ids"
  //               className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
  //             >
  //               Try the IDS
  //             </Link>
  //             <Link
  //               to="/about"
  //               className="bg-primary-700 text-white px-8 py-3 rounded-lg font-semibold hover:bg-primary-600 transition-colors border border-primary-500"
  //             >
  //               Learn More
  //             </Link>
  //           </div>
  //         </div>
  //       </div>
  //     </section>

  //     {/* Stats Section */}
  //     <section className="py-16 bg-white">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //         <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
  //           <StatsCard
  //             title="Total Attacks Detected"
  //             value={
  //               isLoading
  //                 ? 'Loading...'
  //                 : (stats.totalAttacks ?? 0).toLocaleString()
  //             }
  //             icon="🚨"
  //             data-aos="fade-up"
  //             data-aos-delay="0"
  //           />
  //           <StatsCard
  //             title="Model Accuracy"
  //             value={
  //               isLoading
  //                 ? 'Loading...'
  //                 : stats.accuracy ? `${stats.accuracy}%` : 'N/A'
  //             }
  //             icon="🎯"
  //             data-aos="fade-up"
  //             data-aos-delay="100"
  //           />
  //           <StatsCard
  //             title="Threats Detected Today"
  //             value={
  //               isLoading
  //                 ? 'Loading...'
  //                 : (stats.threatsDetected ?? 0).toLocaleString()
  //             }
  //             icon="⚡"
  //             data-aos="fade-up"
  //             data-aos-delay="200"
  //           />
  //         </div>
  //       </div>
  //     </section>

  //     {/* Live Threat Meter & Last Intrusion Section */}
  //     <section className="py-16 bg-gray-50">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
  //           <div data-aos="fade-up" data-aos-delay="0">
  //             <LiveThreatMeter
  //               threatLevel={
  //                 threatPercentage >= 50
  //                   ? 'HIGH'
  //                   : threatPercentage >= 20
  //                   ? 'MEDIUM'
  //                   : 'LOW'
  //               }
  //               threatPercentage={threatPercentage}
  //             />
  //           </div>
  //           <div data-aos="fade-up" data-aos-delay="100">
  //             <LastIntrusionCard recentEvent={recentEvent} isLoading={isLoading} />
  //           </div>
  //         </div>
  //       </div>
  //     </section>

  //     {/* Model Performance Overview */}
  //     {metrics && (
  //       <section className="py-16 bg-white">
  //         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //           <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-up">
  //             Model Performance Overview
  //           </h2>
  //           <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
  //             <StatsCard
  //               title="Accuracy"
  //               value={
  //                 metrics.accuracy
  //                   ? `${(metrics.accuracy * 100).toFixed(2)}%`
  //                   : 'N/A'
  //               }
  //               icon="🎯"
  //               data-aos="fade-up"
  //               data-aos-delay="0"
  //             />
  //             <StatsCard
  //               title="Precision"
  //               value={
  //                 metrics.precision
  //                   ? `${(metrics.precision * 100).toFixed(2)}%`
  //                   : 'N/A'
  //               }
  //               icon="📊"
  //               data-aos="fade-up"
  //               data-aos-delay="100"
  //             />
  //             <StatsCard
  //               title="Recall"
  //               value={
  //                 metrics.recall
  //                   ? `${(metrics.recall * 100).toFixed(2)}%`
  //                   : 'N/A'
  //               }
  //               icon="🔍"
  //               data-aos="fade-up"
  //               data-aos-delay="200"
  //             />
  //             <StatsCard
  //               title="F1-Score"
  //               value={
  //                 metrics.f1
  //                   ? `${(metrics.f1 * 100).toFixed(2)}%`
  //                   : 'N/A'
  //               }
  //               icon="⭐"
  //               data-aos="fade-up"
  //               data-aos-delay="300"
  //             />
  //           </div>
  //         </div>
  //       </section>
  //     )}

  //     {/* Dataset Stats & Weekly Attacks Section */}
  //     <section className="py-16 bg-gray-50">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
  //           <div data-aos="fade-up" data-aos-delay="0">
  //             <DatasetStats datasetStats={datasetStats} isLoading={isLoading} />
  //           </div>
  //           <div data-aos="fade-up" data-aos-delay="100">
  //             <WeeklyAttacks weeklyAttacks={weeklyAttacks} isLoading={isLoading} />
  //           </div>
  //         </div>
  //       </div>
  //     </section>

  //     {/* Key Benefits */}
  //     <section className="py-16 bg-white">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
  //         <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-up">
  //           Why Choose Jharkhand-IDS?
  //         </h2>
  //         <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
  //           {[
  //             {
  //               title: 'AI-Powered Detection',
  //               description:
  //                 'Advanced machine learning algorithms detect intrusions with high accuracy',
  //               icon: '🤖',
  //             },
  //             {
  //               title: 'Real-Time Monitoring',
  //               description:
  //                 'Continuous monitoring of network traffic and system activities',
  //               icon: '⚡',
  //             },
  //             {
  //               title: 'Comprehensive Analysis',
  //               description:
  //                 'Detailed threat analysis and actionable security insights',
  //               icon: '📊',
  //             },
  //           ].map((benefit, idx) => (
  //             <div
  //               key={idx}
  //               className="card text-center hover:shadow-xl transition-shadow duration-300 rounded-xl"
  //               data-aos="fade-up"
  //               data-aos-delay={idx * 100}
  //             >
  //               <div className="text-5xl mb-4">{benefit.icon}</div>
  //               <h3 className="text-xl font-semibold mb-2">{benefit.title}</h3>
  //               <p className="text-gray-600">{benefit.description}</p>
  //             </div>
  //           ))}
  //         </div>
  //       </div>
  //     </section>

  //     {/* CTA Section */}
  //     <section className="py-16 bg-primary-600 text-white">
  //       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
  //         <h2 className="text-3xl font-bold mb-4" data-aos="fade-up">
  //           Ready to Protect Your Infrastructure?
  //         </h2>
  //         <p className="text-xl mb-8" data-aos="fade-up" data-aos-delay="100">
  //           Upload your network flow data and get instant intrusion detection
  //         </p>
  //         <Link
  //           to="/ids"
  //           className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors inline-block"
  //           data-aos="fade-up"
  //           data-aos-delay="200"
  //         >
  //           Get Started Now
  //         </Link>
  //       </div>
  //     </section>
  //   </div>
  // )



  return (
    <div>
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center" data-aos="fade-up">
            <h1 className="hero-title text-5xl md:text-6xl font-bold mb-6">
              🛡️ Jharkhand-IDS
            </h1>
            <p className="hero-subtitle text-xl md:text-2xl mb-16 max-w-3xl mx-auto">
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
              value={
                isLoading
                  ? 'Loading...'
                  : (stats.totalAttacks ?? 0).toLocaleString()
              }
              icon="🚨"
              data-aos="fade-up"
              data-aos-delay="0"
            />
            <StatsCard
              title="Model Accuracy"
              value={
                isLoading
                  ? 'Loading...'
                  : stats.accuracy ? `${stats.accuracy}%` : 'N/A'
              }
              icon="🎯"
              data-aos="fade-up"
              data-aos-delay="100"
            />
            <StatsCard
              title="Threats Detected (Simulated 24h)"
              value={
                isLoading
                  ? 'Loading...'
                  : (stats.threatsDetected ?? 0).toLocaleString()
              }
              icon="⚡"
              data-aos="fade-up"
              data-aos-delay="200"
            />
          </div>

          <p className="text-gray-700 text-sm text-center mt-7 max-w-2xl mx-auto">
            These statistics are generated from the CICIDS2017 dataset and represent computed values rather than real-time traffic. 
            “Threats Detected (Simulated 24h)” is estimated using a scaled 24-hour window for demonstration purposes.
          </p>
        </div>
      </section>

      {/* Live Threat Meter & Last Intrusion Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div data-aos="fade-up" data-aos-delay="0">
              <LiveThreatMeter
                threatLevel={
                  threatPercentage >= 50
                    ? 'HIGH'
                    : threatPercentage >= 20
                    ? 'MEDIUM'
                    : 'LOW'
                }
                threatPercentage={threatPercentage}
              />

              <p className="text-gray-600 text-sm mt-5 text-center">
                The Live Threat Meter reflects the proportion of attacks estimated from a simulated 
                24-hour slice of the dataset. It demonstrates how an IDS would quantify threat severity 
                in a live system, although the underlying data is historical.
              </p>
            </div>

            <div data-aos="fade-up" data-aos-delay="100">
              <LastIntrusionCard recentEvent={recentEvent} isLoading={isLoading} />

              <p className="text-gray-600 text-sm mt-5 text-center">
                The “Last Intrusion Detected” card displays the most recent attack-like event found in 
                sample predictions. Since the dataset is not real-time, this represents the latest event 
                within the dataset's timeline, not the current day.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Model Performance Overview */}
      {metrics && (
        <section className="py-16 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-up">
              Model Performance Overview
            </h2>

            <p className="text-gray-600 text-sm text-center mb-6 max-w-2xl mx-auto">
              These metrics are computed from the trained machine learning model using the CICIDS2017 dataset. 
              They represent the model’s detection capability across accuracy, precision, recall, and F1-score.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <StatsCard
                title="Accuracy"
                value={
                  metrics.accuracy
                    ? `${(metrics.accuracy * 100).toFixed(2)}%`
                    : 'N/A'
                }
                icon="🎯"
                data-aos="fade-up"
                data-aos-delay="0"
              />
              <StatsCard
                title="Precision"
                value={
                  metrics.precision
                    ? `${(metrics.precision * 100).toFixed(2)}%`
                    : 'N/A'
                }
                icon="📊"
                data-aos="fade-up"
                data-aos-delay="100"
              />
              <StatsCard
                title="Recall"
                value={
                  metrics.recall
                    ? `${(metrics.recall * 100).toFixed(2)}%`
                    : 'N/A'
                }
                icon="🔍"
                data-aos="fade-up"
                data-aos-delay="200"
              />
              <StatsCard
                title="F1-Score"
                value={
                  metrics.f1
                    ? `${(metrics.f1 * 100).toFixed(2)}%`
                    : 'N/A'
                }
                icon="⭐"
                data-aos="fade-up"
                data-aos-delay="300"
              />
            </div>
          </div>
        </section>
      )}

      {/* Dataset Stats & Weekly Attacks Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div data-aos="fade-up" data-aos-delay="0">
              <DatasetStats datasetStats={datasetStats} isLoading={isLoading} />

              <p className="text-gray-600 text-sm mt-6 text-center">
                Dataset Statistics provide an overview of the CICIDS2017 traffic distribution. 
                This helps users understand how much of the captured network traffic is normal versus malicious, 
                based on real recorded events.
              </p>
            </div>

            <div data-aos="fade-up" data-aos-delay="100">
              <WeeklyAttacks weeklyAttacks={weeklyAttacks} isLoading={isLoading} />

              <p className="text-gray-600 text-sm mt-6 text-center">
                The “Top 5 Attacks of the Week” section summarizes the most common attack types found in the dataset. 
                Since the dataset is historical, these values are computed by analyzing attack distribution, not live events.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Key Benefits */}
      <section className="py-16 bg-white">
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
                className="card text-center hover:shadow-xl transition-shadow duration-300 rounded-xl"
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
