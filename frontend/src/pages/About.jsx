/**
 * About page describing e-governance in Jharkhand and security challenges
 * Includes infographics and responsive design
 */

import ChartCard from '../components/ChartCard'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js'
import { Doughnut } from 'react-chartjs-2'

ChartJS.register(ArcElement, Tooltip, Legend)

const About = () => {
  // Sample data for infographics
  const securityMeasuresData = {
    labels: ['Firewall', 'IDS/IPS', 'Encryption', 'Access Control', 'Monitoring'],
    datasets: [
      {
        data: [25, 20, 20, 20, 15],
        backgroundColor: [
          '#3b82f6',
          '#10b981',
          '#f59e0b',
          '#ef4444',
          '#8b5cf6',
        ],
      },
    ],
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12" data-aos="fade-up">
        <h1 className="text-4xl font-bold mb-4">About E-Governance in Jharkhand</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Securing digital infrastructure for transparent and efficient governance
        </p>
      </div>

      {/* E-Governance Schemes */}
      <section className="mb-16" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-6">E-Governance Initiatives</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-xl font-semibold mb-3">Digital Services Portal</h3>
            <p className="text-gray-600">
              Comprehensive online platform providing citizens access to government
              services, certificates, and applications 24/7.
            </p>
          </div>
          <div className="card">
            <h3 className="text-xl font-semibold mb-3">Citizen Services</h3>
            <p className="text-gray-600">
              Streamlined processes for birth certificates, land records, tax
              payments, and other essential services.
            </p>
          </div>
          <div className="card">
            <h3 className="text-xl font-semibold mb-3">Financial Management</h3>
            <p className="text-gray-600">
              Digital payment systems and financial transaction monitoring for
              transparent governance.
            </p>
          </div>
          <div className="card">
            <h3 className="text-xl font-semibold mb-3">Data Management</h3>
            <p className="text-gray-600">
              Secure storage and management of citizen data with privacy protection
              and compliance measures.
            </p>
          </div>
        </div>
      </section>

      {/* Current Security Measures */}
      <section className="mb-16" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-6">Current Security Measures</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Security Infrastructure</h3>
            <ul className="space-y-2 text-gray-600">
              <li>✓ Network firewalls and intrusion prevention systems</li>
              <li>✓ End-to-end encryption for sensitive data</li>
              <li>✓ Multi-factor authentication for system access</li>
              <li>✓ Regular security audits and vulnerability assessments</li>
              <li>✓ Incident response and disaster recovery plans</li>
            </ul>
          </div>
          <ChartCard
            title="Security Measures Distribution"
            description="Current security infrastructure allocation"
          >
            <Doughnut data={securityMeasuresData} options={{ maintainAspectRatio: false }} />
          </ChartCard>
        </div>
      </section>

      {/* Challenges */}
      <section className="mb-16" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-6">Security Challenges</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              title: 'Sophisticated Attacks',
              description:
                'Advanced persistent threats and zero-day exploits targeting government systems',
            },
            {
              title: 'Data Privacy',
              description:
                'Protecting sensitive citizen data while maintaining service accessibility',
            },
            {
              title: 'Scalability',
              description:
                'Ensuring security measures scale with growing digital infrastructure',
            },
          ].map((challenge, idx) => (
            <div key={idx} className="card">
              <h3 className="text-xl font-semibold mb-3">{challenge.title}</h3>
              <p className="text-gray-600">{challenge.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Solution */}
      <section className="bg-primary-50 rounded-lg p-8" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-4">How Jharkhand-IDS Helps</h2>
        <p className="text-lg text-gray-700 mb-4">
          Jharkhand-IDS leverages artificial intelligence and machine learning to
          provide real-time intrusion detection, threat analysis, and automated
          response capabilities. Our system continuously monitors network traffic,
          identifies suspicious patterns, and alerts security teams to potential
          threats before they can cause damage.
        </p>
        <p className="text-lg text-gray-700">
          By integrating with existing security infrastructure, Jharkhand-IDS
          enhances protection without disrupting operations, ensuring that
          e-governance services remain secure and accessible to all citizens.
        </p>
      </section>
    </div>
  )
}

export default About

