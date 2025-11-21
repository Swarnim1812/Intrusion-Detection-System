/**
 * Threat Analysis page with common attack types
 * Expandable accordion cards with animations
 */

import { useState } from 'react'

const Threats = () => {
  const [expanded, setExpanded] = useState(null)

  const threats = [
    {
      id: 1,
      name: 'Phishing Attacks',
      icon: '🎣',
      description:
        'Deceptive emails and websites designed to steal credentials and sensitive information',
      example:
        'Fake government portal emails requesting login credentials or personal data',
      details:
        'Phishing attacks target users through social engineering, often using official-looking emails and websites. These attacks can compromise entire systems if credentials are stolen.',
      link: '#',
    },
    {
      id: 2,
      name: 'SQL Injection (SQLi)',
      icon: '💉',
      description:
        'Malicious SQL code injected into database queries to access or manipulate data',
      example:
        'Attacker injects SQL commands through web form inputs to extract citizen data',
      details:
        'SQL injection exploits vulnerabilities in database queries, allowing attackers to read, modify, or delete sensitive data. Proper input validation and parameterized queries are essential defenses.',
      link: '#',
    },
    {
      id: 3,
      name: 'DDoS Attacks',
      icon: '🌊',
      description:
        'Distributed Denial of Service attacks overwhelming servers with traffic',
      example:
        'Coordinated botnet flooding government servers to make services unavailable',
      details:
        'DDoS attacks aim to disrupt services by overwhelming infrastructure with traffic. These attacks can prevent citizens from accessing essential e-governance services.',
      link: '#',
    },
    {
      id: 4,
      name: 'Data Leakage',
      icon: '📤',
      description:
        'Unauthorized access and exfiltration of sensitive citizen or government data',
      example:
        'Insider threat or external breach resulting in exposure of personal records',
      details:
        'Data leakage can occur through various vectors including insider threats, misconfigured databases, or successful breaches. Protecting citizen privacy is paramount.',
      link: '#',
    },
    {
      id: 5,
      name: 'Website Defacement',
      icon: '🎨',
      description:
        'Unauthorized modification of website content to display malicious messages',
      example:
        'Hackers replace homepage content with political messages or propaganda',
      details:
        'Defacement attacks damage public trust and can be used to spread misinformation. These attacks often indicate deeper security vulnerabilities.',
      link: '#',
    },
  ]

  const toggleExpand = (id) => {
    setExpanded(expanded === id ? null : id)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12" data-aos="fade-up">
        <h1 className="text-4xl font-bold mb-4">Threat Analysis & Case Studies</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Understanding common cyber threats targeting e-governance infrastructure
        </p>
      </div>

      <div className="space-y-4">
        {threats.map((threat, idx) => (
          <div
            key={threat.id}
            className="card cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => toggleExpand(threat.id)}
            data-aos="fade-up"
            data-aos-delay={idx * 100}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4 flex-grow">
                <div className="text-4xl">{threat.icon}</div>
                <div className="flex-grow">
                  <h3 className="text-xl font-semibold mb-2">{threat.name}</h3>
                  <p className="text-gray-600 mb-2">{threat.description}</p>
                  <p className="text-sm text-gray-500 italic">
                    Example: {threat.example}
                  </p>
                  {expanded === threat.id && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <p className="text-gray-700 mb-3">{threat.details}</p>
                      <a
                        href={threat.link}
                        className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Learn more →
                      </a>
                    </div>
                  )}
                </div>
              </div>
              <button
                className="text-gray-400 hover:text-gray-600 ml-4"
                aria-label={expanded === threat.id ? 'Collapse' : 'Expand'}
              >
                <svg
                  className={`w-6 h-6 transition-transform ${
                    expanded === threat.id ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Prevention Tips */}
      <section className="mt-16 bg-blue-50 rounded-lg p-8" data-aos="fade-up">
        <h2 className="text-2xl font-bold mb-4">Prevention Best Practices</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ul className="space-y-2 text-gray-700">
            <li>✓ Regular security audits and penetration testing</li>
            <li>✓ Employee training on cybersecurity awareness</li>
            <li>✓ Multi-factor authentication for all systems</li>
            <li>✓ Regular software updates and patch management</li>
          </ul>
          <ul className="space-y-2 text-gray-700">
            <li>✓ Network segmentation and access controls</li>
            <li>✓ Continuous monitoring and intrusion detection</li>
            <li>✓ Incident response planning and drills</li>
            <li>✓ Data encryption at rest and in transit</li>
          </ul>
        </div>
      </section>
    </div>
  )
}

export default Threats

