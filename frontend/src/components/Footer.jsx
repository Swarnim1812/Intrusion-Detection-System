/**
 * Footer component with links and copyright information
 */

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-gray-300 mt-auto" role="contentinfo">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-white font-semibold mb-4">About Jharkhand-IDS</h3>
            <p className="text-sm">
              AI-powered Intrusion Detection System protecting e-governance infrastructure
              in Jharkhand with advanced machine learning algorithms.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="/about" className="hover:text-white transition-colors">
                  About E-Governance
                </a>
              </li>
              <li>
                <a href="/threats" className="hover:text-white transition-colors">
                  Threat Analysis
                </a>
              </li>
              <li>
                <a href="/ids" className="hover:text-white transition-colors">
                  Try IDS
                </a>
              </li>
              <li>
                <a href="/dashboard" className="hover:text-white transition-colors">
                  Analytics Dashboard
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-white font-semibold mb-4">Contact</h3>
            <p className="text-sm">
              For support and inquiries, please contact the Jharkhand-IDS team.
            </p>
          </div>
        </div>

        <div className="border-t border-gray-700 mt-8 pt-8 text-center text-sm">
          <p>&copy; {new Date().getFullYear()} Jharkhand-IDS. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer

