import { useNavigate } from 'react-router-dom'
import NanobotBackground from '../components/common/NanobotBackground'
import SiteFooter from '../components/common/SiteFooter'
import './LegalPage.css'

export default function ContactPage() {
  const navigate = useNavigate()

  return (
    <div className="legal-page">
      <NanobotBackground particleCount={70} connectionDistance={120} />

      <div className="legal-content">
        <button className="back-btn" onClick={() => navigate('/')}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5" />
            <path d="m12 19-7-7 7-7" />
          </svg>
          Home
        </button>

        <header className="legal-header">
          <h1>Contact</h1>
          <p>We are here to help with billing, access, and product questions.</p>
        </header>

        <section className="legal-section">
          <h2>Support Email</h2>
          <p>
            <a href="mailto:support@agentvizora.com">support@agentvizora.com</a>
          </p>
        </section>

        <section className="legal-section">
          <h2>Response Time</h2>
          <p>We typically respond within 1-2 business days.</p>
        </section>

        <section className="legal-section">
          <h2>Business</h2>
          <p>Vizora - AI-powered data analysis and reporting.</p>
        </section>
      </div>

      <SiteFooter />
    </div>
  )
}
