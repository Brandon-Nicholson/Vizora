import { useNavigate } from 'react-router-dom'
import NanobotBackground from '../components/common/NanobotBackground'
import SiteFooter from '../components/common/SiteFooter'
import './LegalPage.css'

export default function TermsPage() {
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
          <h1>Terms of Service</h1>
          <p>These terms govern your use of Vizora and our subscription plans.</p>
        </header>

        <section className="legal-section">
          <h2>Use of the Service</h2>
          <p>
            Vizora provides AI-powered data analysis tools. You agree to use the service in
            compliance with applicable laws and not to misuse the platform.
          </p>
        </section>

        <section className="legal-section">
          <h2>Accounts</h2>
          <ul>
            <li>You are responsible for maintaining the security of your account.</li>
            <li>You are responsible for content you upload and any results you share.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Subscriptions and Billing</h2>
          <ul>
            <li>Paid plans renew automatically until you cancel.</li>
            <li>Prices may change with notice before the next billing cycle.</li>
            <li>Billing is handled securely by Stripe.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Acceptable Use</h2>
          <ul>
            <li>Do not upload illegal, harmful, or unauthorized data.</li>
            <li>Do not attempt to disrupt or reverse engineer the service.</li>
            <li>Do not share access in a way that violates your plan limits.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Service Availability</h2>
          <p>
            We aim for high uptime but do not guarantee uninterrupted service. We may
            modify or discontinue features to improve the product.
          </p>
        </section>

        <section className="legal-section">
          <h2>Limitation of Liability</h2>
          <p>
            Vizora is provided on an as-is basis. We are not liable for indirect damages or
            losses arising from use of the service.
          </p>
        </section>

        <section className="legal-section">
          <h2>Contact</h2>
          <p>
            Questions about these terms? Email us at{' '}
            <a href="mailto:support@agentvizora.com">support@agentvizora.com</a>.
          </p>
        </section>
      </div>

      <SiteFooter />
    </div>
  )
}
