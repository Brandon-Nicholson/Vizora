import { useNavigate } from 'react-router-dom'
import NanobotBackground from '../components/common/NanobotBackground'
import SiteFooter from '../components/common/SiteFooter'
import './LegalPage.css'

export default function RefundsPage() {
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
          <h1>Refund and Cancellation Policy</h1>
          <p>Simple terms for cancelling and requesting a refund.</p>
        </header>

        <section className="legal-section">
          <h2>Refunds</h2>
          <p>
            We offer a full refund within 7 days of your first payment if you are not
            satisfied. Refunds apply to the initial subscription charge only.
          </p>
        </section>

        <section className="legal-section">
          <h2>Cancellations</h2>
          <p>
            You can cancel your subscription at any time. Your Pro access continues until
            the end of your current billing period.
          </p>
        </section>

        <section className="legal-section">
          <h2>How to Request a Refund</h2>
          <p>
            Email us at{' '}
            <a href="mailto:support@agentvizora.com">support@agentvizora.com</a> with the
            email address on your account and the reason for your request. We will confirm
            your refund status by email.
          </p>
        </section>

        <section className="legal-section">
          <h2>Questions</h2>
          <p>
            If you are unsure about your eligibility, contact us and we will help.
          </p>
        </section>
      </div>

      <SiteFooter />
    </div>
  )
}
