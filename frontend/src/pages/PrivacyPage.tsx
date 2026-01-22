import NanobotBackground from '../components/common/NanobotBackground'
import SiteFooter from '../components/common/SiteFooter'
import './LegalPage.css'

export default function PrivacyPage() {
  return (
    <div className="legal-page">
      <NanobotBackground particleCount={70} connectionDistance={120} />

      <div className="legal-content">
        <header className="legal-header">
          <h1>Privacy Policy</h1>
          <p>We respect your data and only use it to operate and improve Vizora.</p>
        </header>

        <section className="legal-section">
          <h2>Information We Collect</h2>
          <ul>
            <li>Account details like name and email address.</li>
            <li>Datasets and inputs you upload for analysis.</li>
            <li>Usage data such as feature interactions and error logs.</li>
            <li>Billing details processed by Stripe (we do not store full card numbers).</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>How We Use Information</h2>
          <ul>
            <li>Provide, maintain, and improve the Vizora service.</li>
            <li>Process payments, manage subscriptions, and handle support.</li>
            <li>Monitor performance and prevent abuse or fraud.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Sharing</h2>
          <p>
            We share data only with service providers needed to operate Vizora, such as
            hosting, analytics, and billing vendors. We do not sell your personal data.
          </p>
        </section>

        <section className="legal-section">
          <h2>Data Retention</h2>
          <p>
            We keep data as long as your account is active or as needed to provide the
            service. You can request deletion of your account and data at any time.
          </p>
        </section>

        <section className="legal-section">
          <h2>Security</h2>
          <p>
            We use reasonable safeguards to protect your information, but no system can be
            guaranteed 100 percent secure.
          </p>
        </section>

        <section className="legal-section">
          <h2>Your Choices</h2>
          <ul>
            <li>Access or update your account information from your profile.</li>
            <li>Cancel your subscription at any time.</li>
            <li>Request deletion by contacting support.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Contact</h2>
          <p>
            Questions about privacy? Email us at{' '}
            <a href="mailto:support@agentvizora.com">support@agentvizora.com</a>.
          </p>
        </section>
      </div>

      <SiteFooter />
    </div>
  )
}
