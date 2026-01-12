import { useNavigate } from 'react-router-dom'
import NanobotBackground from '../components/common/NanobotBackground'
import { useAuth } from '../context/AuthContext'
import { useBilling } from '../context/BillingContext'
import ModelsList from '../components/models/ModelsList'
import './HomePage.css'

export default function HomePage() {
  const navigate = useNavigate()
  const { state: authState } = useAuth()
  const { state: billingState } = useBilling()

  const isPro = billingState.tier === 'pro' || billingState.monthlyLimit === -1
  const isSignedIn = Boolean(authState.user)

  return (
    <div className="home-page">
      <NanobotBackground particleCount={100} connectionDistance={150} />

      <div className="home-content">
        <div className="home-top-actions animate-slide-up" style={{ animationDelay: '0.25s' }}>
          <button
            className="btn btn-secondary home-pricing-btn"
            onClick={() => navigate('/pricing')}
          >
            Pricing
          </button>
          <button
            className="btn btn-secondary home-status-btn"
            type="button"
            aria-disabled="true"
            disabled
          >
            {isSignedIn ? 'Signed In' : 'Signed Out'}
          </button>
        </div>

        <div className="home-layout">
          <div className="home-main">
            <div className="logo-container animate-fade-in">
              <div className="logo-glow"></div>
              <h1 className="logo-text">
                <span className="logo-v">V</span>
                <span className="logo-rest">izora</span>
              </h1>
            </div>

            <p className="tagline animate-slide-up">
              AI-Powered Data Analysis Agent
            </p>

            <p className="subtitle animate-slide-up" style={{ animationDelay: '0.1s' }}>
              Upload your dataset. Define your goal. Let AI do the rest.
            </p>

            <button
              className="btn btn-primary btn-large get-started-btn animate-slide-up"
              style={{ animationDelay: '0.2s' }}
              onClick={() => navigate('/mode')}
            >
              <span>Get Started</span>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14" />
                <path d="m12 5 7 7-7 7" />
              </svg>
            </button>

            <div className="features animate-slide-up" style={{ animationDelay: '0.3s' }}>
              <div className="feature">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12a9 9 0 1 1-9-9" />
                    <path d="M21 3v9h-9" />
                  </svg>
                </div>
                <span>Automated EDA</span>
              </div>
              <div className="feature">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2a10 10 0 1 0 10 10" />
                    <path d="m16 8-4 4-2-2" />
                    <path d="M22 2 12 12" />
                  </svg>
                </div>
                <span>ML Modeling</span>
              </div>
              <div className="feature">
                <div className="feature-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <path d="M7 7h.01" />
                    <path d="M17 7h.01" />
                    <path d="M7 17h.01" />
                    <path d="M17 17h.01" />
                  </svg>
                </div>
                <span>AI Insights</span>
              </div>
            </div>

            <div className="pro-tools animate-slide-up" style={{ animationDelay: '0.35s' }}>
              <div className="pro-tools-header">
                <h2>Pro Tools</h2>
                {!isPro && <span className="pro-pill">Pro</span>}
              </div>
              <div className="pro-tools-grid">
                <div className="pro-tool-card">
                  <h3>Google Sheets Import</h3>
                  <p>Pull data directly from Sheets without CSV downloads.</p>
                  <button
                    className="btn btn-secondary"
                    onClick={() => (isPro ? navigate('/sheets') : navigate('/pricing'))}
                  >
                    {isPro ? 'Try it' : 'Upgrade to Pro'}
                  </button>
                </div>
                <div className="pro-tool-card">
                  <h3>Scheduled Reports</h3>
                  <p>Email recurring PDF reports to stakeholders on a schedule.</p>
                  <button
                    className="btn btn-secondary"
                    onClick={() => (isPro ? navigate('/schedules') : navigate('/pricing'))}
                  >
                    {isPro ? 'Try it' : 'Upgrade to Pro'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <aside className="home-models-side">
          <ModelsList />
        </aside>
      </div>

      <footer className="home-footer">
        <p>Powered by advanced AI orchestration</p>
      </footer>
    </div>
  )
}
