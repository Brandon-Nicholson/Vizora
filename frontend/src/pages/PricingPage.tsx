import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { createCheckoutSession } from '../api/client'
import NanobotBackground from '../components/common/NanobotBackground'
import SiteFooter from '../components/common/SiteFooter'
import './PricingPage.css'

interface PricingTier {
  name: string
  price: string
  period: string
  description: string
  features: string[]
  cta: string
  highlighted: boolean
  tier: 'free' | 'pro'
}

const tiers: PricingTier[] = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    description: 'Perfect for trying out Vizora',
    features: [
      '5 analyses per month',
      'EDA, Predictive, and Hybrid modes',
      'AI-generated insights',
      'Basic visualizations',
      'Model training & evaluation',
    ],
    cta: 'Get Started',
    highlighted: false,
    tier: 'free',
  },
  {
    name: 'Pro',
    price: '$19',
    period: 'per month',
    description: 'For data professionals who need more',
    features: [
      'Unlimited analyses',
      'Everything in Free',
      'PDF report export',
      'Scheduled reports via email',
      'Google Sheets integration',
      'Priority support',
    ],
    cta: 'Upgrade to Pro',
    highlighted: true,
    tier: 'pro',
  },
]

export default function PricingPage() {
  const navigate = useNavigate()
  const { state } = useAuth()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSelectPlan = async (tier: PricingTier) => {
    if (tier.tier === 'free') {
      if (state.user) {
        navigate('/mode')
      } else {
        navigate('/signup')
      }
      return
    }

    // Pro tier - need to be logged in
    if (!state.user) {
      navigate('/signup')
      return
    }

    // Create Stripe checkout session
    setIsLoading(true)
    setError(null)

    try {
      const { checkout_url } = await createCheckoutSession()
      window.location.href = checkout_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start checkout')
      setIsLoading(false)
    }
  }

  return (
    <div className="pricing-page">
      <NanobotBackground particleCount={80} connectionDistance={130} />

      <div className="pricing-content animate-fade-in">
        <button className="back-btn" onClick={() => navigate('/')}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5" />
            <path d="m12 19-7-7 7-7" />
          </svg>
          Home
        </button>

        <div className="demo-banner">
          <span className="demo-pill">Demo mode</span>
          <span className="demo-text">Test payments only. No real charges.</span>
        </div>

        <div className="pricing-header">
          <h1>Simple, Transparent Pricing</h1>
          <p>Start free, upgrade when you need more</p>
        </div>

        {error && (
          <div className="pricing-error animate-slide-up">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        <div className="pricing-tiers">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={`pricing-card ${tier.highlighted ? 'highlighted' : ''}`}
            >
              {tier.highlighted && <div className="popular-badge">Most Popular</div>}

              <div className="card-header">
                <h2>{tier.name}</h2>
                <div className="price">
                  <span className="amount">{tier.price}</span>
                  <span className="period">/{tier.period}</span>
                </div>
                <p className="description">{tier.description}</p>
              </div>

              <ul className="features-list">
                {tier.features.map((feature, index) => (
                  <li key={index}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                className={`btn ${tier.highlighted ? 'btn-primary' : 'btn-secondary'} btn-large`}
                onClick={() => handleSelectPlan(tier)}
                disabled={isLoading}
              >
                {isLoading && tier.tier === 'pro' ? (
                  <>
                    <span className="spinner" />
                    Loading...
                  </>
                ) : (
                  tier.cta
                )}
              </button>
            </div>
          ))}
        </div>

        <div className="pricing-faq">
          <h3>Questions?</h3>
          <div className="faq-grid">
            <div className="faq-item">
              <h4>Can I cancel anytime?</h4>
              <p>Yes, you can cancel your subscription at any time. You'll keep Pro access until the end of your billing period.</p>
            </div>
            <div className="faq-item">
              <h4>What payment methods do you accept?</h4>
              <p>We accept all major credit cards through Stripe's secure payment system.</p>
            </div>
            <div className="faq-item">
              <h4>What happens when I hit my limit?</h4>
              <p>On the Free plan, you'll be prompted to upgrade. Your existing analyses and models are always accessible.</p>
            </div>
            <div className="faq-item">
              <h4>Do you offer refunds?</h4>
              <p>We offer a full refund within 7 days of your first payment if you're not satisfied.</p>
            </div>
          </div>
        </div>
      </div>

      <SiteFooter />
    </div>
  )
}
