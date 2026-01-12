/**
 * Usage Banner Component
 *
 * Shows the user's current usage and warns when approaching limits.
 */

import { useNavigate } from 'react-router-dom'
import './UsageBanner.css'

interface UsageBannerProps {
  usage: number
  limit: number
  tier: 'free' | 'pro'
}

export default function UsageBanner({ usage, limit, tier }: UsageBannerProps) {
  const navigate = useNavigate()

  // Pro users have unlimited, don't show banner
  if (tier === 'pro' || limit === -1) {
    return null
  }

  const percentage = (usage / limit) * 100
  const remaining = limit - usage

  // Don't show if usage is low
  if (percentage < 60) {
    return null
  }

  const isNearLimit = percentage >= 80 && percentage < 100
  const isAtLimit = percentage >= 100

  return (
    <div className={`usage-banner ${isAtLimit ? 'at-limit' : isNearLimit ? 'near-limit' : ''}`}>
      <div className="usage-content">
        <div className="usage-icon">
          {isAtLimit ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          )}
        </div>
        <div className="usage-text">
          {isAtLimit ? (
            <span>You've used all {limit} analyses this month.</span>
          ) : (
            <span>You've used {usage} of {limit} analyses. {remaining} remaining.</span>
          )}
        </div>
        <div className="usage-progress">
          <div
            className="usage-progress-bar"
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </div>
      <button
        className="btn btn-primary btn-small"
        onClick={() => navigate('/pricing')}
      >
        Upgrade to Pro
      </button>
    </div>
  )
}
