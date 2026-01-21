import { useState, FormEvent } from 'react'
import { useNavigate, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import NanobotBackground from '../components/common/NanobotBackground'
import './AuthPages.css'

interface LocationState {
  from?: { pathname: string }
}

export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { signIn, state } = useAuth()

  // Get the redirect path from location state, or default to /mode
  const from = (location.state as LocationState)?.from?.pathname || '/mode'

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setIsLoading(true)

    const { error } = await signIn(email, password)

    if (error) {
      setError(error.message)
      setIsLoading(false)
    } else {
      navigate(from, { replace: true })
    }
  }

  // If not configured, show setup message
  if (!state.isConfigured) {
    return (
      <div className="auth-page">
        <NanobotBackground particleCount={60} connectionDistance={120} />
        <div className="auth-container animate-fade-in">
          <div className="auth-header">
            <Link to="/" className="auth-logo">
              <span className="logo-v">V</span>
              <span className="logo-rest">izora</span>
            </Link>
            <h1>Authentication Not Configured</h1>
            <p className="auth-subtitle">
              Please configure Supabase credentials in your environment variables.
            </p>
          </div>
          <div className="setup-instructions">
            <p>Add these to <code>frontend/.env</code>:</p>
            <pre>
              VITE_SUPABASE_URL=your-url{'\n'}
              VITE_SUPABASE_ANON_KEY=your-key
            </pre>
          </div>
          <Link to="/" className="btn btn-secondary">
            Back to Home
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-page">
      <NanobotBackground particleCount={60} connectionDistance={120} />

      <div className="auth-container animate-fade-in">
        <div className="auth-header">
          <Link to="/" className="auth-logo">
            <span className="logo-v">V</span>
            <span className="logo-rest">izora</span>
          </Link>
          <h1>Welcome Back</h1>
          <p className="auth-subtitle">Sign in to continue your analysis</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {error && (
            <div className="auth-error animate-slide-up">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="email"
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              autoComplete="current-password"
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-large auth-submit"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Don't have an account?{' '}
            <Link to="/signup" className="auth-link">
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
