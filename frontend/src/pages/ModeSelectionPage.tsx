import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAnalysis } from '../context/AnalysisContext'
import NanobotBackground from '../components/common/NanobotBackground'
import type { AnalysisMode } from '../types'
import './ModeSelectionPage.css'

interface ModeOption {
  id: AnalysisMode
  title: string
  description: string
  icon: JSX.Element
  features: string[]
  recommended: boolean
}

const modeOptions: ModeOption[] = [
  {
    id: 'eda',
    title: 'Exploratory Analysis',
    description: 'Understand your data through visualizations and statistical summaries.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 3v18h18" />
        <path d="M7 16l4-8 4 5 5-10" />
      </svg>
    ),
    features: ['Distributions & histograms', 'Correlation analysis', 'Missing data detection'],
    recommended: false
  },
  {
    id: 'predictive',
    title: 'Predictive Modeling',
    description: 'Build and evaluate machine learning models for your target variable.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 2a10 10 0 0 1 10 10" />
        <path d="M12 12l4-4" />
        <circle cx="12" cy="12" r="2" />
      </svg>
    ),
    features: ['Model training & comparison', 'Performance metrics', 'Feature importance'],
    recommended: false
  },
  {
    id: 'hybrid',
    title: 'Full Analysis',
    description: 'Complete end-to-end analysis combining exploration and modeling.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
    features: ['Everything in EDA', 'Plus predictive models', 'AI-generated insights'],
    recommended: true
  },
  {
    id: 'forecast',
    title: 'Time Series Forecast',
    description: 'Predict future values using time series analysis and forecasting models.',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 3v18h18" />
        <path d="M7 14l3-3 3 2 4-4" />
        <path d="M17 9l4 0" strokeDasharray="2 2" />
        <path d="M21 9l0 4" strokeDasharray="2 2" />
      </svg>
    ),
    features: ['Seasonal decomposition', 'Prophet & exponential smoothing', 'Confidence intervals'],
    recommended: false
  }
]

export default function ModeSelectionPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { setMode } = useAnalysis()

  const handleSelectMode = (mode: AnalysisMode) => {
    setMode(mode)
    // Navigate to the next page based on query parameter
    const next = searchParams.get('next')
    if (next === 'sheets') {
      navigate('/sheets')
    } else {
      navigate('/upload')
    }
  }

  return (
    <div className="mode-page">
      <NanobotBackground particleCount={60} connectionDistance={100} />

      <div className="mode-content container">
        <button className="back-btn" onClick={() => navigate('/')}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5" />
            <path d="m12 19-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="mode-header animate-fade-in">
          <h1>Choose Analysis Mode</h1>
          <p>Select the type of analysis that best fits your needs</p>
        </div>

        <div className="mode-cards">
          {modeOptions.map((option, index) => (
            <div
              key={option.id}
              className={`mode-card card card-clickable animate-slide-up${option.recommended ? ' mode-card-recommended' : ''}`}
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => handleSelectMode(option.id)}
            >
              <div className="mode-card-icon">{option.icon}</div>
              <h3 className="mode-card-title">{option.title}</h3>
              <p className="mode-card-description">{option.description}</p>
              <ul className="mode-card-features">
                {option.features.map((feature, i) => (
                  <li key={i}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>
              <div className="mode-card-action">
                <span>Select</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14" />
                  <path d="m12 5 7 7-7 7" />
                </svg>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
