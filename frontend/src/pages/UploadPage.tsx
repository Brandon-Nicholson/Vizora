import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAnalysis } from '../context/AnalysisContext'
import { useBilling } from '../context/BillingContext'
import { usePolling } from '../hooks/usePolling'
import { startAnalysis } from '../api/client'
import NanobotBackground from '../components/common/NanobotBackground'
import LoadingSpinner from '../components/common/LoadingSpinner'
import Dropzone from '../components/upload/Dropzone'
import UsageBanner from '../components/billing/UsageBanner'
import { setActiveModelId, storeModelId } from '../utils/models'
import './UploadPage.css'

const modeLabels: Record<string, string> = {
  eda: 'Exploratory Analysis',
  predictive: 'Predictive Modeling',
  hybrid: 'Full Analysis',
  forecast: 'Time Series Forecast'
}

export default function UploadPage() {
  const navigate = useNavigate()
  const {
    state,
    setFile,
    setGoal,
    setTargetColumn,
    startUpload,
    jobCreated,
    updateProgress,
    analysisComplete,
    analysisError
  } = useAnalysis()
  const { state: billingState, canRunAnalysis, incrementUsage } = useBilling()

  const [localGoal, setLocalGoal] = useState(state.goal)
  const [localTarget, setLocalTarget] = useState(state.targetColumn)

  // Forecast-specific state
  const [forecastHorizon, setForecastHorizon] = useState(30)
  const [forecastFrequency, setForecastFrequency] = useState<'daily' | 'weekly' | 'monthly'>('daily')
  const [dateColumn, setDateColumn] = useState('')

  // Redirect if no mode selected
  useEffect(() => {
    if (!state.mode) {
      navigate('/mode')
    }
  }, [state.mode, navigate])

  // Handle polling for job status
  usePolling({
    jobId: state.jobId,
    onProgress: updateProgress,
    onComplete: (result) => {
      if (state.jobId) {
        storeModelId(state.jobId)
        setActiveModelId(state.jobId)
      }
      analysisComplete(result)
      navigate('/results')
    },
    onError: analysisError
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!state.file || !localGoal.trim()) return

    setGoal(localGoal)
    setTargetColumn(localTarget)
    startUpload()

    try {
      const forecastOptions = state.mode === 'forecast' ? {
        horizon: forecastHorizon,
        frequency: forecastFrequency,
        dateColumn: dateColumn || undefined,
      } : undefined

      const response = await startAnalysis(
        state.file,
        state.mode!,
        localGoal,
        localTarget || undefined,
        forecastOptions
      )
      jobCreated(response.job_id)
      incrementUsage() // Optimistic update
    } catch (err) {
      analysisError(err instanceof Error ? err.message : 'Failed to start analysis')
    }
  }

  const isProcessing = state.status === 'uploading' || state.status === 'processing'
  const needsTarget = state.mode === 'predictive' || state.mode === 'hybrid'
  const isAtLimit = !canRunAnalysis()
  const isPro = billingState.tier === 'pro' || billingState.monthlyLimit === -1
  const showUpgradeCta = Boolean(
    state.error
      && state.error.toLowerCase().includes('upgrade to pro')
  )
  const canSubmit = Boolean(
    state.file
      && localGoal.trim().length >= 10
      && (!needsTarget || localTarget.trim().length > 0)
      && !isProcessing
      && !isAtLimit
  )

  if (isProcessing) {
    return (
      <div className="upload-page">
        <NanobotBackground particleCount={60} />

        <div className="processing-container">
          <LoadingSpinner size="lg" />

          <h2 className="processing-title">
            {state.status === 'uploading' ? 'Uploading...' : 'Analyzing Your Data'}
          </h2>

          {state.progress && (
            <>
              <p className="processing-step">{state.progress.current_step}</p>
              <div className="progress-bar" style={{ width: '300px' }}>
                <div
                  className="progress-bar-fill"
                  style={{ width: `${state.progress.percentage}%` }}
                />
              </div>
              <p className="processing-percent">{state.progress.percentage}%</p>
            </>
          )}

          <p className="processing-hint">This may take 30-60 seconds...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="upload-page">
      <NanobotBackground particleCount={60} connectionDistance={100} />

      <div className="upload-content container">
        <button className="back-btn" onClick={() => navigate('/mode')}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5" />
            <path d="m12 19-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="upload-header animate-fade-in">
          <div className="mode-badge">{state.mode && modeLabels[state.mode]}</div>
          <h1>Upload Your Dataset</h1>
          <p>Provide your CSV file and describe your analysis goal</p>
        </div>

        <UsageBanner
          usage={billingState.monthlyUsage}
          limit={billingState.monthlyLimit}
          tier={billingState.tier}
        />

        {state.error && (
          <div className="error-banner animate-slide-up">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{state.error}</span>
            {showUpgradeCta && (
              <button
                type="button"
                className="btn btn-secondary error-banner-action"
                onClick={() => navigate('/pricing')}
              >
                Upgrade to Pro
              </button>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="upload-form animate-slide-up">
          <div className="form-section">
            <div className="form-section-header">
              <label className="input-label">Dataset</label>
              {isPro && (
                <button
                  type="button"
                  className="btn btn-secondary sheets-link-btn"
                  onClick={() => navigate('/sheets')}
                >
                  Use Google Sheets
                </button>
              )}
            </div>
            <Dropzone
              onFileSelect={setFile}
              selectedFile={state.file}
            />
          </div>

          <div className="form-section">
            <label htmlFor="goal" className="input-label">
              Analysis Goal <span className="required">*</span>
            </label>
            <textarea
              id="goal"
              className="input textarea"
              placeholder="Describe what you want to learn from your data. For example: 'Identify factors that predict customer churn' or 'Understand the distribution of sales across regions'"
              value={localGoal}
              onChange={(e) => setLocalGoal(e.target.value)}
              rows={4}
            />
            <span className="input-hint">
              {localGoal.length < 10
                ? `${10 - localGoal.length} more characters needed`
                : 'Good to go!'}
            </span>
          </div>

          {(state.mode === 'predictive' || state.mode === 'hybrid') && (
            <div className="form-section animate-slide-up">
              <label htmlFor="target" className="input-label">
                Target Column <span className="required">*</span>
              </label>
              <input
                id="target"
                type="text"
                className="input"
                placeholder="e.g., 'churn', 'price', 'outcome'"
                value={localTarget}
                onChange={(e) => setLocalTarget(e.target.value)}
              />
              <span className="input-hint">
                The column you want to predict. We'll try to match similar column names.
              </span>
            </div>
          )}

          {state.mode === 'forecast' && (
            <div className="form-section animate-slide-up forecast-config">
              <label className="input-label">Forecast Configuration</label>

              <div className="forecast-grid">
                <div className="forecast-field">
                  <label htmlFor="forecastTarget" className="input-label-small">
                    Value to Forecast
                  </label>
                  <input
                    id="forecastTarget"
                    type="text"
                    className="input"
                    placeholder="e.g., 'sales', 'revenue', 'demand'"
                    value={localTarget}
                    onChange={(e) => setLocalTarget(e.target.value)}
                  />
                </div>

                <div className="forecast-field">
                  <label htmlFor="dateColumn" className="input-label-small">
                    Date Column (optional)
                  </label>
                  <input
                    id="dateColumn"
                    type="text"
                    className="input"
                    placeholder="Auto-detected if empty"
                    value={dateColumn}
                    onChange={(e) => setDateColumn(e.target.value)}
                  />
                </div>

                <div className="forecast-field">
                  <label htmlFor="horizon" className="input-label-small">
                    Forecast Horizon
                  </label>
                  <input
                    id="horizon"
                    type="number"
                    className="input"
                    min={1}
                    max={365}
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 30)}
                  />
                </div>

                <div className="forecast-field">
                  <label className="input-label-small">Frequency</label>
                  <div className="frequency-buttons">
                    {(['daily', 'weekly', 'monthly'] as const).map((freq) => (
                      <button
                        key={freq}
                        type="button"
                        className={`frequency-btn ${forecastFrequency === freq ? 'active' : ''}`}
                        onClick={() => setForecastFrequency(freq)}
                      >
                        {freq.charAt(0).toUpperCase() + freq.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <span className="input-hint">
                Configure how far into the future to forecast and at what frequency.
              </span>
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary btn-large submit-btn"
            disabled={!canSubmit}
          >
            <span>Begin Analysis</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polygon points="10 8 16 12 10 16 10 8" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}
