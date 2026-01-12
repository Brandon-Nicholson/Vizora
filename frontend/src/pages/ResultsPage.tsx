import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAnalysis } from '../context/AnalysisContext'
import { useBilling } from '../context/BillingContext'
import { exportPdf } from '../api/client'
import NanobotBackground from '../components/common/NanobotBackground'
import FigureGallery from '../components/results/FigureGallery'
import MetricsTable from '../components/results/MetricsTable'
import SummaryPanel from '../components/results/SummaryPanel'
import PlanViewer from '../components/results/PlanViewer'
import PredictPanel from '../components/results/PredictPanel'
import { getActiveModelId, setActiveModelId } from '../utils/models'
import './ResultsPage.css'

type TabId = 'visualizations' | 'metrics' | 'summary' | 'plan' | 'predict'

interface Tab {
  id: TabId
  label: string
  icon: JSX.Element
}

const tabs: Tab[] = [
  {
    id: 'visualizations',
    label: 'Visualizations',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <path d="M21 15l-5-5L5 21" />
      </svg>
    )
  },
  {
    id: 'metrics',
    label: 'Model Metrics',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 3v18h18" />
        <path d="M7 16l4-8 4 5 5-10" />
      </svg>
    )
  },
  {
    id: 'summary',
    label: 'AI Summary',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <path d="M14 2v6h6" />
        <path d="M16 13H8" />
        <path d="M16 17H8" />
      </svg>
    )
  },
  {
    id: 'plan',
    label: 'Execution Plan',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 11l3 3L22 4" />
        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
      </svg>
    )
  },
  {
    id: 'predict',
    label: 'Predict',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 3v6" />
        <path d="M9 9h6" />
        <path d="M5 21h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-4l-2-2h-4l-2 2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2z" />
      </svg>
    )
  }
]

export default function ResultsPage() {
  const navigate = useNavigate()
  const { state, reset } = useAnalysis()
  const { canExportPdf, refreshBilling } = useBilling()
  const [activeTab, setActiveTab] = useState<TabId>('summary')
  const [isExporting, setIsExporting] = useState(false)
  const activeModelId = useMemo(() => getActiveModelId(), [])
  const hasResult = Boolean(state.result)

  // Redirect if no results
  useEffect(() => {
    if (!state.result && !activeModelId) {
      navigate('/')
    }
  }, [state.result, activeModelId, navigate])

  useEffect(() => {
    if (!hasResult) {
      setActiveTab('predict')
    }
  }, [hasResult])

  if (!state.result && !activeModelId) {
    return null
  }

  const figures = state.result?.figures || []
  const metrics = state.result?.metrics || null
  const summary_markdown = state.result?.summary_markdown || ''
  const plan = state.result?.plan || {}
  const errors = state.result?.errors || []

  const handleNewAnalysis = () => {
    reset()
    navigate('/mode')
  }

  const handleHome = () => {
    reset()
    navigate('/')
  }

  const handleMorePredictions = () => {
    if (state.jobId) {
      setActiveModelId(state.jobId)
    }
    setActiveTab('predict')
  }

  const handleExportPdf = async () => {
    if (!state.jobId) {
      return
    }

    setIsExporting(true)
    try {
      await refreshBilling()
      const blob = await exportPdf(state.jobId)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `vizora_report_${state.jobId.slice(0, 8)}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      if (err && typeof err === 'object' && 'status' in err && err.status === 402) {
        navigate('/pricing')
      } else {
        console.error('Failed to export PDF', err)
      }
    } finally {
      setIsExporting(false)
    }
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'visualizations':
        return <FigureGallery figures={figures} />
      case 'metrics':
        return <MetricsTable metrics={metrics} />
      case 'summary':
        return <SummaryPanel markdown={summary_markdown} />
      case 'plan':
        return <PlanViewer plan={plan} errors={errors} />
      case 'predict':
        return <PredictPanel modelId={state.jobId || activeModelId} />
      default:
        return null
    }
  }

  const visibleTabs = hasResult ? tabs : tabs.filter((tab) => tab.id === 'predict')

  return (
    <div className="results-page">
      <NanobotBackground particleCount={40} connectionDistance={80} />

      <div className="results-content container">
        {/* Header */}
        <div className="results-header animate-fade-in">
          <div className="header-left">
            {hasResult ? (
              <>
                <div className="success-badge">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                  Analysis Complete
                </div>
                <h1>Your Results</h1>
                <p className="results-summary">
                  Generated {figures.length} visualization{figures.length !== 1 ? 's' : ''}
                  {metrics && Object.keys(metrics).length > 0 &&
                    ` • Trained ${Object.keys(metrics).length} model${Object.keys(metrics).length !== 1 ? 's' : ''}`
                  }
                </p>
              </>
            ) : (
              <>
                <div className="success-badge">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 3v18" />
                    <path d="M5 12h14" />
                  </svg>
                  Saved Model
                </div>
                <h1>Run Predictions</h1>
                <p className="results-summary">
                  Upload a new CSV to score with your saved model.
                </p>
              </>
            )}
          </div>
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={handleHome}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 9l9-7 9 7" />
                <path d="M9 22V12h6v10" />
              </svg>
              Home
            </button>
            {hasResult && (
              <>
                <button
                  className={`btn ${canExportPdf() ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={handleExportPdf}
                  disabled={isExporting}
                  title={canExportPdf() ? 'Export as PDF' : 'Upgrade to Pro to export PDF reports'}
                >
                  {isExporting ? (
                    <>
                      <span className="btn-spinner" />
                      Exporting...
                    </>
                  ) : (
                    <>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="12" y1="18" x2="12" y2="12" />
                        <line x1="9" y1="15" x2="15" y2="15" />
                      </svg>
                      Export PDF
                      {!canExportPdf() && <span className="pro-badge">PRO</span>}
                    </>
                  )}
                </button>
                <button className="btn btn-secondary" onClick={handleMorePredictions}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 12h18" />
                    <path d="M12 3v18" />
                  </svg>
                  Use This Model for More Predictions
                </button>
              </>
            )}
            <button className="btn btn-secondary" onClick={handleNewAnalysis}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 5v14" />
                <path d="M5 12h14" />
              </svg>
              New Analysis
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="results-tabs animate-slide-up">
          {visibleTabs.map((tab) => (
            <button
              key={tab.id}
              className={`results-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {tab.id === 'visualizations' && figures.length > 0 && (
                <span className="tab-badge">{figures.length}</span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="results-tab-content animate-fade-in">
          {renderTabContent()}
        </div>
      </div>
    </div>
  )
}
