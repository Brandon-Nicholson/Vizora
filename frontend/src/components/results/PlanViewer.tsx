import { useState } from 'react'
import './PlanViewer.css'

interface PlanViewerProps {
  plan: Record<string, unknown>
  errors: string[]
}

interface StepItem {
  action?: string
  type?: string
  section?: string
  [key: string]: unknown
}

export default function PlanViewer({ plan, errors }: PlanViewerProps) {
  const [showRaw, setShowRaw] = useState(false)

  const directSteps = (plan.steps || plan.actions || []) as StepItem[]
  const notes = (plan.notes || []) as string[]

  const sectionOrder = ['cleaning', 'eda', 'analysis', 'preprocessing', 'modeling', 'evaluation']
  const derivedSteps: StepItem[] = sectionOrder.flatMap((section) => {
    const items = (plan as Record<string, unknown>)[section]
    if (!Array.isArray(items)) {
      return []
    }
    return items.map((item) => ({
      ...(item as Record<string, unknown>),
      section
    }))
  })

  const steps = directSteps.length > 0 ? directSteps : derivedSteps

  const getActionIcon = (action: string) => {
    if (action.includes('histogram') || action.includes('plot') || action.includes('chart')) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 3v18h18" />
          <rect x="7" y="10" width="3" height="8" />
          <rect x="14" y="6" width="3" height="12" />
        </svg>
      )
    }
    if (action.includes('train') || action.includes('model')) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
      )
    }
    if (action.includes('encode') || action.includes('scale') || action.includes('clean')) {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 3v18" />
          <path d="M18 9l-6-6-6 6" />
          <path d="M6 15l6 6 6-6" />
        </svg>
      )
    }
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    )
  }

  return (
    <div className="plan-viewer">
      {/* Steps/Actions */}
      <div className="plan-section">
        <div className="section-header">
          <h3>Execution Steps</h3>
          <span className="step-count">{steps.length} steps</span>
        </div>
        <div className="steps-list">
          {steps.map((step, index) => (
            <div key={index} className="step-item">
              <div className="step-number">{index + 1}</div>
              <div className="step-icon">
                {getActionIcon(step.action || step.type || '')}
              </div>
              <div className="step-content">
                <span className="step-action">{step.action || step.type}</span>
                {'section' in step && (
                  <span className="step-section">{String(step.section)}</span>
                )}
                {Object.entries(step).filter(([k]) => k !== 'action' && k !== 'type' && k !== 'section').length > 0 && (
                  <span className="step-params">
                    {Object.entries(step)
                      .filter(([k]) => k !== 'action' && k !== 'type' && k !== 'section')
                      .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
                      .join(', ')}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Notes */}
      {notes.length > 0 && (
        <div className="plan-section">
          <h3>Notes & Assumptions</h3>
          <ul className="notes-list">
            {notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Errors/Warnings */}
      {errors.length > 0 && (
        <div className="plan-section errors-section">
          <h3>Warnings & Errors</h3>
          <ul className="errors-list">
            {errors.map((error, index) => (
              <li key={index}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                {error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Raw JSON toggle */}
      <div className="plan-section">
        <button className="toggle-raw" onClick={() => setShowRaw(!showRaw)}>
          {showRaw ? 'Hide' : 'Show'} Raw JSON
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            style={{ transform: showRaw ? 'rotate(180deg)' : 'none' }}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
        {showRaw && (
          <pre className="raw-json">
            {JSON.stringify(plan, null, 2)}
          </pre>
        )}
      </div>
    </div>
  )
}
