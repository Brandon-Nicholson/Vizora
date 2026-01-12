import type { ModelMetrics } from '../../types'
import './MetricsTable.css'

interface MetricsTableProps {
  metrics: Record<string, ModelMetrics> | null
}

export default function MetricsTable({ metrics }: MetricsTableProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M3 9h18" />
          <path d="M9 21V9" />
        </svg>
        <p>No model metrics available</p>
        <span>Run predictive or hybrid analysis to see model performance</span>
      </div>
    )
  }

  const modelNames = Object.keys(metrics)
  const firstModel = metrics[modelNames[0]]
  const metricNames = Object.keys(firstModel)

  const formatMetricName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase())
  }

  const formatValue = (value: number | string) => {
    if (typeof value === 'number') {
      return value.toFixed(4)
    }
    return value
  }

  const getMetricClass = (name: string, value: number) => {
    // Highlight good metrics
    if (name === 'accuracy' || name === 'f1' || name === 'roc_auc') {
      if (value >= 0.9) return 'metric-excellent'
      if (value >= 0.8) return 'metric-good'
      if (value >= 0.7) return 'metric-ok'
    }
    if (name === 'brier_score') {
      if (value <= 0.1) return 'metric-excellent'
      if (value <= 0.15) return 'metric-good'
      if (value <= 0.25) return 'metric-ok'
    }
    return ''
  }

  return (
    <div className="metrics-container">
      <div className="metrics-table-wrapper">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Metric</th>
              {modelNames.map((name) => (
                <th key={name}>{formatMetricName(name)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricNames.map((metricName) => (
              <tr key={metricName}>
                <td className="metric-name">{formatMetricName(metricName)}</td>
                {modelNames.map((modelName) => {
                  const value = metrics[modelName][metricName]
                  return (
                    <td
                      key={modelName}
                      className={typeof value === 'number' ? getMetricClass(metricName, value) : ''}
                    >
                      {formatValue(value)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="metrics-legend">
        <span className="legend-item">
          <span className="legend-dot metric-excellent"></span>
          Excellent
        </span>
        <span className="legend-item">
          <span className="legend-dot metric-good"></span>
          Good
        </span>
        <span className="legend-item">
          <span className="legend-dot metric-ok"></span>
          Acceptable
        </span>
      </div>
    </div>
  )
}
