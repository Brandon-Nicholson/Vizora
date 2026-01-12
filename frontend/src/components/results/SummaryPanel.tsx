import ReactMarkdown from 'react-markdown'
import './SummaryPanel.css'

interface SummaryPanelProps {
  markdown: string
}

export default function SummaryPanel({ markdown }: SummaryPanelProps) {
  if (!markdown) {
    return (
      <div className="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <path d="M14 2v6h6" />
          <path d="M16 13H8" />
          <path d="M16 17H8" />
          <path d="M10 9H8" />
        </svg>
        <p>No summary available</p>
      </div>
    )
  }

  return (
    <div className="summary-panel">
      <div className="summary-content">
        <ReactMarkdown>{markdown}</ReactMarkdown>
      </div>
    </div>
  )
}
