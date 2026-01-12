import { useState } from 'react'
import type { FigureData } from '../../types'
import './FigureGallery.css'

interface FigureGalleryProps {
  figures: FigureData[]
}

export default function FigureGallery({ figures }: FigureGalleryProps) {
  const [selectedFigure, setSelectedFigure] = useState<FigureData | null>(null)

  if (figures.length === 0) {
    return (
      <div className="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <path d="M21 15l-5-5L5 21" />
        </svg>
        <p>No visualizations generated</p>
      </div>
    )
  }

  return (
    <>
      <div className="figure-gallery">
        {figures.map((fig) => (
          <div
            key={fig.id}
            className="figure-card"
            onClick={() => setSelectedFigure(fig)}
          >
            <img src={fig.base64_png} alt={fig.name} loading="lazy" />
            <div className="figure-info">
              <span className="figure-type">{fig.type}</span>
              <span className="figure-name">{fig.name}</span>
            </div>
          </div>
        ))}
      </div>

      {selectedFigure && (
        <div className="lightbox" onClick={() => setSelectedFigure(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setSelectedFigure(null)}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
            <img src={selectedFigure.base64_png} alt={selectedFigure.name} />
            <div className="lightbox-info">
              <span className="figure-type">{selectedFigure.type}</span>
              <span className="figure-name">{selectedFigure.name}</span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
