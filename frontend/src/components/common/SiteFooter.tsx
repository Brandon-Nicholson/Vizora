import { Link } from 'react-router-dom'
import './SiteFooter.css'

interface SiteFooterProps {
  className?: string
}

export default function SiteFooter({ className = '' }: SiteFooterProps) {
  const footerClassName = ['site-footer', className].filter(Boolean).join(' ')

  return (
    <footer className={footerClassName}>
      <div className="site-footer-inner">
        <div className="site-footer-brand">
          <span className="site-footer-logo">Vizora</span>
          <span className="site-footer-tagline">
            AI-powered data analysis for teams that move fast.
          </span>
        </div>

        <nav className="site-footer-links" aria-label="Footer">
          <Link to="/pricing">Pricing</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/terms">Terms</Link>
          <Link to="/refunds">Refunds</Link>
          <Link to="/contact">Contact</Link>
        </nav>

        <div className="site-footer-contact">
          <span>Support:</span>{' '}
          <a href="mailto:support@agentvizora.com">support@agentvizora.com</a>
        </div>
      </div>
    </footer>
  )
}
