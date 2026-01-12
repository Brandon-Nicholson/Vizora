import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useBilling } from '../context/BillingContext'
import { useAnalysis } from '../context/AnalysisContext'
import {
  getGoogleStatus,
  getGoogleAuthUrl,
  disconnectGoogle,
  listSpreadsheets,
  getSpreadsheetDetails,
  getSheetPreview,
  importGoogleSheet,
  type Spreadsheet,
  type SpreadsheetDetails,
  type SheetPreview
} from '../api/client'
import NanobotBackground from '../components/common/NanobotBackground'
import './SheetsPage.css'

export default function SheetsPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { state: billingState } = useBilling()
  const { setFile, setMode } = useAnalysis()

  const [isConnected, setIsConnected] = useState(false)
  const [isConfigured, setIsConfigured] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const [spreadsheets, setSpreadsheets] = useState<Spreadsheet[]>([])
  const [selectedSpreadsheet, setSelectedSpreadsheet] = useState<SpreadsheetDetails | null>(null)
  const [selectedSheet, setSelectedSheet] = useState<string | null>(null)
  const [preview, setPreview] = useState<SheetPreview | null>(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const [isImporting, setIsImporting] = useState(false)

  const isPro = billingState.tier === 'pro' || billingState.monthlyLimit === -1
  const homeButton = (
    <button className="btn btn-secondary home-btn" onClick={() => navigate('/')}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 9l9-7 9 7" />
        <path d="M9 22V12h6v10" />
      </svg>
      Home
    </button>
  )

  // Check connection status on mount
  useEffect(() => {
    checkStatus()

    // Handle OAuth callback
    const connected = searchParams.get('connected')
    const callbackError = searchParams.get('error')

    if (connected === 'true') {
      setSuccess('Successfully connected to Google Sheets!')
      setIsConnected(true)
      loadSpreadsheets()
    } else if (callbackError) {
      setError(`Connection failed: ${callbackError}`)
    }
  }, [searchParams])

  const checkStatus = async () => {
    try {
      const status = await getGoogleStatus()
      setIsConnected(status.connected)
      setIsConfigured(status.configured)
      if (status.connected) {
        loadSpreadsheets()
      }
    } catch (err) {
      console.error('Failed to check status', err)
    } finally {
      setIsLoading(false)
    }
  }

  const loadSpreadsheets = async () => {
    try {
      const data = await listSpreadsheets()
      setSpreadsheets(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load spreadsheets')
    }
  }

  const handleConnect = async () => {
    setIsConnecting(true)
    setError(null)

    try {
      const { auth_url } = await getGoogleAuthUrl()
      window.location.href = auth_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect')
      setIsConnecting(false)
    }
  }

  const handleDisconnect = async () => {
    if (!confirm('Are you sure you want to disconnect Google Sheets?')) return

    try {
      await disconnectGoogle()
      setIsConnected(false)
      setSpreadsheets([])
      setSelectedSpreadsheet(null)
      setPreview(null)
      setSuccess('Disconnected from Google Sheets')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to disconnect')
    }
  }

  const handleSelectSpreadsheet = async (spreadsheet: Spreadsheet) => {
    setIsLoadingPreview(true)
    setSelectedSheet(null)
    setPreview(null)

    try {
      const details = await getSpreadsheetDetails(spreadsheet.id)
      setSelectedSpreadsheet(details)

      // Load preview of first sheet
      if (details.sheets.length > 0) {
        const firstSheet = details.sheets[0].name
        setSelectedSheet(firstSheet)
        const previewData = await getSheetPreview(spreadsheet.id, firstSheet)
        setPreview(previewData)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load spreadsheet')
    } finally {
      setIsLoadingPreview(false)
    }
  }

  const handleSelectSheet = async (sheetName: string) => {
    if (!selectedSpreadsheet) return

    setSelectedSheet(sheetName)
    setIsLoadingPreview(true)

    try {
      const previewData = await getSheetPreview(selectedSpreadsheet.id, sheetName)
      setPreview(previewData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sheet preview')
    } finally {
      setIsLoadingPreview(false)
    }
  }

  const handleImport = async () => {
    if (!selectedSpreadsheet || !selectedSheet) return

    setIsImporting(true)
    setError(null)

    try {
      const imported = await importGoogleSheet(selectedSpreadsheet.id, selectedSheet)

      // Create a File object from the CSV data
      const blob = new Blob([imported.data], { type: 'text/csv' })
      const file = new File([blob], imported.filename, { type: 'text/csv' })

      // Set file in analysis context and navigate
      setFile(file)
      setMode('hybrid')
      navigate('/upload')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import sheet')
      setIsImporting(false)
    }
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Unknown'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  if (!isPro) {
    return (
      <div className="sheets-page">
        <NanobotBackground particleCount={40} />
        <div className="sheets-content container">
          <div className="page-nav">
            {homeButton}
          </div>
          <div className="upgrade-prompt">
            <div className="upgrade-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M3 9h18" />
                <path d="M9 21V9" />
              </svg>
            </div>
            <h2>Google Sheets Integration</h2>
            <p>
              Connect your Google account to import spreadsheets directly into Vizora.
              Analyze your Google Sheets data without downloading and uploading CSV files.
            </p>
            <button className="btn btn-primary btn-large" onClick={() => navigate('/pricing')}>
              Upgrade to Pro
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="sheets-page">
        <NanobotBackground particleCount={40} />
        <div className="sheets-content container">
          <div className="page-nav">
            {homeButton}
          </div>
          <div className="loading-state">
            <div className="spinner" />
            <p>Loading...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="sheets-page">
      <NanobotBackground particleCount={40} connectionDistance={80} />

      <div className="sheets-content container">
        <div className="page-nav">
          {homeButton}
        </div>
        <div className="sheets-header animate-fade-in">
          <div className="header-left">
            <h1>Google Sheets</h1>
            <p>Import spreadsheets directly from Google Drive</p>
          </div>
          {isConnected && (
            <button className="btn btn-secondary" onClick={handleDisconnect}>
              Disconnect
            </button>
          )}
        </div>

        {error && (
          <div className="error-banner animate-slide-up">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="error-dismiss">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18" />
                <path d="M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        {success && (
          <div className="success-banner animate-slide-up">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            <span>{success}</span>
            <button onClick={() => setSuccess(null)} className="success-dismiss">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18" />
                <path d="M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        {!isConnected ? (
          <div className="connect-prompt animate-fade-in">
            <div className="google-icon">
              <svg viewBox="0 0 24 24" width="64" height="64">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
            </div>
            <h2>Connect Google Sheets</h2>
            <p>
              Sign in with Google to access your spreadsheets.
              We only request read-only access to your files.
            </p>
            {!isConfigured ? (
              <p className="config-warning">
                Google Sheets integration is not configured. Contact your administrator.
              </p>
            ) : (
              <button
                className="btn btn-primary btn-large google-btn"
                onClick={handleConnect}
                disabled={isConnecting}
              >
                {isConnecting ? (
                  <>
                    <span className="btn-spinner" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <svg viewBox="0 0 24 24" width="20" height="20">
                      <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                      <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                      <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                      <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                    Sign in with Google
                  </>
                )}
              </button>
            )}
          </div>
        ) : (
          <div className="sheets-browser animate-slide-up">
            <div className="browser-sidebar">
              <h3>Your Spreadsheets</h3>
              {spreadsheets.length === 0 ? (
                <p className="no-sheets">No spreadsheets found</p>
              ) : (
                <div className="spreadsheet-list">
                  {spreadsheets.map((sheet) => (
                    <button
                      key={sheet.id}
                      className={`spreadsheet-item ${selectedSpreadsheet?.id === sheet.id ? 'selected' : ''}`}
                      onClick={() => handleSelectSpreadsheet(sheet)}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <path d="M3 9h18" />
                        <path d="M9 21V9" />
                      </svg>
                      <div className="spreadsheet-info">
                        <span className="spreadsheet-name">{sheet.name}</span>
                        <span className="spreadsheet-date">{formatDate(sheet.modified_at)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="browser-main">
              {!selectedSpreadsheet ? (
                <div className="select-prompt">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M15 15l6 6" />
                    <circle cx="10" cy="10" r="7" />
                  </svg>
                  <p>Select a spreadsheet to preview</p>
                </div>
              ) : isLoadingPreview ? (
                <div className="loading-state">
                  <div className="spinner" />
                  <p>Loading preview...</p>
                </div>
              ) : (
                <>
                  <div className="preview-header">
                    <h3>{selectedSpreadsheet.name}</h3>
                    {selectedSpreadsheet.sheets.length > 1 && (
                      <div className="sheet-tabs">
                        {selectedSpreadsheet.sheets.map((sheet) => (
                          <button
                            key={sheet.id}
                            className={`sheet-tab ${selectedSheet === sheet.name ? 'active' : ''}`}
                            onClick={() => handleSelectSheet(sheet.name)}
                          >
                            {sheet.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {preview && (
                    <>
                      <div className="preview-table-wrapper">
                        <table className="preview-table">
                          <thead>
                            <tr>
                              {preview.columns.map((col, i) => (
                                <th key={i}>{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {preview.rows.map((row, i) => (
                              <tr key={i}>
                                {row.map((cell, j) => (
                                  <td key={j}>{cell ?? ''}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div className="preview-footer">
                        <span className="row-count">
                          {preview.total_rows.toLocaleString()} rows total
                        </span>
                        <button
                          className="btn btn-primary"
                          onClick={handleImport}
                          disabled={isImporting}
                        >
                          {isImporting ? (
                            <>
                              <span className="btn-spinner" />
                              Importing...
                            </>
                          ) : (
                            <>
                              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                              </svg>
                              Import & Analyze
                            </>
                          )}
                        </button>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
