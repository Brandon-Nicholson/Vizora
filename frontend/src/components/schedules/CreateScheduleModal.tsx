import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import {
  createSchedule,
  listSpreadsheets,
  getSpreadsheetDetails,
  getGoogleStatus,
  type CreateScheduleRequest,
  type Spreadsheet,
  type Sheet
} from '../../api/client'
import './CreateScheduleModal.css'

interface Props {
  onClose: () => void
  onCreated: () => void
}

export default function CreateScheduleModal({ onClose, onCreated }: Props) {
  const navigate = useNavigate()
  const { state: authState } = useAuth()
  const [name, setName] = useState('')
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'monthly'>('weekly')
  const [email, setEmail] = useState(authState.user?.email || '')
  const [hour, setHour] = useState(9)
  const [dayOfWeek, setDayOfWeek] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Google Sheets state
  const [googleConnected, setGoogleConnected] = useState(false)
  const [spreadsheets, setSpreadsheets] = useState<Spreadsheet[]>([])
  const [selectedSpreadsheetId, setSelectedSpreadsheetId] = useState('')
  const [sheets, setSheets] = useState<Sheet[]>([])
  const [selectedSheetName, setSelectedSheetName] = useState('')
  const [analysisMode, setAnalysisMode] = useState<'eda' | 'predictive'>('eda')
  const [analysisGoal, setAnalysisGoal] = useState('')
  const [targetColumn, setTargetColumn] = useState('')
  const [isLoadingSheets, setIsLoadingSheets] = useState(false)

  useEffect(() => {
    checkGoogleStatus()
  }, [])

  const checkGoogleStatus = async () => {
    try {
      const status = await getGoogleStatus()
      setGoogleConnected(status.connected)
      if (status.connected) {
        loadSpreadsheets()
      }
    } catch (err) {
      console.error('Failed to check Google status', err)
    }
  }

  const loadSpreadsheets = async () => {
    try {
      const data = await listSpreadsheets()
      setSpreadsheets(data)
      if (data.length > 0) {
        setSelectedSpreadsheetId(data[0].id)
        loadSheets(data[0].id)
      }
    } catch (err) {
      console.error('Failed to load spreadsheets', err)
    }
  }

  const loadSheets = async (spreadsheetId: string) => {
    setIsLoadingSheets(true)
    try {
      const details = await getSpreadsheetDetails(spreadsheetId)
      setSheets(details.sheets)
      if (details.sheets.length > 0) {
        setSelectedSheetName(details.sheets[0].name)
      }
    } catch (err) {
      console.error('Failed to load sheets', err)
    } finally {
      setIsLoadingSheets(false)
    }
  }

  const handleSpreadsheetChange = (spreadsheetId: string) => {
    setSelectedSpreadsheetId(spreadsheetId)
    setSelectedSheetName('')
    setSheets([])
    loadSheets(spreadsheetId)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!name.trim() || !email.trim()) {
      setError('Please fill in all required fields')
      return
    }

    if (!googleConnected) {
      setError('Connect Google Sheets to enable scheduled reports')
      return
    }

    if (!selectedSpreadsheetId) {
      setError('Please select a Google Sheet')
      return
    }

    if (!analysisGoal.trim()) {
      setError('Please enter an analysis goal')
      return
    }

    if (analysisMode === 'predictive' && !targetColumn.trim()) {
      setError('Please enter a target column for prediction')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Convert local hour to UTC
      // Get the timezone offset in hours (negative for west of UTC)
      const timezoneOffsetHours = new Date().getTimezoneOffset() / 60
      // Convert local hour to UTC (add offset because getTimezoneOffset returns minutes behind UTC as positive)
      let utcHour = hour + timezoneOffsetHours
      // Handle wrap-around
      if (utcHour < 0) utcHour += 24
      if (utcHour >= 24) utcHour -= 24

      const data: CreateScheduleRequest = {
        name: name.trim(),
        frequency,
        email: email.trim(),
        hour: Math.round(utcHour),
      }

      if (frequency === 'weekly') {
        data.day_of_week = dayOfWeek
      }

      data.spreadsheet_id = selectedSpreadsheetId
      data.sheet_name = selectedSheetName || undefined
      data.analysis_mode = analysisMode
      data.analysis_goal = analysisGoal.trim()
      if (analysisMode === 'predictive' && targetColumn.trim()) {
        data.target_column = targetColumn.trim()
      }

      await createSchedule(data)
      onCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create schedule')
      setIsLoading(false)
    }
  }

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create Scheduled Report</h2>
          <button className="modal-close" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18" />
              <path d="M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          {error && (
            <div className="form-error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="name">Schedule Name</label>
            <input
              id="name"
              type="text"
              className="input"
              placeholder="e.g., Weekly Sales Report"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Data Source</label>
            <div className="frequency-options">
              <button
                type="button"
                className="frequency-btn active"
                disabled
              >
                Google Sheets
              </button>
            </div>
            {!googleConnected && (
              <p className="form-hint">
                Connect Google Sheets to enable live data.
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => navigate('/sheets')}
                >
                  Go to Sheets
                </button>
              </p>
            )}
          </div>

          <>
            <div className="form-group">
              <label htmlFor="spreadsheet">Google Sheet</label>
              {spreadsheets.length === 0 ? (
                <div className="no-models">No spreadsheets found in your Google Drive</div>
              ) : (
                <select
                  id="spreadsheet"
                  className="input select"
                  value={selectedSpreadsheetId}
                  onChange={(e) => handleSpreadsheetChange(e.target.value)}
                  required
                >
                  {spreadsheets.map((sheet) => (
                    <option key={sheet.id} value={sheet.id}>
                      {sheet.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {sheets.length > 0 && (
              <div className="form-group">
                <label htmlFor="sheetName">Sheet Tab</label>
                {isLoadingSheets ? (
                  <div className="loading-select">Loading sheets...</div>
                ) : (
                  <select
                    id="sheetName"
                    className="input select"
                    value={selectedSheetName}
                    onChange={(e) => setSelectedSheetName(e.target.value)}
                  >
                    {sheets.map((sheet) => (
                      <option key={sheet.id} value={sheet.name}>
                        {sheet.name} ({sheet.row_count} rows)
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}

            <div className="form-group">
              <label>Analysis Type</label>
              <div className="frequency-options">
                <button
                  type="button"
                  className={`frequency-btn ${analysisMode === 'eda' ? 'active' : ''}`}
                  onClick={() => setAnalysisMode('eda')}
                >
                  Explore
                </button>
                <button
                  type="button"
                  className={`frequency-btn ${analysisMode === 'predictive' ? 'active' : ''}`}
                  onClick={() => setAnalysisMode('predictive')}
                >
                  Predict
                </button>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="goal">Analysis Goal</label>
              <input
                id="goal"
                type="text"
                className="input"
                placeholder="e.g., Analyze weekly sales trends"
                value={analysisGoal}
                onChange={(e) => setAnalysisGoal(e.target.value)}
                required
              />
            </div>

            {analysisMode === 'predictive' && (
              <div className="form-group">
                <label htmlFor="targetColumn">Target Column</label>
                <input
                  id="targetColumn"
                  type="text"
                  className="input"
                  placeholder="e.g., sales_amount"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  required
                />
              </div>
            )}
          </>

          <div className="form-group">
            <label htmlFor="frequency">Frequency</label>
            <div className="frequency-options">
              {(['daily', 'weekly', 'monthly'] as const).map((freq) => (
                <button
                  key={freq}
                  type="button"
                  className={`frequency-btn ${frequency === freq ? 'active' : ''}`}
                  onClick={() => setFrequency(freq)}
                >
                  {freq.charAt(0).toUpperCase() + freq.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {frequency === 'weekly' && (
            <div className="form-group">
              <label htmlFor="dayOfWeek">Day of Week</label>
              <select
                id="dayOfWeek"
                className="input select"
                value={dayOfWeek}
                onChange={(e) => setDayOfWeek(Number(e.target.value))}
              >
                {days.map((day, i) => (
                  <option key={day} value={i}>{day}</option>
                ))}
              </select>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="hour">Time (Hour)</label>
            <select
              id="hour"
              className="input select"
              value={hour}
              onChange={(e) => setHour(Number(e.target.value))}
            >
              {Array.from({ length: 24 }, (_, i) => (
                <option key={i} value={i}>
                  {i === 0 ? '12:00 AM' : i < 12 ? `${i}:00 AM` : i === 12 ? '12:00 PM' : `${i - 12}:00 PM`}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="email">Send Report To</label>
            <input
              id="email"
              type="email"
              className="input"
              placeholder="email@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="btn-spinner" />
                  Creating...
                </>
              ) : (
                'Create Schedule'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
