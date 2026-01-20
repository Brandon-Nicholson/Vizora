import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useBilling } from '../context/BillingContext'
import {
  listSchedules,
  deleteSchedule,
  pauseSchedule,
  resumeSchedule,
  runScheduleNow,
  type Schedule
} from '../api/client'
import NanobotBackground from '../components/common/NanobotBackground'
import CreateScheduleModal from '../components/schedules/CreateScheduleModal'
import './SchedulesPage.css'

export default function SchedulesPage() {
  const navigate = useNavigate()
  const { state: billingState } = useBilling()
  const [schedules, setSchedules] = useState<Schedule[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)

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

  useEffect(() => {
    loadSchedules()
  }, [])

  const loadSchedules = async () => {
    try {
      const { schedules: data } = await listSchedules()
      setSchedules(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load schedules')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (scheduleId: string) => {
    if (!confirm('Are you sure you want to delete this schedule?')) return

    setActionLoading(scheduleId)
    try {
      await deleteSchedule(scheduleId)
      setSchedules((prev) => prev.filter((s) => s.id !== scheduleId))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete schedule')
    } finally {
      setActionLoading(null)
    }
  }

  const handleTogglePause = async (schedule: Schedule) => {
    setActionLoading(schedule.id)
    try {
      if (schedule.is_active) {
        await pauseSchedule(schedule.id)
      } else {
        await resumeSchedule(schedule.id)
      }
      await loadSchedules()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update schedule')
    } finally {
      setActionLoading(null)
    }
  }

  const handleRunNow = async (scheduleId: string) => {
    setActionLoading(scheduleId)
    try {
      await runScheduleNow(scheduleId)
      setError(null)
      // Show success briefly
      const el = document.getElementById(`schedule-${scheduleId}`)
      if (el) {
        el.classList.add('flash-success')
        setTimeout(() => el.classList.remove('flash-success'), 1000)
      }
      await loadSchedules()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run schedule')
    } finally {
      setActionLoading(null)
    }
  }

  const formatNextRun = (dateStr: string | null) => {
    if (!dateStr) return 'Not scheduled'
    const date = new Date(dateStr)
    return date.toLocaleString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    })
  }

  const formatFrequency = (schedule: Schedule) => {
    const hour = schedule.hour
    const hourStr = `${hour > 12 ? hour - 12 : hour || 12}:00 ${hour >= 12 ? 'PM' : 'AM'}`
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    switch (schedule.frequency) {
      case 'daily':
        return `Daily at ${hourStr}`
      case 'weekly':
        return `Every ${days[schedule.day_of_week || 0]} at ${hourStr}`
      case 'monthly':
        return `1st of month at ${hourStr}`
      default:
        return schedule.frequency
    }
  }

  if (!isPro) {
    return (
      <div className="schedules-page">
        <NanobotBackground particleCount={40} />
        <div className="schedules-content container">
          <div className="page-nav">
            {homeButton}
          </div>
          <div className="upgrade-prompt">
            <div className="upgrade-icon">
              <svg width="48\" height="48\" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4" />
                <path d="M12 18v4" />
                <path d="M4.93 4.93l2.83 2.83" />
                <path d="M16.24 16.24l2.83 2.83" />
                <path d="M2 12h4" />
                <path d="M18 12h4" />
                <path d="M4.93 19.07l2.83-2.83" />
                <path d="M16.24 7.76l2.83-2.83" />
              </svg>
            </div>
            <h2>Scheduled Reports</h2>
            <p>
              Automate your data analysis with scheduled reports delivered directly to your inbox.
              Set up daily, weekly, or monthly schedules and never miss an update.
            </p>
            <button className="btn btn-primary btn-large" onClick={() => navigate('/pricing')}>
              Upgrade to Pro
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="schedules-page">
      <NanobotBackground particleCount={40} connectionDistance={80} />

      <div className="schedules-content container">
        <div className="page-nav">
          {homeButton}
        </div>
        <div className="schedules-header animate-fade-in">
          <div className="header-left">
            <h1>Scheduled Reports</h1>
            <p>Automate your analysis reports with scheduled email delivery</p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14" />
              <path d="M5 12h14" />
            </svg>
            New Schedule
          </button>
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

        {isLoading ? (
          <div className="loading-state">
            <div className="spinner" />
            <p>Loading schedules...</p>
          </div>
        ) : schedules.length === 0 ? (
          <div className="empty-state animate-fade-in">
            <div className="empty-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="4" width="18" height="18" rx="2" />
                <path d="M16 2v4" />
                <path d="M8 2v4" />
                <path d="M3 10h18" />
                <path d="M8 14h.01" />
                <path d="M12 14h.01" />
                <path d="M16 14h.01" />
                <path d="M8 18h.01" />
                <path d="M12 18h.01" />
              </svg>
            </div>
            <h3>No Scheduled Reports Yet</h3>
            <p>Create your first schedule to automatically receive analysis reports via email.</p>
            <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
              Create Schedule
            </button>
          </div>
        ) : (
          <div className="schedules-list animate-slide-up">
            {schedules.map((schedule) => (
              <div
                key={schedule.id}
                id={`schedule-${schedule.id}`}
                className={`schedule-card ${!schedule.is_active ? 'paused' : ''}`}
              >
                <div className="schedule-header">
                  <div className="schedule-name">
                    <h3>{schedule.name}</h3>
                    {!schedule.is_active && <span className="status-badge paused">Paused</span>}
                  </div>
                  <div className="schedule-actions">
                    <button
                      className={`btn-action ${actionLoading === schedule.id ? 'loading' : ''}`}
                      onClick={() => handleRunNow(schedule.id)}
                      disabled={actionLoading === schedule.id}
                      title="Run now"
                    >
                      {actionLoading === schedule.id ? (
                        <>
                          <span className="btn-spinner" />
                          Running...
                        </>
                      ) : (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                          </svg>
                          Run Now
                        </>
                      )}
                    </button>
                    <button
                      className="btn-action"
                      onClick={() => handleTogglePause(schedule)}
                      disabled={actionLoading === schedule.id}
                      title={schedule.is_active ? 'Pause' : 'Resume'}
                    >
                      {schedule.is_active ? (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="6" y="4" width="4" height="16" />
                            <rect x="14" y="4" width="4" height="16" />
                          </svg>
                          Pause
                        </>
                      ) : (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          Resume
                        </>
                      )}
                    </button>
                    <button
                      className="btn-action danger"
                      onClick={() => handleDelete(schedule.id)}
                      disabled={actionLoading === schedule.id}
                      title="Delete"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M3 6h18" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                      Delete
                    </button>
                  </div>
                </div>

                <div className="schedule-details">
                  <div className="detail-item">
                    <span className="detail-label">Data Source</span>
                    <span className="detail-value">
                      {schedule.spreadsheet_id ? 'Google Sheets (Live)' : 'Saved Analysis'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Frequency</span>
                    <span className="detail-value">{formatFrequency(schedule)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Recipient</span>
                    <span className="detail-value">{schedule.email}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Next Run</span>
                    <span className="detail-value">
                      {schedule.is_active ? formatNextRun(schedule.next_run_at) : 'Paused'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Reports Sent</span>
                    <span className="detail-value">{schedule.run_count}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {showCreateModal && (
        <CreateScheduleModal
          onClose={() => setShowCreateModal(false)}
          onCreated={() => {
            setShowCreateModal(false)
            loadSchedules()
          }}
        />
      )}
    </div>
  )
}
