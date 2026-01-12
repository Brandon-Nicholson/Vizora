import type { JobStatus, JobCreatedResponse, ModelMeta } from '../types'
import { supabase } from '../lib/supabase'

const API_BASE = import.meta.env.VITE_API_URL || ''

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Get the current access token from Supabase session
 */
async function getAuthHeaders(): Promise<HeadersInit> {
  const { data: { session } } = await supabase.auth.getSession()
  if (session?.access_token) {
    return {
      'Authorization': `Bearer ${session.access_token}`
    }
  }
  return {}
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Request failed')
  }
  return response.json()
}

export async function startAnalysis(
  file: File,
  mode: string,
  goal: string,
  targetColumn?: string
): Promise<JobCreatedResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('mode', mode)
  formData.append('goal', goal)
  if (targetColumn) {
    formData.append('target_column', targetColumn)
  }

  const authHeaders = await getAuthHeaders()

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    headers: authHeaders,
    body: formData
  })

  return handleResponse<JobCreatedResponse>(response)
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    headers: authHeaders
  })
  return handleResponse<JobStatus>(response)
}

export async function cancelJob(jobId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    method: 'DELETE',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to cancel job')
  }
}

export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_BASE}/api/health`)
  return handleResponse(response)
}

export async function getModelMeta(modelId: string): Promise<ModelMeta> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/models/${modelId}`, {
    headers: authHeaders
  })
  return handleResponse<ModelMeta>(response)
}

export async function listModels(): Promise<ModelMeta[]> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/models`, {
    headers: authHeaders
  })
  return handleResponse<ModelMeta[]>(response)
}

export async function deleteModel(modelId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/models/${modelId}`, {
    method: 'DELETE',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to delete model')
  }
}

// Billing API functions
export interface UserProfile {
  id: string
  email: string
  name: string | null
  tier: 'free' | 'pro'
  monthly_usage: number
  monthly_limit: number
}

export interface CheckoutSession {
  checkout_url: string
  session_id: string
}

export async function getUserProfile(): Promise<UserProfile> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/auth/profile`, {
    headers: authHeaders
  })
  return handleResponse<UserProfile>(response)
}

export async function createCheckoutSession(): Promise<CheckoutSession> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/billing/create-checkout`, {
    method: 'POST',
    headers: {
      ...authHeaders,
      'Content-Type': 'application/json'
    }
  })
  return handleResponse<CheckoutSession>(response)
}

export async function getSubscriptionStatus(): Promise<{
  tier: string
  status: string
  current_period_end: string | null
}> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/billing/subscription`, {
    headers: authHeaders
  })
  return handleResponse(response)
}

export async function createPortalSession(): Promise<{ portal_url: string }> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/billing/portal`, {
    method: 'POST',
    headers: authHeaders
  })
  return handleResponse(response)
}

export async function exportPdf(jobId: string): Promise<Blob> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/export-pdf`, {
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to export PDF')
  }

  return response.blob()
}

// Schedule API functions
export interface Schedule {
  id: string
  name: string
  job_id: string | null
  frequency: 'daily' | 'weekly' | 'monthly'
  email: string
  hour: number
  day_of_week: number | null
  is_active: boolean
  next_run_at: string | null
  last_run_at: string | null
  run_count: number
  created_at: string
  // Google Sheets fields
  spreadsheet_id: string | null
  sheet_name: string | null
  analysis_mode: string | null
  analysis_goal: string | null
  target_column: string | null
  data_source: 'model' | 'google_sheets' | null
}

export interface CreateScheduleRequest {
  name: string
  job_id?: string
  frequency: 'daily' | 'weekly' | 'monthly'
  email: string
  hour?: number
  day_of_week?: number
  // Google Sheets fields
  spreadsheet_id?: string
  sheet_name?: string
  analysis_mode?: string
  analysis_goal?: string
  target_column?: string
}

export async function createSchedule(data: CreateScheduleRequest): Promise<Schedule> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules`, {
    method: 'POST',
    headers: {
      ...authHeaders,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  return handleResponse<Schedule>(response)
}

export async function listSchedules(): Promise<{ schedules: Schedule[] }> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules`, {
    headers: authHeaders
  })
  return handleResponse<{ schedules: Schedule[] }>(response)
}

export async function getSchedule(scheduleId: string): Promise<Schedule> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}`, {
    headers: authHeaders
  })
  return handleResponse<Schedule>(response)
}

export async function updateSchedule(
  scheduleId: string,
  data: Partial<CreateScheduleRequest & { is_active: boolean }>
): Promise<Schedule> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}`, {
    method: 'PATCH',
    headers: {
      ...authHeaders,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  return handleResponse<Schedule>(response)
}

export async function deleteSchedule(scheduleId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}`, {
    method: 'DELETE',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to delete schedule')
  }
}

export async function pauseSchedule(scheduleId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}/pause`, {
    method: 'POST',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to pause schedule')
  }
}

export async function resumeSchedule(scheduleId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}/resume`, {
    method: 'POST',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to resume schedule')
  }
}

export async function runScheduleNow(scheduleId: string): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/schedules/${scheduleId}/run-now`, {
    method: 'POST',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to run schedule')
  }
}

// Google Sheets API functions
export interface GoogleConnectionStatus {
  connected: boolean
  configured: boolean
}

export interface Spreadsheet {
  id: string
  name: string
  modified_at: string | null
  owner: string | null
}

export interface Sheet {
  id: number
  name: string
  row_count: number
  column_count: number
}

export interface SpreadsheetDetails {
  id: string
  name: string
  sheets: Sheet[]
}

export interface SheetPreview {
  columns: string[]
  rows: (string | number | null)[][]
  total_rows: number
}

export interface ImportedSheet {
  filename: string
  data: string
  rows: number
  columns: string[]
}

export async function getGoogleStatus(): Promise<GoogleConnectionStatus> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/google/status`, {
    headers: authHeaders
  })
  return handleResponse<GoogleConnectionStatus>(response)
}

export async function getGoogleAuthUrl(): Promise<{ auth_url: string }> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/google/connect`, {
    headers: authHeaders
  })
  return handleResponse<{ auth_url: string }>(response)
}

export async function disconnectGoogle(): Promise<void> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/google/disconnect`, {
    method: 'POST',
    headers: authHeaders
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Failed to disconnect')
  }
}

export async function listSpreadsheets(limit = 20): Promise<Spreadsheet[]> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/google/spreadsheets?limit=${limit}`, {
    headers: authHeaders
  })
  return handleResponse<Spreadsheet[]>(response)
}

export async function getSpreadsheetDetails(spreadsheetId: string): Promise<SpreadsheetDetails> {
  const authHeaders = await getAuthHeaders()
  const response = await fetch(`${API_BASE}/api/google/spreadsheets/${spreadsheetId}`, {
    headers: authHeaders
  })
  return handleResponse<SpreadsheetDetails>(response)
}

export async function getSheetPreview(
  spreadsheetId: string,
  sheetName?: string,
  rows = 5
): Promise<SheetPreview> {
  const authHeaders = await getAuthHeaders()
  const params = new URLSearchParams({ rows: rows.toString() })
  if (sheetName) params.set('sheet', sheetName)

  const response = await fetch(
    `${API_BASE}/api/google/spreadsheets/${spreadsheetId}/preview?${params}`,
    { headers: authHeaders }
  )
  return handleResponse<SheetPreview>(response)
}

export async function importGoogleSheet(
  spreadsheetId: string,
  sheetName?: string
): Promise<ImportedSheet> {
  const authHeaders = await getAuthHeaders()
  const params = new URLSearchParams()
  if (sheetName) params.set('sheet', sheetName)

  const response = await fetch(
    `${API_BASE}/api/google/spreadsheets/${spreadsheetId}/import?${params}`,
    {
      method: 'POST',
      headers: authHeaders
    }
  )
  return handleResponse<ImportedSheet>(response)
}

export { ApiError }
