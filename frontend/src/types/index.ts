// API Response Types

export interface ProgressInfo {
  current_step: string
  percentage: number
}

export interface FigureData {
  id: string
  type: string
  name: string
  base64_png: string
}

export interface ModelMetrics {
  [key: string]: number | string
}

export interface AnalysisResult {
  figures: FigureData[]
  metrics: Record<string, ModelMetrics> | null
  summary_markdown: string
  plan: Record<string, unknown>
  errors: string[]
  preprocessing: Record<string, unknown>
}

export interface ModelMeta {
  model_id: string
  created_at: string
  task_type: 'classification' | 'regression'
  model_type?: string | null
  target_column?: string | null
  feature_columns?: string[]
  metrics?: Record<string, ModelMetrics> | null
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: ProgressInfo | null
  result: AnalysisResult | null
  error_message: string | null
}

export interface JobCreatedResponse {
  job_id: string
  status: 'queued'
}

// Analysis State Types

export type AnalysisMode = 'eda' | 'predictive' | 'hybrid' | 'forecast'

export interface ForecastConfig {
  dateColumn?: string
  horizon: number  // days to forecast
  frequency: 'daily' | 'weekly' | 'monthly'
}

export interface AnalysisState {
  mode: AnalysisMode | null
  goal: string
  targetColumn: string
  file: File | null
  jobId: string | null
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error'
  progress: ProgressInfo | null
  result: AnalysisResult | null
  error: string | null
}

export type AnalysisAction =
  | { type: 'SET_MODE'; payload: AnalysisMode }
  | { type: 'SET_GOAL'; payload: string }
  | { type: 'SET_TARGET_COLUMN'; payload: string }
  | { type: 'SET_FILE'; payload: File | null }
  | { type: 'START_UPLOAD' }
  | { type: 'JOB_CREATED'; payload: string }
  | { type: 'UPDATE_PROGRESS'; payload: ProgressInfo }
  | { type: 'ANALYSIS_COMPLETE'; payload: AnalysisResult }
  | { type: 'ANALYSIS_ERROR'; payload: string }
  | { type: 'RESET' }
