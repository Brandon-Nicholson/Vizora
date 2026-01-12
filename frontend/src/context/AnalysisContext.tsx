import { createContext, useContext, useReducer, ReactNode } from 'react'
import type { AnalysisState, AnalysisAction, AnalysisMode, ProgressInfo, AnalysisResult } from '../types'

const initialState: AnalysisState = {
  mode: null,
  goal: '',
  targetColumn: '',
  file: null,
  jobId: null,
  status: 'idle',
  progress: null,
  result: null,
  error: null
}

function analysisReducer(state: AnalysisState, action: AnalysisAction): AnalysisState {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, mode: action.payload }
    case 'SET_GOAL':
      return { ...state, goal: action.payload }
    case 'SET_TARGET_COLUMN':
      return { ...state, targetColumn: action.payload }
    case 'SET_FILE':
      return { ...state, file: action.payload }
    case 'START_UPLOAD':
      return { ...state, status: 'uploading', error: null }
    case 'JOB_CREATED':
      return { ...state, jobId: action.payload, status: 'processing' }
    case 'UPDATE_PROGRESS':
      return { ...state, progress: action.payload }
    case 'ANALYSIS_COMPLETE':
      return { ...state, status: 'completed', result: action.payload }
    case 'ANALYSIS_ERROR':
      return { ...state, status: 'error', error: action.payload }
    case 'RESET':
      return initialState
    default:
      return state
  }
}

interface AnalysisContextType {
  state: AnalysisState
  setMode: (mode: AnalysisMode) => void
  setGoal: (goal: string) => void
  setTargetColumn: (column: string) => void
  setFile: (file: File | null) => void
  startUpload: () => void
  jobCreated: (jobId: string) => void
  updateProgress: (progress: ProgressInfo) => void
  analysisComplete: (result: AnalysisResult) => void
  analysisError: (error: string) => void
  reset: () => void
}

const AnalysisContext = createContext<AnalysisContextType | null>(null)

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(analysisReducer, initialState)

  const value: AnalysisContextType = {
    state,
    setMode: (mode) => dispatch({ type: 'SET_MODE', payload: mode }),
    setGoal: (goal) => dispatch({ type: 'SET_GOAL', payload: goal }),
    setTargetColumn: (column) => dispatch({ type: 'SET_TARGET_COLUMN', payload: column }),
    setFile: (file) => dispatch({ type: 'SET_FILE', payload: file }),
    startUpload: () => dispatch({ type: 'START_UPLOAD' }),
    jobCreated: (jobId) => dispatch({ type: 'JOB_CREATED', payload: jobId }),
    updateProgress: (progress) => dispatch({ type: 'UPDATE_PROGRESS', payload: progress }),
    analysisComplete: (result) => dispatch({ type: 'ANALYSIS_COMPLETE', payload: result }),
    analysisError: (error) => dispatch({ type: 'ANALYSIS_ERROR', payload: error }),
    reset: () => dispatch({ type: 'RESET' })
  }

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysis() {
  const context = useContext(AnalysisContext)
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider')
  }
  return context
}
