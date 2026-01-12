/**
 * Billing Context
 *
 * Provides billing state management including subscription tier and usage.
 */

import {
  createContext,
  useContext,
  useReducer,
  useEffect,
  ReactNode,
  useCallback,
} from 'react'
import { useAuth } from './AuthContext'
import { getUserProfile, type UserProfile } from '../api/client'

// Types
interface BillingState {
  tier: 'free' | 'pro'
  monthlyUsage: number
  monthlyLimit: number
  isLoading: boolean
  error: string | null
}

type BillingAction =
  | { type: 'SET_BILLING'; payload: Partial<BillingState> }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'INCREMENT_USAGE' }
  | { type: 'RESET' }

interface BillingContextType {
  state: BillingState
  refreshBilling: () => Promise<UserProfile | null>
  canRunAnalysis: () => boolean
  canExportPdf: () => boolean
  incrementUsage: () => void
}

// Initial state
const initialState: BillingState = {
  tier: 'free',
  monthlyUsage: 0,
  monthlyLimit: 5,
  isLoading: false,
  error: null,
}

// Reducer
function billingReducer(state: BillingState, action: BillingAction): BillingState {
  switch (action.type) {
    case 'SET_BILLING':
      return { ...state, ...action.payload, isLoading: false, error: null }
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false }
    case 'INCREMENT_USAGE':
      return { ...state, monthlyUsage: state.monthlyUsage + 1 }
    case 'RESET':
      return initialState
    default:
      return state
  }
}

// Context
const BillingContext = createContext<BillingContextType | undefined>(undefined)

// Provider
export function BillingProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(billingReducer, initialState)
  const { state: authState } = useAuth()

  // Refresh billing info from the API
  const refreshBilling = useCallback(async () => {
    if (!authState.user) {
      dispatch({ type: 'RESET' })
      return null
    }

    dispatch({ type: 'SET_LOADING', payload: true })

    try {
      const profile = await getUserProfile()
      dispatch({
        type: 'SET_BILLING',
        payload: {
          tier: profile.tier as 'free' | 'pro',
          monthlyUsage: profile.monthly_usage,
          monthlyLimit: profile.monthly_limit,
        },
      })
      return profile
    } catch (error) {
      // If billing info fails to load, use defaults
      dispatch({
        type: 'SET_BILLING',
        payload: {
          tier: 'free',
          monthlyUsage: 0,
          monthlyLimit: 5,
        },
      })
      return null
    }
  }, [authState.user])

  // Fetch billing info when user changes
  useEffect(() => {
    if (authState.user && !authState.isLoading) {
      refreshBilling()
    } else if (!authState.user && !authState.isLoading) {
      dispatch({ type: 'RESET' })
    }
  }, [authState.user, authState.isLoading, refreshBilling])

  // Check if user can run an analysis
  const canRunAnalysis = useCallback(() => {
    // Pro users have unlimited
    if (state.tier === 'pro' || state.monthlyLimit === -1) {
      return true
    }
    return state.monthlyUsage < state.monthlyLimit
  }, [state.tier, state.monthlyUsage, state.monthlyLimit])

  // Check if user can export PDF (Pro feature)
  const canExportPdf = useCallback(() => {
    return state.tier === 'pro' || state.monthlyLimit === -1
  }, [state.tier, state.monthlyLimit])

  // Increment usage locally (optimistic update)
  const incrementUsage = useCallback(() => {
    dispatch({ type: 'INCREMENT_USAGE' })
  }, [])

  const value: BillingContextType = {
    state,
    refreshBilling,
    canRunAnalysis,
    canExportPdf,
    incrementUsage,
  }

  return (
    <BillingContext.Provider value={value}>{children}</BillingContext.Provider>
  )
}

// Hook
export function useBilling(): BillingContextType {
  const context = useContext(BillingContext)
  if (context === undefined) {
    throw new Error('useBilling must be used within a BillingProvider')
  }
  return context
}
