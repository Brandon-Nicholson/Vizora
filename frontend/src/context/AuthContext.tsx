/**
 * Authentication Context
 *
 * Provides authentication state management using Supabase Auth.
 */

import {
  createContext,
  useContext,
  useReducer,
  useEffect,
  ReactNode,
} from 'react'
import { Session, User, AuthError } from '@supabase/supabase-js'
import { supabase, isSupabaseConfigured } from '../lib/supabase'

// Types
interface AuthState {
  user: User | null
  session: Session | null
  isLoading: boolean
  isConfigured: boolean
  error: string | null
}

type AuthAction =
  | { type: 'SET_SESSION'; payload: { user: User | null; session: Session | null } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SIGN_OUT' }

interface AuthContextType {
  state: AuthState
  signUp: (email: string, password: string, name?: string) => Promise<{ error: AuthError | null }>
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>
  signOut: () => Promise<void>
  getAccessToken: () => string | null
}

// Initial state
const initialState: AuthState = {
  user: null,
  session: null,
  isLoading: true,
  isConfigured: isSupabaseConfigured(),
  error: null,
}

// Reducer
function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'SET_SESSION':
      return {
        ...state,
        user: action.payload.user,
        session: action.payload.session,
        isLoading: false,
        error: null,
      }
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false }
    case 'SIGN_OUT':
      return { ...state, user: null, session: null, error: null }
    default:
      return state
  }
}

// Context
const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Provider
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(authReducer, initialState)

  // Initialize auth state on mount
  useEffect(() => {
    if (!state.isConfigured) {
      dispatch({ type: 'SET_LOADING', payload: false })
      return
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      dispatch({
        type: 'SET_SESSION',
        payload: { user: session?.user ?? null, session },
      })
    })

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      dispatch({
        type: 'SET_SESSION',
        payload: { user: session?.user ?? null, session },
      })
    })

    return () => {
      subscription.unsubscribe()
    }
  }, [state.isConfigured])

  // Sign up with email and password
  const signUp = async (
    email: string,
    password: string,
    name?: string
  ): Promise<{ error: AuthError | null }> => {
    dispatch({ type: 'SET_LOADING', payload: true })

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { name },
      },
    })

    if (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message })
      return { error }
    }

    return { error: null }
  }

  // Sign in with email and password
  const signIn = async (
    email: string,
    password: string
  ): Promise<{ error: AuthError | null }> => {
    dispatch({ type: 'SET_LOADING', payload: true })

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })

    if (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message })
      return { error }
    }

    return { error: null }
  }

  // Sign out
  const signOut = async () => {
    await supabase.auth.signOut()
    dispatch({ type: 'SIGN_OUT' })
  }

  // Get the current access token
  const getAccessToken = (): string | null => {
    return state.session?.access_token ?? null
  }

  const value: AuthContextType = {
    state,
    signUp,
    signIn,
    signOut,
    getAccessToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Hook
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
