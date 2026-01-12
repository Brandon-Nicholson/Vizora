/**
 * Protected Route Component
 *
 * Wraps routes that require authentication.
 * Redirects to login if user is not authenticated.
 */

import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { state } = useAuth()
  const location = useLocation()

  // If auth is not configured, allow access (development mode)
  if (!state.isConfigured) {
    return <>{children}</>
  }

  // Show nothing while loading
  if (state.isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
      </div>
    )
  }

  // Redirect to login if not authenticated
  if (!state.user) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
