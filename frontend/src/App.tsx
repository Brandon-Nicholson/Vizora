import { Routes, Route } from 'react-router-dom'
import { AnalysisProvider } from './context/AnalysisContext'
import { AuthProvider } from './context/AuthContext'
import { BillingProvider } from './context/BillingContext'
import ProtectedRoute from './components/auth/ProtectedRoute'
import HomePage from './pages/HomePage'
import ModeSelectionPage from './pages/ModeSelectionPage'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import PricingPage from './pages/PricingPage'
import SchedulesPage from './pages/SchedulesPage'
import SheetsPage from './pages/SheetsPage'

function App() {
  return (
    <AuthProvider>
      <BillingProvider>
        <AnalysisProvider>
          <div className="app">
            <Routes>
              {/* Public routes */}
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/signup" element={<SignupPage />} />
              <Route path="/pricing" element={<PricingPage />} />

              {/* Protected routes */}
              <Route
                path="/mode"
                element={
                  <ProtectedRoute>
                    <ModeSelectionPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/upload"
                element={
                  <ProtectedRoute>
                    <UploadPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/results"
                element={
                  <ProtectedRoute>
                    <ResultsPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/schedules"
                element={
                  <ProtectedRoute>
                    <SchedulesPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/sheets"
                element={
                  <ProtectedRoute>
                    <SheetsPage />
                  </ProtectedRoute>
                }
              />
            </Routes>
          </div>
        </AnalysisProvider>
      </BillingProvider>
    </AuthProvider>
  )
}

export default App
