import { useEffect, useRef, useCallback } from 'react'
import { getJobStatus } from '../api/client'
import type { JobStatus, AnalysisResult, ProgressInfo } from '../types'

interface UsePollingOptions {
  jobId: string | null
  onProgress?: (progress: ProgressInfo) => void
  onComplete: (result: AnalysisResult) => void
  onError: (error: string) => void
  interval?: number
}

export function usePolling({
  jobId,
  onProgress,
  onComplete,
  onError,
  interval = 2000
}: UsePollingOptions) {
  const intervalRef = useRef<number | null>(null)
  const isPollingRef = useRef(false)

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    isPollingRef.current = false
  }, [])

  const poll = useCallback(async () => {
    if (!jobId || !isPollingRef.current) return

    try {
      const status: JobStatus = await getJobStatus(jobId)

      if (status.progress && onProgress) {
        onProgress(status.progress)
      }

      if (status.status === 'completed' && status.result) {
        stopPolling()
        onComplete(status.result)
      } else if (status.status === 'failed') {
        stopPolling()
        onError(status.error_message || 'Analysis failed')
      }
    } catch (err) {
      stopPolling()
      onError(err instanceof Error ? err.message : 'Failed to check job status')
    }
  }, [jobId, onProgress, onComplete, onError, stopPolling])

  useEffect(() => {
    if (!jobId) {
      stopPolling()
      return
    }

    isPollingRef.current = true

    // Initial poll
    poll()

    // Set up interval
    intervalRef.current = window.setInterval(poll, interval)

    return () => {
      stopPolling()
    }
  }, [jobId, interval, poll, stopPolling])

  return { stopPolling }
}
