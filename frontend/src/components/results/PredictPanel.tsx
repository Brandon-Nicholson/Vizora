import { useEffect, useState } from 'react'
import Dropzone from '../upload/Dropzone'
import './PredictPanel.css'

const API_BASE = import.meta.env.VITE_API_URL || ''

interface PredictPanelProps {
  modelId: string | null
}

export default function PredictPanel({ modelId }: PredictPanelProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [missingColumns, setMissingColumns] = useState<string[]>([])
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [downloadName, setDownloadName] = useState<string>('')
  const [previewHeaders, setPreviewHeaders] = useState<string[]>([])
  const [previewRows, setPreviewRows] = useState<string[][]>([])

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl)
      }
    }
  }, [downloadUrl])

  const handleFileSelect = (selected: File | null) => {
    setFile(selected)
    setPreviewHeaders([])
    setPreviewRows([])
    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl)
      setDownloadUrl(null)
      setDownloadName('')
    }
  }

  const parseCsvPreview = (csvText: string, maxRows = 5) => {
    const rows: string[][] = []
    let current: string[] = []
    let field = ''
    let inQuotes = false

    const pushField = () => {
      current.push(field)
      field = ''
    }

    const pushRow = () => {
      if (current.length > 0) {
        rows.push(current)
      }
      current = []
    }

    for (let i = 0; i < csvText.length; i++) {
      const char = csvText[i]
      const next = csvText[i + 1]

      if (char === '"' && inQuotes && next === '"') {
        field += '"'
        i++
        continue
      }

      if (char === '"') {
        inQuotes = !inQuotes
        continue
      }

      if (char === ',' && !inQuotes) {
        pushField()
        continue
      }

      if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && next === '\n') {
          i++
        }
        pushField()
        pushRow()
        if (rows.length >= maxRows + 1) {
          break
        }
        continue
      }

      field += char
    }

    if (field.length > 0 || current.length > 0) {
      pushField()
      pushRow()
    }

    const [headerRow, ...dataRows] = rows
    setPreviewHeaders(headerRow || [])
    setPreviewRows(dataRows.slice(0, maxRows))
  }

  const handlePredict = async () => {
    if (!file || !modelId) {
      return
    }

    setIsRunning(true)
    setError(null)
    setMissingColumns([])

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/api/runs/${modelId}/predict_csv`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        if (payload.error === 'missing_columns') {
          setMissingColumns(payload.missing || [])
          setError('Missing required columns in the uploaded CSV.')
        } else if (payload.error === 'run_not_found') {
          setError('Run not found. Please rerun the analysis.')
        } else if (payload.error === 'invalid_csv') {
          setError('Unable to parse the CSV file.')
        } else {
          setError('Prediction failed. Please try again.')
        }
        return
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl)
      }
      setDownloadUrl(url)
      setDownloadName(`predictions_${modelId}.csv`)
      try {
        const text = await blob.text()
        parseCsvPreview(text, 5)
      } catch {
        setPreviewHeaders([])
        setPreviewRows([])
      }
    } catch (err) {
      setError('Prediction failed. Please try again.')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="predict-panel">
      <div className="predict-card">
        <div className="predict-header">
          <h3>Score New Data</h3>
          <p>Upload a CSV with the same feature columns to generate predictions.</p>
        </div>

        <div className="predict-body">
          <Dropzone onFileSelect={handleFileSelect} selectedFile={file} />

          <button
            className="btn btn-primary"
            disabled={!file || !modelId || isRunning}
            onClick={handlePredict}
          >
            {isRunning ? 'Running...' : 'Run Predictions'}
          </button>

          {downloadUrl && (
            <a className="btn btn-secondary btn-glow" href={downloadUrl} download={downloadName}>
              Download Predictions CSV
            </a>
          )}

          {previewHeaders.length > 0 && previewRows.length > 0 && (
            <div className="predict-preview">
              <h4>Preview</h4>
              <div className="preview-table">
                <table>
                  <thead>
                    <tr>
                      {previewHeaders.map((header, index) => (
                        <th key={`${header}-${index}`}>{header}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row, rowIndex) => (
                      <tr key={`row-${rowIndex}`}>
                        {previewHeaders.map((_, colIndex) => (
                          <td key={`cell-${rowIndex}-${colIndex}`}>{row[colIndex] ?? ''}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {error && (
            <div className="predict-error">
              <strong>{error}</strong>
              {missingColumns.length > 0 && (
                <div className="missing-columns">
                  Missing: {missingColumns.join(', ')}
                </div>
              )}
            </div>
          )}

          {!modelId && (
            <div className="predict-error">
              <strong>Run ID unavailable. Complete a training run first.</strong>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
