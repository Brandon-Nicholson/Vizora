import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { deleteModel, getModelMeta } from '../../api/client'
import type { ModelMeta, ModelMetrics } from '../../types'
import {
  getStoredModelIds,
  removeModelId,
  setActiveModelId
} from '../../utils/models'
import './ModelsList.css'

const metricPriority = ['accuracy', 'roc_auc', 'f1', 'r2', 'rmse', 'mae']

function pickMetric(metrics?: Record<string, ModelMetrics> | null, modelType?: string | null) {
  if (!metrics) return null
  const candidate = modelType && metrics[modelType] ? metrics[modelType] : metrics[Object.keys(metrics)[0]]
  if (!candidate) return null

  for (const key of metricPriority) {
    const value = candidate[key]
    if (typeof value === 'number') {
      return { label: key.replace('_', ' '), value: value.toFixed(4) }
    }
  }

  const fallbackKey = Object.keys(candidate)[0]
  const fallbackValue = candidate[fallbackKey]
  if (fallbackKey && (typeof fallbackValue === 'number' || typeof fallbackValue === 'string')) {
    return { label: fallbackKey.replace('_', ' '), value: String(fallbackValue) }
  }
  return null
}

export default function ModelsList() {
  const navigate = useNavigate()
  const [models, setModels] = useState<ModelMeta[]>([])
  const [loading, setLoading] = useState(true)

  const refreshModels = async () => {
    const ids = getStoredModelIds()
    if (ids.length === 0) {
      setModels([])
      setLoading(false)
      return
    }

    const results = await Promise.all(
      ids.map(async (id) => {
        try {
          return await getModelMeta(id)
        } catch {
          removeModelId(id)
          return null
        }
      })
    )

    const filtered = results.filter((item): item is ModelMeta => Boolean(item))
    setModels(filtered)
    setLoading(false)
  }

  useEffect(() => {
    refreshModels()
  }, [])

  const handleUseModel = (modelId: string) => {
    setActiveModelId(modelId)
    navigate('/results')
  }

  const handleDelete = async (modelId: string) => {
    try {
      await deleteModel(modelId)
    } finally {
      removeModelId(modelId)
      refreshModels()
    }
  }

  if (loading) {
    return (
      <div className="models-panel">
        <h2>My Models</h2>
        <p className="models-empty">Loading saved models...</p>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="models-panel">
        <h2>My Models</h2>
        <p className="models-empty">No saved models yet. Run an analysis to create one.</p>
      </div>
    )
  }

  return (
    <div className="models-panel">
      <div className="models-header">
        <h2>My Models</h2>
        <span className="models-count">{models.length}</span>
      </div>
      <div className="models-list">
        {models.map((model) => {
          const metric = pickMetric(model.metrics, model.model_type || null)
          return (
            <div key={model.model_id} className="model-card">
              <div className="model-main">
                <div className="model-title">
                  <span>{model.model_type || 'Model'}</span>
                  <span className="model-task">{model.task_type}</span>
                </div>
                <div className="model-meta">
                  <span>Target: {model.target_column || 'Unknown'}</span>
                  <span>{new Date(model.created_at).toLocaleString()}</span>
                </div>
                {metric && (
                  <div className="model-metric">
                    <span>{metric.label}</span>
                    <strong>{metric.value}</strong>
                  </div>
                )}
              </div>
              <div className="model-actions">
                <button className="btn btn-secondary" onClick={() => handleUseModel(model.model_id)}>
                  Use for Predictions
                </button>
                <button className="btn btn-secondary model-delete" onClick={() => handleDelete(model.model_id)}>
                  Delete Model
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
