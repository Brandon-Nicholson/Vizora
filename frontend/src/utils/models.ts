export const MODELS_STORAGE_KEY = 'vizora_models'
export const ACTIVE_MODEL_KEY = 'vizora_active_model'

export function getStoredModelIds(): string[] {
  const raw = localStorage.getItem(MODELS_STORAGE_KEY)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function storeModelId(modelId: string) {
  const current = getStoredModelIds()
  if (!current.includes(modelId)) {
    const next = [...current, modelId]
    localStorage.setItem(MODELS_STORAGE_KEY, JSON.stringify(next))
  }
}

export function removeModelId(modelId: string) {
  const current = getStoredModelIds()
  const next = current.filter((id) => id !== modelId)
  localStorage.setItem(MODELS_STORAGE_KEY, JSON.stringify(next))
  if (getActiveModelId() === modelId) {
    clearActiveModelId()
  }
}

export function setActiveModelId(modelId: string) {
  localStorage.setItem(ACTIVE_MODEL_KEY, modelId)
}

export function getActiveModelId(): string | null {
  return localStorage.getItem(ACTIVE_MODEL_KEY)
}

export function clearActiveModelId() {
  localStorage.removeItem(ACTIVE_MODEL_KEY)
}
