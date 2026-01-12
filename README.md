# Vizora
Vizora is an agentic data analysis engine that cleans, analyzes, models, and visualizes datasets using deterministic pipelines guided by LLM-based judgment.

## Training and Inference
- Run a predictive or hybrid analysis to train a model. Artifacts are saved under `artifacts/{run_id}/` as `model.joblib` and `meta.json`.
- Batch CSV scoring: `POST /api/runs/{run_id}/predict_csv` with `multipart/form-data` field `file` to download `predictions_{run_id}.csv`.
- JSON scoring: `POST /api/runs/{run_id}/predict` with a JSON object or list of objects to receive `predictions` (and optional `probabilities`).
