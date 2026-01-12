"""
Inference utilities for loading artifacts and scoring new data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class InferencePreprocessor(BaseEstimator, TransformerMixin):
    """
    Apply the same preprocessing steps used during training.
    """

    def __init__(
        self,
        encodings: list[dict[str, Any]],
        scaling: Optional[dict[str, Any]],
        label_encoders: dict[str, Any],
        onehot_columns: dict[str, list[str]],
        scaler: Any
    ) -> None:
        self.encodings = encodings or []
        self.scaling = scaling or None
        self.label_encoders = label_encoders or {}
        self.onehot_columns = onehot_columns or {}
        self.scaler = scaler

    def fit(self, X, y=None):  # noqa: N802 - sklearn API
        return self

    def transform(self, X):  # noqa: N802 - sklearn API
        df = X.copy()

        for enc in self.encodings:
            column = enc.get("column")
            method = enc.get("method")

            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found for encoding")

            if method == "label":
                encoder = self.label_encoders.get(column)
                if encoder is None:
                    raise ValueError(f"Missing label encoder for '{column}'")
                df[column] = encoder.transform(df[column].astype(str))
            elif method == "onehot":
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                output_columns = self.onehot_columns.get(column) or enc.get("output_columns")
                if not output_columns:
                    raise ValueError(f"Missing one-hot columns for '{column}'")

                for col in output_columns:
                    if col not in dummies.columns:
                        dummies[col] = 0
                dummies = dummies[output_columns]

                df = df.drop(columns=[column])
                df = pd.concat([df, dummies], axis=1)
            else:
                raise ValueError(f"Unknown encoding method: {method}")

        if self.scaling:
            columns = self.scaling.get("columns", [])
            if columns:
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns for scaling: {missing}")
                if self.scaler is None:
                    raise ValueError("Missing scaler for numeric scaling")
                df[columns] = self.scaler.transform(df[columns])

        return df


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = find_project_root()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_model_pipeline(ctx) -> Any:
    """Return a fitted model or a preprocessing+model pipeline."""
    model = ctx.model
    if model is None:
        return None

    has_preprocessing = bool(ctx.preprocessing.get("encodings")) or bool(ctx.preprocessing.get("scaling"))
    if not has_preprocessing:
        return model

    preprocessor = InferencePreprocessor(
        encodings=ctx.preprocessing.get("encodings", []),
        scaling=ctx.preprocessing.get("scaling"),
        label_encoders=ctx.preprocessing_artifacts.get("label_encoders", {}),
        onehot_columns=ctx.preprocessing_artifacts.get("onehot_columns", {}),
        scaler=ctx.preprocessing_artifacts.get("scaler")
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


def build_metadata(ctx, run_id: str) -> dict[str, Any]:
    """Build metadata for a trained run."""
    model = ctx.model
    is_classification = bool(ctx.preprocessing.get("target_encoding"))
    if not is_classification and model is not None and hasattr(model, "predict_proba"):
        is_classification = True

    feature_columns = ctx.feature_columns or (ctx.X_train.columns.tolist() if ctx.X_train is not None else [])
    class_labels = None
    if is_classification:
        if ctx.preprocessing.get("target_encoding"):
            class_labels = ctx.preprocessing["target_encoding"].get("original_classes")
        elif model is not None and hasattr(model, "classes_"):
            class_labels = list(model.classes_)

    model_type = ctx.current_model_name or (model.__class__.__name__ if model is not None else None)
    metrics = ctx.results.get("model_metrics") if hasattr(ctx, "results") else None

    meta = {
        "model_id": run_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_column": ctx.target_column,
        "task_type": "classification" if is_classification else "regression",
        "feature_columns": feature_columns,
        "model_type": model_type,
        "metrics": _sanitize_for_json(metrics),
        "class_labels": _sanitize_for_json(class_labels),
        "preprocessing": _sanitize_for_json(ctx.preprocessing or {}),
    }

    if ctx.preprocessing.get("target_encoding"):
        meta["label_mapping"] = _sanitize_for_json(
            ctx.preprocessing["target_encoding"].get("mapping")
        )
        meta["positive_label"] = _sanitize_for_json(
            ctx.preprocessing["target_encoding"].get("positive_label")
        )

    return meta


def list_models() -> list[dict[str, Any]]:
    """Return metadata for all saved models."""
    if not ARTIFACTS_DIR.exists():
        return []

    models: list[dict[str, Any]] = []
    for item in ARTIFACTS_DIR.iterdir():
        if not item.is_dir():
            continue
        meta_path = item / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        models.append(meta)

    def sort_key(meta: dict[str, Any]) -> str:
        return str(meta.get("created_at") or "")

    models.sort(key=sort_key, reverse=True)
    return models


def delete_artifacts(run_id: str) -> bool:
    """Delete a model's artifact folder."""
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    for item in run_dir.iterdir():
        item.unlink()
    run_dir.rmdir()
    return True


def save_artifacts(run_id: str, model: Any, meta: dict[str, Any]) -> None:
    """Persist model pipeline and metadata for a run."""
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(run_id: str) -> tuple[Any, dict[str, Any]]:
    """Load model pipeline and metadata for a run."""
    run_dir = ARTIFACTS_DIR / run_id
    model_path = run_dir / "model.joblib"
    meta_path = run_dir / "meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Artifacts missing for run {run_id}")

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta


def load_metadata(run_id: str) -> dict[str, Any]:
    """Load metadata for a run without loading the model."""
    run_dir = ARTIFACTS_DIR / run_id
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata missing for run {run_id}")
    return json.loads(meta_path.read_text())


def validate_and_prepare_features(
    df: pd.DataFrame,
    meta: dict[str, Any]
) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Validate required columns and return ordered features."""
    feature_columns = meta.get("feature_columns") or []
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return None, missing
    if not feature_columns:
        return df.copy(), []
    return df[feature_columns].copy(), []


def _to_list(values: Any) -> list[Any]:
    return values.tolist() if hasattr(values, "tolist") else list(values)


def predict_with_optional_proba(
    model: Any,
    X: pd.DataFrame,
    meta: dict[str, Any]
) -> tuple[list[Any], Optional[list[float]]]:
    """Return predictions and optional probabilities."""
    preds = model.predict(X)

    mapped_preds: list[Any] = []
    label_mapping = meta.get("label_mapping") or {}
    if label_mapping:
        inverse = {v: k for k, v in label_mapping.items()}
        for p in preds:
            try:
                key = int(p)
            except (TypeError, ValueError):
                key = p
            mapped_preds.append(inverse.get(key, p))
    else:
        mapped_preds = _to_list(preds)

    probs = None
    if meta.get("task_type") == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "shape", None) and proba.shape[1] == 2:
            positive_index = 1
            if meta.get("positive_label") and label_mapping:
                encoded = label_mapping.get(meta["positive_label"])
                if encoded is not None and hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    if encoded in classes:
                        positive_index = classes.index(encoded)
            probs = _to_list(proba[:, positive_index])

    return mapped_preds, probs
