"""
Dataset Profiler - Builds a compact JSON profile for the orchestrator agent.

Design principles:
1. MINIMAL - Only include what the agent needs to make planning decisions
2. STRUCTURED - Consistent format that's easy to reference
3. ACTIONABLE - Focus on signals that inform which steps to take
"""

import pandas as pd
import re
from typing import Optional, Literal
from rapidfuzz import process, fuzz
from sklearn.feature_selection import mutual_info_classif

# normalize column names for fuzzy matching
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# perform fuzzy matching on user input to find the best matching column if no exact match is found
def resolve_target_column(df_columns, user_target, threshold=85):
    if not user_target:
        raise ValueError("No target column provided.")

    norm_cols = {_norm(c): c for c in df_columns}
    target_norm = _norm(user_target)

    # exact normalized match
    if target_norm in norm_cols:
        return norm_cols[target_norm]

    # fuzzy match
    matches = process.extract(
        target_norm,
        norm_cols.keys(),
        scorer=fuzz.WRatio,
        limit=5,
    )

    best_norm, best_score, _ = matches[0]
    best_col = norm_cols[best_norm]

    # if the best score is greater than the threshold, return the best column
    if best_score >= threshold:
        print(
            f"Warning: interpreting target '{user_target}' as column '{best_col}' "
            f"(similarity {best_score:.1f})"
        )
        return best_col

    suggestions = ", ".join(
        f"{norm_cols[n]} ({s:.1f})" for n, s, _ in matches
    )
    # if the best score is less than the threshold, raise an error
    raise ValueError(
        f"Could not match target '{user_target}' to any column.\n"
        f"Top candidates: {suggestions}"
    )

# ============================================================================
# SEMANTIC TYPE INFERENCE
# ============================================================================

def _infer_semantic_type(series: pd.Series, n_rows: int) -> Literal["numeric", "categorical", "binary", "text"]:
    """Classify a column into one of 4 semantic types for planning purposes."""
    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique == 2:
            return "binary"
        if n_unique <= 15 or (n_unique / max(n_rows, 1)) < 0.05:
            return "categorical"
        return "numeric"

    # Object dtype - check if it's low-cardinality categorical or free text
    n_unique = series.nunique(dropna=True)
    if n_unique <= 50 or (n_unique / max(n_rows, 1)) < 0.1:
        return "categorical"
    return "text"


def _infer_task_type(target_series: pd.Series, n_rows: int) -> Literal["binary_classification", "multiclass_classification", "regression"]:
    """Infer the ML task type from the target column."""
    n_unique = target_series.nunique(dropna=True)

    if n_unique == 2:
        return "binary_classification"
    if n_unique <= 20 or (n_unique / max(n_rows, 1)) < 0.05:
        return "multiclass_classification"
    return "regression"


# ============================================================================
# COMPACT PROFILE BUILDERS
# ============================================================================

def _build_column_summary(df: pd.DataFrame, target_column: Optional[str] = None) -> list[dict]:
    """
    Build a compact column summary - just what the agent needs for planning.
    Returns a list of column info dicts with essential metadata only.
    """
    n_rows = len(df)
    columns = []

    for col in df.columns:
        series = df[col]
        sem_type = _infer_semantic_type(series, n_rows)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        col_info = {
            "name": col,
            "type": sem_type,
            "missing": n_missing,
            "unique": n_unique,
        }

        # Add is_target flag only if this is the target
        if col == target_column:
            col_info["is_target"] = True

        # Add compact stats based on type
        if sem_type == "numeric":
            desc = series.describe()
            col_info["stats"] = {
                "min": round(float(desc["min"]), 2),
                "max": round(float(desc["max"]), 2),
                "mean": round(float(desc["mean"]), 2),
                "std": round(float(desc["std"]), 2),
            }
        elif sem_type in ("categorical", "binary"):
            top_vals = series.value_counts(dropna=True).head(3)
            col_info["top_values"] = [str(v) for v in top_vals.index.tolist()]

        columns.append(col_info)

    return columns


def _build_target_info(df: pd.DataFrame, target_column: str) -> dict:
    """Build compact target column info for modeling tasks."""
    series = df[target_column]
    task_type = _infer_task_type(series, len(df))

    info = {"task_type": task_type}

    if task_type in ("binary_classification", "multiclass_classification"):
        counts = series.value_counts(dropna=True)
        info["classes"] = counts.index.tolist()
        info["class_counts"] = counts.tolist()
        # Imbalance ratio (max/min)
        info["imbalance_ratio"] = round(counts.max() / max(counts.min(), 1), 2)
    else:  # regression
        desc = series.describe()
        info["range"] = [round(float(desc["min"]), 2), round(float(desc["max"]), 2)]
        info["mean"] = round(float(desc["mean"]), 2)

    return info


def _build_quality_flags(df: pd.DataFrame, columns: list[dict]) -> list[str]:
    """
    Generate actionable quality flags that inform cleaning decisions.
    These are simple string flags the agent can use to decide what cleaning steps are needed.
    """
    flags = []
    n_rows = len(df)

    # Check for missing data
    cols_with_missing = [c["name"] for c in columns if c["missing"] > 0]
    if cols_with_missing:
        pct = sum(c["missing"] for c in columns) / (n_rows * len(columns)) * 100
        flags.append(f"missing_data:{len(cols_with_missing)}_cols:{pct:.1f}%_total")

    # Check for high cardinality categoricals
    high_card = [c["name"] for c in columns if c["type"] == "categorical" and c["unique"] > 50]
    if high_card:
        flags.append(f"high_cardinality:{','.join(high_card[:3])}")

    # Check for potential ID columns (unique ratio > 0.9)
    id_cols = [c["name"] for c in columns if c["unique"] / max(n_rows, 1) > 0.9 and c["type"] != "numeric"]
    if id_cols:
        flags.append(f"potential_id_columns:{','.join(id_cols[:3])}")

    # Check for constant columns
    constant = [c["name"] for c in columns if c["unique"] <= 1]
    if constant:
        flags.append(f"constant_columns:{','.join(constant)}")

    return flags


def _build_top_correlations(df: pd.DataFrame, target_column: Optional[str], columns: list[dict], top_k: int = 5) -> list[dict]:
    """
    Get top feature-target correlations/associations.
    Returns a compact list for the agent to prioritize visualizations.
    """
    if not target_column:
        return []

    target_series = df[target_column]
    numeric_cols = [c["name"] for c in columns if c["type"] == "numeric" and c["name"] != target_column]

    if not numeric_cols:
        return []

    # Encode target if categorical
    if not pd.api.types.is_numeric_dtype(target_series):
        target_encoded = target_series.astype("category").cat.codes
    else:
        target_encoded = target_series

    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(target_encoded)
        if pd.notna(corr):
            correlations.append({
                "feature": col,
                "corr": round(float(corr), 3),
                "abs": round(abs(float(corr)), 3)
            })

    # Sort by absolute correlation and return top_k
    correlations.sort(key=lambda x: x["abs"], reverse=True)
    return correlations[:top_k]


def _build_feature_correlations(df: pd.DataFrame, columns: list[dict], top_k: int = 5) -> list[dict]:
    """Get top feature-feature correlations for multicollinearity awareness."""
    numeric_cols = [c["name"] for c in columns if c["type"] == "numeric"]

    if len(numeric_cols) < 2:
        return []

    corr_matrix = df[numeric_cols].corr(method="pearson")
    pairs = []

    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i+1:]:
            corr = corr_matrix.loc[col_a, col_b]
            if pd.notna(corr) and abs(corr) > 0.3:  # Only include notable correlations
                pairs.append({
                    "a": col_a,
                    "b": col_b,
                    "corr": round(float(corr), 3)
                })

    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return pairs[:top_k]


# ============================================================================
# MAIN PROFILE BUILDER
# ============================================================================

def build_dataset_profile(
    df: pd.DataFrame,
    user_goal: str,
    target_column: Optional[str] = None,
    analysis_mode: Literal["eda", "predictive", "hybrid"] = "eda"
) -> dict:
    """
    Build a compact dataset profile optimized for the orchestrator agent.

    The profile is designed to be:
    - Small enough for fast LLM processing (< 2KB typically)
    - Complete enough for informed planning decisions
    - Structured for easy reference by column name

    Args:
        df: The pandas DataFrame to profile
        user_goal: The user's stated analysis objective
        target_column: Optional target column for predictive tasks
        analysis_mode: One of "eda", "predictive", or "hybrid"

    Returns:
        A dict ready to be JSON-serialized and sent to the orchestrator
    """
    n_rows, n_cols = df.shape

    # Build column summaries first (used by other builders)
    columns = _build_column_summary(df, target_column)

    # Core profile structure
    profile = {
        "goal": user_goal,
        "mode": analysis_mode,
        "shape": {"rows": n_rows, "cols": n_cols},
        "columns": columns,
    }

    # Add quality flags for cleaning decisions
    quality_flags = _build_quality_flags(df, columns)
    if quality_flags:
        profile["quality_flags"] = quality_flags

    # Add target info for predictive/hybrid modes
    if target_column and analysis_mode in ("predictive", "hybrid"):
        profile["target"] = _build_target_info(df, target_column)

        # Add feature-target correlations
        target_corrs = _build_top_correlations(df, target_column, columns, top_k=5)
        if target_corrs:
            profile["target_correlations"] = target_corrs

    # Add feature-feature correlations (useful for both EDA and modeling)
    feature_corrs = _build_feature_correlations(df, columns, top_k=5)
    if feature_corrs:
        profile["feature_correlations"] = feature_corrs

    return profile