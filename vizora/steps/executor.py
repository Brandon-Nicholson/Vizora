"""
Plan Executor - Executes the JSON plan returned by the orchestrator agent.

This module provides a registry-based action executor that maps action types
to their implementations. Each action type defined in the orchestrator schema
has a corresponding handler function.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Optional
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, brier_score_loss,
    confusion_matrix as sk_confusion_matrix, roc_curve as sk_roc_curve
)


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

@dataclass
class ExecutionContext:
    """
    Holds state during plan execution.

    Attributes:
        df: The working DataFrame (modified by cleaning actions)
        original_df: The original unmodified DataFrame
        X_train, X_test, y_train, y_test: Split data (set by train_test_split)
        models: Dict of trained model instances (name -> model)
        current_model_name: Name of the most recently trained model
        predictions: Model predictions on test set (from current model)
        figures: List of generated matplotlib figures
        results: Dict of statistical results
        errors: List of error messages
        preprocessing: Dict tracking all preprocessing steps for reproducibility
    """
    df: pd.DataFrame
    original_df: pd.DataFrame = None
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None
    target_column: str = None
    models: dict = field(default_factory=dict)  # name -> trained model
    current_model_name: str = None
    predictions: dict = field(default_factory=dict)  # model_name -> predictions
    figures: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    label_mapping: dict = field(default_factory=dict)  # e.g. {"Absence": 0, "Presence": 1}
    positive_label: str | None = None
    # Preprocessing artifacts for reproducibility
    preprocessing: dict = field(default_factory=lambda: {
        "encodings": [],      # List of {"column": str, "method": "label"|"onehot", "classes": list}
        "scaling": None,      # {"method": str, "columns": list}
        "target_encoding": None,  # {"original_values": list, "mapping": dict}
        "train_test_split": None,  # {"test_size": float, "stratify": bool, "random_state": int}
    })

    def __post_init__(self):
        if self.original_df is None:
            self.original_df = self.df.copy()

    @property
    def model(self):
        """Get the current/most recent model."""
        if self.current_model_name:
            return self.models.get(self.current_model_name)
        return None


# =============================================================================
# ACTION REGISTRY
# =============================================================================

ACTION_HANDLERS: dict[str, Callable] = {}


def register_action(name: str):
    """Decorator to register an action handler."""
    def decorator(func: Callable):
        ACTION_HANDLERS[name] = func
        return func
    return decorator


# =============================================================================
# CLEANING ACTIONS
# =============================================================================

@register_action("drop_columns")
def action_drop_columns(ctx: ExecutionContext, spec: dict) -> str:
    """Drop specified columns from the DataFrame."""
    columns = spec.get("columns", [])
    existing = [c for c in columns if c in ctx.df.columns]
    if existing:
        ctx.df = ctx.df.drop(columns=existing)
        return f"Dropped columns: {existing}"
    return "No columns to drop"


@register_action("fill_missing")
def action_fill_missing(ctx: ExecutionContext, spec: dict) -> str:
    """Fill missing values in a column."""
    column = spec.get("column")
    strategy = spec.get("strategy", "mean")

    if column not in ctx.df.columns:
        return f"Column '{column}' not found"

    if strategy == "mean":
        ctx.df[column] = ctx.df[column].fillna(ctx.df[column].mean())
    elif strategy == "median":
        ctx.df[column] = ctx.df[column].fillna(ctx.df[column].median())
    elif strategy == "mode":
        ctx.df[column] = ctx.df[column].fillna(ctx.df[column].mode().iloc[0])
    elif strategy == "constant":
        value = spec.get("value", 0)
        ctx.df[column] = ctx.df[column].fillna(value)

    return f"Filled missing in '{column}' with {strategy}"


@register_action("drop_missing_rows")
def action_drop_missing_rows(ctx: ExecutionContext, spec: dict) -> str:
    """Drop rows with missing values."""
    before = len(ctx.df)

    if "columns" in spec:
        ctx.df = ctx.df.dropna(subset=spec["columns"])
    elif "threshold" in spec:
        thresh = int(len(ctx.df.columns) * (1 - spec["threshold"]))
        ctx.df = ctx.df.dropna(thresh=thresh)
    else:
        ctx.df = ctx.df.dropna()

    dropped = before - len(ctx.df)
    return f"Dropped {dropped} rows with missing values"


@register_action("encode_categorical")
def action_encode_categorical(ctx: ExecutionContext, spec: dict) -> str:
    """Encode a categorical column. If train/test split exists, applies to X_train/X_test."""
    column = spec.get("column")
    method = spec.get("method", "label")

    if not column:
        return "No column specified"

    # Never encode the target column
    if ctx.target_column and column == ctx.target_column:
        return f"Skipped encoding target column '{column}'"

    encoding_info = {"column": column, "method": method, "classes": []}

    # If we have split data, apply to X_train/X_test instead of ctx.df
    if ctx.X_train is not None:
        if column not in ctx.X_train.columns:
            return f"Column '{column}' not found in training data"

        if method == "label":
            le = LabelEncoder()
            # Fit on train, transform both
            le.fit(ctx.X_train[column].astype(str))
            encoding_info["classes"] = le.classes_.tolist()
            ctx.X_train[column] = le.transform(ctx.X_train[column].astype(str))
            ctx.X_test[column] = le.transform(ctx.X_test[column].astype(str))
        elif method == "onehot":
            # Get dummies for train
            train_dummies = pd.get_dummies(ctx.X_train[column], prefix=column, drop_first=True)
            test_dummies = pd.get_dummies(ctx.X_test[column], prefix=column, drop_first=True)
            encoding_info["classes"] = ctx.X_train[column].unique().tolist()
            encoding_info["output_columns"] = train_dummies.columns.tolist()

            # Ensure test has same columns as train (handle unseen categories)
            for col in train_dummies.columns:
                if col not in test_dummies.columns:
                    test_dummies[col] = 0
            test_dummies = test_dummies[train_dummies.columns]

            ctx.X_train = ctx.X_train.drop(columns=[column])
            ctx.X_train = pd.concat([ctx.X_train, train_dummies], axis=1)
            ctx.X_test = ctx.X_test.drop(columns=[column])
            ctx.X_test = pd.concat([ctx.X_test, test_dummies], axis=1)
        else:
            return f"Unknown encoding method: {method}"
    else:
        # Pre-split encoding (for EDA mode)
        if column not in ctx.df.columns:
            return f"Column '{column}' not found"

        if method == "label":
            le = LabelEncoder()
            le.fit(ctx.df[column].astype(str))
            encoding_info["classes"] = le.classes_.tolist()
            ctx.df[column] = le.transform(ctx.df[column].astype(str))
        elif method == "onehot":
            encoding_info["classes"] = ctx.df[column].unique().tolist()
            dummies = pd.get_dummies(ctx.df[column], prefix=column, drop_first=True)
            encoding_info["output_columns"] = dummies.columns.tolist()
            ctx.df = ctx.df.drop(columns=[column])
            ctx.df = pd.concat([ctx.df, dummies], axis=1)
        else:
            return f"Unknown encoding method: {method}"

    # Track encoding for reproducibility
    ctx.preprocessing["encodings"].append(encoding_info)
    return f"Encoded '{column}' with {method}"



@register_action("scale_numeric")
def action_scale_numeric(ctx: ExecutionContext, spec: dict) -> str:
    """Scale numeric columns. If train/test split exists, applies to X_train/X_test."""
    columns = spec.get("columns", [])
    method = spec.get("method", "standard")

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return f"Unknown scaling method: {method}"

    # If we have split data, apply to X_train/X_test
    if ctx.X_train is not None:
        existing = [c for c in columns if c in ctx.X_train.columns]
        if not existing:
            return "No valid columns to scale in training data"

        # Fit on train, transform both
        scaler.fit(ctx.X_train[existing])
        ctx.X_train[existing] = scaler.transform(ctx.X_train[existing])
        ctx.X_test[existing] = scaler.transform(ctx.X_test[existing])
    else:
        # Pre-split scaling (for EDA mode)
        existing = [c for c in columns if c in ctx.df.columns]
        if not existing:
            return "No valid columns to scale"

        ctx.df[existing] = scaler.fit_transform(ctx.df[existing])

    # Track scaling for reproducibility
    ctx.preprocessing["scaling"] = {"method": method, "columns": existing}
    return f"Scaled {existing} with {method}"


# =============================================================================
# VISUALIZATION ACTIONS
# =============================================================================

@register_action("histogram")
def action_histogram(ctx: ExecutionContext, spec: dict) -> str:
    """Create a histogram."""
    column = spec.get("column")
    bins = spec.get("bins", 20)

    if column not in ctx.df.columns:
        return f"Column '{column}' not found"

    fig, ax = plt.subplots(figsize=(8, 5))
    ctx.df[column].hist(bins=bins, ax=ax, edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {column}")
    ctx.figures.append(("histogram", column, fig))
    return f"Created histogram for '{column}'"


@register_action("boxplot")
def action_boxplot(ctx: ExecutionContext, spec: dict) -> str:
    """Create a boxplot."""
    column = spec.get("column")
    by = spec.get("by")

    if column not in ctx.df.columns:
        return f"Column '{column}' not found"

    fig, ax = plt.subplots(figsize=(8, 5))
    if by and by in ctx.df.columns:
        ctx.df.boxplot(column=column, by=by, ax=ax)
        ax.set_title(f"{column} by {by}")
    else:
        ctx.df.boxplot(column=column, ax=ax)
        ax.set_title(f"Boxplot of {column}")
    plt.suptitle("")  # Remove automatic title
    ctx.figures.append(("boxplot", column, fig))
    return f"Created boxplot for '{column}'"


@register_action("countplot")
def action_countplot(ctx: ExecutionContext, spec: dict) -> str:
    """Create a count plot."""
    column = spec.get("column")

    if column not in ctx.df.columns:
        return f"Column '{column}' not found"

    fig, ax = plt.subplots(figsize=(8, 5))
    ctx.df[column].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Count of {column}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ctx.figures.append(("countplot", column, fig))
    return f"Created countplot for '{column}'"


@register_action("scatterplot")
def action_scatterplot(ctx: ExecutionContext, spec: dict) -> str:
    """Create a scatter plot."""
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")

    if x not in ctx.df.columns or y not in ctx.df.columns:
        return f"Columns '{x}' or '{y}' not found"

    fig, ax = plt.subplots(figsize=(8, 6))
    if hue and hue in ctx.df.columns:
        sns.scatterplot(data=ctx.df, x=x, y=y, hue=hue, ax=ax)
    else:
        sns.scatterplot(data=ctx.df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    ctx.figures.append(("scatterplot", f"{x}_vs_{y}", fig))
    return f"Created scatterplot: {y} vs {x}"


@register_action("heatmap")
def action_heatmap(ctx: ExecutionContext, spec: dict) -> str:
    """Create a correlation heatmap."""
    columns = spec.get("columns", [])

    if columns:
        cols = [c for c in columns if c in ctx.df.columns]
    else:
        cols = ctx.df.select_dtypes(include=[np.number]).columns.tolist()

    if len(cols) < 2:
        return "Not enough numeric columns for heatmap"

    corr = ctx.df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    ctx.figures.append(("heatmap", "correlation", fig))
    return f"Created correlation heatmap for {len(cols)} columns"


@register_action("violinplot")
def action_violinplot(ctx: ExecutionContext, spec: dict) -> str:
    """Create a violin plot."""
    x = spec.get("x")
    y = spec.get("y")

    if x not in ctx.df.columns or y not in ctx.df.columns:
        return f"Columns '{x}' or '{y}' not found"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=ctx.df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} by {x}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ctx.figures.append(("violinplot", f"{y}_by_{x}", fig))
    return f"Created violinplot: {y} by {x}"


@register_action("barplot")
def action_barplot(ctx: ExecutionContext, spec: dict) -> str:
    """Create a bar plot."""
    x = spec.get("x")
    y = spec.get("y")
    estimator = spec.get("estimator", "mean")

    if x not in ctx.df.columns:
        return f"Column '{x}' not found"

    fig, ax = plt.subplots(figsize=(8, 5))

    if estimator == "count":
        ctx.df[x].value_counts().plot(kind='bar', ax=ax)
    else:
        est_func = np.mean if estimator == "mean" else np.sum
        sns.barplot(data=ctx.df, x=x, y=y, estimator=est_func, ax=ax)

    ax.set_title(f"{y} by {x}" if y else f"Count of {x}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ctx.figures.append(("barplot", f"{x}", fig))
    return f"Created barplot for '{x}'"


# =============================================================================
# STATISTICAL ACTIONS
# =============================================================================

@register_action("describe")
def action_describe(ctx: ExecutionContext, spec: dict) -> str:
    """Get descriptive statistics."""
    columns = spec.get("columns", [])

    if columns:
        cols = [c for c in columns if c in ctx.df.columns]
        desc = ctx.df[cols].describe()
    else:
        desc = ctx.df.describe()

    ctx.results["describe"] = desc.to_dict()
    return f"Generated descriptive statistics for {len(desc.columns)} columns"


@register_action("value_counts")
def action_value_counts(ctx: ExecutionContext, spec: dict) -> str:
    """Get value counts for a column."""
    column = spec.get("column")
    top_n = spec.get("top_n", 10)

    if column not in ctx.df.columns:
        return f"Column '{column}' not found"

    counts = ctx.df[column].value_counts().head(top_n)
    ctx.results[f"value_counts_{column}"] = counts.to_dict()
    return f"Got value counts for '{column}' (top {top_n})"


@register_action("correlation_matrix")
def action_correlation_matrix(ctx: ExecutionContext, spec: dict) -> str:
    """Compute correlation matrix."""
    columns = spec.get("columns", [])

    if columns:
        cols = [c for c in columns if c in ctx.df.columns]
    else:
        cols = ctx.df.select_dtypes(include=[np.number]).columns.tolist()

    if len(cols) < 2:
        return "Not enough numeric columns"

    corr = ctx.df[cols].corr()
    ctx.results["correlation_matrix"] = corr.to_dict()
    return f"Computed correlation matrix for {len(cols)} columns"


@register_action("crosstab")
def action_crosstab(ctx: ExecutionContext, spec: dict) -> str:
    """Create a crosstab."""
    index = spec.get("index")
    columns = spec.get("columns")

    if index not in ctx.df.columns or columns not in ctx.df.columns:
        return f"Columns '{index}' or '{columns}' not found"

    ct = pd.crosstab(ctx.df[index], ctx.df[columns])
    ctx.results[f"crosstab_{index}_{columns}"] = ct.to_dict()
    return f"Created crosstab: {index} x {columns}"


# =============================================================================
# MODELING ACTIONS
# =============================================================================

@register_action("train_test_split")
def action_train_test_split(ctx: ExecutionContext, spec: dict) -> str:
    """Split data into train/test sets."""
    test_size = spec.get("test_size", 0.2)
    stratify = spec.get("stratify", False)

    if not ctx.target_column:
        return "No target column specified"

    X = ctx.df.drop(columns=[ctx.target_column])
    y = ctx.df[ctx.target_column]

    # Determine if this looks like classification
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique() <= 20)

    strat = y if stratify and is_classification and y.nunique() < 50 else None

    ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=strat
    )

    # Track split parameters for reproducibility
    ctx.preprocessing["train_test_split"] = {
        "test_size": test_size,
        "stratify": strat is not None,
        "random_state": 42,
        "train_samples": len(ctx.X_train),
        "test_samples": len(ctx.X_test)
    }

    # If classification and y is non-numeric, encode to 0/1 (or 0..K-1)
    if is_classification and not pd.api.types.is_numeric_dtype(ctx.y_train):
        le = LabelEncoder()
        le.fit(ctx.y_train.astype(str))

        ctx.label_mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}

        pos_candidates = {"1", "true", "yes", "y", "positive", "presence", "present", "disease"}
        pos = None
        for cls in le.classes_:
            if str(cls).strip().lower() in pos_candidates:
                pos = str(cls)
                break
        ctx.positive_label = pos or str(le.classes_[-1])

        ctx.y_train = pd.Series(le.transform(ctx.y_train.astype(str)), index=ctx.y_train.index)
        ctx.y_test = pd.Series(le.transform(ctx.y_test.astype(str)), index=ctx.y_test.index)

        # Track target encoding for reproducibility
        ctx.preprocessing["target_encoding"] = {
            "original_classes": le.classes_.tolist(),
            "mapping": ctx.label_mapping,
            "positive_label": ctx.positive_label
        }

    return f"Split data: {len(ctx.X_train)} train, {len(ctx.X_test)} test"

@register_action("train_model")
def action_train_model(ctx: ExecutionContext, spec: dict) -> str:
    """Train a model."""
    model_name = spec.get("model", "logistic_regression")
    params = spec.get("params", {})

    if ctx.X_train is None:
        return "Must run train_test_split first"

    # Default params for each model - user params override these
    default_params = {
        "logistic_regression": {"max_iter": 1000},
        "random_forest": {"n_estimators": 100, "random_state": 42},
        "xgboost": {"n_estimators": 100, "random_state": 42},
        "linear_regression": {},
        "ridge": {},
        "lasso": {},
    }

    model_classes = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": RandomForestClassifier,  # Fallback if xgboost not installed
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
    }

    if model_name not in model_classes:
        return f"Unknown model: {model_name}"

    # Merge defaults with user params (user params take precedence)
    final_params = {**default_params.get(model_name, {}), **params}
    model = model_classes[model_name](**final_params)
    model.fit(ctx.X_train, ctx.y_train)
    preds = model.predict(ctx.X_test)

    # Store model and predictions
    ctx.models[model_name] = model
    ctx.predictions[model_name] = preds
    ctx.current_model_name = model_name

    return f"Trained {model_name}"


@register_action("evaluate_model")
def action_evaluate_model(ctx: ExecutionContext, spec: dict) -> str:
    """Evaluate all trained models with test metrics, cross-validation, and calibration."""
    metrics = spec.get("metrics", ["accuracy"])

    if not ctx.models:
        return "Must train model first"

    all_results = {}
    y_true = ctx.y_test

    # Reconstruct full X and y for cross-validation
    X_full = pd.concat([ctx.X_train, ctx.X_test])
    y_full = pd.concat([ctx.y_train, ctx.y_test])

    for model_name, model in ctx.models.items():
        y_pred = ctx.predictions[model_name]
        results = {}

        # Standard test set metrics
        for metric in metrics:
            try:
                if metric == "accuracy":
                    results["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
                elif metric == "precision":
                    results["precision"] = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif metric == "recall":
                    results["recall"] = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif metric == "f1":
                    results["f1"] = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif metric == "roc_auc":
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(ctx.X_test)
                        if y_prob.shape[1] == 2:
                            results["roc_auc"] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
                elif metric == "mse":
                    results["mse"] = round(mean_squared_error(y_true, y_pred), 4)
                elif metric == "rmse":
                    results["rmse"] = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
                elif metric == "mae":
                    results["mae"] = round(mean_absolute_error(y_true, y_pred), 4)
                elif metric == "r2":
                    results["r2"] = round(r2_score(y_true, y_pred), 4)
            except Exception as e:
                results[metric] = f"Error: {e}"

        # Cross-validation scores (5-fold) for model credibility
        try:
            cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
            results["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
            results["cv_accuracy_std"] = round(cv_scores.std(), 4)
        except Exception:
            pass  # Skip CV if it fails (e.g., not enough samples)

        # Brier score for probability calibration (binary classification only)
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(ctx.X_test)
                if y_prob.shape[1] == 2:
                    brier = brier_score_loss(y_true, y_prob[:, 1])
                    results["brier_score"] = round(brier, 4)
            except Exception:
                pass

        all_results[model_name] = results

    ctx.results["model_metrics"] = all_results
    return f"Evaluated {len(all_results)} models: {list(all_results.keys())}"


@register_action("confusion_matrix")
def action_confusion_matrix(ctx: ExecutionContext, spec: dict) -> str:
    """Create a confusion matrix plot for the current model."""
    if not ctx.models:
        return "Must train model first"

    model_name = ctx.current_model_name
    y_pred = ctx.predictions[model_name]

    cm = sk_confusion_matrix(ctx.y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    ctx.figures.append(("confusion_matrix", model_name, fig))
    return f"Created confusion matrix for {model_name}"


@register_action("roc_curve")
def action_roc_curve(ctx: ExecutionContext, spec: dict) -> str:
    """Create ROC curve plot for all models that support predict_proba."""
    if not ctx.models:
        return "Must train model first"

    fig, ax = plt.subplots(figsize=(8, 6))
    curves_plotted = 0

    for model_name, model in ctx.models.items():
        if not hasattr(model, 'predict_proba'):
            continue

        y_prob = model.predict_proba(ctx.X_test)
        if y_prob.shape[1] != 2:
            continue

        fpr, tpr, _ = sk_roc_curve(ctx.y_test, y_prob[:, 1])
        roc_auc = roc_auc_score(ctx.y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        curves_plotted += 1

    if curves_plotted == 0:
        plt.close(fig)
        return "No models support ROC curve (need predict_proba for binary classification)"

    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    ctx.figures.append(("roc_curve", "comparison", fig))
    return f"Created ROC curves for {curves_plotted} models"


@register_action("feature_importance")
def action_feature_importance(ctx: ExecutionContext, spec: dict) -> str:
    """Plot feature importance for the current model."""
    top_n = spec.get("top_n", 10)

    if not ctx.models:
        return "Must train model first"

    model_name = ctx.current_model_name
    model = ctx.models[model_name]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return f"Model {model_name} doesn't support feature importance"

    features = ctx.X_train.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances ({model_name})")
    plt.tight_layout()
    ctx.figures.append(("feature_importance", model_name, fig))
    ctx.results[f"feature_importance_{model_name}"] = importance_df.to_dict('records')
    return f"Created feature importance plot for {model_name} (top {top_n})"


@register_action("residual_plot")
def action_residual_plot(ctx: ExecutionContext, spec: dict) -> str:
    """Create residual plot for regression."""
    if not ctx.models:
        return "Must train model first"

    model_name = ctx.current_model_name
    y_pred = ctx.predictions[model_name]
    residuals = ctx.y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot ({model_name})")
    plt.tight_layout()
    ctx.figures.append(("residual_plot", model_name, fig))
    return f"Created residual plot for {model_name}"


@register_action("prediction_vs_actual")
def action_prediction_vs_actual(ctx: ExecutionContext, spec: dict) -> str:
    """Create prediction vs actual plot for regression."""
    if not ctx.models:
        return "Must train model first"

    model_name = ctx.current_model_name
    y_pred = ctx.predictions[model_name]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ctx.y_test, y_pred, alpha=0.5)
    min_val = min(ctx.y_test.min(), y_pred.min())
    max_val = max(ctx.y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Prediction vs Actual ({model_name})")
    ax.legend()
    plt.tight_layout()
    ctx.figures.append(("prediction_vs_actual", model_name, fig))
    return f"Created prediction vs actual plot for {model_name}"


# =============================================================================
# PLAN EXECUTOR
# =============================================================================

def execute_plan(
    df: pd.DataFrame,
    plan: dict,
    target_column: Optional[str] = None,
    show_progress: bool = True
) -> ExecutionContext:
    """
    Execute a plan generated by the orchestrator.

    Args:
        df: The DataFrame to analyze
        plan: The JSON plan from the orchestrator
        target_column: Optional target column for modeling
        show_progress: Whether to print progress messages

    Returns:
        ExecutionContext with all results, figures, and errors
    """
    ctx = ExecutionContext(df=df.copy(), target_column=target_column)

    # Define execution order
    sections = ["cleaning", "eda", "analysis", "preprocessing", "modeling", "evaluation"]

    for section in sections:
        if section not in plan:
            continue

        if show_progress:
            print(f"\n=== {section.upper()} ===")

        for action_item in plan[section]:
            action_type = action_item.get("action")
            spec = action_item.get("spec", {})
            reason = action_item.get("reason", "")

            if action_type not in ACTION_HANDLERS:
                msg = f"Unknown action: {action_type}"
                ctx.errors.append(msg)
                if show_progress:
                    print(f"  [!] {msg}")
                continue

            try:
                handler = ACTION_HANDLERS[action_type]
                result = handler(ctx, spec)
                if show_progress:
                    print(f"  [✓] {action_type}: {result}")
            except Exception as e:
                msg = f"{action_type} failed: {e}"
                ctx.errors.append(msg)
                if show_progress:
                    print(f"  [✗] {msg}")

    return ctx
