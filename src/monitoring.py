"""
Privacy-Conscious Prediction Logging and Model Monitoring Module
Day 20: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

This module handles:
1. Event logging for student performance predictions with strict privacy safeguards.
2. Subjective user feedback collection (clearly separated from verified outcomes).
3. Summary analytics and monitoring metrics computation.
4. Resilient, non-blocking I/O that prevents monitoring errors from interrupting prediction.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Set up module logger
logger = logging.getLogger("model_monitoring")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

# Project root directory and default storage location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "monitoring"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "prediction_events.csv"

# Canonical log schema
STANDARD_LOG_COLUMNS = [
    "timestamp",
    "prediction_id",
    "model_id",
    "predicted_category",
    "proba_High",
    "proba_Low",
    "proba_Medium",
    "feedback"
]


def get_default_log_path() -> Path:
    """Return the resolved default path for the monitoring log CSV."""
    return DEFAULT_LOG_FILE


def ensure_storage_ready(log_path: Optional[Path] = None) -> Path:
    """
    Ensure the parent directory and log CSV exist with appropriate headers.
    """
    target_path = Path(log_path) if log_path else DEFAULT_LOG_FILE
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        empty_df = pd.DataFrame(columns=STANDARD_LOG_COLUMNS)
        empty_df.to_csv(target_path, index=False)

    return target_path


def log_prediction_event(
    predicted_category: str,
    probabilities: Optional[Dict[str, float]] = None,
    model_id: str = "Decision_Tree_v1",
    prediction_id: Optional[str] = None,
    feedback: Optional[str] = None,
    input_features: Optional[Dict[str, Any]] = None,
    log_features: bool = False,
    log_path: Optional[Path] = None,
    enabled: bool = True
) -> Tuple[bool, str]:
    """
    Record a prediction event to local storage with strict privacy preservation.

    Privacy & Operational Guardrails:
    ---------------------------------
    1. If `enabled` is False, no record is written.
    2. Student identifiers (names, IDs, emails, contact info) are NEVER recorded.
    3. Input feature attributes are excluded by default. Only if `log_features=True`
       is explicitly enabled will non-target attributes be saved.
    4. G3 and performance_category are strictly excluded from logged features.
    5. I/O failures are caught and logged without raising exceptions, ensuring the
       user-facing prediction workflow is never disrupted by monitoring failures.

    Parameters:
    -----------
    predicted_category : str
        Predicted tier ('Low', 'Medium', or 'High').
    probabilities : dict, optional
        Dictionary mapping class names to estimated probabilities.
    model_id : str
        Identifier of the deployed model pipeline.
    prediction_id : str, optional
        Unique UUID for the event. Generated automatically if omitted.
    feedback : str, optional
        Initial subjective feedback value, if supplied.
    input_features : dict, optional
        Feature values dictionary (only logged if log_features is True).
    log_features : bool
        Whether to store non-target input features (default False).
    log_path : Path, optional
        Custom path for log CSV.
    enabled : bool
        Whether logging is active.

    Returns:
    --------
    tuple (bool, str)
        (True, prediction_id) on success, or (False, error_message) on failure/disabled.
    """
    if not enabled:
        return False, "Logging is disabled by user configuration."

    event_id = prediction_id or uuid.uuid4().hex[:12]
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        resolved_path = ensure_storage_ready(log_path)

        # Prepare standard event dictionary
        event_record = {
            "timestamp": timestamp_utc,
            "prediction_id": event_id,
            "model_id": str(model_id),
            "predicted_category": str(predicted_category),
            "proba_High": np.nan,
            "proba_Low": np.nan,
            "proba_Medium": np.nan,
            "feedback": str(feedback) if feedback else "None"
        }

        # Populate probability fields safely
        if probabilities and isinstance(probabilities, dict):
            for cls_name in ["High", "Low", "Medium"]:
                col_key = f"proba_{cls_name}"
                val = probabilities.get(cls_name)
                if val is None:
                    val = probabilities.get(f"model_estimated_proba_{cls_name}")
                if val is not None:
                    event_record[col_key] = float(val)

        # Opt-in feature logging (strictly non-target, non-PII)
        if log_features and input_features and isinstance(input_features, dict):
            for feat_name, feat_val in input_features.items():
                # Strict target leakage protection: never log G3
                from src.dataset import CANONICAL_FEATURES
                if feat_name not in CANONICAL_FEATURES:
                    continue
                event_record[f"feat_{feat_name}"] = feat_val

        # Create 1-row DataFrame and append to CSV
        new_row_df = pd.DataFrame([event_record])

        # If new columns exist (e.g. opt-in features), align with existing CSV
        if resolved_path.exists() and resolved_path.stat().st_size > 0:
            existing_cols = pd.read_csv(resolved_path, nrows=0).columns.tolist()
            # If new row has identical columns to existing CSV, simple append
            if set(new_row_df.columns) == set(existing_cols):
                new_row_df[existing_cols].to_csv(resolved_path, mode="a", header=False, index=False)
            else:
                # Merge schema
                existing_df = pd.read_csv(resolved_path)
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                combined_df.to_csv(resolved_path, index=False)
        else:
            new_row_df.to_csv(resolved_path, index=False)

        return True, event_id

    except Exception as e:
        logger.warning(f"Failed to record prediction event safely: {e}")
        return False, f"Monitoring storage error: {str(e)}"


def update_prediction_feedback(
    prediction_id: str,
    feedback_value: str,
    log_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Update subjective user feedback for an existing prediction event.

    Important Framing:
    ------------------
    This function records subjective user impression (e.g., 'Helpful', 'Neutral',
    'Unhelpful'). It must NOT be interpreted or reported as verified ground truth
    or true model accuracy.

    Parameters:
    -----------
    prediction_id : str
        Unique identifier of the prediction event.
    feedback_value : str
        Subjective feedback score or rating.
    log_path : Path, optional
        Custom path for log CSV.

    Returns:
    --------
    tuple (bool, str)
        (True, message) on success, or (False, error_message) on failure.
    """
    target_path = Path(log_path) if log_path else DEFAULT_LOG_FILE
    if not target_path.exists():
        return False, f"Log file does not exist at {target_path}"

    try:
        df = pd.read_csv(target_path, keep_default_na=False)
        if "prediction_id" not in df.columns:
            return False, "Malformed log schema: 'prediction_id' column missing."

        df["prediction_id"] = df["prediction_id"].astype(str)
        df["feedback"] = df["feedback"].astype(str)

        mask = df["prediction_id"] == str(prediction_id)
        if not mask.any():
            return False, f"Event with prediction_id '{prediction_id}' not found."

        df.loc[mask, "feedback"] = str(feedback_value)
        df.to_csv(target_path, index=False)
        return True, f"Feedback '{feedback_value}' recorded for prediction {prediction_id}."

    except Exception as e:
        logger.warning(f"Failed to update feedback for prediction {prediction_id}: {e}")
        return False, f"Feedback update error: {str(e)}"


def load_prediction_logs(log_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all logged prediction events as a Pandas DataFrame.
    Returns an empty DataFrame with the standard schema if no logs exist.
    """
    target_path = Path(log_path) if log_path else DEFAULT_LOG_FILE
    if not target_path.exists() or target_path.stat().st_size == 0:
        return pd.DataFrame(columns=STANDARD_LOG_COLUMNS)

    try:
        df = pd.read_csv(target_path, keep_default_na=False)
        # Convert numeric probability columns back to float
        for col in ["proba_High", "proba_Low", "proba_Medium"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "timestamp" in df.columns and len(df) > 0:
            df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        logger.warning(f"Error loading logs from {target_path}: {e}")
        return pd.DataFrame(columns=STANDARD_LOG_COLUMNS)


def get_monitoring_summary(log_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Compute aggregate operational statistics and monitoring summaries from event logs.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'is_empty': bool indicating if log has records
        - 'total_events': int total count of logged predictions
        - 'class_counts': pd.Series count of predictions per tier
        - 'class_proportions': pd.Series percentage of predictions per tier
        - 'average_probabilities': dict mean probability for each class
        - 'feedback_counts': pd.Series subjective feedback breakdown
        - 'feedback_summary': dict with feedback counts and disclaimer
        - 'recent_events': pd.DataFrame latest 10 events
    """
    df = load_prediction_logs(log_path)

    if len(df) == 0:
        return {
            "is_empty": True,
            "total_events": 0,
            "class_counts": pd.Series(dtype=int),
            "class_proportions": pd.Series(dtype=float),
            "average_probabilities": {"High": np.nan, "Low": np.nan, "Medium": np.nan},
            "feedback_counts": pd.Series(dtype=int),
            "feedback_summary": {
                "total_feedback_events": 0,
                "disclaimer": "Subjective user feedback only; not verified ground-truth accuracy."
            },
            "recent_events": pd.DataFrame(columns=STANDARD_LOG_COLUMNS)
        }

    # Prediction class distributions
    class_counts = df["predicted_category"].value_counts()
    class_proportions = (df["predicted_category"].value_counts(normalize=True) * 100).round(2)

    # Average class probabilities where available
    avg_probas = {}
    for cls in ["High", "Low", "Medium"]:
        col = f"proba_{cls}"
        if col in df.columns and df[col].notna().any():
            avg_probas[cls] = round(float(df[col].mean()), 4)
        else:
            avg_probas[cls] = np.nan

    # Feedback breakdown (excluding unassigned 'None')
    feedback_series = df["feedback"].dropna()
    active_feedback = feedback_series[feedback_series != "None"]
    feedback_counts = active_feedback.value_counts()

    return {
        "is_empty": False,
        "total_events": len(df),
        "class_counts": class_counts,
        "class_proportions": class_proportions,
        "average_probabilities": avg_probas,
        "feedback_counts": feedback_counts,
        "feedback_summary": {
            "total_feedback_events": int(len(active_feedback)),
            "feedback_participation_rate_pct": round(float(len(active_feedback) / len(df) * 100), 2),
            "disclaimer": "Subjective user impression only. Does not measure statistical accuracy, precision, or recall."
        },
        "recent_events": df.tail(10)
    }


def clear_monitoring_logs(log_path: Optional[Path] = None) -> bool:
    """Clear existing log records (useful for automated testing or reset)."""
    target_path = Path(log_path) if log_path else DEFAULT_LOG_FILE
    if target_path.exists():
        try:
            target_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to clear log file {target_path}: {e}")
            return False
    return True
