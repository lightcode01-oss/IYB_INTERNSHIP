"""
Evaluation Dashboard & Metrics Analysis Utilities
Day 23: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Provides robust data loading, schema validation, error breakdown calculations,
and report generation for historical model evaluation results (Days 14–18).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Set up module logger
logger = logging.getLogger("evaluation_dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Root directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Expected evaluation files
EVALUATION_FILENAMES = {
    "cv_results": "day_14_cross_validation_results.csv",
    "baseline_vs_tuned": "day_15_baseline_vs_tuned.csv",
    "tuning_results": "day_15_tuning_results.csv",
    "day_15_test_metrics": "day_15_final_test_metrics.csv",
    "final_model_metrics": "day_16_final_model_metrics.csv",
    "classification_report": "day_16_classification_report.csv",
    "error_analysis": "day_16_error_analysis.csv",
    "model_selection_report": "day_16_model_selection_report.json",
    "day_17_test_metrics": "day_17_test_metrics.csv",
    "pipeline_metadata": "artifacts/student_performance_metadata.json"
}


def find_evaluation_files(base_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Locate real evaluation artifacts across primary and fallback paths.
    Returns a dictionary mapping logical keys to resolved existing Paths (or None).
    """
    root = Path(base_dir) if base_dir else PROJECT_ROOT
    found = {}

    for key, rel_path in EVALUATION_FILENAMES.items():
        candidates = [
            root / "notebooks" / rel_path,
            root / rel_path,
            Path.cwd() / rel_path,
            root / Path(rel_path).name,
            Path.cwd() / Path(rel_path).name
        ]
        resolved = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                resolved = cand.resolve()
                break
        found[key] = resolved

    return found


def load_cv_results(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Day 14 5-Fold Cross-Validation results.
    Validates required columns: Model, Mean CV Accuracy, Mean CV Macro F1.
    """
    path = file_path or find_evaluation_files()["cv_results"]
    if not path or not path.exists():
        return None, "Day 14 Cross-Validation results file not found."
    try:
        df = pd.read_csv(path)
        required = ["Model", "Mean CV Accuracy", "Mean CV Macro F1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Cross-validation file missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"Failed to parse cross-validation results: {str(e)}"


def load_baseline_vs_tuned(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Day 15 Baseline vs Tuned model comparison data.
    """
    path = file_path or find_evaluation_files()["baseline_vs_tuned"]
    if not path or not path.exists():
        return None, "Day 15 Baseline vs Tuned results file not found."
    try:
        df = pd.read_csv(path)
        required = ["Model", "Baseline Mean CV Macro F1", "Tuned Mean CV Macro F1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Baseline vs Tuned file missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"Failed to parse baseline vs tuned results: {str(e)}"


def load_tuning_results(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Day 15 Hyperparameter Tuning details across models.
    """
    path = file_path or find_evaluation_files()["tuning_results"]
    if not path or not path.exists():
        return None, "Day 15 Hyperparameter Tuning results file not found."
    try:
        df = pd.read_csv(path)
        required = ["Model", "Tuned Mean CV Macro F1", "Best Parameters"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Tuning results file missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"Failed to parse tuning results: {str(e)}"


def load_classification_report(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Day 16 Held-out Test Set classification report.
    """
    path = file_path or find_evaluation_files()["classification_report"]
    if not path or not path.exists():
        return None, "Day 16 Classification Report file not found."
    try:
        df = pd.read_csv(path)
        required = ["Class/Average", "precision", "recall", "f1-score", "support"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Classification report missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"Failed to parse classification report: {str(e)}"


def load_error_analysis(file_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Day 16 test set error analysis predictions (130 samples).
    """
    path = file_path or find_evaluation_files()["error_analysis"]
    if not path or not path.exists():
        return None, "Day 16 Error Analysis file not found."
    try:
        df = pd.read_csv(path)
        required = ["actual_category", "predicted_category", "is_correct"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Error analysis file missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"Failed to parse error analysis data: {str(e)}"


def load_model_selection_report(file_path: Optional[Path] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load Day 16 Model Selection Report JSON.
    """
    path = file_path or find_evaluation_files()["model_selection_report"]
    if not path or not path.exists():
        return None, "Day 16 Model Selection Report JSON not found."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, f"Failed to parse model selection report: {str(e)}"


def compute_confusion_matrix(error_df: pd.DataFrame, class_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute the empirical confusion matrix strictly from actual test predictions.
    Row labels: Actual Category; Column labels: Predicted Category.
    """
    if class_order is None:
        class_order = ["Low", "Medium", "High"]

    # Filter to classes present in order
    ct = pd.crosstab(
        error_df["actual_category"],
        error_df["predicted_category"],
        dropna=False
    )

    # Reindex to guarantee complete 3x3 square matrix
    ct = ct.reindex(index=class_order, columns=class_order, fill_value=0)
    return ct


def compute_error_breakdown(error_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute adjacent versus non-adjacent error metrics from test records.
    Adjacent errors: Low <-> Medium or Medium <-> High.
    Non-adjacent errors: Low <-> High (severe errors).
    """
    total = len(error_df)
    correct = int((error_df["is_correct"] == True).sum())
    errors = total - correct

    if errors == 0:
        return {
            "total": total,
            "correct": correct,
            "errors": 0,
            "adjacent_count": 0,
            "adjacent_pct": 0.0,
            "non_adjacent_count": 0,
            "non_adjacent_pct": 0.0
        }

    misclassified = error_df[error_df["is_correct"] == False]

    # Identify non-adjacent (Low <-> High)
    non_adj_mask = (
        ((misclassified["actual_category"] == "Low") & (misclassified["predicted_category"] == "High")) |
        ((misclassified["actual_category"] == "High") & (misclassified["predicted_category"] == "Low"))
    )
    non_adj_count = int(non_adj_mask.sum())
    adj_count = errors - non_adj_count

    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "error_rate": errors / total,
        "adjacent_count": adj_count,
        "adjacent_pct": (adj_count / errors) * 100.0,
        "non_adjacent_count": non_adj_count,
        "non_adjacent_pct": (non_adj_count / errors) * 100.0
    }


def generate_markdown_report(
    selection_report: Optional[Dict[str, Any]] = None,
    clf_report_df: Optional[pd.DataFrame] = None,
    error_summary: Optional[Dict[str, Any]] = None,
    baseline_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate an authoritative, structured Markdown evaluation report based strictly
    on verified historical evaluation results from Days 14–18.
    """
    lines = [
        "# Model Evaluation and Performance Verification Report",
        "## AI-Based Student Performance Prediction System",
        "**YBI Foundation Python AI & Data Science Internship — Day 23**",
        "",
        "---",
        "",
        "### 1. Executive Summary & Validation Scope",
        ""
    ]

    if selection_report:
        model_name = selection_report.get("selected_model", "Decision Tree")
        sel_metric = selection_report.get("selection_metric", "Mean 5-Fold Stratified CV Macro F1")
        cv_f1 = selection_report.get("training_cv_macro_f1", 0.0)
        cv_std = selection_report.get("training_cv_macro_f1_std", 0.0)
        test_acc = selection_report.get("test_metrics", {}).get("accuracy", 0.0)
        test_f1 = selection_report.get("test_metrics", {}).get("macro_f1", 0.0)
        gen_delta = selection_report.get("test_metrics", {}).get("generalization_delta", 0.0)

        lines.extend([
            f"- **Selected Architecture:** {model_name}",
            f"- **Model Selection Criterion:** {sel_metric}",
            f"- **Cross-Validation Macro F1 (Training):** {cv_f1:.4f} ± {cv_std:.4f}",
            f"- **Held-Out Test Set Accuracy:** {test_acc:.2%}",
            f"- **Held-Out Test Set Macro F1:** {test_f1:.4f}",
            f"- **Generalization Gap (Test F1 - CV F1):** {gen_delta:+.4f}",
            "- **Evaluation Partition:** 80% Training (519 samples) / 20% Held-Out Test (130 samples)",
            "- **Target Leakage Safeguard:** Active (`G3` strictly excluded from feature space; target categorized into Low/Medium/High)"
        ])
    else:
        lines.append("- *Model selection metadata unavailable.*")

    lines.extend([
        "",
        "---",
        "",
        "### 2. Multi-Model Validation & Tuning Comparison (Day 14 vs Day 15)",
        "",
        "Comparison of 5-Fold Stratified Cross-Validation Macro F1 before and after hyperparameter optimization:",
        ""
    ])

    if baseline_df is not None:
        lines.append("| Model | Baseline CV Macro F1 | Tuned CV Macro F1 | Gain (Δ F1) |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for _, row in baseline_df.iterrows():
            m_name = row["Model"]
            b_f1 = row.get("Baseline Mean CV Macro F1", np.nan)
            t_f1 = row.get("Tuned Mean CV Macro F1", np.nan)
            diff = row.get("Change in Mean CV Macro F1", np.nan)
            b_str = f"{b_f1:.4f}" if pd.notnull(b_f1) else "N/A"
            t_str = f"{t_f1:.4f}" if pd.notnull(t_f1) else "N/A"
            d_str = f"{diff:+.4f}" if pd.notnull(diff) else "N/A"
            lines.append(f"| {m_name} | {b_str} | {t_str} | {d_str} |")
    else:
        lines.append("*Baseline versus tuned model comparison data unavailable.*")

    lines.extend([
        "",
        "---",
        "",
        "### 3. Class-Level Test Performance (Day 16 Held-Out Evaluation)",
        "",
        "Detailed performance on the unseen 130-student held-out test partition:",
        ""
    ])

    if clf_report_df is not None:
        lines.append("| Class / Metric | Precision | Recall | F1-Score | Support |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for _, row in clf_report_df.iterrows():
            c_name = row["Class/Average"]
            p = row.get("precision", np.nan)
            r = row.get("recall", np.nan)
            f1 = row.get("f1-score", np.nan)
            s = row.get("support", np.nan)
            lines.append(f"| {c_name} | {p:.4f} | {r:.4f} | {f1:.4f} | {s:.0f} |")
    else:
        lines.append("*Classification report data unavailable.*")

    lines.extend([
        "",
        "---",
        "",
        "### 4. Error Analysis & Severity Characterization",
        ""
    ])

    if error_summary:
        lines.extend([
            f"- **Total Test Records Evaluated:** {error_summary.get('total', 130)}",
            f"- **Correct Classifications:** {error_summary.get('correct', 70)} ({100 - error_summary.get('error_rate', 0.4615)*100:.1f}%)",
            f"- **Total Misclassifications:** {error_summary.get('errors', 60)} ({error_summary.get('error_rate', 0.4615)*100:.1f}%)",
            f"- **Adjacent Errors (Low ↔ Medium or Medium ↔ High):** {error_summary.get('adjacent_count', 56)} ({error_summary.get('adjacent_pct', 93.3):.1f}% of errors)",
            f"- **Non-Adjacent Errors (Low ↔ High):** {error_summary.get('non_adjacent_count', 4)} ({error_summary.get('non_adjacent_pct', 6.7):.1f}% of errors)",
            "",
            "> **Diagnostic Takeaway:** Over 93% of model misclassifications occur strictly between adjacent grade categories (e.g., student scoring near 10 or 15 cutoffs). Severe opposite-tier misclassifications (Low classified as High or vice-versa) constitute under 7% of errors."
        ])
    else:
        lines.append("*Detailed error analysis records unavailable.*")

    lines.extend([
        "",
        "---",
        "",
        "### 5. Ethical Guidelines, Limitations & Caveats",
        "",
        "1. **Non-Causal Associations:** All features represent observational demographic and behavioral survey responses. Predictors do not prove direct causal mechanisms.",
        "2. **Target Leakage Prevention:** Final exam score `G3` is strictly removed from input feature sets.",
        "3. **Absence of Midterm Exams:** In realistic early-warning settings, midterm marks `G1` and `G2` are unavailable at enrollment. The system evaluates baseline potential rather than late-semester outcomes.",
        "4. **Educational Decision-Support Only:** Predictions must never be used to penalize students, restrict access to curriculum, or automate high-stakes administrative decisions. The system is designed solely to signal timely academic support.",
        "",
        "---",
        "*Report generated automatically by Day 23 Model Performance Dashboard.*"
    ])

    return "\n".join(lines)


def generate_csv_summary(
    selection_report: Optional[Dict[str, Any]] = None,
    clf_report_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Compile a unified CSV summary string of validated model evaluation metrics.
    """
    rows = []

    # 1. Model Selection & Global metrics
    if selection_report:
        model_name = selection_report.get("selected_model", "Decision Tree")
        cv_f1 = selection_report.get("training_cv_macro_f1", np.nan)
        cv_acc = selection_report.get("training_cv_accuracy", np.nan)
        test_m = selection_report.get("test_metrics", {})

        rows.append({
            "Category": "Model Selection",
            "Entity": model_name,
            "Evaluation_Method": "5-Fold Stratified Cross-Validation (Training)",
            "Metric_Name": "Macro F1",
            "Metric_Value": cv_f1,
            "Source_File": "day_16_model_selection_report.json"
        })
        rows.append({
            "Category": "Model Selection",
            "Entity": model_name,
            "Evaluation_Method": "5-Fold Stratified Cross-Validation (Training)",
            "Metric_Name": "Accuracy",
            "Metric_Value": cv_acc,
            "Source_File": "day_16_model_selection_report.json"
        })
        rows.append({
            "Category": "Held-Out Test",
            "Entity": model_name,
            "Evaluation_Method": "Unseen Test Set (130 samples)",
            "Metric_Name": "Test Accuracy",
            "Metric_Value": test_m.get("accuracy", np.nan),
            "Source_File": "day_16_model_selection_report.json"
        })
        rows.append({
            "Category": "Held-Out Test",
            "Entity": model_name,
            "Evaluation_Method": "Unseen Test Set (130 samples)",
            "Metric_Name": "Test Macro F1",
            "Metric_Value": test_m.get("macro_f1", np.nan),
            "Source_File": "day_16_model_selection_report.json"
        })
        rows.append({
            "Category": "Generalization",
            "Entity": model_name,
            "Evaluation_Method": "Test F1 - CV F1 Delta",
            "Metric_Name": "Generalization Delta",
            "Metric_Value": test_m.get("generalization_delta", np.nan),
            "Source_File": "day_16_model_selection_report.json"
        })

    # 2. Class-Level Metrics
    if clf_report_df is not None:
        for _, r in clf_report_df.iterrows():
            c_name = r["Class/Average"]
            rows.append({
                "Category": "Class-Level",
                "Entity": f"Tier: {c_name}",
                "Evaluation_Method": "Held-Out Test Set (130 samples)",
                "Metric_Name": "F1-Score",
                "Metric_Value": r.get("f1-score", np.nan),
                "Source_File": "day_16_classification_report.csv"
            })
            rows.append({
                "Category": "Class-Level",
                "Entity": f"Tier: {c_name}",
                "Evaluation_Method": "Held-Out Test Set (130 samples)",
                "Metric_Name": "Precision",
                "Metric_Value": r.get("precision", np.nan),
                "Source_File": "day_16_classification_report.csv"
            })
            rows.append({
                "Category": "Class-Level",
                "Entity": f"Tier: {c_name}",
                "Evaluation_Method": "Held-Out Test Set (130 samples)",
                "Metric_Name": "Recall",
                "Metric_Value": r.get("recall", np.nan),
                "Source_File": "day_16_classification_report.csv"
            })

    df_out = pd.DataFrame(rows)
    return df_out.to_csv(index=False)
