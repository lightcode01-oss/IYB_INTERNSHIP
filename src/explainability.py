"""
Model Explainability and Feature Importance Module
Day 24: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Provides:
1. Permutation feature importance on the pipeline using raw input columns.
2. Native feature importance (MDI / Gini importance) from the fitted Decision Tree.
3. Target leakage verification (strict exclusion of G3 and performance_category).
4. Individual prediction explanation via decision rule paths and cohort comparisons.
5. Visualization and report generation utilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logger
logger = logging.getLogger("model_explainability")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "artifacts" / "student_performance_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "student_performance_metadata.json"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "day_16_error_analysis.csv"

# Canonical feature list (30 input features)
CANONICAL_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

FORBIDDEN_LEAKAGE_COLUMNS = ["G3", "performance_category"]


def load_explainability_pipeline(model_path: Optional[Path] = None) -> Any:
    """
    Load and validate the pre-trained end-to-end student performance pipeline.
    """
    path = Path(model_path) if model_path else DEFAULT_PIPELINE_PATH
    if not path.exists():
        # Fallback candidates
        candidates = [
            Path.cwd() / "artifacts" / "student_performance_pipeline.joblib",
            PROJECT_ROOT / "artifacts" / "student_performance_pipeline.joblib",
            Path.cwd() / "student_performance_pipeline.joblib"
        ]
        for cand in candidates:
            if cand.exists():
                path = cand
                break

    if not path.exists():
        raise FileNotFoundError(f"Model pipeline artifact not found at {path}")

    pipeline = joblib.load(path)
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("Loaded object is not a valid scikit-learn Pipeline with 'preprocessor' and 'classifier' steps.")

    return pipeline


def load_reference_dataset(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the authentic reference test dataset (Day 16 held-out test split of 130 instances).
    Enforces strict target leakage prevention: G3 and performance_category are separated from X.

    Returns:
        X (pd.DataFrame): 30 input features.
        y (pd.Series): Ground-truth target categories ('Low', 'Medium', 'High').
    """
    path = Path(data_path) if data_path else DEFAULT_TEST_DATA_PATH
    if not path.exists():
        candidates = [
            PROJECT_ROOT / "day_16_error_analysis.csv",
            Path.cwd() / "day_16_error_analysis.csv"
        ]
        for cand in candidates:
            if cand.exists():
                path = cand
                break

    if not path.exists():
        raise FileNotFoundError(f"Reference evaluation dataset not found at {path}")

    df = pd.read_csv(path)

    # Identify target column
    if "actual_category" in df.columns:
        y = df["actual_category"].copy()
    elif "performance_category" in df.columns:
        y = df["performance_category"].copy()
    elif "G3" in df.columns:
        # Derive target if only G3 is present
        y = df["G3"].apply(lambda s: "Low" if s < 10 else ("Medium" if s < 15 else "High"))
    else:
        raise ValueError("Could not find a valid target column ('actual_category' or 'performance_category') in dataset.")

    # Extract 30 input features
    features_present = [c for c in CANONICAL_FEATURES if c in df.columns]
    if len(features_present) != len(CANONICAL_FEATURES):
        missing = set(CANONICAL_FEATURES) - set(features_present)
        raise ValueError(f"Reference dataset is missing required input features: {missing}")

    X = df[CANONICAL_FEATURES].copy()

    # Strict leakage validation
    for col in FORBIDDEN_LEAKAGE_COLUMNS:
        if col in X.columns:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Forbidden column '{col}' is present in feature matrix X!")

    return X, y


def validate_feature_schema(X: pd.DataFrame) -> bool:
    """
    Validate that feature matrix X conforms to the project's input schema and contains no leakage.
    """
    missing = [f for f in CANONICAL_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required features: {missing}")

    leakage = [col for col in FORBIDDEN_LEAKAGE_COLUMNS if col in X.columns]
    if leakage:
        raise ValueError(f"Target leakage violation: Forbidden columns present in feature inputs: {leakage}")

    return True


def compute_permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute permutation feature importance on the fitted pipeline using raw feature columns.
    Measures the decrease in Macro F1 score when a feature's values are randomly shuffled.

    Args:
        pipeline: Fitted scikit-learn Pipeline with preprocessor and classifier.
        X: Unprocessed input DataFrame (30 raw features).
        y: True categorical labels ('Low', 'Medium', 'High').
        scoring: Metric to evaluate (default 'f1_macro').
        n_repeats: Number of permutation iterations per feature.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs (default None for robust cross-platform execution).

    Returns:
        pd.DataFrame: Columns ['feature', 'importance_mean', 'importance_std', 'rank'].
    """
    validate_feature_schema(X)

    logger.info(f"Computing permutation importance with {n_repeats} repeats (scoring='{scoring}', seed={random_state})...")
    res = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    df_importance = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    df_importance["rank"] = df_importance.index + 1
    return df_importance


def compute_native_feature_importance(pipeline: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract native tree feature importance (MDI / Gini importance) from the fitted Decision Tree.

    Returns:
        (df_transformed, df_aggregated):
            df_transformed: Importance per one-hot/passthrough column (56 features).
            df_aggregated: Importance summed back to original 30 input columns.
    """
    clf = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(clf, "feature_importances_"):
        raise AttributeError("Classifier does not support native feature_importances_.")

    importances = clf.feature_importances_
    transformed_names = list(preprocessor.get_feature_names_out())

    df_transformed = pd.DataFrame({
        "transformed_feature": transformed_names,
        "native_importance": importances
    }).sort_values(by="native_importance", ascending=False).reset_index(drop=True)

    # Aggregate back to original 30 input feature names
    agg_scores = {f: 0.0 for f in CANONICAL_FEATURES}
    for feat_name, imp in zip(transformed_names, importances):
        # Transformed names follow format: 'num_ord__<name>' or 'cat__<name>_<value>'
        clean_name = feat_name.replace("num_ord__", "").replace("cat__", "")
        # Match to original feature prefix
        for orig in CANONICAL_FEATURES:
            if clean_name == orig or clean_name.startswith(orig + "_"):
                agg_scores[orig] += imp
                break

    df_aggregated = pd.DataFrame([
        {"feature": k, "native_importance": v} for k, v in agg_scores.items()
    ]).sort_values(by="native_importance", ascending=False).reset_index(drop=True)
    df_aggregated["rank"] = df_aggregated.index + 1

    return df_transformed, df_aggregated


def explain_individual_prediction(
    pipeline: Any,
    student_record: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    reference_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Provide a transparent, faithful explanation of an individual student prediction
    using the Decision Tree's traversed decision rules and cohort comparisons.

    Args:
        pipeline: Fitted pipeline.
        student_record: Single student feature vector (dict, Series, or 1-row DataFrame).
        reference_df: Optional reference DataFrame (e.g. test set) for cohort percentile context.

    Returns:
        dict: Predicted class, probabilities, traversed decision rules, and key feature comparison.
    """
    if isinstance(student_record, dict):
        rec_df = pd.DataFrame([student_record])
    elif isinstance(student_record, pd.Series):
        rec_df = student_record.to_frame().T
    else:
        rec_df = student_record.copy()

    validate_feature_schema(rec_df)

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    # Predict class & probabilities
    pred_class = pipeline.predict(rec_df)[0]
    pred_proba = pipeline.predict_proba(rec_df)[0]
    class_probs = {cls_name: float(p) for cls_name, p in zip(clf.classes_, pred_proba)}

    # Trace tree decision path
    X_trans = preprocessor.transform(rec_df)
    node_indicator = clf.decision_path(X_trans)
    leave_id = clf.apply(X_trans)[0]
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rules = []
    for node_id in node_index:
        if leave_id == node_id:
            continue
        feat_idx = clf.tree_.feature[node_id]
        thresh = clf.tree_.threshold[node_id]
        raw_feat_name = transformed_feature_names[feat_idx]
        val = X_trans[0, feat_idx]
        op = "<=" if val <= thresh else ">"

        # Human-readable rule cleanup
        readable_feat = raw_feat_name.replace("num_ord__", "").replace("cat__", "")
        if "=" in readable_feat or "_" in readable_feat:
            readable_rule = f"Feature '{readable_feat}' was {val:.1f} ({op} {thresh:.2f})"
        else:
            readable_rule = f"{readable_feat} = {val:.1f} ({op} {thresh:.2f})"

        rules.append({
            "node_id": int(node_id),
            "feature": readable_feat,
            "student_value": float(val),
            "threshold": float(thresh),
            "operator": op,
            "rule_description": readable_rule
        })

    # Cohort comparison on top actionable attributes
    cohort_stats = {}
    key_features = ["failures", "absences", "studytime", "Medu", "Fedu", "goout", "health", "Dalc", "Walc"]
    for kf in key_features:
        stu_val = rec_df[kf].iloc[0]
        if hasattr(stu_val, "item"):
            stu_val = stu_val.item()
        stat_item = {"student_value": stu_val}
        if reference_df is not None and kf in reference_df.columns:
            ref_col = reference_df[kf]
            stat_item["cohort_mean"] = round(float(ref_col.mean()), 2)
            stat_item["cohort_median"] = float(ref_col.median())
        cohort_stats[kf] = stat_item

    return {
        "predicted_category": str(pred_class),
        "predicted_probabilities": class_probs,
        "traversed_depth": len(rules),
        "decision_rules": rules,
        "cohort_comparison": cohort_stats,
        "methodological_notice": (
            "This explanation traces the exact sequence of decision boundaries evaluated by the "
            "champion Decision Tree for this specific student. Observed rules indicate classification logic, "
            "not deterministic causes of academic performance."
        )
    }


def plot_permutation_importance(
    perm_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Features by Permutation Importance (Macro F1 Decrease)",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate a publication-grade horizontal bar chart for top permutation importance features.
    """
    top_df = perm_df.head(top_n).iloc[::-1]  # Invert so highest is on top

    fig, ax = plt.subplots(figsize=(8, max(4.5, top_n * 0.45)))
    y_pos = np.arange(len(top_df))

    bars = ax.barh(
        y_pos,
        top_df["importance_mean"],
        xerr=top_df["importance_std"],
        align="center",
        color="#2563EB",
        ecolor="#1E293B",
        capsize=4,
        alpha=0.88
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"], fontsize=10)
    ax.set_xlabel("Mean Macro F1 Decrease (±1 Std Dev)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.axvline(x=0, color="#64748B", linestyle="--", linewidth=1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        x_text = w + 0.003 if w >= 0 else w - 0.003
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{w:+.4f}",
                va="center", ha=ha, fontsize=8, color="#0F172A", fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_native_importance(
    native_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Features by Decision Tree Native Importance (MDI / Gini)",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate horizontal bar chart for native Decision Tree Gini importance.
    """
    top_df = native_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4.5, top_n * 0.45)))
    y_pos = np.arange(len(top_df))

    bars = ax.barh(
        y_pos,
        top_df["native_importance"],
        align="center",
        color="#0D9488",
        alpha=0.88
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"], fontsize=10)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                va="center", ha="left", fontsize=8, color="#0F172A", fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def save_day24_outputs(
    output_dir: Optional[Path] = None,
    n_repeats: int = 15,
    random_state: int = 42
) -> Dict[str, Path]:
    """
    Compute, visualize, and persist Day 24 explainability artifacts to reports/day_24/.
    """
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / "day_24"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_explainability_pipeline()
    X_test, y_test = load_reference_dataset()

    # 1. Permutation Importance
    perm_df = compute_permutation_importance(
        pipeline=pipeline,
        X=X_test,
        y=y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state
    )
    perm_csv = out_dir / "permutation_importance.csv"
    perm_df.to_csv(perm_csv, index=False)

    perm_png = out_dir / "permutation_importance.png"
    fig_perm = plot_permutation_importance(perm_df, top_n=10, save_path=perm_png)
    plt.close(fig_perm)

    # 2. Native Tree Importance
    df_trans, df_agg = compute_native_feature_importance(pipeline)
    native_csv = out_dir / "native_feature_importance.csv"
    df_agg.to_csv(native_csv, index=False)

    native_png = out_dir / "native_feature_importance.png"
    fig_nat = plot_native_importance(df_agg, top_n=10, save_path=native_png)
    plt.close(fig_nat)

    # 3. Individual Explanations for 3 representative test cases (Low, Medium, High)
    sample_explanations = []
    for tier in ["Low", "Medium", "High"]:
        tier_indices = y_test[y_test == tier].index
        if len(tier_indices) > 0:
            sample_rec = X_test.loc[tier_indices[0]]
            exp = explain_individual_prediction(pipeline, sample_rec, reference_df=X_test)
            exp["actual_category"] = tier
            exp["sample_index"] = int(tier_indices[0])
            sample_explanations.append(exp)

    ind_json = out_dir / "sample_individual_explanations.json"
    with open(ind_json, "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2)

    logger.info(f"Saved all Day 24 explainability reports to: {out_dir}")
    return {
        "permutation_csv": perm_csv,
        "permutation_png": perm_png,
        "native_csv": native_csv,
        "native_png": native_png,
        "individual_json": ind_json
    }


def load_precomputed_results(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load pre-calculated explainability tables and plots if they exist in reports/day_24/.
    Enables instant rendering on the Streamlit dashboard without expensive recomputation.

    Returns:
        dict: Containing 'has_precomputed' bool, 'permutation_df', 'native_df', 'individual_samples'.
    """
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / "day_24"
    res = {
        "has_precomputed": False,
        "permutation_df": None,
        "native_df": None,
        "sample_explanations": None,
        "permutation_png": None,
        "native_png": None
    }

    perm_csv = out_dir / "permutation_importance.csv"
    native_csv = out_dir / "native_feature_importance.csv"
    ind_json = out_dir / "sample_individual_explanations.json"
    perm_png = out_dir / "permutation_importance.png"
    native_png = out_dir / "native_feature_importance.png"

    if perm_csv.exists() and native_csv.exists():
        try:
            res["permutation_df"] = pd.read_csv(perm_csv)
            res["native_df"] = pd.read_csv(native_csv)
            res["has_precomputed"] = True
            if ind_json.exists():
                with open(ind_json, "r", encoding="utf-8") as f:
                    res["sample_explanations"] = json.load(f)
            if perm_png.exists():
                res["permutation_png"] = perm_png
            if native_png.exists():
                res["native_png"] = native_png
        except Exception as e:
            logger.warning(f"Error loading precomputed explainability files from {out_dir}: {e}")
            res["has_precomputed"] = False

    return res
