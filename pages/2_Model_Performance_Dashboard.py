"""
Model Performance Dashboard
Day 23: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Interactive model evaluation dashboard for inspecting cross-validation results,
hyperparameter tuning gains, test set metrics, class-level reports, and error analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_dashboard import (
    find_evaluation_files,
    load_cv_results,
    load_baseline_vs_tuned,
    load_tuning_results,
    load_classification_report,
    load_error_analysis,
    load_model_selection_report,
    compute_confusion_matrix,
    compute_error_breakdown,
    generate_markdown_report,
    generate_csv_summary
)


def render_dashboard():
    # Page configuration
    st.set_page_config(
        page_title="Model Performance Dashboard | Student AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
    <style>
        .dash-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .dash-subtitle {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-val {
            font-size: 1.7rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.2rem;
        }
        .source-tag {
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #E0E7FF;
            color: #3730A3;
            font-weight: 600;
        }
        .source-tag-test {
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #FEF3C7;
            color: #92400E;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="dash-title">📊 Model Performance & Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-subtitle">'
        'Comprehensive empirical evaluation, model selection analytics, error characterization, '
        'and exportable audit reports (Days 14–18).'
        '</div>',
        unsafe_allow_html=True
    )

    # Discover real evaluation files
    available_files = find_evaluation_files(PROJECT_ROOT)

    # Load data safely
    selection_report, err_sel = load_model_selection_report()
    cv_df, err_cv = load_cv_results()
    baseline_df, err_base = load_baseline_vs_tuned()
    tuning_df, err_tune = load_tuning_results()
    clf_report_df, err_clf = load_classification_report()
    error_df, err_err = load_error_analysis()

    # Sidebar: Evaluation context and file status
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=65)
        st.title("Evaluation Context")
        st.markdown("**AI-Based Student Performance Prediction System**")
        st.caption("YBI Foundation Python AI & DS Internship — Day 23")

        st.markdown("---")
        st.subheader("Navigation")
        try:
            st.page_link("app.py", label="Prediction & Monitoring App", icon="🎓")
        except Exception:
            st.caption("Navigate to app.py via the multipage menu.")

        st.markdown("---")
        st.subheader("Data Integrity Safeguards")
        st.markdown("""
        - **Target:** 3 Tiers (`Low`, `Medium`, `High`)
        - **Leakage Prevention:** `G3` strictly excluded from features
        - **Model Selection:** Stratified 5-Fold CV on training split (519 records)
        - **Unbiased Assessment:** 130 unseen held-out test records
        """)

        st.markdown("---")
        st.subheader("Evaluation Files Status")
        for key, path in available_files.items():
            if path:
                st.caption(f"✅ `{path.name}`")
            else:
                st.caption(f"⚠️ `{key}` (Not found)")

        st.markdown("---")
        st.caption("Strict separation between cross-validation and test set evaluation.")

    # -------------------------------------------------------------------------
    # SECTION A: OVERVIEW & VALIDATION SCOPE
    # -------------------------------------------------------------------------
    st.markdown("### 1. Evaluation Overview & Key Metrics")
    st.markdown(
        "Empirical performance summary comparing internal training cross-validation with held-out generalization testing. "
        "Each metric clearly states its evaluation partition to maintain strict data integrity."
    )

    if selection_report:
        model_name = selection_report.get("selected_model", "Decision Tree")
        cv_f1 = selection_report.get("training_cv_macro_f1", np.nan)
        cv_std = selection_report.get("training_cv_macro_f1_std", np.nan)
        cv_acc = selection_report.get("training_cv_accuracy", np.nan)
        test_acc = selection_report.get("test_metrics", {}).get("accuracy", np.nan)
        test_f1 = selection_report.get("test_metrics", {}).get("macro_f1", np.nan)
        gen_delta = selection_report.get("test_metrics", {}).get("generalization_delta", np.nan)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{model_name}</div>
                <div class="metric-lbl">Selected Architecture</div>
                <div style="margin-top:6px;"><span class="source-tag">Day 16 Report</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{cv_f1:.4f}</div>
                <div class="metric-lbl">5-Fold CV Macro F1 (±{cv_std:.4f})</div>
                <div style="margin-top:6px;"><span class="source-tag">Training Split (519)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{cv_acc:.2%}</div>
                <div class="metric-lbl">5-Fold CV Accuracy</div>
                <div style="margin-top:6px;"><span class="source-tag">Training Split (519)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{test_f1:.4f}</div>
                <div class="metric-lbl">Held-Out Test Macro F1</div>
                <div style="margin-top:6px;"><span class="source-tag-test">Test Set (130)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            delta_color = "#DC2626" if gen_delta < -0.05 else "#2563EB"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{delta_color};">{gen_delta:+.4f}</div>
                <div class="metric-lbl">Generalization Gap (Test - CV)</div>
                <div style="margin-top:6px;"><span class="source-tag-test">Generalization Delta</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model selection report (`day_16_model_selection_report.json`) was not found or could not be parsed.")

    st.info(
        "ℹ️ **Evaluation Method Transparency:** Cross-Validation metrics were computed across 5 stratified folds on the training dataset (519 students). "
        "Held-Out Test metrics were measured once on the isolated 130-student test split. "
        "Model selection was finalized strictly on cross-validation performance to preserve test set purity."
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION B: MODEL COMPARISON (BASELINE VS TUNED)
    # -------------------------------------------------------------------------
    st.markdown("### 2. Multi-Model Comparison (Baseline vs. Tuned)")
    st.markdown(
        "Comparative evaluation across all candidate architectures evaluated during Day 14 and Day 15. "
        "Macro F1 is the primary selection metric due to class distribution imbalance."
    )

    if baseline_df is not None:
        metric_choice = st.selectbox(
            "Select Evaluation Metric to Compare:",
            options=["Macro F1 (Primary)", "Cross-Validation Accuracy", "Tuning Parameter Comparison"],
            index=0
        )

        if metric_choice == "Macro F1 (Primary)":
            col_tbl, col_chart = st.columns([1, 1])
            with col_tbl:
                st.markdown("##### 5-Fold Stratified CV Macro F1 Gains")
                display_df = baseline_df.copy()
                display_df["Baseline CV F1"] = display_df["Baseline Mean CV Macro F1"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "Missing")
                display_df["Tuned CV F1"] = display_df["Tuned Mean CV Macro F1"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "Missing")
                display_df["Δ F1 Gain"] = display_df["Change in Mean CV Macro F1"].apply(lambda x: f"{x:+.4f}" if pd.notnull(x) else "Missing")

                st.dataframe(
                    display_df[["Model", "Baseline CV F1", "Tuned CV F1", "Δ F1 Gain"]],
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("Data source: `day_15_baseline_vs_tuned.csv`. Standard deviations tracked in validation logs.")

            with col_chart:
                st.markdown("##### Macro F1: Baseline vs Tuned")
                fig, ax = plt.subplots(figsize=(6, 3.8))

                models = baseline_df["Model"].tolist()
                x = np.arange(len(models))
                width = 0.35

                b_vals = baseline_df["Baseline Mean CV Macro F1"].fillna(0)
                t_vals = baseline_df["Tuned Mean CV Macro F1"].fillna(0)
                b_err = baseline_df["Baseline Macro F1 Standard Deviation"].fillna(0)
                t_err = baseline_df["Tuned Macro F1 Standard Deviation"].fillna(0)

                rects1 = ax.bar(x - width/2, b_vals, width, yerr=b_err, label="Baseline CV F1", color="#93C5FD", capsize=3)
                rects2 = ax.bar(x + width/2, t_vals, width, yerr=t_err, label="Tuned CV F1", color="#1D4ED8", capsize=3)

                ax.set_ylabel("Mean CV Macro F1", fontsize=9)
                ax.set_title("5-Fold CV Macro F1 Comparison (with ±1 Std Dev)", fontsize=10, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif metric_choice == "Cross-Validation Accuracy":
            if cv_df is not None:
                st.markdown("##### Day 14 Baseline Cross-Validation Metrics")
                st.dataframe(
                    cv_df.style.format({
                        "Mean CV Accuracy": "{:.4f}",
                        "Standard Deviation CV Accuracy": "{:.4f}",
                        "Mean CV Macro Precision": "{:.4f}",
                        "Mean CV Macro Recall": "{:.4f}",
                        "Mean CV Macro F1": "{:.4f}",
                        "Standard Deviation CV Macro F1": "{:.4f}",
                        "Mean Training Macro F1": "{:.4f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("⚠️ Day 14 cross-validation data unavailable.")

        else: # Tuning Parameter Comparison
            if tuning_df is not None:
                st.markdown("##### Day 15 Hyperparameter Optimization Search Space & Optimal Configurations")
                tune_display = tuning_df[["Model", "Best Parameters", "Tuned Mean CV Macro F1", "Combinations Evaluated"]].copy()
                st.dataframe(tune_display, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Day 15 tuning results unavailable.")

    else:
        st.warning("⚠️ Baseline vs. Tuned evaluation results (`day_15_baseline_vs_tuned.csv`) not found.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION C: CLASS-LEVEL PERFORMANCE
    # -------------------------------------------------------------------------
    st.markdown("### 3. Class-Level Performance (Held-Out Test Set)")
    st.markdown(
        "Evaluation across the three academic performance tiers (`Low`, `Medium`, `High`) on the 130-student held-out test split. "
        "Class-level metrics are separated from macro and weighted averages."
    )

    if clf_report_df is not None:
        col_c_tbl, col_c_chart = st.columns([1, 1])

        with col_c_tbl:
            st.markdown("##### Genuine Test Classification Report")
            # Separate class-level rows from summary rows
            class_rows = clf_report_df[clf_report_df["Class/Average"].isin(["Low", "Medium", "High"])]
            avg_rows = clf_report_df[~clf_report_df["Class/Average"].isin(["Low", "Medium", "High"])]

            st.markdown("**Per-Class Breakdown:**")
            st.dataframe(
                class_rows.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Aggregate Averages:**")
            st.dataframe(
                avg_rows.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            st.caption("Source: `day_16_classification_report.csv`. Support reflects test split sample size.")

        with col_c_chart:
            st.markdown("##### Precision, Recall, and F1 by Performance Tier")
            fig, ax = plt.subplots(figsize=(6, 3.8))

            tiers = class_rows["Class/Average"].tolist()
            x = np.arange(len(tiers))
            width = 0.25

            p_vals = class_rows["precision"].tolist()
            r_vals = class_rows["recall"].tolist()
            f1_vals = class_rows["f1-score"].tolist()

            ax.bar(x - width, p_vals, width, label="Precision", color="#3B82F6")
            ax.bar(x, r_vals, width, label="Recall", color="#10B981")
            ax.bar(x + width, f1_vals, width, label="F1-Score", color="#8B5CF6")

            ax.set_ylabel("Score", fontsize=9)
            ax.set_title("Test Metrics Across Performance Tiers", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(tiers, fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "📌 **Class Imbalance Note:** The `Medium` tier represents ~64.6% (84/130) of the test dataset, "
            "whereas `Low` (20/130) and `High` (26/130) are minority classes. The model achieved a balanced recall of 45.0% on Low "
            "and 63.1% on Medium, successfully flagging students needing academic intervention."
        )
    else:
        st.warning("⚠️ Classification report (`day_16_classification_report.csv`) not found.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION D: ERROR ANALYSIS & MISCLASSIFICATIONS
    # -------------------------------------------------------------------------
    st.markdown("### 4. Error Analysis & Misclassification Inspection")
    st.markdown(
        "Inspection of actual test predictions, confusion matrix counts, and misclassification severity. "
        "All calculations derive strictly from the 130 individual test records."
    )

    if error_df is not None:
        eb = compute_error_breakdown(error_df)
        cm = compute_confusion_matrix(error_df)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Test Samples", f"{eb['total']}")
        with col_m2:
            st.metric("Correct Predictions", f"{eb['correct']} ({100-eb['error_rate']*100:.1f}%)")
        with col_m3:
            st.metric("Adjacent Errors (Low ↔ Med / Med ↔ High)", f"{eb['adjacent_count']} ({eb['adjacent_pct']:.1f}%)")
        with col_m4:
            st.metric("Non-Adjacent Errors (Low ↔ High)", f"{eb['non_adjacent_count']} ({eb['non_adjacent_pct']:.1f}%)")

        col_cm, col_dist = st.columns([1, 1])
        with col_cm:
            st.markdown("##### Empirical Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 11, "weight": "bold"},
                ax=ax
            )
            ax.set_xlabel("Predicted Academic Tier", fontsize=9, fontweight="bold")
            ax.set_ylabel("Actual Academic Tier", fontsize=9, fontweight="bold")
            ax.set_title("Held-Out Test Set Confusion Matrix (N=130)", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_dist:
            st.markdown("##### Error Severity Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.8))
            labels = ["Adjacent Errors (Low↔Med, Med↔High)", "Non-Adjacent Errors (Low↔High)"]
            sizes = [eb["adjacent_count"], eb["non_adjacent_count"]]
            colors = ["#60A5FA", "#F87171"]
            explode = (0, 0.1)

            ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
            ax.set_title("Misclassification Breakdown by Proximity", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("##### Interactive Test Prediction Browser")
        filter_opt = st.radio(
            "Filter Test Records:",
            options=["All Test Records (130)", "Misclassified Records Only (60)", "Severe Errors Only (Low ↔ High) (4)"],
            horizontal=True
        )

        if filter_opt == "Misclassified Records Only (60)":
            view_df = error_df[error_df["is_correct"] == False]
        elif filter_opt == "Severe Errors Only (Low ↔ High) (4)":
            view_df = error_df[
                ((error_df["actual_category"] == "Low") & (error_df["predicted_category"] == "High")) |
                ((error_df["actual_category"] == "High") & (error_df["predicted_category"] == "Low"))
            ]
        else:
            view_df = error_df

        st.dataframe(
            view_df[["actual_category", "predicted_category", "is_correct", "studytime", "failures", "absences", "schoolsup", "famsup", "goout", "health"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Showing {len(view_df)} records from `day_16_error_analysis.csv`.")
    else:
        st.warning("⚠️ Error analysis data (`day_16_error_analysis.csv`) not found.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION E: DOWNLOADABLE EVALUATION REPORT
    # -------------------------------------------------------------------------
    st.markdown("### 5. Downloadable Evaluation Reports")
    st.markdown(
        "Export the verified evaluation results in standard Markdown or CSV formats for archiving, auditing, or sharing with educational stakeholders."
    )

    err_summary = compute_error_breakdown(error_df) if error_df is not None else None
    md_report = generate_markdown_report(
        selection_report=selection_report,
        clf_report_df=clf_report_df,
        error_summary=err_summary,
        baseline_df=baseline_df
    )
    csv_report = generate_csv_summary(
        selection_report=selection_report,
        clf_report_df=clf_report_df
    )

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            label="📄 Download Markdown Evaluation Report",
            data=md_report,
            file_name="model_evaluation_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="📊 Download Evaluation Summary CSV",
            data=csv_report,
            file_name="model_evaluation_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_d3:
        if error_df is not None:
            csv_err = error_df.to_csv(index=False)
            st.download_button(
                label="🔍 Download Test Error Predictions CSV",
                data=csv_err,
                file_name="test_error_analysis_records.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION F: METHODOLOGICAL INTEGRITY & ETHICAL GOVERNANCE
    # -------------------------------------------------------------------------
    st.markdown("### 6. Methodological Integrity & Ethical Governance")
    st.markdown("""
    - **No Target Leakage:** The final evaluation grade `G3` is strictly removed from input features. Models predict performance solely based on demographic, lifestyle, and early behavioral indicators.
    - **Correlation vs. Causation:** All identified relationships in the UCI student dataset are empirical statistical associations. Factors such as alcohol consumption, failures, or study hours should not be assumed to have direct, isolated causal effects.
    - **Decision-Support Purpose:** This machine learning system is designed solely as an early-warning signal to help educators identify students who may benefit from academic tutoring or counseling. It must **never** be used for punitive tracking, automated grading, or high-stakes administrative decisions.
    """)


if __name__ == "__main__":
    render_dashboard()
