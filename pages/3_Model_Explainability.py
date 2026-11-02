"""
Model Explainability and Feature Importance Dashboard
Day 24: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Provides transparent, faithful explanations of model predictions:
1. Permutation Feature Importance (Macro F1 impact on held-out test data).
2. Native Decision Tree Feature Importance (MDI / Gini Impurity).
3. Local Prediction Explanations via Decision Tree Rules & Cohort Baselines.
4. Methodological Safeguards & Ethical Guidance (Causation vs Association).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import (
    load_explainability_pipeline,
    load_reference_dataset,
    load_precomputed_results,
    compute_permutation_importance,
    compute_native_feature_importance,
    explain_individual_prediction,
    plot_permutation_importance,
    plot_native_importance,
    CANONICAL_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS
)


@st.cache_data(show_spinner=False)
def get_explainability_data():
    """
    Cached data retrieval: Loads precomputed artifacts or falls back to
    on-demand computation with deterministic seed.
    """
    pre = load_precomputed_results()
    if pre["has_precomputed"]:
        pipeline = None
        X_test, y_test = None, None
        try:
            pipeline = load_explainability_pipeline()
            X_test, y_test = load_reference_dataset()
        except Exception:
            pass
        return pre["permutation_df"], pre["native_df"], pre["sample_explanations"], pipeline, X_test, y_test

    # Compute on demand if missing
    try:
        pipeline = load_explainability_pipeline()
        X_test, y_test = load_reference_dataset()
        perm_df = compute_permutation_importance(
            pipeline=pipeline,
            X=X_test,
            y=y_test,
            scoring="f1_macro",
            n_repeats=10,
            random_state=42,
            n_jobs=1
        )
        _, nat_df = compute_native_feature_importance(pipeline)

        # Sample explanations
        samples = []
        for tier in ["Low", "Medium", "High"]:
            indices = y_test[y_test == tier].index
            if len(indices) > 0:
                rec = X_test.loc[indices[0]]
                exp = explain_individual_prediction(pipeline, rec, reference_df=X_test)
                exp["actual_category"] = tier
                exp["sample_index"] = int(indices[0])
                samples.append(exp)
        return perm_df, nat_df, samples, pipeline, X_test, y_test
    except Exception as e:
        st.error(f"Failed to initialize explainability module: {e}")
        return None, None, None, None, None, None


def render_explainability_page():
    st.set_page_config(
        page_title="Model Explainability | Student AI",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styling
    st.markdown("""
    <style>
        .exp-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .exp-subtitle {
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
            font-size: 1.6rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.2rem;
        }
        .rule-box {
            background-color: #EFF6FF;
            border-left: 4px solid #3B82F6;
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            font-family: monospace;
            font-size: 0.95rem;
            color: #1E293B;
        }
        .notice-box {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 0.85rem 1.1rem;
            border-radius: 4px;
            margin: 1rem 0;
            color: #92400E;
            font-size: 0.92rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="exp-title">🔍 Model Explainability & Feature Importance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="exp-subtitle">Day 24: Model Transparency, Global Influence Rankings, and Local Decision Tree Paths</div>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Explainability Settings")
    top_n = st.sidebar.slider("Top N Features to Display", min_value=5, max_value=25, value=10, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Leakage Safeguards")
    st.sidebar.success("✅ **G3 Exam Mark**: Strictly Excluded\n\n✅ **Target**: Derived Post-Split (`Low`, `Medium`, `High`)\n\n✅ **Pipeline Encapsulation**: Active")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Evaluation Reference")
    st.sidebar.info("**Dataset:** Cortez & Silva (2008)\n\n**Held-Out Test:** $N=130$ instances\n\n**Selected Model:** Decision Tree (`max_depth=10`, `min_samples_split=10`)")

    # Load data
    perm_df, nat_df, sample_exps, pipeline, X_test, y_test = get_explainability_data()

    if perm_df is None or nat_df is None:
        st.error("Model explainability artifacts could not be loaded. Please ensure the pipeline and test datasets exist.")
        return

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    top_perm_feature = perm_df.iloc[0]["feature"]
    top_perm_score = perm_df.iloc[0]["importance_mean"]
    top_nat_feature = nat_df.iloc[0]["feature"]
    top_nat_score = nat_df.iloc[0]["native_importance"]

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{top_perm_feature}</div>
            <div class="metric-lbl">#1 Permutation Feature (Macro F1)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">+{top_perm_score:.4f}</div>
            <div class="metric-lbl">Mean Macro F1 Drop on Shuffling</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{top_nat_feature}</div>
            <div class="metric-lbl">#1 Native Tree Feature (Gini MDI)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{top_nat_score:.4f}</div>
            <div class="metric-lbl">Native Impurity Reduction Share</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Permutation Importance",
        "⚖️ Permutation vs Native MDI",
        "🎯 Local Prediction Explainer",
        "📖 Methodology & Limitations"
    ])

    with tab1:
        st.subheader("Global Permutation Feature Importance")
        st.write(
            "Permutation importance measures how much the model's test performance (**Macro F1**) decreases when a feature's "
            "values are randomly shuffled across the held-out test set ($N=130$). Because shuffling breaks the relationship between "
            "the feature and target, a large drop indicates the model relied heavily on that feature for generalization."
        )

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig_perm = plot_permutation_importance(
                perm_df,
                top_n=top_n,
                title=f"Top {top_n} Features by Permutation Importance (Macro F1 Impact)"
            )
            st.pyplot(fig_perm, use_container_width=True)
            plt.close(fig_perm)

        with col_table:
            st.markdown(f"**Top {top_n} Feature Metrics**")
            display_perm = perm_df.head(top_n).copy()
            display_perm["Mean F1 Drop"] = display_perm["importance_mean"].apply(lambda v: f"{v:+.4f}")
            display_perm["Std Dev"] = display_perm["importance_std"].apply(lambda v: f"±{v:.4f}")
            st.dataframe(
                display_perm[["rank", "feature", "Mean F1 Drop", "Std Dev"]].rename(columns={"rank": "Rank", "feature": "Feature"}),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="📥 Download Full Permutation CSV (30 Features)",
                data=perm_df.to_csv(index=False),
                file_name="day_24_permutation_importance.csv",
                mime="text/csv"
            )

        st.markdown("""
        > **Understanding Permutation Values:**
        > - **Positive ($> 0$):** Shuffling degraded model performance; the model relies on this feature.
        > - **Near Zero ($\approx 0$):** Shuffling had no meaningful impact; the feature contributes little unique information.
        > - **Negative ($< 0$):** Shuffling slightly improved the metric on the test sample; typical for noisy, low-importance, or collinear features in finite test sets.
        """)

    with tab2:
        st.subheader("Method Comparison: Permutation vs Native Tree Importance (MDI)")
        st.write(
            "Different explainability techniques evaluate different aspects of the model. "
            "Here we contrast **Permutation Importance** (model-agnostic, computed on unseen test data) "
            "with **Native Gini Impurity Reduction (MDI)** (tree-specific, computed over training splits)."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Permutation Importance (Test Set)")
            fig_p = plot_permutation_importance(perm_df, top_n=top_n, title=f"Top {top_n} Permutation Features")
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)

        with c2:
            st.markdown("#### Native Decision Tree MDI (Training Splits)")
            fig_n = plot_native_importance(nat_df, top_n=top_n, title=f"Top {top_n} Native Gini Features")
            st.pyplot(fig_n, use_container_width=True)
            plt.close(fig_n)

        # Comparative Insights
        st.markdown("### Key Discrepancies & Educational Interpretation")
        st.markdown("""
        1. **`failures` (Past Class Failures):**
           - Ranks **#1 in Permutation Importance** ($+0.0629$) and **#2 in Native MDI** ($0.0893$).
           - Strongly aligned across both methods: students with prior academic struggles are heavily weighted by the model.
        2. **`health` & `reason`:**
           - Rank high in Native MDI (#1 and #5) because the tree made several splits on them during training.
           - However, their Permutation Importance on held-out test data is near zero or slightly negative. This indicates that while the tree partitioned training samples on these features, they did not contribute positively to generalization on unseen test data.
        3. **`Mjob` (Mother's Job) & `freetime`:**
           - High permutation impact (#2 and #3) indicates socio-demographic and extracurricular balance have genuine predictive utility across unseen cohorts.
        """)

    with tab3:
        st.subheader("Local Prediction Explanations: Transparent Decision Paths")
        st.write(
            "For individual students, we trace the exact path evaluated through the Decision Tree "
            "and contrast actionable features against reference cohort baselines."
        )

        if sample_exps:
            selected_tier = st.selectbox(
                "Select a Representative Cohort Case to Inspect:",
                options=["Low Tier Student (Actual: Low)", "Medium Tier Student (Actual: Medium)", "High Tier Student (Actual: High)"],
                index=0
            )
            tier_key = "Low" if "Low" in selected_tier else ("Medium" if "Medium" in selected_tier else "High")

            exp_match = next((e for e in sample_exps if e.get("actual_category") == tier_key), sample_exps[0])

            c_pred1, c_pred2, c_pred3 = st.columns(3)
            with c_pred1:
                st.metric("Predicted Performance Tier", exp_match["predicted_category"])
            with c_pred2:
                st.metric("Ground Truth (Actual)", exp_match.get("actual_category", "N/A"))
            with c_pred3:
                st.metric("Traversed Tree Depth", f"{exp_match['traversed_depth']} split nodes")

            st.markdown("#### Class Probabilities at Terminal Leaf")
            probs = exp_match["predicted_probabilities"]
            st.progress(probs.get("High", 0.0), text=f"High: {probs.get('High', 0.0):.1%}")
            st.progress(probs.get("Medium", 0.0), text=f"Medium: {probs.get('Medium', 0.0):.1%}")
            st.progress(probs.get("Low", 0.0), text=f"Low: {probs.get('Low', 0.0):.1%}")

            st.markdown("#### Decision Tree Boundary Sequence (Trace Path)")
            st.caption("The sequence of boolean splits leading from the root node to this student's leaf node:")
            for rule in exp_match["decision_rules"]:
                st.markdown(f'<div class="rule-box">Node {rule["node_id"]}: <b>{rule["rule_description"]}</b></div>', unsafe_allow_html=True)

            st.markdown("#### Cohort Context for Key Features")
            cohort_data = exp_match.get("cohort_comparison", {})
            if cohort_data:
                rows = []
                for feat, stats in cohort_data.items():
                    rows.append({
                        "Feature": feat,
                        "Student Value": stats.get("student_value"),
                        "Cohort Mean": stats.get("cohort_mean", "—"),
                        "Cohort Median": stats.get("cohort_median", "—")
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="notice-box">
                <b>Methodological Notice:</b> {exp_match.get("methodological_notice", "")}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Sample explanations are currently not available. Run `save_day24_outputs()` to precompute.")

    with tab4:
        st.subheader("Methodology, Leakage Safeguards, and Ethical Limitations")
        st.markdown("""
        ### 1. Leakage Prevention Protocol
        To guarantee academic integrity and prevent optimistic evaluation bias:
        - The target label `performance_category` is derived from final grade `G3` (`Low < 10`, `Medium 10-14`, `High >= 15`).
        - **`G3` is strictly excluded from the 30 model input features.**
        - Neither `G3` nor `performance_category` are ever permuted or evaluated as input features.

        ### 2. Permutation Feature Importance Principles
        - Permutation importance is a **model-agnostic** diagnostic tool.
        - It quantifies the loss in predictive capability when a single feature is broken by shuffling.
        - Because feature transformations (One-Hot Encoding, Ordinal Encoding, Scaling) are encapsulated inside the `sklearn.pipeline.Pipeline`, permutation is conducted on the **raw input columns**, preserving realistic interaction structures.

        ### 3. Correlation vs. Causation
        > **Critical Educational Disclaimer:** High feature importance indicates that a variable has high predictive power within this specific model. **It does NOT prove that changing this variable will cause an improvement in student grades.**

        For example:
        - High `failures` is strongly associated with `Low` performance, but the underlying causal driver may be reading comprehension, personal hardship, or curriculum mismatch.
        - High `absences` reflects attendance patterns, but mandatory physical attendance alone does not ensure mastery of coursework.

        ### 4. Impact of Correlated Features
        When two features are collinear (such as mother's education `Medu` and father's education `Fedu`, or weekday alcohol `Dalc` and weekend alcohol `Walc`):
        - Permuting one feature leaves the correlated feature intact, allowing the model to leverage the unpermuted variable as a surrogate.
        - Consequently, permutation importance can underestimate the true joint contribution of correlated feature groups.

        ### 5. Reproducibility & Technical Details
        - **Metric:** Scikit-Learn `f1_macro` across 3 multi-class tiers (`Low`, `Medium`, `High`).
        - **Permutation Repeats:** 10–15 iterations per feature.
        - **Random Seed:** Fixed seed (`random_state=42`).
        """)


if __name__ == "__main__":
    render_explainability_page()
