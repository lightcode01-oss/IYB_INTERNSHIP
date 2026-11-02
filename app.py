"""
Student Performance Prediction System — Streamlit Web Application
Day 20: AI-Based Student Performance Prediction System with Model Monitoring
YBI Foundation Python AI & Data Science Internship
"""

import os
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from src.prediction import predict as serve_prediction, get_pipeline
from src.validation import validate_inputs
from src.batch import parse_csv

# Import privacy-conscious monitoring module
from src.monitoring import (
    log_prediction_event,
    update_prediction_feedback,
    get_monitoring_summary,
    clear_monitoring_logs,
    load_prediction_logs,
    DEFAULT_LOG_FILE
)


# -----------------------------------------------------------------------------
# Model & Metadata Loading Helper
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prediction_pipeline():
    """
    Safely locate and deserialize the pre-trained model pipeline and metadata.
    Anchors to the project root directory relative to this script for reliable deployment.
    """
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"

    pipeline_path = artifacts_dir / "student_performance_pipeline.joblib"
    metadata_path = artifacts_dir / "student_performance_metadata.json"

    # Fallback search if launched from differing working directories
    if not pipeline_path.exists():
        fallback_candidates = [
            Path.cwd() / "artifacts" / "student_performance_pipeline.joblib",
            Path.cwd() / "student_performance_pipeline.joblib",
            base_dir / "student_performance_pipeline.joblib"
        ]
        for cand in fallback_candidates:
            if cand.exists():
                pipeline_path = cand
                break

    if not metadata_path.exists():
        fallback_meta = [
            Path.cwd() / "artifacts" / "student_performance_metadata.json",
            Path.cwd() / "student_performance_metadata.json",
            base_dir / "student_performance_metadata.json"
        ]
        for cand in fallback_meta:
            if cand.exists():
                metadata_path = cand
                break

    if not pipeline_path.exists():
        return None, None, (
            "Model artifact 'student_performance_pipeline.joblib' was not found in 'artifacts/'. "
            "Please ensure that the trained pipeline exists or execute Day 17 to regenerate it."
        )

    try:
        pipeline = get_pipeline()
    except Exception as e:
        return None, None, f"Failed to deserialize pipeline artifact: {str(e)}"

    metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    return pipeline, metadata, None


# -----------------------------------------------------------------------------
# Main Application Routine
# -----------------------------------------------------------------------------
def main():
    # Page Configuration
    st.set_page_config(
        page_title="Student Performance Prediction & Monitoring",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.warning("Portfolio demonstration only. Do not use these predictions to grade students or determine access to education. Model probabilities are uncalibrated estimates.")

    # Custom CSS for polished, professional styling
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .prediction-box-high {
            background-color: #ECFDF5;
            border-left: 6px solid #10B981;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .prediction-box-medium {
            background-color: #EFF6FF;
            border-left: 6px solid #3B82F6;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .prediction-box-low {
            background-color: #FEF2F2;
            border-left: 6px solid #EF4444;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load artifacts
    pipeline, metadata, load_error = load_prediction_pipeline()

    # Sidebar: System Context & Monitoring Controls
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/student-center.png", width=70)
        st.title("System Overview")
        st.markdown("**AI-Based Student Performance Prediction System**")
        st.caption("YBI Foundation Python AI & DS Internship — Day 22 (Cloud Deployment)")

        st.markdown("---")
        st.subheader("Model Information")
        if metadata:
            model_name = metadata.get("model_selection", {}).get("selected_model", "Decision Tree")
            cv_f1 = metadata.get("model_selection", {}).get("training_cv_macro_f1", 0.5237)
            test_acc = metadata.get("test_metrics", {}).get("accuracy", 0.5385)
            st.write(f"**Architecture:** {model_name}")
            st.write(f"**5-Fold CV Macro F1:** {cv_f1:.4f}")
            st.write(f"**Test Accuracy:** {test_acc:.2%}")
            st.write("**Leakage Safeguard:** Active (`G3` excluded)")
        else:
            model_name = "Decision_Tree_v1"
            st.info("Operating in standalone pipeline mode.")

        st.markdown("---")
        st.subheader("Navigation & Evaluation")
        try:
            st.page_link("pages/2_Model_Performance_Dashboard.py", label="Performance Dashboard", icon="📊")
        except Exception:
            st.caption("Visit pages/2_Model_Performance_Dashboard.py for model evaluation.")

        st.markdown("---")
        st.subheader("Monitoring & Privacy Controls")
        enable_logging = st.checkbox(
            "Enable Prediction Logging",
            value=False,
            help="Record anonymous operational telemetry for post-deployment monitoring."
        )
        log_features = st.checkbox(
            "Log Input Features (Opt-in)",
            value=False,
            help="Record non-target input attributes to enable input distribution monitoring. G3 is strictly excluded."
        )

        with st.expander("Privacy & Storage Safeguards"):
            st.caption(
                "• No personal identifying information (PII) is recorded.\n"
                "• G3 final grade and target categories are strictly excluded.\n"
                "• Stored locally at `data/monitoring/prediction_events.csv`.\n"
                "• Logging can be toggled off at any time."
            )

        st.markdown("---")
        st.subheader("Performance Tiers")
        st.markdown("""
        - **High:** $G3 \\ge 15$
        - **Medium:** $10 \\le G3 < 15$
        - **Low:** $G3 < 10$ (Academic Intervention Needed)
        """)

        st.markdown("---")
        st.caption("Educational Machine Learning Demonstration. Not intended for consequential high-stakes tracking.")

    # Main Header
    st.markdown('<div class="main-title">Student Performance Prediction & Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Early-warning academic diagnostic tool with integrated post-deployment monitoring, '
        'prediction logging, and subjective user feedback.'
        '</div>',
        unsafe_allow_html=True
    )

    if load_error:
        st.error(f"⚠️ {load_error}")
        st.warning("Please ensure that Day 17 has been executed to generate the model artifact at `artifacts/student_performance_pipeline.joblib`.")
        st.stop()

    # Tabs (Now includes Tab 3: Model Monitoring & Analytics)
    tab_single, tab_batch, tab_monitoring, tab_about = st.tabs([
        "Single Student Prediction",
        "Batch CSV Prediction",
        "Model Monitoring & Analytics",
        "About & Ethical Guidelines"
    ])

    # =========================================================================
    # TAB 1: Single Student Prediction Form
    # =========================================================================
    with tab_single:
        st.markdown("Use demonstration data to explore a model estimate. All 30 fields are required. Do not enter names, contact details or final grades. The dataset covers two Portuguese schools and may not represent your context.")

        with st.form("single_student_prediction_form"):
            col1, col2 = st.columns(2)

            # Section 1: Demographics & School Profile
            with col1:
                st.markdown("#### 1. Demographics & School Profile")
                school = st.selectbox(
                    "School",
                    options=["GP", "MS"],
                    format_func=lambda x: "GP — Gabriel Pereira" if x == "GP" else "MS — Mousinho da Silveira",
                    help="Student's secondary school."
                )
                sex = st.selectbox(
                    "Sex (as recorded in the source dataset)",
                    options=["F", "M"],
                    format_func=lambda x: "Female" if x == "F" else "Male"
                )
                age = st.slider("Age (years)", min_value=15, max_value=22, value=17, step=1)
                address = st.selectbox(
                    "Home Address Type",
                    options=["U", "R"],
                    format_func=lambda x: "U — Urban" if x == "U" else "R — Rural"
                )
                famsize = st.selectbox(
                    "Family Size",
                    options=["GT3", "LE3"],
                    format_func=lambda x: "GT3 — Greater than 3 members" if x == "GT3" else "LE3 — Less than or equal to 3 members"
                )
                Pstatus = st.selectbox(
                    "Parents Cohabitation Status",
                    options=["T", "A"],
                    format_func=lambda x: "T — Living together" if x == "T" else "A — Living apart"
                )

            # Section 2: Family Background
            with col2:
                st.markdown("#### 2. Family Background & Support")
                Medu = st.selectbox(
                    "Mother's Education (Medu)",
                    options=[0, 1, 2, 3, 4],
                    index=2,
                    format_func=lambda x: {
                        0: "0 — None",
                        1: "1 — Primary (4th grade)",
                        2: "2 — 5th to 9th grade",
                        3: "3 — Secondary education",
                        4: "4 — Higher education"
                    }[x]
                )
                Fedu = st.selectbox(
                    "Father's Education (Fedu)",
                    options=[0, 1, 2, 3, 4],
                    index=2,
                    format_func=lambda x: {
                        0: "0 — None",
                        1: "1 — Primary (4th grade)",
                        2: "2 — 5th to 9th grade",
                        3: "3 — Secondary education",
                        4: "4 — Higher education"
                    }[x]
                )
                Mjob = st.selectbox(
                    "Mother's Job (Mjob)",
                    options=["at_home", "health", "other", "services", "teacher"],
                    index=2
                )
                Fjob = st.selectbox(
                    "Father's Job (Fjob)",
                    options=["at_home", "health", "other", "services", "teacher"],
                    index=2
                )
                reason = st.selectbox(
                    "Reason for Choosing School",
                    options=["course", "home", "other", "reputation"],
                    index=0
                )
                guardian = st.selectbox(
                    "Legal Guardian",
                    options=["mother", "father", "other"],
                    index=0
                )

            st.markdown("---")
            col3, col4 = st.columns(2)

            # Section 3: Academic Habits & Support
            with col3:
                st.markdown("#### 3. Academic Habits & Support")
                traveltime = st.selectbox(
                    "Home to School Travel Time",
                    options=[1, 2, 3, 4],
                    index=0,
                    format_func=lambda x: {
                        1: "1 — Less than 15 min",
                        2: "2 — 15 to 30 min",
                        3: "3 — 30 min to 1 hour",
                        4: "4 — More than 1 hour"
                    }[x]
                )
                studytime = st.selectbox(
                    "Weekly Study Time",
                    options=[1, 2, 3, 4],
                    index=1,
                    format_func=lambda x: {
                        1: "1 — Less than 2 hours",
                        2: "2 — 2 to 5 hours",
                        3: "3 — 5 to 10 hours",
                        4: "4 — More than 10 hours"
                    }[x]
                )
                failures = st.selectbox(
                    "Number of Past Class Failures",
                    options=[0, 1, 2, 3, 4],
                    index=0,
                    format_func=lambda x: f"{x} failures"
                )
                schoolsup = st.selectbox("Extra Educational Support from School", options=["no", "yes"])
                famsup = st.selectbox("Family Educational Support", options=["yes", "no"])
                paid = st.selectbox("Extra Paid Classes in Course Subject", options=["no", "yes"])
                higher = st.selectbox("Wants to Pursue Higher Education", options=["yes", "no"], index=0)
                absences = st.number_input("School Absences (count)", min_value=0, max_value=93, value=4, step=1)

            # Section 4: Extracurriculars, Social Life & Health
            with col4:
                st.markdown("#### 4. Social Life, Health & Lifestyle")
                activities = st.selectbox("Extracurricular Activities", options=["no", "yes"])
                nursery = st.selectbox("Attended Nursery School", options=["yes", "no"])
                internet = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0)
                romantic = st.selectbox("In a Romantic Relationship", options=["no", "yes"])
                famrel = st.slider("Quality of Family Relationships (1=Very bad to 5=Excellent)", 1, 5, 4)
                freetime = st.slider("Free Time After School (1=Very low to 5=Very high)", 1, 5, 3)
                goout = st.slider("Going Out with Friends (1=Very low to 5=Very high)", 1, 5, 3)
                Dalc = st.slider("Workday Alcohol Consumption (1=Very low to 5=Very high)", 1, 5, 1)
                Walc = st.slider("Weekend Alcohol Consumption (1=Very low to 5=Very high)", 1, 5, 1)
                health = st.slider("Current Health Status (1=Very bad to 5=Very good)", 1, 5, 4)

            submit_btn = st.form_submit_button("Predict Performance Tier", use_container_width=True)

        # Prediction Processing
        if submit_btn:
            student_record = {
                "school": school,
                "sex": sex,
                "age": int(age),
                "address": address,
                "famsize": famsize,
                "Pstatus": Pstatus,
                "Medu": int(Medu),
                "Fedu": int(Fedu),
                "Mjob": Mjob,
                "Fjob": Fjob,
                "reason": reason,
                "guardian": guardian,
                "traveltime": int(traveltime),
                "studytime": int(studytime),
                "failures": int(failures),
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
                "nursery": nursery,
                "higher": higher,
                "internet": internet,
                "romantic": romantic,
                "famrel": int(famrel),
                "freetime": int(freetime),
                "goout": int(goout),
                "Dalc": int(Dalc),
                "Walc": int(Walc),
                "health": int(health),
                "absences": int(absences)
            }

            expected_cols = list(pipeline.feature_names_in_)
            input_df = pd.DataFrame([student_record])[expected_cols]

            try:
                served = serve_prediction(input_df)
                predicted_category = served.predicted_category.iloc[0]

                probas = None
                classes = list(pipeline.classes_)
                if hasattr(pipeline, "predict_proba"):
                    probas = np.array([served["model_estimated_proba_"+label].iloc[0] for label in classes])

                # Generate event ID for monitoring and feedback
                current_prediction_id = uuid.uuid4().hex[:10]
                st.session_state["active_prediction_id"] = current_prediction_id
                st.session_state["active_prediction_category"] = predicted_category

                # Privacy-conscious event logging
                if enable_logging:
                    prob_dict = {classes[i]: float(probas[i]) for i in range(len(classes))} if probas is not None else None
                    logged, log_info = log_prediction_event(
                        predicted_category=predicted_category,
                        probabilities=prob_dict,
                        model_id=model_name if metadata else "DecisionTree_v1",
                        prediction_id=current_prediction_id,
                        input_features=student_record,
                        log_features=log_features,
                        enabled=enable_logging
                    )
                    if logged:
                        st.caption(f"🔒 Anonymous prediction event logged (ID: `{current_prediction_id}`).")

                st.markdown("### Prediction Results")

                if predicted_category == "High":
                    st.markdown(
                        f'<div class="prediction-box-high">'
                        f'<h3>🎓 Predicted Category: HIGH (G3 ≥ 15)</h3>'
                        f'<p>The model assigned the High category for this input. This is a statistical estimate, not a guarantee of a future grade.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                elif predicted_category == "Medium":
                    st.markdown(
                        f'<div class="prediction-box-medium">'
                        f'<h3>📘 Predicted Category: MEDIUM (10 ≤ G3 < 15)</h3>'
                        f'<p>The model assigned the Medium category for this input. It cannot determine individual ability or future achievement.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:  # Low
                    st.markdown(
                        f'<div class="prediction-box-low">'
                        f'<h3>⚠️ Predicted Category: LOW (G3 < 10)</h3>'
                        f'<p>The model assigned the Low category for this input. Review model limitations before interpreting this estimate; it is not an individual assessment.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                if probas is not None:
                    st.markdown("#### Model-Estimated Class Probabilities")
                    st.caption("Note: These scores represent empirical likelihoods derived from decision tree leaf distributions, not deterministic certainties.")

                    prob_cols = st.columns(len(classes))
                    for i, cls_name in enumerate(classes):
                        score = probas[i]
                        with prob_cols[i]:
                            st.metric(label=f"Class: {cls_name}", value=f"{score:.1%}")
                            st.progress(float(score))

                st.info("ℹ️ **Educational Disclaimer:** Predictions represent statistical associations from survey data and should not be used as the sole basis for high-stakes academic decisions.")

            except Exception as err:
                st.error(f"Prediction failed: {str(err)}")

        # Subjective Feedback Interface (Active after a prediction is made)
        if "active_prediction_id" in st.session_state:
            active_id = st.session_state["active_prediction_id"]
            st.markdown("---")
            st.markdown("#### Optional User Feedback (Subjective Impression)")
            st.caption(
                "How does this prediction align with your subjective domain judgment? "
                "**Important Note:** Feedback is subjective user perception and is NOT verified ground-truth accuracy."
            )

            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])
            with fb_col1:
                if st.button("👍 Plausible / Useful", key=f"fb_helpful_{active_id}"):
                    ok, msg = update_prediction_feedback(active_id, "Helpful")
                    if ok:
                        st.success("Feedback recorded: 'Helpful'. Thank you!")
            with fb_col2:
                if st.button("👎 Doubtful / Unhelpful", key=f"fb_unhelpful_{active_id}"):
                    ok, msg = update_prediction_feedback(active_id, "Unhelpful")
                    if ok:
                        st.info("Feedback recorded: 'Unhelpful'. Thank you!")
            with fb_col3:
                if st.button("➖ Neutral / Unsure", key=f"fb_neutral_{active_id}"):
                    ok, msg = update_prediction_feedback(active_id, "Neutral")
                    if ok:
                        st.caption("Feedback recorded: 'Neutral'.")

    # =========================================================================
    # TAB 2: Batch CSV Prediction
    # =========================================================================
    with tab_batch:
        st.markdown("### Batch Student Prediction via CSV")
        st.markdown(
            "Upload a `.csv` file containing multiple student records to evaluate them simultaneously. "
            "The CSV must contain exactly the 30 predictor columns, at most 500 rows and be no larger than 1 MiB. Invalid batches are rejected in full. `G3` and `performance_category` must not be included."
        )

        uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = parse_csv(uploaded_file.getvalue())
                st.write(f"Uploaded {len(batch_df)} rows and {batch_df.shape[1]} columns.")

                expected_cols = list(pipeline.feature_names_in_)
                missing_cols = set(expected_cols) - set(batch_df.columns)

                if missing_cols:
                    st.error(f"❌ Missing {len(missing_cols)} required columns: {sorted(missing_cols)}")
                else:
                    df_to_predict = validate_inputs(batch_df)

                    if df_to_predict.isnull().any().any():
                        st.warning("⚠️ Warning: Uploaded data contains missing (NaN) values. Rows with missing data may cause errors.")

                    if st.button("Run Batch Prediction", key="run_batch_btn"):
                        with st.spinner("Generating batch predictions..."):
                            served = serve_prediction(df_to_predict)
                            preds = served.predicted_category.to_numpy()
                            results_df = batch_df.copy()
                            results_df["predicted_category"] = preds

                            if hasattr(pipeline, "predict_proba"):
                                probas = served[["model_estimated_proba_"+label for label in pipeline.classes_]].to_numpy()
                                for idx, cls_name in enumerate(pipeline.classes_):
                                    results_df[f"model_estimated_proba_{cls_name}"] = np.round(probas[:, idx], 4)

                            # Log batch summary event if logging is active
                            if enable_logging:
                                for pred_val in preds:
                                    log_prediction_event(
                                        predicted_category=pred_val,
                                        model_id=f"{model_name}_batch" if metadata else "DecisionTree_batch",
                                        enabled=enable_logging
                                    )

                            st.success(f"Successfully processed {len(results_df)} student records!")
                            st.dataframe(results_df.head(10), use_container_width=True)

                            csv_data = results_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_data,
                                file_name="student_performance_batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

            except Exception as e:
                st.error(f"Failed to process CSV file: {str(e)}")

    # =========================================================================
    # TAB 3: Model Monitoring & Analytics (NEW: Day 20)
    # =========================================================================
    with tab_monitoring:
        st.markdown("### Post-Deployment Monitoring & Operational Telemetry")
        st.markdown(
            "This dashboard aggregates operational telemetry from prediction activity to monitor "
            "prediction distributions, probability patterns, and subjective user feedback over time."
        )

        summary = get_monitoring_summary()

        if summary["is_empty"]:
            st.info("ℹ️ **No prediction events have been recorded yet.**")
            st.markdown(
                "When predictions are run with **Enable Prediction Logging** checked, operational events "
                "will appear here automatically."
            )
        else:
            # Top Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Total Predictions Logged", summary["total_events"])
            with m_col2:
                top_class = summary["class_counts"].index[0] if len(summary["class_counts"]) > 0 else "N/A"
                top_pct = summary["class_proportions"].iloc[0] if len(summary["class_proportions"]) > 0 else 0.0
                st.metric("Top Predicted Tier", f"{top_class} ({top_pct}%)")
            with m_col3:
                fb_total = summary["feedback_summary"]["total_feedback_events"]
                st.metric("Subjective Feedback Received", fb_total)
            with m_col4:
                fb_rate = summary["feedback_summary"].get("feedback_participation_rate_pct", 0.0)
                st.metric("Feedback Rate", f"{fb_rate}%")

            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### Predicted Category Distribution")
                st.bar_chart(summary["class_counts"])

                # Proportions table
                dist_df = pd.DataFrame({
                    "Count": summary["class_counts"],
                    "Proportion (%)": summary["class_proportions"]
                })
                st.dataframe(dist_df, use_container_width=True)

            with chart_col2:
                st.markdown("#### Subjective User Feedback")
                st.caption("Subjective perception from users, NOT verified ground truth accuracy.")

                if len(summary["feedback_counts"]) > 0:
                    st.bar_chart(summary["feedback_counts"])
                    st.dataframe(pd.DataFrame({"Count": summary["feedback_counts"]}), use_container_width=True)
                else:
                    st.write("No feedback entries submitted yet.")

                st.markdown("#### Mean Estimated Class Probabilities")
                proba_df = pd.DataFrame(
                    list(summary["average_probabilities"].items()),
                    columns=["Class Tier", "Mean Probability"]
                )
                st.dataframe(proba_df, use_container_width=True)

            st.markdown("---")
            with st.expander("Telemetry Audit Log (Strict Privacy: Anonymized Event Telemetry)", expanded=False):
                st.markdown("#### Recent Anonymous Prediction Events")
                st.caption("Latest 10 logged prediction records (strict privacy: no student names, IDs, or contact details are recorded or displayed).")
                recent_df = summary["recent_events"]
                display_cols = [c for c in ["timestamp", "prediction_id", "predicted_category", "proba_High", "proba_Low", "proba_Medium", "feedback"] if c in recent_df.columns]
                st.dataframe(recent_df[display_cols], use_container_width=True)

            # Maintenance & reset options
            with st.expander("Storage Management & Reset"):
                st.caption(f"Log storage path: `{DEFAULT_LOG_FILE.resolve()}`")
                if st.button("Clear Logged Prediction Events", type="secondary"):
                    if clear_monitoring_logs():
                        st.success("Monitoring logs cleared successfully.")
                        st.rerun()

    # =========================================================================
    # TAB 4: About & Ethical Safeguards
    # =========================================================================
    with tab_about:
        st.markdown("### About the AI-Based Student Performance Prediction System")
        st.markdown("""
        This project is part of the **YBI Foundation Python AI & Data Science Internship**.
        It demonstrates how supervised machine learning can be applied responsibly in educational contexts.

        #### 1. Target Formulation & Leakage Prevention
        - The ground-truth performance categories are defined based on the final grade $G3$:
          - **Low:** $G3 < 10$
          - **Medium:** $10 \\le G3 < 15$
          - **High:** $G3 \\ge 15$
        - **Target Leakage Safeguard:** Because $G3$ directly generates the target, including it as a predictor would allow trivial 100% memorization while failing completely in real-world early prediction. $G3$ is strictly excluded from all inference schemas.

        #### 2. Selected Machine Learning Architecture
        - **Champion Model:** Tuned **Decision Tree Classifier** (`criterion='gini'`, `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=10`).
        - **Evidence Basis:** Selected dynamically based on highest 5-Fold Stratified Cross-Validation Macro F1 ($0.5237 \\pm 0.0389$) on Day 16, outperforming Logistic Regression, Random Forest, SVM, KNN, and Gaussian Naive Bayes.
        - **Held-Out Test Performance ($N=130$):** Macro F1: $0.4498$, Accuracy: $0.5385$.

        #### 3. Post-Deployment Monitoring Principles (Day 20)
        - **Privacy First:** The system logs anonymous event telemetry (timestamp, predicted tier, probabilities). No student PII (names, IDs, emails) is ever logged.
        - **Subjective Feedback vs. Verified Ground Truth:** User ratings ('Helpful', 'Unhelpful') reflect subjective human perception. They are **never** used to compute real model accuracy, precision, or recall.
        - **Operational Telemetry:** Monitoring class distributions helps detect operational shifts, but changes in prediction frequencies alone do not prove model concept drift or degradation without verified labels.

        #### 4. Ethical and Decision-Support Safeguards
        - **Exploratory Diagnostic Tool:** Predictions provide early empirical signals, not deterministic guarantees.
        - **Non-Causal Interpretability:** Features such as study time, parental education, and absences represent correlational survey indicators, not direct causal levers.
        - **Strict Operational Boundary:** This system must **never** be used as the sole basis for punitive academic sanctions, tracking, or resource denial.
        """)
        st.caption("Developed for educational research and diagnostic exploration. © 2026 YBI Foundation Internship.")


if __name__ == "__main__":
    main()
