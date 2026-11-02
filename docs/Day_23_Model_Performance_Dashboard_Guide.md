# Day 23: Model Performance Dashboard & Evaluation Report Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System

---

## 1. Overview & Purpose

The **Model Performance Dashboard** (`pages/2_Model_Performance_Dashboard.py`) provides an interactive, transparent, and auditable environment for inspecting the empirical evaluation results produced across Days 14–18 of the internship.

### Key Objectives:
- **Empirical Transparency:** Surface genuine cross-validation and held-out test metrics without synthesizing data or altering historical results.
- **Model Comparison:** Compare baseline architectures against tuned counterparts using **Macro F1** under 5-Fold Stratified Cross-Validation.
- **Class-Level Evaluation:** Inspect precision, recall, and F1 across academic tiers (`Low`, `Medium`, `High`) to understand model behavior across imbalanced subgroups.
- **Error Characterization:** Interactively examine test set misclassifications, empirical confusion matrix counts, and distinguish benign adjacent errors from severe non-adjacent errors.
- **Exportable Governance:** Allow practitioners, educators, and auditors to download structured Markdown and CSV evaluation reports.

---

## 2. System Architecture & File Organization

The Day 23 additions integrate smoothly with the existing Streamlit application and codebase:

```
IYB_internship/
├── app.py                                   # Primary prediction & monitoring app (Day 19–22)
├── pages/
│   └── 2_Model_Performance_Dashboard.py     # Interactive Streamlit evaluation dashboard (Day 23)
├── src/
│   ├── monitoring.py                        # Privacy-conscious event logging (Day 20)
│   └── evaluation_dashboard.py              # Robust data loader, matrix math & report export (Day 23)
├── tests/
│   ├── test_deployment_readiness.py         # Deployment readiness test suite (Updated Day 23)
│   ├── test_monitoring.py                   # Monitoring & telemetry unit tests (Day 20)
│   └── test_evaluation_dashboard.py         # Data loading, confusion matrix & AppTest smoke tests (Day 23)
├── notebooks/
│   └── Day_23_Model_Performance_Dashboard.ipynb  # Interactive Colab-compatible documentation notebook
├── docs/
│   ├── Day_23_Model_Performance_Dashboard_Guide.md  # Comprehensive operator manual (This document)
│   ├── Day_22_Streamlit_Deployment_Guide.md
│   ├── Day_21_Deployment_Guide.md
│   ├── Day_20_Monitoring_Guide.md
│   └── Student_Performance_Prediction_User_Guide.md
└── artifacts/
    ├── student_performance_pipeline.joblib  # Serialized scikit-learn Decision Tree pipeline (20.6 KB)
    └── student_performance_metadata.json    # Machine-readable evaluation schema & metadata
```

---

## 3. Verified Historical Evaluation Results

All dashboard metrics are drawn directly from the authentic project evaluation files:

| Source File | Evaluation Method | Key Metrics Captured |
| :--- | :--- | :--- |
| **`day_14_cross_validation_results.csv`** | 5-Fold Stratified CV ($N=519$) | Baseline metrics for 6 models (Accuracy, Macro Precision, Recall, F1) |
| **`day_15_baseline_vs_tuned.csv`** | 5-Fold Stratified CV ($N=519$) | Baseline vs Tuned Macro F1, std devs, and gains ($\Delta$ F1) |
| **`day_15_tuning_results.csv`** | GridSearchCV on 5-Fold CV | Optimal hyperparameters, combination counts, fit durations |
| **`day_16_classification_report.csv`** | Unseen Held-Out Test ($N=130$) | Per-class metrics (Low, Medium, High) + Macro & Weighted averages |
| **`day_16_error_analysis.csv`** | Unseen Held-Out Test ($N=130$) | Prediction records for every test student with features & correctness |
| **`day_16_model_selection_report.json`** | Comprehensive Validation Audit | Model selection justification, test metrics, error summary, limits |
| **`artifacts/student_performance_metadata.json`** | Production Pipeline Artifact | Serialized pipeline schema, features, target mapping, leakage checks |

### Core Evaluation Findings:
- **Selected Model:** Decision Tree (`criterion='gini'`, `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=10`).
- **Selection Metric:** 5-Fold Stratified Cross-Validation Macro F1 = **0.5237 ± 0.0389** (Highest across all evaluated algorithms).
- **Held-Out Test Accuracy:** **53.85%** (70 correct predictions out of 130 test instances).
- **Held-Out Test Macro F1:** **0.4498** (Generalization gap: $-0.0739$).
- **Misclassification Proximity:** **93.3%** of errors are strictly adjacent (Low $\leftrightarrow$ Medium or Medium $\leftrightarrow$ High). Only **6.7%** (4 students) are severe opposite-tier errors (Low $\leftrightarrow$ High).

---

## 4. Dashboard Sections & Operational Features

The dashboard at `pages/2_Model_Performance_Dashboard.py` is organized into six clear, functional sections:

### Section 1: Evaluation Overview & Key Metrics
- **KPI Summary Cards:** Selected architecture, training CV Macro F1 with standard deviation, training CV Accuracy, held-out Test Macro F1, and Generalization Gap.
- **Evaluation Source Tags:** Explicitly labels metrics as either *Cross-Validation (Training Split)* or *Held-Out Test Set*.
- **Method Transparency Callout:** Explains why training cross-validation was used to select the champion model without peeking at the test set.

### Section 2: Multi-Model Comparison (Baseline vs. Tuned)
- **Selectable Views:** Users can toggle between:
  1. *Macro F1 (Primary)*: Tabular comparison with gains and interactive grouped bar charts with $\pm 1$ std dev error bars.
  2. *Cross-Validation Accuracy*: Baseline metrics across all 6 models from Day 14.
  3. *Tuning Parameter Comparison*: Detailed optimal hyperparameters and search spaces evaluated during Day 15.
- **Explicit Missing Values:** Any unavailable metrics are marked explicitly rather than artificially replaced with zero.

### Section 3: Class-Level Performance (Held-Out Test Set)
- **Separated Reports:** Clear visual separation between per-class metrics (`Low`, `Medium`, `High`) and aggregate averages (`accuracy`, `macro avg`, `weighted avg`).
- **Tier Comparison Chart:** Grouped bar chart comparing Precision, Recall, and F1 across tiers.
- **Class Imbalance Commentary:** Highlights the distribution of the test split (`Medium`: 84 students / 64.6%, `Low`: 20 students / 15.4%, `High`: 26 students / 20.0%) and discusses its influence on minority-class recall.

### Section 4: Error Analysis & Misclassification Inspection
- **Empirical Confusion Matrix:** Heatmap with exact integer counts derived strictly from the 130 real test predictions:
  - Actual Low ($N=20$): Predicted Low: 9, Medium: 9, High: 2
  - Actual Medium ($N=84$): Predicted Low: 17, Medium: 53, High: 14
  - Actual High ($N=26$): Predicted Low: 2, Medium: 16, High: 8
- **Error Severity Distribution:** Pie chart quantifying that 93.3% of errors are between neighboring grade tiers.
- **Interactive Test Prediction Browser:** Filter test records by:
  - *All Test Records (130)*
  - *Misclassified Records Only (60)*
  - *Severe Errors Only (Low $\leftrightarrow$ High) (4)*
  Inspect key student attributes such as `studytime`, `failures`, `absences`, and `schoolsup` for each case.

### Section 5: Downloadable Evaluation Reports
- **Download Markdown Report:** Generates `model_evaluation_report.md` summarizing the full validation methodology, metrics, and ethical notes.
- **Download Evaluation Summary CSV:** Exports `model_evaluation_summary.csv` tracking metric values, model names, validation methods, and source files.
- **Download Test Error Records CSV:** Exports the 130 genuine test prediction records for external audit.

### Section 6: Methodological Integrity & Ethical Governance
- **Strict Leakage Prevention:** Confirms that final grade `G3` is omitted from inputs.
- **Correlation vs. Causation:** Disclaims that survey associations do not prove causal interventions.
- **Decision-Support Scope:** Reaffirms that model outputs are early warning indicators for academic tutoring, never punitive tracking.

---

## 5. How to Run the Application

### Option A: Launch Primary App with Multi-Page Navigation
Start the primary application from the project root:
```bash
streamlit run app.py
```
1. Open your browser at `http://localhost:8501`.
2. In the Streamlit sidebar, select **"2 Model Performance Dashboard"** (or click the **"Performance Dashboard"** link in the sidebar).
3. Seamlessly return to the prediction application using the **"Prediction & Monitoring App"** sidebar link.

### Option B: Launch Performance Dashboard Directly
You can also launch the dashboard page directly:
```bash
streamlit run pages/2_Model_Performance_Dashboard.py
```

---

## 6. Verification & Automated Testing

The evaluation dashboard is validated through automated test suites in `tests/test_evaluation_dashboard.py` and `tests/test_deployment_readiness.py`:

```bash
python -m unittest discover -s tests
```

### Test Coverage Summary:
- **`test_01_all_historical_evaluation_files_discovered`:** Validates discovery and non-empty status of all 10 historical evaluation files.
- **`test_02_evaluation_data_loading_and_columns`:** Validates schema parsing and required columns across all data loaders.
- **`test_03_missing_file_graceful_handling`:** Confirms loaders return `(None, error_msg)` without throwing exceptions when files are absent.
- **`test_04_malformed_columns_graceful_handling`:** Confirms loaders detect missing schema columns and report actionable error messages.
- **`test_05_confusion_matrix_accuracy_parity`:** Confirms confusion matrix math aligns exactly with 130 test instances, 70 correct predictions, and 53.85% accuracy.
- **`test_06_error_severity_breakdown`:** Confirms calculation of 56 adjacent errors (93.3%) and 4 non-adjacent errors (6.7%).
- **`test_07_report_generation`:** Validates Markdown and CSV export generators.
- **`test_08_streamlit_dashboard_smoke_test`:** Runs headless `AppTest` to verify that `pages/2_Model_Performance_Dashboard.py` executes without unhandled runtime exceptions.

All 26 unit and integration tests across the repository pass cleanly.

---

## 7. Ethical Limitations & Academic Guidance

1. **Early Diagnostic Tooling:** The system provides probabilistic classification to assist educators in offering proactive academic help.
2. **Absence of Consequential Authority:** Under no circumstances should automated model estimates be used for high-stakes tracking, course exclusions, or disciplinary action.
3. **Observational Bounds:** Features reflect demographic and self-reported lifestyle attributes. Improving a behavioral attribute does not guarantee grade improvements due to complex confounding socio-economic factors.
