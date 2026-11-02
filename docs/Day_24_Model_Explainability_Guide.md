# Day 24: Model Explainability & Feature Importance Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System

---

## 1. Overview & Purpose

The **Model Explainability & Feature Importance Module** provides transparency, mathematical interpretability, and ethical accountability for the student academic performance prediction system.

In predictive systems deployed within educational institutions, "black box" decisions can reinforce educational bias, cause misallocation of tutoring resources, or lead to misinterpretations by school counselors. Day 24 implements both **global feature attribution** and **local prediction explanation** without retraining or replacing the established champion Decision Tree pipeline (`artifacts/student_performance_pipeline.joblib`).

### Key Objectives:
1. **Global Permutation Feature Importance:** Measure the drop in model generalization (**Macro F1**) when each of the 30 raw input features is permuted across the authentic held-out test set ($N=130$).
2. **Native Model Feature Importance:** Extract and aggregate the Decision Tree's internal Mean Decrease in Impurity (Gini importance) across 56 transformed pipeline features back to canonical input columns.
3. **Comparative Diagnostic:** Contrast permutation importance (test set impact) against native MDI (training split frequency) to highlight overfitting risks and pseudo-predictive features.
4. **Local Prediction Explanation:** Trace exact tree decision rules (feature thresholds, operators, node splits) and benchmark student attributes against reference cohort distributions for individual predictions.
5. **Leakage & Ethical Safeguards:** Strictly guarantee that final exam score `G3` and target label `performance_category` remain excluded from all explanatory input spaces, while clarifying that predictive importance does **not** imply academic causation.

---

## 2. System Architecture & File Organization

The Day 24 deliverables are structured cleanly within the existing repository layout:

```
IYB_internship/
├── artifacts/
│   ├── student_performance_pipeline.joblib   # Champion Decision Tree pipeline (Day 17)
│   └── student_performance_metadata.json     # Pipeline schema & target boundaries
├── day_16_error_analysis.csv                 # Held-out test set with 30 inputs (N=130)
├── docs/
│   ├── Day_24_Model_Explainability_Guide.md  # Comprehensive operator manual (This document)
│   ├── Day_23_Model_Performance_Dashboard_Guide.md
│   └── Student_Performance_Prediction_User_Guide.md
├── notebooks/
│   ├── Day_24_Model_Explainability_Feature_Importance.ipynb # Colab-compatible notebook
│   └── Day_23_Model_Performance_Dashboard.ipynb
├── pages/
│   ├── 2_Model_Performance_Dashboard.py      # Model performance dashboard (Day 23)
│   └── 3_Model_Explainability.py             # Model explainability dashboard (Day 24)
├── reports/
│   └── day_24/
│       ├── permutation_importance.csv        # 30-feature permutation importance metrics
│       ├── permutation_importance.png        # Top 10 permutation horizontal bar chart
│       ├── native_feature_importance.csv     # Decision Tree Gini MDI aggregated table
│       ├── native_feature_importance.png     # Top 10 native Gini importance chart
│       └── sample_individual_explanations.json # Local decision paths for Low, Med, High
├── src/
│   ├── evaluation_dashboard.py               # Performance dashboard backend (Day 23)
│   ├── explainability.py                     # Reusable explainability engine (Day 24)
│   └── monitoring.py                         # Telemetry & event logging (Day 20)
└── tests/
    ├── test_deployment_readiness.py          # Deployment tests
    ├── test_evaluation_dashboard.py          # Evaluation tests
    ├── test_explainability.py                # 10-test explainability verification suite (Day 24)
    └── test_monitoring.py                    # Monitoring tests
```

---

## 3. Explainability Methodologies

### 3.1 Permutation Feature Importance (Model-Agnostic)

Permutation feature importance (`sklearn.inspection.permutation_importance`) evaluates how dependent a fitted model's predictions are on a given feature.

#### Execution Setup:
- **Model Pipeline:** The pre-trained scikit-learn `Pipeline` (ColumnTransformer preprocessor + DecisionTreeClassifier) is evaluated intact. No retraining or weight adjustment occurs.
- **Reference Dataset:** The authentic held-out test split of 130 students (`day_16_error_analysis.csv`) established in Cortez and Silva (2008) and partitioned during Day 14–16.
- **Target Metric:** **Macro F1** across the three academic tiers (`Low`, `Medium`, `High`). Macro F1 treats minority tiers (`Low` and `High`) with equal weight to prevent majority-class masking.
- **Permutation Process:** For each feature $j \in \{1, \dots, 30\}$:
  1. Feature $j$ is randomly shuffled across the 130 test instances, severing its relationship with the target $y$ while preserving the marginal distribution of other features.
  2. The pipeline computes predictions on the shuffled matrix and calculates the new Macro F1 score.
  3. The importance metric is defined as:
     $$\Delta \text{Macro F1}_j = \text{Macro F1}_{\text{baseline}} - \text{Macro F1}_{\text{permuted}, j}$$
  4. The process is repeated across 15 iterations with a fixed random seed (`random_state=42`) to compute the mean importance and standard deviation.

#### Why Permute on Raw Input Features?
Because the model's preprocessing is encapsulated inside the pipeline, permutation is conducted on the raw 30 input columns. This avoids the severe pitfall of permuting isolated one-hot dummy flags (which produces impossible feature vectors, such as a student whose mother works simultaneously in health, services, and at home).

### 3.2 Native Decision Tree Importance (Gini MDI)

The Decision Tree classifier calculates feature importance via **Mean Decrease in Impurity (MDI)** during training:
- At each split node $t$ that branches on feature $k$, the Gini impurity reduction is calculated:
  $$\Delta i(t) = i(t) - \frac{N_{t_L}}{N_t} i(t_L) - \frac{N_{t_R}}{N_t} i(t_R)$$
- The overall importance of feature $k$ is the sum of impurity decreases weighted by the fraction of training samples reaching node $t$, normalized so all feature importances sum to $1.0$.
- Because categorical columns were one-hot encoded into 56 transformed features, the module aggregates the importances of all dummy levels (e.g., `Mjob_health`, `Mjob_services`, `Mjob_at_home`) back to their parent canonical attribute (`Mjob`).

### 3.3 Local Prediction Explanation (Decision Paths)

For an individual student, standard global rankings do not reveal why the model assigned a particular category. The explainability module traces the exact boolean boundary path through the fitted tree:
1. Transforms the student record using the pipeline's fitted `ColumnTransformer`.
2. Uses `clf.decision_path(X)` to extract node traversal indices from root to terminal leaf.
3. Decodes the split threshold, comparison operator (`<=` or `>`), and student attribute value at each branch.
4. Retrieves class probabilities at the terminal leaf.
5. Provides cohort comparison: compares student values for actionable variables (`failures`, `absences`, `studytime`, `Dalc`, `Walc`) against test cohort mean and median baselines.

---

## 4. Empirical Findings & Result Interpretation

### 4.1 Permutation Importance Rankings (Held-Out Test Set)

Computed with 15 repeats on the 130-student test split (`scoring='f1_macro'`, `random_state=42`):

| Rank | Feature | Mean Macro F1 Drop | Std Dev | Educational / Analytical Significance |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `failures` | **+0.0629** | ±0.0186 | Past academic failure count is the single most vital predictor of student performance tier. |
| **2** | `Mjob` | **+0.0437** | ±0.0094 | Mother's employment status strongly guides intermediate classification branches. |
| **3** | `freetime` | **+0.0286** | ±0.0241 | Free time after school balances study and extracurricular commitments. |
| **4** | `romantic` | **+0.0225** | ±0.0124 | Romantic involvement flags emotional and time-allocation shifts in secondary school. |
| **5** | `famrel` | **+0.0166** | ±0.0119 | Quality of family relationships provides strong baseline stability. |
| **6** | `Fjob` | **+0.0148** | ±0.0066 | Father's employment sector provides secondary socio-economic context. |
| **7** | `absences` | **+0.0144** | ±0.0123 | School absenteeism directly impairs test-set tier classification when shuffled. |
| **8** | `Pstatus` | **+0.0112** | ±0.0044 | Parent cohabitation status ('Together' vs 'Apart') contributes consistent predictive signal. |
| **9** | `Medu` | **+0.0107** | ±0.0151 | Mother's education level provides long-term academic environment cues. |
| **10** | `sex` | **+0.0107** | ±0.0054 | Student gender carries slight predictive partitioning within this historical dataset. |

### 4.2 Interpreting Importance Value Signs

- **Positive Importance ($> 0$):** Shuffling the feature degrades the model's test Macro F1. The larger the positive drop, the more dependent the model is on that feature for generalization.
- **Zero / Near-Zero Importance ($\approx 0$):** Shuffling has virtually no effect on Macro F1 (e.g., `paid`, `famsup`, `nursery`, `higher`). The model does not rely on these features for unseen predictions, even if they appeared in earlier exploratory analyses.
- **Negative Importance ($< 0$):** Shuffling the feature slightly *improves* test performance (e.g., `reason` $-0.0388$, `Dalc` $-0.0164$, `health` $-0.0157$).
  > **Crucial Rule:** We do **not** convert negative values to zero. In finite test sets ($N=130$), a negative permutation score typically reflects features that the tree overfit on during training splits, where injecting random noise actually eliminates spurious branching errors.

### 4.3 Native Tree MDI vs. Permutation Discrepancies

Comparing the two methodologies reveals important diagnostic insights:

| Feature | Permutation Rank (Test Set) | Native MDI Rank (Training Splits) | Diagnostic Insight |
| :--- | :---: | :---: | :--- |
| `failures` | **#1** (+0.0629) | **#2** (0.0893) | **Consistent Champion:** Crucial in both training splits and unseen test generalization. |
| `health` | **#28** (-0.0157) | **#1** (0.1029) | **Overfitting Flag:** Used heavily to partition training leaves, but provides no generalization benefit on test data. |
| `studytime` | **#14** (+0.0027) | **#3** (0.0798) | **Surrogate Split:** Highly split by tree during training, but collinear with absences and failures on test data. |
| `Mjob` | **#2** (+0.0437) | **#8** (0.0522) | **High Generalization Value:** Modest split frequency during training yields substantial generalization power on test set. |

---

## 5. Methodological & Ethical Constraints

### 5.1 Importance Does NOT Imply Causation
A statistical model's reliance on a feature does **not** indicate that manipulating that feature will directly alter student performance.
- **Correlation $\neq$ Intervention:** Shuffling `failures` incurs a large F1 drop because past failures indicate prolonged academic struggle. However, artificially passing a struggling student does not magically impart coursework mastery.
- **Confounding Variables:** Variables such as `Mjob` or `address` correlate with access to resources, tutoring, and family stability. Modifying the recorded value does not change the student's underlying socioeconomic reality.

### 5.2 Pitfalls of Correlated Features
When two features are collinear:
- Examples in this dataset include parental education (`Medu` and `Fedu`), or alcohol consumption (`Dalc` and `Walc`).
- When `Medu` is permuted, the model can still partially infer parental support from `Fedu`.
- Consequently, permutation importance can underestimate the true combined contribution of correlated feature clusters.

### 5.3 Finite Sample Size
The held-out evaluation dataset comprises 130 students ($Low=24$, $Medium=80$, $High=26$). While statistically sound for validation, standard deviations across 15 permutation repeats range from $\pm 0.004$ to $\pm 0.025$. Users must view rankings as probabilistic tiers rather than rigid decimal hierarchies.

---

## 6. Target Leakage Prevention Protocol

Academic integrity demands strict target leakage prevention:
1. **Target Derivation:** The target `performance_category` is derived from the final period exam mark `G3`:
   - `Low`: $G3 < 10$
   - `Medium`: $10 \le G3 < 15$
   - `High`: $G3 \ge 15$
2. **Complete Input Exclusion:**
   - Neither `G3` nor `performance_category` is ever permitted inside the input feature matrix $X$.
   - Any attempt to pass a DataFrame containing `G3` or `performance_category` to `validate_feature_schema()` raises an immediate `ValueError: Target leakage violation`.
   - Neither column is ever included in permutation importance calculations.

---

## 7. Interactive Streamlit Explainability Dashboard

The explainability module is exposed through the interactive multi-page Streamlit interface at:

`pages/3_Model_Explainability.py`

### Page Features:
1. **Top KPI Metrics:** Highlights top permutation feature, mean F1 drop, top native tree feature, and Gini reduction share.
2. **Interactive Slider:** Allows users to dynamically adjust the number of top features displayed (5 to 25).
3. **Tab 1: Global Permutation Importance:** Publication-quality horizontal bar chart with error bars, full metrics table, and one-click CSV export.
4. **Tab 2: Permutation vs Native MDI:** Side-by-side comparative charts with detailed explanations of why differences emerge.
5. **Tab 3: Local Prediction Explainer:** Case studies for Low, Medium, and High tier students showing leaf probabilities, traversed tree decision rules, and cohort benchmark tables.
6. **Tab 4: Methodology & Limitations:** Detailed documentation of safeguards, correlation vs causation guidance, and technical parameters.
7. **Performance Caching:** Precomputed results in `reports/day_24/` are loaded instantly via `@st.cache_data`, preventing slow recomputations on every widget rerun.

---

## 8. Verification & How to Rerun the Analysis

### 8.1 Running the Automated Test Suite
Execute the dedicated Day 24 explainability unit tests:

```powershell
python -m unittest tests/test_explainability.py -v
```

Execute all repository tests to ensure no regressions:

```powershell
python -m unittest discover -s tests -v
```

### 8.2 Recomputing Explainability Reports
To re-run the permutation importance calculations and regenerate all figures:

```powershell
python -c "from src.explainability import save_day24_outputs; save_day24_outputs()"
```

### 8.3 Launching the Streamlit Application
To launch the full interactive web application:

```powershell
streamlit run app.py
```

Navigate to **"3 Model Explainability"** in the sidebar to interact with the explainability dashboard.

---

## 9. Development Log & Audit Summary

| Checkpoint | Status | Execution Detail |
| :--- | :---: | :--- |
| **Model Artifact Inspection** | **PASSED** | Loaded `artifacts/student_performance_pipeline.joblib` (Full Pipeline: ColumnTransformer + DecisionTree). |
| **Metadata Audit** | **PASSED** | Verified `artifacts/student_performance_metadata.json` (30 features, G3 leakage excluded). |
| **Dataset Discovery** | **PASSED** | Authentic held-out test set `day_16_error_analysis.csv` ($N=130$) loaded. |
| **Target Leakage Safeguard** | **PASSED** | Confirmed `G3` and `performance_category` excluded; deliberate injection test caught by `validate_feature_schema`. |
| **Permutation Importance** | **PASSED** | Successfully executed with 15 repeats on test set (`f1_macro` scoring, seed 42). Top feature: `failures` (+0.0629). |
| **Native Tree MDI** | **PASSED** | Extracted from Decision Tree, summed to 1.0, aggregated from 56 one-hot features to 30 raw features. |
| **Local Interpretability** | **PASSED** | Traced exact tree split paths and cohort statistics for Low, Medium, and High test cases. |
| **Streamlit Page Integration** | **PASSED** | Created `pages/3_Model_Explainability.py` with multi-tab layout and cached fast loading. |
| **Colab Notebook** | **PASSED** | Created `notebooks/Day_24_Model_Explainability_Feature_Importance.ipynb`. |
| **Automated Testing** | **PASSED** | 10/10 Day 24 tests passed in 2.0s; 36/36 full project tests passed in 6.8s. |
