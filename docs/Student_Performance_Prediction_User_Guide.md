# Student Performance Prediction System — User Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System

---

## 1. Project Purpose & System Overview

The **AI-Based Student Performance Prediction System** provides educators, academic counselors, and institutional administrators with an early-warning diagnostic mechanism to forecast student academic achievement before final examinations occur.

### Target Formulation
Student academic performance is categorized into three discrete performance tiers based on the final grade $G3$ ($0 \le G3 \le 20$):
- **Low:** $G3 < 10$ (Students at risk of failing or requiring immediate foundational academic intervention)
- **Medium:** $10 \le G3 < 15$ (Students on a satisfactory, standard academic trajectory)
- **High:** $G3 \ge 15$ (Students exhibiting superior academic proficiency and mastery)

### Target Leakage Elimination
> **Important:** The final grade `G3` is used solely to define the ground-truth target category during model training. **`G3` is strictly excluded from all prediction inputs and the inference schema.** The model relies entirely on baseline demographic, familial, social, and academic behavioral attributes. Intermediate assessment scores ($G1, G2$) are not present in this dataset.

---

## 2. Environment & System Requirements

The prediction workflow requires Python 3.10+ and standard data science libraries. Because scikit-learn serialized estimators (`.joblib`) rely on internal object structures, using aligned package versions across training and inference environments is strongly recommended.

### Core Dependencies
```bash
pip install "scikit-learn>=1.4.0" "pandas>=2.0.0" "numpy>=1.26.0" "joblib>=1.3.0"
```

| Package | Tested Version | Role |
| :--- | :--- | :--- |
| **Python** | `3.14.2` | Runtime environment |
| **scikit-learn** | `1.7.dev0` / `1.4+` | Pipeline, transformers, and classifier execution |
| **pandas** | `3.0.3` / `2.0+` | Tabular data manipulation and schema ordering |
| **numpy** | `2.4.6` / `1.26+` | Array operations and probability thresholding |
| **joblib** | `1.5.3` / `1.3+` | Deserialization of the fitted pipeline artifact |

---

## 3. Model Artifacts

The system is encapsulated within two production-ready artifacts stored in the `artifacts/` directory:

1. **`artifacts/student_performance_pipeline.joblib`**
   - The serialized, end-to-end scikit-learn `Pipeline`.
   - Contains both the fitted `ColumnTransformer` (handling numerical/ordinal passthrough and categorical one-hot encoding) and the champion **Decision Tree Classifier** (`criterion='gini'`, `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=10`).
   - Self-contained: incoming raw feature records require no manual scaling or manual dummy encoding prior to calling `.predict()`.

2. **`artifacts/student_performance_metadata.json`**
   - Complete machine-readable audit trail containing:
     - Dataset identification and sample sizes ($N=649$ total, $N=519$ train, $N=130$ test)
     - Optimal hyperparameters and selection criteria (5-Fold CV Macro F1: $0.5237 \pm 0.0389$)
     - Held-out test set metrics (Accuracy: $0.5385$, Macro F1: $0.4498$)
     - Canonical list and order of all 30 input features
     - Model class labels in actual internal order (`['High', 'Low', 'Medium']`)
     - Target leakage prevention confirmation (`g3_excluded_from_features: true`)

---

## 4. Input Schema Specification

The model expects exactly **30 features** representing student attributes. Neither `G3` nor `performance_category` should be provided.

### Feature Taxonomy (30 Features Total)

#### A. Numerical Features (2 features)
- **`age`** *(integer/float)*: Student age in years ($\ge 0$, typically 15–22).
- **`absences`** *(integer/float)*: Number of school absences ($\ge 0$, typically 0–93).

#### B. Ordinal Features (11 features)
- **`Medu`** *(integer, 0–4)*: Mother's education level ($0$: none, $1$: primary (4th grade), $2$: 5th to 9th grade, $3$: secondary education, $4$: higher education).
- **`Fedu`** *(integer, 0–4)*: Father's education level ($0$: none, $1$: primary, $2$: 5th to 9th grade, $3$: secondary, $4$: higher education).
- **`traveltime`** *(integer, 1–4)*: Home to school travel time ($1$: $<15$ min, $2$: $15$–$30$ min, $3$: $30$ min to $1$ hour, $4$: $>1$ hour).
- **`studytime`** *(integer, 1–4)*: Weekly study time ($1$: $<2$ hours, $2$: $2$–$5$ hours, $3$: $5$–$10$ hours, $4$: $>10$ hours).
- **`failures`** *(integer, 0–4)*: Number of past class failures ($n$ if $0 \le n < 3$, else $4$).
- **`famrel`** *(integer, 1–5)*: Quality of family relationships (from $1$: very bad to $5$: excellent).
- **`freetime`** *(integer, 1–5)*: Free time after school (from $1$: very low to $5$: very high).
- **`goout`** *(integer, 1–5)*: Going out with friends (from $1$: very low to $5$: very high).
- **`Dalc`** *(integer, 1–5)*: Workday alcohol consumption (from $1$: very low to $5$: very high).
- **`Walc`** *(integer, 1–5)*: Weekend alcohol consumption (from $1$: very low to $5$: very high).
- **`health`** *(integer, 1–5)*: Current health status (from $1$: very bad to $5$: very good).

#### C. Binary Categorical Features (13 features)
- **`school`** *(string, 'GP' or 'MS')*: Student's school ('GP': Gabriel Pereira, 'MS': Mousinho da Silveira).
- **`sex`** *(string, 'F' or 'M')*: Student sex ('F': female, 'M': male).
- **`address`** *(string, 'U' or 'R')*: Student home address type ('U': urban, 'R': rural).
- **`famsize`** *(string, 'LE3' or 'GT3')*: Family size ('LE3': $\le 3$, 'GT3': $> 3$).
- **`Pstatus`** *(string, 'T' or 'A')*: Parent's cohabitation status ('T': living together, 'A': apart).
- **`schoolsup`** *(string, 'yes' or 'no')*: Extra educational support.
- **`famsup`** *(string, 'yes' or 'no')*: Family educational support.
- **`paid`** *(string, 'yes' or 'no')*: Extra paid classes within the course subject.
- **`activities`** *(string, 'yes' or 'no')*: Extra-curricular activities.
- **`nursery`** *(string, 'yes' or 'no')*: Attended nursery school.
- **`higher`** *(string, 'yes' or 'no')*: Wants to take higher education.
- **`internet`** *(string, 'yes' or 'no')*: Internet access at home.
- **`romantic`** *(string, 'yes' or 'no')*: In a romantic relationship.

#### D. Nominal Categorical Features (4 features)
- **`Mjob`** *(string)*: Mother's job ('teacher', 'health' care, civil 'services', 'at_home', 'other').
- **`Fjob`** *(string)*: Father's job ('teacher', 'health' care, civil 'services', 'at_home', 'other').
- **`reason`** *(string)*: Reason to choose school ('home': close to home, 'reputation': school reputation, 'course': course preference, 'other').
- **`guardian`** *(string)*: Student's guardian ('mother', 'father', 'other').

---

## 5. How to Make Predictions

### Python API Example

Below is a complete, self-contained Python script demonstrating how to load the pipeline, validate inputs, and make single-student and batch predictions:

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Load the trusted fitted pipeline
pipeline_path = Path("artifacts/student_performance_pipeline.joblib")
pipeline = joblib.load(pipeline_path)

# 2. Define the reusable prediction function
def predict_student_performance(student_data, fitted_pipeline):
    """
    Generate student performance predictions and estimated class probabilities.
    """
    if isinstance(student_data, dict):
        df_input = pd.DataFrame([student_data])
    elif isinstance(student_data, pd.DataFrame):
        df_input = student_data.copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(student_data).__name__}")

    expected_cols = list(fitted_pipeline.feature_names_in_)
    missing_cols = set(expected_cols) - set(df_input.columns)
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {sorted(missing_cols)}")

    unexpected_cols = set(df_input.columns) - set(expected_cols)
    if unexpected_cols:
        raise ValueError(f"Unexpected columns found: {sorted(unexpected_cols)}")

    df_ordered = df_input[expected_cols]
    preds = fitted_pipeline.predict(df_ordered)
    results_df = pd.DataFrame({"predicted_category": preds}, index=df_input.index)

    if hasattr(fitted_pipeline, "predict_proba"):
        probas = fitted_pipeline.predict_proba(df_ordered)
        for idx, cls_name in enumerate(fitted_pipeline.classes_):
            results_df[f"model_estimated_proba_{cls_name}"] = np.round(probas[:, idx], 4)

    return results_df

# 3. Example Single-Student Prediction (Dictionary Input)
sample_student = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "services", "Fjob": "teacher",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 3,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1,
    "Walc": 1, "health": 4, "absences": 2
}

prediction = predict_student_performance(sample_student, pipeline)
print("Single Student Prediction:")
print(prediction)
```

---

## 6. Batch Predictions

For institutional or classroom-level evaluations, pass a `pandas.DataFrame` containing multiple records:

```python
# Create batch DataFrame from list of student records
batch_students = pd.DataFrame([sample_student, sample_student])

# Generate batch predictions
batch_results = predict_student_performance(batch_students, pipeline)
print("\nBatch Predictions (Shape: {}):".format(batch_results.shape))
print(batch_results)
```

---

## 7. Input Validation Policy & Error Handling

To safeguard data integrity, user inputs should be validated prior to inference using the following established policy:

| Issue | Validation Action | Rationale |
| :--- | :--- | :--- |
| **Missing required feature** | Raises `ValueError` | The model cannot make principled predictions without complete feature sets. |
| **Unexpected feature column** | Raises `ValueError` | Prevents misaligned columns or unintended leakage of non-approved attributes. |
| **Null / NaN values** | Raises `ValueError` | No silent imputation is performed; missing values must be investigated by the user. |
| **Negative numeric values** | Raises `ValueError` | `age < 0` or `absences < 0` represent physical impossibilities and data corruption. |
| **Out-of-range ordinal values** | Raises `ValueError` | e.g., `studytime=8` violates the 1–4 survey definition. |
| **Unfamiliar categorical values** | Emits `UserWarning` | `OneHotEncoder(handle_unknown='ignore')` prevents pipeline crash, but accuracy may drop. |

---

## 8. Probability Interpretation & Ethical Safeguards

### Understanding Estimated Probabilities
- Columns titled `model_estimated_proba_High`, `model_estimated_proba_Low`, and `model_estimated_proba_Medium` represent **empirical frequency estimates** produced by the decision tree leaves.
- They are **model-estimated likelihoods**, not deterministic guarantees or personal certainties.

### Ethical Guidelines & Operational Boundaries
1. **Supportive, Not Punitive:** The system is engineered solely as an **early-warning diagnostic tool** to identify students who may benefit from proactive tutoring, mentoring, or counseling.
2. **Never for High-Stakes Sanctions:** The predictions must **never** be used as the sole or primary basis for punitive actions, denial of academic resources, or compulsory tracking.
3. **Non-Causal Associations:** Features such as parent education, study hours, and absences reflect statistical correlations in historical survey data, **not direct causal mechanisms**.
4. **Human-in-the-Loop:** All model outputs should be interpreted in conjunction with qualitative evaluations by qualified educators and counselors.
5. **Security Notice:** Serialized `.joblib` artifacts execute arbitrary code upon deserialization. Only load model artifacts from trusted, authenticated sources.

---

## 9. Model Performance Dashboard & Evaluation Reporting (Day 23)

To inspect authentic model validation metrics, cross-validation tuning comparisons, class-level classification reports, and test set error analyses:

1. Launch the interactive web application:
   ```bash
   streamlit run app.py
   ```
2. Navigate to **"2 Model Performance Dashboard"** in the multi-page sidebar (or click the **"Performance Dashboard"** navigation link).
3. Alternatively, launch the performance dashboard directly:
   ```bash
   streamlit run pages/2_Model_Performance_Dashboard.py
   ```
4. For comprehensive architectural details and operator instructions, consult [Day_23_Model_Performance_Dashboard_Guide.md](file:///c:/Users/ABHINAV%20DASH/Downloads/IYB_internship/docs/Day_23_Model_Performance_Dashboard_Guide.md).
