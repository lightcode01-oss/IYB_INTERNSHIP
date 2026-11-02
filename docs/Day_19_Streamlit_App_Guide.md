# Streamlit Prediction App — User and Developer Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System (Day 19)

---

## 1. Overview & Objectives

The **Streamlit Student Performance Prediction App** (`app.py`) provides an interactive, beginner-friendly web interface for educators, school counselors, and administrators to evaluate student performance tiers using the project's trained and tested machine-learning pipeline.

### Core Capabilities:
- **Interactive Single-Student Evaluation:** Input student demographic, familial, academic, and behavioral features via organized forms with intuitive labels and helper tooltips.
- **Instant Three-Tier Prediction:** Predicts student achievement tier:
  - **Low:** Final grade $G3 < 10$ (Early-warning indicator for academic intervention)
  - **Medium:** Final grade $10 \le G3 < 15$ (Standard academic progression)
  - **High:** Final grade $G3 \ge 15$ (Superior academic mastery)
- **Model-Estimated Class Probabilities:** Displays likelihood estimates for all three categories (`High`, `Low`, `Medium`) dynamically aligned with the pipeline's internal class order.
- **Batch CSV Inference:** Upload a multi-student CSV to receive batch predictions with a downloadable results file.
- **Target Leakage Safeguard:** The application strictly enforces the project's leakage prevention policy—**final grade `G3` and `performance_category` are never requested as prediction inputs**.

---

## 2. Prerequisites & Environment Setup

### System Requirements
- Python `3.10` or higher (tested on Python `3.14.2`)
- Streamlit `1.30.0` or higher (installed: `1.65.0`)
- scikit-learn `1.4.0` or higher
- pandas `2.0.0` or higher
- numpy `1.26.0` or higher
- joblib `1.3.0` or higher

### Installing Dependencies
From the repository root directory, install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Required Model Artifacts

The application loads the pre-trained, pre-tested pipeline without retraining. Ensure these files are present in the `artifacts/` directory relative to the repository root:

```
IYB_internship/
├── app.py
├── requirements.txt
├── artifacts/
│   ├── student_performance_pipeline.joblib   # Fitted ColumnTransformer + Decision Tree Classifier (20.6 KB)
│   └── student_performance_metadata.json     # Complete architecture and schema audit metadata (3.2 KB)
└── docs/
    └── Day_19_Streamlit_App_Guide.md         # This run guide
```

If the pipeline artifact is missing, execute the Day 17 notebook (`notebooks/Day_17_Student_Performance_Prediction_System.ipynb`) to fit and persist the model pipeline.

---

## 4. How to Launch the App Locally

From your terminal or PowerShell prompt, navigate to the repository root directory and run:

```bash
streamlit run app.py
```

### Expected Startup Output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## 5. Using the Application Interface

### A. Single Student Interactive Prediction Form
The input form is divided into four clean, logical categories:

1. **Student Demographics & School Profile:**
   - `school`: School attended ('GP' = Gabriel Pereira, 'MS' = Mousinho da Silveira)
   - `sex`: Gender ('F' = Female, 'M' = Male)
   - `age`: Student age (15–22)
   - `address`: Home location ('U' = Urban, 'R' = Rural)
   - `famsize`: Family size ('LE3' = $\le 3$, 'GT3' = $> 3$)
   - `Pstatus`: Cohabitation status ('T' = Living together, 'A' = Apart)

2. **Family Background & Parental Support:**
   - `Medu` & `Fedu`: Mother's and Father's education ($0$ = None, $1$ = Primary 4th grade, $2$ = 5th–9th grade, $3$ = Secondary, $4$ = Higher education)
   - `Mjob` & `Fjob`: Mother's and Father's occupation ('teacher', 'health', 'services', 'at_home', 'other')
   - `reason`: Reason for choosing school ('home', 'reputation', 'course', 'other')
   - `guardian`: Legal guardian ('mother', 'father', 'other')
   - `famrel`: Family relationship quality ($1$ = Very bad to $5$ = Excellent)

3. **Academic Habits & School Support:**
   - `traveltime`: Home to school travel time ($1$: $<15$ min, $2$: $15$–$30$ min, $3$: $30$ min to $1$ hr, $4$: $>1$ hr)
   - `studytime`: Weekly study hours ($1$: $<2$ hrs, $2$: $2$–$5$ hrs, $3$: $5$–$10$ hrs, $4$: $>10$ hrs)
   - `failures`: Past class failures ($0$ to $4$)
   - `schoolsup` & `famsup`: Extra school and family educational support ('yes', 'no')
   - `paid`: Extra paid tutoring ('yes', 'no')
   - `higher`: Plans for higher education ('yes', 'no')
   - `absences`: Total school absences count ($0$ to $93$)

4. **Extracurriculars, Social Life & Health:**
   - `activities`: Extracurricular activities ('yes', 'no')
   - `nursery`: Attended nursery school ('yes', 'no')
   - `internet`: Internet access at home ('yes', 'no')
   - `romantic`: In a romantic relationship ('yes', 'no')
   - `freetime`: Free time after school ($1$ = Very low to $5$ = Very high)
   - `goout`: Going out with friends ($1$ = Very low to $5$ = Very high)
   - `Dalc` & `Walc`: Workday and weekend alcohol consumption ($1$ = Very low to $5$ = Very high)
   - `health`: Current health rating ($1$ = Very bad to $5$ = Very good)

Click **Predict Performance** to compute the prediction.

### B. Prediction Results & Probabilities
- **Predicted Tier Banner:** A styled alert banner displays the outcome (`Low`, `Medium`, or `High`).
- **Probability Breakdown:** Visual progress bars display the model's estimated likelihood for each category, calculated using the decision tree leaf node distributions.

### C. Batch CSV Prediction
- Select the **Batch CSV Prediction** tab in the app.
- Upload a `.csv` file containing the 30 required feature columns.
- The app validates column presence and orders them to match the pipeline.
- View preview predictions and click **Download Predictions CSV** to save results.

---

## 6. Troubleshooting & Common Issues

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **"Model artifact not found"** | Missing `artifacts/student_performance_pipeline.joblib` | Run `notebooks/Day_17_Student_Performance_Prediction_System.ipynb` to regenerate the artifact. |
| **Port 8501 already in use** | Another Streamlit process is running | Run `streamlit run app.py --server.port 8502` to use an alternate port. |
| **Unpickling / Deserialization error** | Incompatible scikit-learn version | Ensure your environment has `scikit-learn>=1.4.0` installed matching the training environment. |
| **Missing feature error in batch CSV** | Uploaded CSV lacks one of the 30 columns | Check column names against the input schema specification in Section 5. |

---

## 7. Educational Purpose & Limitations

1. **Non-Causal Diagnostic Tool:** The predictions reflect empirical correlations identified in historical Portuguese secondary school survey data. They do **not** represent causal laws. For example, changing a student's `freetime` or `studytime` does not guarantee an immediate change in academic performance.
2. **Early Warning, Not Deterministic Destiny:** Predictions must be treated as **exploratory early indicators**, never as certainties or deterministic judgments.
3. **Strict Ethical Guardrail:** The system must **never** be used as the sole basis for punitive actions, school tracking, course denial, or resource withholding. Its sole intended purpose is to identify at-risk students who may benefit from proactive academic tutoring and counseling support.
