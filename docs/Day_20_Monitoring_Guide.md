# Model Monitoring & Prediction Logging Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System (Day 20)

---

## 1. Overview & Purpose

Post-deployment model monitoring is an essential discipline in responsible machine learning. Once an ML model is exposed to users, system owners must observe its operational telemetry, prediction distributions, and user interactions over time to detect potential distribution shifts, usage spikes, or unexpected system behavior.

### Day 20 Monitoring Scope:
- **Anonymous Prediction Event Logging:** Safely records essential operational telemetry for each prediction without collecting identifying information.
- **Subjective User Feedback Interface:** Allows end users to optionally express whether a model estimate was helpful or plausible, without conflating subjective impressions with ground-truth accuracy.
- **Operational Dashboard:** An integrated Streamlit monitoring tab visualizing prediction frequencies, average class probabilities, feedback counts, and recent telemetry events.
- **Strict Privacy Safeguards:** No Personally Identifiable Information (PII) is ever recorded, and target variable `G3` is strictly quarantined.

---

## 2. Privacy Architecture: What is Logged vs. Excluded

Student data in academic environments is sensitive. Our monitoring architecture enforces strict minimization principles:

| Telemetry Attribute | Logged? | Description / Rationale |
| :--- | :---: | :--- |
| **`timestamp`** | **YES** | ISO 8601 UTC timestamp recording when the inference took place. |
| **`prediction_id`** | **YES** | Anonymized unique hexadecimal identifier (e.g., `e557fdf5ee5d`) linking events to feedback. |
| **`model_id`** | **YES** | Identifier of the active pipeline architecture (e.g., `Decision Tree v1.0`). |
| **`predicted_category`** | **YES** | The model's categorical forecast (`High`, `Medium`, or `Low`). |
| **`proba_High`, `proba_Low`, `proba_Medium`** | **YES** | Estimated class probability distribution from the model. |
| **`feedback`** | **YES** | Optional subjective user rating (`Helpful`, `Neutral`, `Unhelpful`, or `None`). |
| **Student Names / IDs / Contact Info** | **NEVER** | Personally Identifiable Information is strictly excluded by design. |
| **Final Grade `G3`** | **NEVER** | Quarantined to prevent target leakage and preserve early-warning integrity. |
| **Input Features (Demographics/Behaviors)** | **OPT-IN** | Excluded by default. Only recorded if the practitioner explicitly enables feature logging. |

---

## 3. Storage Location & Git Protection

### Local File Location
All prediction events are stored in a local, standard CSV file:
```
data/monitoring/prediction_events.csv
```

### Git Protection (`.gitignore`)
Local operational telemetry should **never** be committed to version control. The repository `.gitignore` explicitly excludes monitoring data:
```gitignore
# Local monitoring logs and data
data/monitoring/
*.log
*.sqlite
*.sqlite3
```

---

## 4. How to Use & Configure Monitoring in Streamlit

### A. Enabling or Disabling Logging
In the application sidebar (`app.py`), users can control telemetry:
- **Enable Prediction Logging:** Checked by default. When unchecked, no data is written to disk.
- **Log Input Features (Opt-in):** Unchecked by default. When enabled, non-target input features are included in the telemetry record for input drift inspection.
- **Privacy Notice Expander:** Explains storage paths and data minimization rules.

### B. Viewing the Monitoring Dashboard
1. Open the application (`streamlit run app.py`).
2. Navigate to the **Model Monitoring & Analytics** tab.
3. If no predictions have been run, an informative notice is shown.
4. Once predictions are generated, the dashboard displays:
   - **Total Predictions Logged:** Overall count of inference events.
   - **Top Predicted Tier:** Dominant achievement category and proportion.
   - **Subjective Feedback Count:** Total feedback submissions and participation rate.
   - **Category Distribution Chart:** Bar chart and frequency table for `Low`, `Medium`, and `High`.
   - **Mean Estimated Probabilities:** Average model confidence across all processed students.
   - **Recent Anonymous Prediction Events:** Scrollable table showing the 10 most recent predictions.

### C. Providing Optional Feedback
Immediately following any single-student prediction, the app renders a subjective feedback module:
- **"👍 Plausible / Useful"**
- **"👎 Doubtful / Unhelpful"**
- **"➖ Neutral / Unsure"**

Submitting feedback immediately updates the corresponding event in `prediction_events.csv` without rerunning the model.

---

## 5. Critical Methodological Distinctions

### Subjective Feedback $\ne$ Verified Model Accuracy
> **Crucial Rule:** User feedback represents a subjective human impression from an educator or student. It does **not** represent verified ground truth.
- A user marking a prediction as "Helpful" does not prove that the student will achieve that final grade.
- A user marking a prediction as "Doubtful" does not prove the model is wrong.
- **Real model evaluation metrics (Accuracy, Precision, Recall, Macro F1) must NEVER be computed from subjective feedback.** True performance evaluation requires authoritative, verified examination grades collected through an institutional audit.

### Prediction Class Distributions $\ne$ Model Drift
- A change in the proportion of `High`, `Medium`, or `Low` predictions over time indicates an operational shift in incoming queries.
- It is a **monitoring signal** for human inspection, **not proof** that the model has degraded or that concept drift has occurred.

---

## 6. How to Launch and Test the App

From the repository root directory:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Run unit tests
python -m unittest tests/test_monitoring.py

# 3. Launch Streamlit application
streamlit run app.py
```

### Resetting Monitoring Logs
To clear logs during local testing:
- Click the **Clear Logged Prediction Events** button inside the **Model Monitoring & Analytics** tab in the Streamlit UI, or
- Delete `data/monitoring/prediction_events.csv` directly.
