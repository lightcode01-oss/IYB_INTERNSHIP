# Day 21: Deployment Guide & Release Readiness
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System

---

## 1. Overview & Deployment Scope

This guide provides step-by-step instructions for preparing, verifying, and deploying the **AI-Based Student Performance Prediction System** (`app.py`).

### Deployment Scope:
- **Application Architecture:** Streamlit interactive web application with single-student inference, batch CSV scoring, privacy-conscious operational monitoring, and ethical disclaimers.
- **Model Pipeline:** Scikit-learn tuned Decision Tree classifier with categorical and numeric transformers encapsulated in a single serialized joblib pipeline (`artifacts/student_performance_pipeline.joblib`, 20.6 KB).
- **Target Formulation:** 3-tier academic classification (`Low`: $G3 < 10$, `Medium`: $10 \le G3 < 15$, `High`: $G3 \ge 15$) with strict $G3$ target leakage prevention.
- **Release Status:** **Deployment Preparation & Release Readiness** (fully verified locally and ready for hosting platform configuration; not yet publicly deployed to cloud).

---

## 2. Clean-Environment Setup Instructions

To ensure absolute reproducibility across different operating systems (Windows, Linux, macOS) or cloud platforms, follow this clean-environment setup.

### Step 1: Clone or Open the Repository
Open a terminal in the root folder of the repository:
```bash
git clone <repository-url>
cd IYB_internship
```

### Step 2: Create an Isolated Python Virtual Environment
Use Python 3.10, 3.11, 3.12, 3.13, or 3.14:

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS / Bash:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Upgrade Package Installer
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies from `requirements.txt`
```bash
pip install -r requirements.txt
```

---

## 3. Verifying Artifacts & System Pre-requisites

Before starting the web server, confirm that the necessary pipeline artifacts are present and intact:

```bash
# Verify artifact presence and size (Windows PowerShell)
Get-ChildItem -Path artifacts

# Verify artifact presence (Linux/macOS)
ls -lh artifacts/
```

### Required Artifact Checklist:
1. `artifacts/student_performance_pipeline.joblib`:
   - Pre-trained scikit-learn pipeline (Imputer + OneHotEncoder/StandardScaler + DecisionTreeClassifier).
   - Expected size: ~20.6 KB.
   - Requires 30 predictor features (strictly excludes $G3$).
2. `artifacts/student_performance_metadata.json`:
   - Metadata containing dataset origin, 5-fold cross-validation metrics, held-out test metrics, and feature schemas.
3. `data/monitoring/`:
   - Local directory initialized on demand by `src/monitoring.py` for anonymous operational telemetry.

### Artifact Regeneration (If Missing):
If the model artifact is absent, regenerate it by executing Day 17:
```bash
jupyter execute notebooks/Day_17_Student_Performance_Prediction_System.ipynb
```
*Note: Do not retrain or alter hyperparameters unless authorized.*

---

## 4. Running the App Locally

Launch the Streamlit web server directly from the repository root:

```bash
streamlit run app.py
```

### Expected Startup Output:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Once loaded in your browser:
1. **Single Student Prediction:** Fill in student survey indicators (study time, parental education, absences) and click **Predict Performance Tier**.
2. **Batch CSV Prediction:** Upload a sample CSV (e.g., `day_17_prediction_demo.csv` with target columns omitted) to obtain multi-student estimates with probability scores and downloadable results.
3. **Model Monitoring & Analytics:** Check the operational telemetry dashboard to observe aggregated class distributions, average model confidence, and optional subjective feedback without exposing student records.

---

## 5. Automated Pre-Release Validation Checks

Before committing code or preparing a release, run the automated test suites:

```bash
# Run all project test suites (Monitoring + Deployment Readiness)
python -m unittest discover tests
```

### Test Coverage Summary:
- `tests/test_monitoring.py` (10 tests):
  - Storage initialization and standard CSV headers.
  - Anonymous prediction logging with probability scores.
  - Strict PII and $G3$ leakage filtering.
  - Subjective feedback recording and non-blocking error handling.
- `tests/test_deployment_readiness.py` (8 tests):
  - Pipeline file size boundaries (<25 MB Git threshold).
  - Metadata schema verification (30 features, no target leakage).
  - Live model deserialization and synthetic inference.
  - `requirements.txt` and `.gitignore` policy compliance.
  - Bytecode compilation and Streamlit `AppTest` headless smoke test.

---

## 6. Hosting Platforms & Deployment Options

When ready to publish the app to an online platform, consider the following manual deployment options:

### Option A: Streamlit Community Cloud (Recommended for Demos)
- **Prerequisites:** GitHub repository containing `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `artifacts/`.
- **Steps:**
  1. Push your repository to GitHub.
  2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
  3. Click **"New app"**.
  4. Select repository, branch (`main` or `master`), and main file path (`app.py`).
  5. Click **"Deploy"**.
- **Cold-Start Time:** ~1–2 minutes for initial dependency installation.
- **Resource Limits:** 1 GB RAM, free tier suitable for lightweight educational demos.

### Option B: Containerized Deployment (Docker)
For hosting on AWS ECS, Google Cloud Run, Azure Container Apps, or Render:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option C: Virtual Machine / Cloud VPS (Ubuntu / Debian / EC2)
1. Provision a lightweight Ubuntu 22.04 LTS instance.
2. Install Python 3.11+ and Nginx reverse proxy.
3. Configure `systemd` service for background app persistence:
   ```ini
   [Unit]
   Description=Student Performance Prediction Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/IYB_internship
   ExecStart=/home/ubuntu/IYB_internship/venv/bin/streamlit run app.py --server.port=8501 --server.headless=true
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

---

## 7. Artifact & Repository Safety Considerations

### A. Model Artifact Size & Tracking Policy
- The model artifact `student_performance_pipeline.joblib` is **20.6 KB** in size.
- Because it is far below GitHub's 100 MB hard file limit and 25 MB warning threshold, it is safe to commit directly to Git for one-click hosting on platforms like Streamlit Community Cloud.
- **Alternative (External Storage):** In enterprise environments prohibiting binary model weights in Git, upload `student_performance_pipeline.joblib` to an AWS S3 bucket, Google Cloud Storage, or GitHub Release Asset, and retrieve it using an initialization script (`download_artifacts.py`) prior to launching `streamlit run app.py`.

### B. Data & Monitoring Privacy Protection
- `.gitignore` strictly excludes local operational logs (`data/monitoring/`), `.log` files, SQLite databases, and environment secrets (`.streamlit/secrets.toml`, `.env*`).
- **Telemetry Policy:** The deployed app logs anonymous prediction event IDs, predicted categories, and leaf probabilities. Raw student demographics, names, student IDs, and grades are **never** stored in public logs.
- The monitoring dashboard displays aggregated frequency counts and wraps individual anonymous telemetry events in a collapsed operator drawer.

---

## 8. Common Troubleshooting & Error Resolution

| Issue / Error | Root Cause | Resolution |
| :--- | :--- | :--- |
| `Model artifact 'student_performance_pipeline.joblib' was not found` | The artifact file is missing from `artifacts/`. | Verify the file exists or re-run Day 17 to export the trained pipeline. |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment is not activated or dependencies not installed. | Activate virtual environment (`.\venv\Scripts\Activate.ps1` or `source venv/bin/activate`) and run `pip install -r requirements.txt`. |
| `InconsistentVersionWarning: Trying to unpickle estimator...` | Scikit-learn version in deployment differs from serialization environment. | Pin `scikit-learn` in `requirements.txt` to the compatible major version (`>=1.4.0,<2.0.0`). |
| `Missing required columns: {'absences', ...}` | Uploaded CSV in batch prediction tab is missing predictor columns. | Ensure CSV contains all 30 expected feature headers (refer to `student_performance_metadata.json`). |
| `Permission denied: data/monitoring/prediction_events.csv` | File locked by another process or read-only filesystem. | The app handles I/O errors gracefully in non-blocking mode; for write access, ensure the directory has write permissions. |

---

## 9. Deployment Readiness Verdict

- **Code Integrity:** PASSED (clean compilation, zero syntax errors).
- **Artifact Compatibility:** PASSED (deserialization and inference verified).
- **Unit & Integration Tests:** PASSED (18/18 tests passed).
- **Public Hosting Status:** **PREPARED / READY FOR RELEASE** (no unauthorized external deployment performed).
