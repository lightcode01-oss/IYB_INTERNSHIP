# Day 22: Streamlit Cloud Deployment Guide
## YBI Foundation — Python AI & Data Science Internship
### Project: AI-Based Student Performance Prediction System

> **STATUS: Deployment instructions — not yet live**
> *This document provides step-by-step instructions for deploying the validated application to Streamlit Community Cloud. In accordance with safety policies, no cloud account logins or automated deployments are executed without user action.*

---

## 1. Hosting Platform Selection: Why Streamlit Community Cloud?

For deploying the **AI-Based Student Performance Prediction System**, **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io/)) was chosen as the premier hosting platform:

| Criteria | Streamlit Community Cloud | Traditional VPS / Docker | Cloud PaaS (Render / Heroku) |
| :--- | :---: | :---: | :---: |
| **Setup Overhead** | **Zero infrastructure** (pure GitHub integration) | High (Dockerfile, Nginx, Linux management) | Medium (Procfile, buildpacks) |
| **Cost** | **Free Tier** for public GitHub repositories | Paid or limited free compute | Free tiers sleep or have cold starts |
| **Continuous Delivery** | **Automatic hot reload** on every `git push` | Requires CI/CD pipeline | Requires webhook setup |
| **Model Serving Fit** | Ideal for lightweight ML models (1 GB RAM limit) | Customizable RAM/CPU | 512 MB – 1 GB RAM limits |
| **Security & Secrets** | Built-in Secrets Management UI | `.env` files / environment variables | Platform environment variables |

### Key Platform Fit:
Our model pipeline (`artifacts/student_performance_pipeline.joblib`) is only **20.6 KB** and memory footprint during inference is **< 150 MB**, making Streamlit Community Cloud an ideal, reliable, and cost-free hosting environment.

---

## 2. Repository & Artifact Prerequisites

Before initiating cloud deployment, verify that your repository contains the following:

```text
IYB_internship/
├── app.py                                   # Streamlit application entry point
├── requirements.txt                         # Production runtime dependencies
├── .streamlit/
│   └── config.toml                          # Server and theme configuration
├── src/
│   └── monitoring.py                        # Operational telemetry module
├── artifacts/                               # Pre-trained ML pipeline and metadata
│   ├── student_performance_pipeline.joblib  # Scikit-learn champion pipeline (20.6 KB)
│   └── student_performance_metadata.json    # Pipeline schema & validation metrics
└── .gitignore                               # Privacy, logs & secrets protection
```

### Critical Artifact Handling:
- **Pipeline Size:** 20,595 bytes (**20.6 KB**).
- **Git Tracking Requirement:** Because Streamlit Community Cloud clones the GitHub repository at build time, the model artifact must be present in the repository for the app to deserialize it out-of-the-box.
- **Stage Model Artifacts (if policy allows):**
  ```bash
  git add artifacts/student_performance_pipeline.joblib
  git add artifacts/student_performance_metadata.json
  ```
- **Alternative (External Storage):** If company policy strictly prohibits `.joblib` files in Git, store the file in an AWS S3 bucket or GitHub Release Asset and add an initialization script (`download_artifacts.py`) called at the start of `app.py`.

---

## 3. Exact Deployment Settings

When creating the app on Streamlit Community Cloud, enter the following exact settings:

| Setting Field | Value / Configuration |
| :--- | :--- |
| **Repository** | `your-github-username/IYB_internship` (or your chosen repository name) |
| **Branch** | `main` (or `master`) |
| **Main file path** | `app.py` |
| **App URL (Custom Subdomain)** | `student-performance-ai` (results in: `https://student-performance-ai.streamlit.app`) |
| **Advanced settings (Python Version)** | `3.11` or `3.12` |
| **Secrets (Environment Variables)** | *None required* (zero external credentials needed) |

---

## 4. Step-by-Step Cloud Deployment Procedure

Follow these manual steps to launch the app:

### Step 1: Push Local Commits to Your GitHub Repository
Ensure your local branch is up to date and pushed to GitHub:
```bash
# Verify git status
git status

# If remote is not yet added:
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main

# Stage and commit Day 22 updates
git add notebooks/Day_22_Streamlit_App_Deployment.ipynb
git add docs/Day_22_Streamlit_Deployment_Guide.md
git add app.py
git add .streamlit/config.toml
git add artifacts/student_performance_pipeline.joblib
git add artifacts/student_performance_metadata.json
git commit -m "Day 22: Deploy student performance prediction app"

# Push to GitHub
git push -u origin main
```

### Step 2: Sign In to Streamlit Community Cloud
1. Navigate to **[https://share.streamlit.io/](https://share.streamlit.io/)**.
2. Click **"Continue with GitHub"** and authenticate.
3. Grant Streamlit read access to your public or private repositories.

### Step 3: Create a New Application
1. Click the blue **"Create app"** (or **"New app"**) button on the top right.
2. Select **"I already have an app"**.
3. In the deployment form:
   - **Repository:** Select your GitHub repository.
   - **Branch:** Select `main` (or `master`).
   - **Main file path:** Type `app.py`.
   - **App URL:** Customize your app subdomain (e.g., `student-performance-prediction`).
4. Click **"Deploy!"**.

### Step 4: Monitor the Build Logs
1. Click the **"Manage app"** flyout icon on the bottom right.
2. Streamlit Cloud will:
   - Spin up an isolated Debian-based container.
   - Clone your GitHub repository.
   - Run `pip install -r requirements.txt`.
   - Launch `streamlit run app.py`.
3. If successful, the terminal output displays: `You can now view your Streamlit app in your browser.`

---

## 5. Post-Deployment Verification Checklist

Once the app is running live, execute the following verification checklist:

- [ ] **1. Cloud App Loads:** Navigate to `https://<your-subdomain>.streamlit.app`. Verify that the page title, sidebar, and tabs load cleanly.
- [ ] **2. Model Artifact Deserialization:** Verify that no red error box appears stating `Model artifact 'student_performance_pipeline.joblib' was not found`.
- [ ] **3. Sidebar Model Metrics:** Confirm that the sidebar displays:
  - Architecture: `Decision Tree`
  - 5-Fold CV Macro F1: `0.5237`
  - Test Accuracy: `53.85%`
  - Leakage Safeguard: `Active (G3 excluded)`
- [ ] **4. Single-Student Prediction Form:**
  - Select student attributes (e.g., Mother's education: 3, Study time: 2, Absences: 4).
  - Click **"Predict Performance Tier"**.
  - Verify that a performance tier (`High`, `Medium`, or `Low`) is displayed along with confidence bar charts.
- [ ] **5. Target Leakage Quarantine:** Confirm that the prediction form **never** asks for $G3$ or the target classification.
- [ ] **6. Batch Prediction via CSV:**
  - Switch to **"Batch CSV Prediction"** tab.
  - Upload `day_17_prediction_demo.csv` (with target columns removed).
  - Click **"Run Batch Prediction"** and verify that predictions and download buttons function.
- [ ] **7. Model Monitoring & Operational Telemetry:**
  - Switch to **"Model Monitoring & Analytics"** tab.
  - Verify that operational prediction counts increment and class distributions are charted.
  - Expand the **"Telemetry Audit Log"** expander and confirm that records show only anonymized IDs and timestamps without personal identifiers.
- [ ] **8. Subjective Feedback Submission:**
  - Click 👍 or 👎 on a prediction result and verify that the feedback confirmation appears without errors.

---

## 6. Common Deployment Errors and Resolutions

| Symptom / Cloud Log Message | Root Cause | Fix / Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: student_performance_pipeline.joblib` | The model artifact was not committed or pushed to GitHub. | Run `git add artifacts/` and push to GitHub, or use external storage initialization. |
| `ModuleNotFoundError: No module named 'xyz'` | Dependency missing from `requirements.txt`. | Ensure all imported libraries are listed in `requirements.txt` and push to GitHub. |
| `InconsistentVersionWarning: unpickle estimator` | Cloud scikit-learn version mismatch. | Verify `requirements.txt` contains `scikit-learn>=1.4.0,<2.0.0` to preserve binary tree schema. |
| `Memory limit exceeded (OOM)` | Container exceeded 1 GB RAM quota. | App cache (`@st.cache_resource`) loads model once. Ensure large datasets are not held in memory. |
| `Streamlit encountered an unexpected error` | Python version conflict or deprecated parameter. | In Streamlit Cloud settings, verify Python version is set to 3.11 or 3.12. |

---

## 7. App Updates, Maintenance, and Rollback

### How to Update After Code Changes:
Streamlit Community Cloud continuously monitors your GitHub repository branch:
1. Make code or UI improvements locally.
2. Commit and push: `git push origin main`.
3. Streamlit Cloud automatically detects the push, pulls changes, and refreshes the live app in seconds.

### How to Roll Back a Faulty Deployment:
If a commit introduces an issue in production:
```bash
# Revert to previous stable commit
git revert HEAD
git push origin main
```
Alternatively, in the Streamlit Cloud dashboard:
1. Click **"Manage app"** -> **"..." (Settings)**.
2. Click **"Reboot app"** to clear memory cache.

### How to Take the App Offline:
1. Open [share.streamlit.io](https://share.streamlit.io/).
2. Locate the app card.
3. Click the three dots `...` and select **"Delete"** or **"Pause app"**.

---

## 8. Responsible AI & Deployment Governance

1. **Student Privacy Protection:** The application collects zero student PII (names, emails, IDs). Telemetry logs record only statistical categories and random IDs.
2. **Subjective Feedback vs Ground Truth:** User ratings ('Helpful' / 'Unhelpful') reflect educator impressions. They are **never** used to compute model accuracy or precision.
3. **Ethical Decision-Support:** The hosted tool provides early exploratory guidance. It must **never** be used for punitive tracking, resource withholding, or automated grading.
