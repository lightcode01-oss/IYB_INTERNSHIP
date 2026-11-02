# Reproduce the environment and results

Use Python 3.14. From the repository root run `python scripts/setup.py`. This creates .venv and installs requirements-lock.txt without changing machine-wide packages. Activate .venv, then run `python -m pip check` and `python -m pytest -q`. Runtime-only, notebook/test and API dependency subsets are also available as requirements.txt, requirements-dev.txt and requirements-api.txt. The lock captures the actual environment used for validation on Windows; the CI workflow checks Linux compatibility separately.

The saved artifact is from scikit-learn 1.9.0, which is pinned for compatible deserialization. Never load an untrusted joblib file. The cached public dataset and split_indices.json make serving tests independent of UCI availability. Existing training notebooks use UCI’s fetch function and need internet access when rerun.

For a fresh notebook execution, run `python scripts/execute_notebooks.py notebooks/Day_25_Explainability_Validation.ipynb`. You can list multiple notebook paths. The script runs each in a separate kernel from the repository root and retains genuine outputs or failure traces. Days 14–17 reproduce cross-validation, tuning, evaluation and artifact creation in that order. Training reruns may change generated files and timestamps; inspect differences before committing. No test data is used for model selection.

In Colab, open a daily notebook from this repository. Its setup locates or clones the repository. Install requirements-dev.txt in the notebook’s active environment (plus requirements-api.txt for API notebooks), then run cells in order. The checked-in execution evidence was generated locally, not claimed as a Colab runtime run.
