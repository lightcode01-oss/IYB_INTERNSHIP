"""
Deployment Readiness & Reproducibility Test Suite
Day 21: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Validates:
1. Artifact presence, integrity, and file size boundaries.
2. Metadata schema consistency and target leakage prevention.
3. Pipeline deserialization and test inference reproducibility.
4. Feature schema concordance between app and trained pipeline.
5. Dependency specification in requirements.txt.
6. Git safety rules in .gitignore.
7. Clean bytecode compilation of application and modules.
8. Headless Streamlit smoke test execution via AppTest API.
"""

import os
import sys
import json
import py_compile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestDeploymentReadiness(unittest.TestCase):
    """Test suite validating production deployment readiness and release integrity."""

    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = ROOT_DIR / "artifacts"
        cls.pipeline_path = cls.artifacts_dir / "student_performance_pipeline.joblib"
        cls.metadata_path = cls.artifacts_dir / "student_performance_metadata.json"
        cls.requirements_path = ROOT_DIR / "requirements.txt"
        cls.gitignore_path = ROOT_DIR / ".gitignore"
        cls.app_path = ROOT_DIR / "app.py"

    def test_01_model_artifact_exists_and_size_safe(self):
        """Verify model pipeline artifact exists, is non-empty, and under 25MB Git warning limit."""
        self.assertTrue(self.pipeline_path.exists(), "Model pipeline artifact not found.")
        file_size_bytes = self.pipeline_path.stat().st_size
        file_size_kb = file_size_bytes / 1024
        self.assertGreater(file_size_bytes, 1024, "Artifact file is suspiciously small (<1KB).")
        # Ensure under 25 MB (GitHub recommends <25 MB for regular git tracking)
        self.assertLess(file_size_kb, 25 * 1024, f"Artifact is too large for git tracking: {file_size_kb:.1f} KB")

    def test_02_metadata_artifact_exists_and_schema_valid(self):
        """Verify metadata JSON exists, parses cleanly, and confirms G3 exclusion."""
        self.assertTrue(self.metadata_path.exists(), "Metadata artifact not found.")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("pipeline_schema", meta)
        features = meta["pipeline_schema"]["input_features"]
        self.assertEqual(len(features), 30, f"Expected 30 predictor features, got {len(features)}")
        self.assertNotIn("G3", features, "Critical leakage: G3 must NOT be in predictor features.")
        self.assertNotIn("performance_category", features, "Target must NOT be in predictor features.")

    def test_03_pipeline_deserialization_and_inference(self):
        """Verify model deserializes and runs inference on a synthetic valid input record."""
        pipeline = joblib.load(self.pipeline_path)
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "predict_proba"))
        self.assertEqual(len(pipeline.feature_names_in_), 30)

        # Create valid synthetic row with default values
        synthetic_record = {
            "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
            "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other",
            "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
            "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
            "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes",
            "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
            "Walc": 1, "health": 4, "absences": 4
        }
        df_input = pd.DataFrame([synthetic_record])[list(pipeline.feature_names_in_)]
        pred = pipeline.predict(df_input)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], ["High", "Low", "Medium"])

        probas = pipeline.predict_proba(df_input)
        self.assertEqual(probas.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(probas)), 1.0, places=4)

    def test_04_app_feature_schema_matches_pipeline(self):
        """Verify the feature schema in metadata matches the serialized pipeline exactly."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pipeline = joblib.load(self.pipeline_path)
        meta_features = meta["pipeline_schema"]["input_features"]
        pipe_features = list(pipeline.feature_names_in_)
        self.assertEqual(meta_features, pipe_features)

    def test_05_requirements_txt_completeness(self):
        """Verify requirements.txt specifies all essential deployment packages."""
        self.assertTrue(self.requirements_path.exists())
        with open(self.requirements_path, "r", encoding="utf-8") as f:
            req_content = f.read().lower()

        essential_pkgs = ["streamlit", "scikit-learn", "pandas", "numpy", "joblib"]
        for pkg in essential_pkgs:
            self.assertIn(pkg, req_content, f"Missing essential dependency '{pkg}' in requirements.txt")

    def test_06_gitignore_safeguards(self):
        """Verify .gitignore includes critical patterns for privacy, logs, and secrets."""
        self.assertTrue(self.gitignore_path.exists())
        with open(self.gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()

        required_patterns = ["data/monitoring/", "*.log", ".env", "secrets.toml"]
        for pat in required_patterns:
            self.assertIn(pat, gi_content, f"Missing safeguard pattern '{pat}' in .gitignore")

    def test_07_py_compile_syntax(self):
        """Verify app.py, monitoring module, and tests compile cleanly without syntax errors."""
        modules_to_test = [
            self.app_path,
            ROOT_DIR / "src" / "monitoring.py",
            ROOT_DIR / "src" / "evaluation_dashboard.py",
            ROOT_DIR / "pages" / "2_Model_Performance_Dashboard.py",
            ROOT_DIR / "tests" / "test_monitoring.py",
            ROOT_DIR / "tests" / "test_evaluation_dashboard.py"
        ]
        for mod in modules_to_test:
            self.assertTrue(mod.exists(), f"Module {mod} not found.")
            py_compile.compile(str(mod), doraise=True)

    def test_08_streamlit_app_test_execution(self):
        """Run headless Streamlit AppTest smoke test to ensure app starts without unhandled errors."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1 not available in this Streamlit version.")

        app_test = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        self.assertFalse(app_test.exception, f"AppTest raised an unhandled exception: {app_test.exception}")


if __name__ == "__main__":
    unittest.main()
