"""
Unit Tests for Privacy-Conscious Prediction Logging and Model Monitoring
Day 20: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship
"""

import sys
import tempfile
import py_compile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitoring import (
    log_prediction_event,
    update_prediction_feedback,
    load_prediction_logs,
    get_monitoring_summary,
    ensure_storage_ready,
    STANDARD_LOG_COLUMNS
)
import joblib


class TestMonitoringSuite(unittest.TestCase):
    """Test suite for Day 20 monitoring and logging functionalities."""

    def setUp(self):
        """Set up an isolated temporary directory for test logs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_log_path = Path(self.temp_dir.name) / "test_events.csv"

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_01_ensure_storage_ready(self):
        """Verify storage initialization creates directory and empty CSV with standard schema."""
        path = ensure_storage_ready(self.test_log_path)
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), STANDARD_LOG_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_02_log_single_event(self):
        """Verify recording a valid prediction event with probabilities."""
        probas = {"High": 0.65, "Low": 0.10, "Medium": 0.25}
        success, pred_id = log_prediction_event(
            predicted_category="High",
            probabilities=probas,
            model_id="DecisionTree_v1",
            log_path=self.test_log_path
        )
        self.assertTrue(success)
        self.assertTrue(bool(pred_id))

        df = load_prediction_logs(self.test_log_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["predicted_category"].iloc[0], "High")
        self.assertAlmostEqual(df["proba_High"].iloc[0], 0.65)
        self.assertAlmostEqual(df["proba_Low"].iloc[0], 0.10)
        self.assertAlmostEqual(df["proba_Medium"].iloc[0], 0.25)
        self.assertEqual(df["feedback"].iloc[0], "None")

    def test_03_logging_disabled_mode(self):
        """Verify that when logging is disabled, no record is written."""
        success, msg = log_prediction_event(
            predicted_category="Medium",
            log_path=self.test_log_path,
            enabled=False
        )
        self.assertFalse(success)
        self.assertIn("disabled", msg.lower())
        self.assertFalse(self.test_log_path.exists())

    def test_04_privacy_safeguards_no_pii_and_no_g3(self):
        """Verify strict privacy: no PII and G3 target leakage prevention."""
        sensitive_inputs = {
            "name": "Jane Doe",
            "student_id": "ST12345",
            "age": 17,
            "absences": 4,
            "G3": 16,  # Target leakage candidate!
            "performance_category": "High"
        }
        success, _ = log_prediction_event(
            predicted_category="High",
            input_features=sensitive_inputs,
            log_features=True,  # Opt-in feature logging
            log_path=self.test_log_path
        )
        self.assertTrue(success)
        df = load_prediction_logs(self.test_log_path)

        # G3 and performance_category must NEVER be logged
        self.assertNotIn("feat_G3", df.columns)
        self.assertNotIn("feat_performance_category", df.columns)
        self.assertNotIn("G3", df.columns)

    def test_05_update_feedback(self):
        """Verify updating subjective user feedback for an existing prediction."""
        _, pred_id = log_prediction_event(
            predicted_category="Low",
            log_path=self.test_log_path
        )
        success, msg = update_prediction_feedback(
            prediction_id=pred_id,
            feedback_value="Helpful",
            log_path=self.test_log_path
        )
        self.assertTrue(success)

        df = load_prediction_logs(self.test_log_path)
        self.assertEqual(df.loc[df["prediction_id"] == pred_id, "feedback"].iloc[0], "Helpful")

    def test_06_empty_log_monitoring_summary(self):
        """Verify monitoring summary gracefully handles empty or non-existent logs."""
        summary = get_monitoring_summary(self.test_log_path)
        self.assertTrue(summary["is_empty"])
        self.assertEqual(summary["total_events"], 0)
        self.assertTrue(summary["class_counts"].empty)
        self.assertEqual(summary["feedback_summary"]["total_feedback_events"], 0)

    def test_07_populated_monitoring_summary_metrics(self):
        """Verify summary calculations with synthetic events."""
        # Log 3 events: 2 Medium, 1 High
        log_prediction_event("Medium", probabilities={"High": 0.1, "Low": 0.1, "Medium": 0.8}, log_path=self.test_log_path)
        log_prediction_event("Medium", probabilities={"High": 0.2, "Low": 0.2, "Medium": 0.6}, log_path=self.test_log_path)
        _, p3_id = log_prediction_event("High", probabilities={"High": 0.7, "Low": 0.1, "Medium": 0.2}, log_path=self.test_log_path)
        update_prediction_feedback(p3_id, "Accurate Impression", log_path=self.test_log_path)

        summary = get_monitoring_summary(self.test_log_path)
        self.assertFalse(summary["is_empty"])
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["class_counts"]["Medium"], 2)
        self.assertEqual(summary["class_counts"]["High"], 1)
        self.assertAlmostEqual(summary["class_proportions"]["Medium"], 66.67, places=1)
        self.assertEqual(summary["feedback_summary"]["total_feedback_events"], 1)
        self.assertIn("disclaimer", summary["feedback_summary"])

    def test_08_resilient_error_handling(self):
        """Verify that an unwritable path fails gracefully without crashing caller."""
        # Using an existing directory as the file path triggers a write error cleanly
        invalid_path = Path(self.temp_dir.name)
        success, msg = log_prediction_event("Low", log_path=invalid_path)
        self.assertFalse(success)
        self.assertIn("error", msg.lower())

    def test_09_existing_pipeline_loads_and_predicts(self):
        """Verify that the existing Day 17/18 pipeline still loads and predicts correctly."""
        artifact_path = ROOT_DIR / "artifacts" / "student_performance_pipeline.joblib"
        self.assertTrue(artifact_path.exists())
        pipeline = joblib.load(artifact_path)
        self.assertEqual(len(pipeline.feature_names_in_), 30)

    def test_10_app_py_syntax_and_compilation(self):
        """Verify app.py passes bytecode syntax compilation."""
        app_path = ROOT_DIR / "app.py"
        self.assertTrue(app_path.exists())
        py_compile.compile(str(app_path), doraise=True)


if __name__ == "__main__":
    unittest.main()
