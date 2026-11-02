"""
Unit and Integration Tests for Model Performance Dashboard
Day 23: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Validates:
1. Historical evaluation artifact discovery and schema validity (Days 14–18).
2. Resilient data loading and graceful error handling for missing/malformed files.
3. Confusion matrix computation and statistical parity with classification report.
4. Error severity classification (adjacent vs non-adjacent misclassifications).
5. Dynamic report generation (Markdown & CSV formats).
6. Headless Streamlit smoke test for pages/2_Model_Performance_Dashboard.py.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_dashboard import (
    find_evaluation_files,
    load_cv_results,
    load_baseline_vs_tuned,
    load_tuning_results,
    load_classification_report,
    load_error_analysis,
    load_model_selection_report,
    compute_confusion_matrix,
    compute_error_breakdown,
    generate_markdown_report,
    generate_csv_summary
)


class TestEvaluationDashboard(unittest.TestCase):
    """Test suite for evaluation dashboard data ingestion, computation, and rendering."""

    def test_01_all_historical_evaluation_files_discovered(self):
        """Confirm all expected evaluation artifacts from Days 14-18 exist."""
        files = find_evaluation_files(PROJECT_ROOT)
        critical_keys = [
            "cv_results",
            "baseline_vs_tuned",
            "tuning_results",
            "classification_report",
            "error_analysis",
            "model_selection_report"
        ]
        for key in critical_keys:
            self.assertIsNotNone(files.get(key), f"Critical evaluation file for '{key}' not found.")
            self.assertTrue(files[key].exists(), f"Path for '{key}' does not exist on disk.")

    def test_02_evaluation_data_loading_and_columns(self):
        """Verify each data loader returns a valid DataFrame/dict with expected schemas."""
        cv_df, err_cv = load_cv_results()
        self.assertIsNone(err_cv)
        self.assertIsNotNone(cv_df)
        self.assertIn("Mean CV Macro F1", cv_df.columns)
        self.assertIn("Model", cv_df.columns)

        base_df, err_base = load_baseline_vs_tuned()
        self.assertIsNone(err_base)
        self.assertIsNotNone(base_df)
        self.assertIn("Change in Mean CV Macro F1", base_df.columns)

        tune_df, err_tune = load_tuning_results()
        self.assertIsNone(err_tune)
        self.assertIsNotNone(tune_df)
        self.assertIn("Best Parameters", tune_df.columns)

        clf_df, err_clf = load_classification_report()
        self.assertIsNone(err_clf)
        self.assertIsNotNone(clf_df)
        self.assertIn("Class/Average", clf_df.columns)
        self.assertIn("f1-score", clf_df.columns)

        err_df, err_err = load_error_analysis()
        self.assertIsNone(err_err)
        self.assertIsNotNone(err_df)
        self.assertEqual(len(err_df), 130)

        sel_meta, err_sel = load_model_selection_report()
        self.assertIsNone(err_sel)
        self.assertIsNotNone(sel_meta)
        self.assertEqual(sel_meta.get("selected_model"), "Decision Tree")

    def test_03_missing_file_graceful_handling(self):
        """Ensure loaders return (None, error_msg) when given nonexistent paths without throwing."""
        fake_path = PROJECT_ROOT / "nonexistent_evaluation_data.csv"
        df, err = load_cv_results(fake_path)
        self.assertIsNone(df)
        self.assertIsNotNone(err)

        df, err = load_classification_report(fake_path)
        self.assertIsNone(df)
        self.assertIsNotNone(err)

    def test_04_malformed_columns_graceful_handling(self):
        """Verify loaders detect missing required schema columns and return clean error messages."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("WrongCol1,WrongCol2\n1,2\n")
            tmp_path = Path(tmp.name)

        try:
            df, err = load_cv_results(tmp_path)
            self.assertIsNone(df)
            self.assertIn("missing required columns", err)

            df, err = load_error_analysis(tmp_path)
            self.assertIsNone(df)
            self.assertIn("missing required columns", err)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_05_confusion_matrix_accuracy_parity(self):
        """Verify confusion matrix derived from test predictions matches test metrics exactly."""
        err_df, _ = load_error_analysis()
        self.assertIsNotNone(err_df)

        cm = compute_confusion_matrix(err_df)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(list(cm.index), ["Low", "Medium", "High"])
        self.assertEqual(list(cm.columns), ["Low", "Medium", "High"])

        total_samples = cm.values.sum()
        correct_predictions = np.trace(cm.values)
        self.assertEqual(total_samples, 130)
        self.assertEqual(correct_predictions, 70)

        accuracy = correct_predictions / total_samples
        self.assertAlmostEqual(accuracy, 0.5384615, places=4)

    def test_06_error_severity_breakdown(self):
        """Verify empirical error severity numbers match Day 16 report."""
        err_df, _ = load_error_analysis()
        self.assertIsNotNone(err_df)

        eb = compute_error_breakdown(err_df)
        self.assertEqual(eb["total"], 130)
        self.assertEqual(eb["correct"], 70)
        self.assertEqual(eb["errors"], 60)
        self.assertEqual(eb["adjacent_count"], 56)
        self.assertEqual(eb["non_adjacent_count"], 4)
        self.assertAlmostEqual(eb["adjacent_pct"], 93.3333, places=2)
        self.assertAlmostEqual(eb["non_adjacent_pct"], 6.6667, places=2)

    def test_07_report_generation(self):
        """Verify markdown and CSV report exports contain required headers and metric values."""
        sel_meta, _ = load_model_selection_report()
        clf_df, _ = load_classification_report()
        err_df, _ = load_error_analysis()
        base_df, _ = load_baseline_vs_tuned()
        eb = compute_error_breakdown(err_df)

        md = generate_markdown_report(sel_meta, clf_df, eb, base_df)
        self.assertIn("# Model Evaluation and Performance Verification Report", md)
        self.assertIn("Decision Tree", md)
        self.assertIn("0.5237", md)
        self.assertIn("G3", md)
        self.assertIn("93.3%", md)

        csv_str = generate_csv_summary(sel_meta, clf_df)
        self.assertIn("Category,Entity,Evaluation_Method,Metric_Name,Metric_Value,Source_File", csv_str)
        self.assertIn("Decision Tree", csv_str)
        self.assertIn("Macro F1", csv_str)

    def test_08_streamlit_dashboard_smoke_test(self):
        """Verify pages/2_Model_Performance_Dashboard.py launches via AppTest without exceptions."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1 not available in this environment.")

        dash_path = PROJECT_ROOT / "pages" / "2_Model_Performance_Dashboard.py"
        self.assertTrue(dash_path.exists())
        app_test = AppTest.from_file(str(dash_path), default_timeout=20).run()
        self.assertFalse(app_test.exception, f"Dashboard AppTest raised: {app_test.exception}")


if __name__ == "__main__":
    unittest.main()
