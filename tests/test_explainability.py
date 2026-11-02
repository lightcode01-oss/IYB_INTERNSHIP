"""
Unit and Integration Tests for Model Explainability and Feature Importance
Day 24: AI-Based Student Performance Prediction System
YBI Foundation Python AI & Data Science Internship

Validates:
1. Pipeline and reference dataset loading.
2. Feature schema validation and strict target leakage prevention (G3 exclusion).
3. Permutation feature importance computation with Macro F1 scoring.
4. Reproducibility using fixed random seed.
5. Native Decision Tree feature importance extraction and aggregation.
6. Local prediction explanation structure and decision path tracing.
7. Graceful degradation and error handling on malformed inputs.
8. Precomputed report discovery and Streamlit page headless compilation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import (
    load_explainability_pipeline,
    load_reference_dataset,
    validate_feature_schema,
    compute_permutation_importance,
    compute_native_feature_importance,
    explain_individual_prediction,
    load_precomputed_results,
    CANONICAL_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS
)


class TestModelExplainability(unittest.TestCase):
    """Test suite for Day 24 explainability module and feature importance."""

    @classmethod
    def setUpClass(cls):
        """Set up model pipeline and reference dataset for testing."""
        cls.pipeline = load_explainability_pipeline()
        cls.X_ref, cls.y_ref = load_reference_dataset()

    def test_01_pipeline_structure_and_steps(self):
        """Verify the loaded pipeline contains preprocessor and classifier."""
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(hasattr(self.pipeline, "named_steps"))
        self.assertIn("preprocessor", self.pipeline.named_steps)
        self.assertIn("classifier", self.pipeline.named_steps)
        clf = self.pipeline.named_steps["classifier"]
        self.assertEqual(clf.__class__.__name__, "DecisionTreeClassifier")

    def test_02_reference_dataset_schema_and_size(self):
        """Ensure reference dataset contains 130 held-out instances with 30 features."""
        self.assertEqual(len(self.X_ref), 130)
        self.assertEqual(len(self.y_ref), 130)
        self.assertEqual(self.X_ref.shape[1], 30)
        self.assertListEqual(sorted(self.X_ref.columns.tolist()), sorted(CANONICAL_FEATURES))
        unique_targets = set(self.y_ref.unique())
        self.assertTrue(unique_targets.issubset({"Low", "Medium", "High"}))

    def test_03_target_leakage_safeguards_enforced(self):
        """Confirm G3 and performance_category are excluded from inputs and trigger error if present."""
        for col in FORBIDDEN_LEAKAGE_COLUMNS:
            self.assertNotIn(col, self.X_ref.columns)

        leaky_df = self.X_ref.copy()
        leaky_df["G3"] = 12
        with self.assertRaises(ValueError) as ctx:
            validate_feature_schema(leaky_df)
        self.assertIn("Target leakage violation", str(ctx.exception))

        leaky_df2 = self.X_ref.copy()
        leaky_df2["performance_category"] = "Medium"
        with self.assertRaises(ValueError) as ctx2:
            validate_feature_schema(leaky_df2)
        self.assertIn("Target leakage violation", str(ctx2.exception))

    def test_04_missing_features_validation_fails(self):
        """Confirm validation fails when required features are missing."""
        incomplete_df = self.X_ref.drop(columns=["failures", "absences"])
        with self.assertRaises(ValueError) as ctx:
            validate_feature_schema(incomplete_df)
        self.assertIn("missing required features", str(ctx.exception))

    def test_05_permutation_importance_computation(self):
        """Verify permutation importance returns 30 ranked features with valid numeric values."""
        df_perm = compute_permutation_importance(
            pipeline=self.pipeline,
            X=self.X_ref,
            y=self.y_ref,
            scoring="f1_macro",
            n_repeats=3,
            random_state=42,
            n_jobs=1
        )
        self.assertIsInstance(df_perm, pd.DataFrame)
        self.assertEqual(len(df_perm), 30)
        self.assertListEqual(df_perm.columns.tolist(), ["feature", "importance_mean", "importance_std", "rank"])

        self.assertTrue(df_perm["importance_mean"].is_monotonic_decreasing)
        self.assertEqual(df_perm["rank"].iloc[0], 1)
        self.assertEqual(df_perm["rank"].iloc[-1], 30)

        self.assertFalse(df_perm["importance_mean"].isna().any())
        self.assertFalse(df_perm["importance_std"].isna().any())

    def test_06_permutation_importance_reproducibility(self):
        """Confirm identical random seed produces identical importance values."""
        res1 = compute_permutation_importance(
            self.pipeline, self.X_ref, self.y_ref, scoring="f1_macro", n_repeats=3, random_state=99, n_jobs=1
        )
        res2 = compute_permutation_importance(
            self.pipeline, self.X_ref, self.y_ref, scoring="f1_macro", n_repeats=3, random_state=99, n_jobs=1
        )
        pd.testing.assert_frame_equal(res1, res2)

    def test_07_native_feature_importance_extraction(self):
        """Verify native tree feature importance sums to 1.0 and aggregates to 30 features."""
        df_trans, df_agg = compute_native_feature_importance(self.pipeline)

        self.assertIsInstance(df_trans, pd.DataFrame)
        self.assertGreater(len(df_trans), 30)
        self.assertAlmostEqual(df_trans["native_importance"].sum(), 1.0, places=4)

        self.assertIsInstance(df_agg, pd.DataFrame)
        self.assertEqual(len(df_agg), 30)
        self.assertAlmostEqual(df_agg["native_importance"].sum(), 1.0, places=4)
        self.assertTrue(df_agg["native_importance"].is_monotonic_decreasing)

    def test_08_individual_prediction_explanation(self):
        """Verify local explanation generates predicted probabilities, decision rules, and cohort context."""
        sample_student = self.X_ref.iloc[0]
        explanation = explain_individual_prediction(self.pipeline, sample_student, reference_df=self.X_ref)

        self.assertIn("predicted_category", explanation)
        self.assertIn(explanation["predicted_category"], ["Low", "Medium", "High"])
        self.assertIn("predicted_probabilities", explanation)
        self.assertEqual(len(explanation["predicted_probabilities"]), 3)
        self.assertAlmostEqual(sum(explanation["predicted_probabilities"].values()), 1.0, places=4)

        self.assertIn("decision_rules", explanation)
        self.assertGreater(len(explanation["decision_rules"]), 0)
        rule = explanation["decision_rules"][0]
        self.assertIn("node_id", rule)
        self.assertIn("feature", rule)
        self.assertIn("threshold", rule)
        self.assertIn("operator", rule)
        self.assertIn("rule_description", rule)

        self.assertIn("cohort_comparison", explanation)
        self.assertIn("failures", explanation["cohort_comparison"])
        self.assertIn("absences", explanation["cohort_comparison"])
        self.assertIn("cohort_mean", explanation["cohort_comparison"]["failures"])

    def test_09_precomputed_results_loading(self):
        """Verify precomputed outputs in reports/day_24/ are discoverable and load correctly."""
        pre = load_precomputed_results()
        self.assertTrue(pre["has_precomputed"])
        self.assertIsNotNone(pre["permutation_df"])
        self.assertIsNotNone(pre["native_df"])
        self.assertEqual(len(pre["permutation_df"]), 30)
        self.assertEqual(len(pre["native_df"]), 30)
        self.assertIsNotNone(pre["sample_explanations"])
        self.assertEqual(len(pre["sample_explanations"]), 3)

    def test_10_streamlit_page_syntax(self):
        """Confirm pages/3_Model_Explainability.py compiles without error."""
        page_path = PROJECT_ROOT / "pages" / "3_Model_Explainability.py"
        self.assertTrue(page_path.exists())
        with open(page_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(page_path), "exec")
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
