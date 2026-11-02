# Day 36 — Model and Data Version Documentation

## Objective

Save data provenance, model hashes, original split indices and environment versions.

## Executed results

```text
Project: IYB_INTERNSHIP

{
  "dataset": {
    "rows": 649,
    "train_rows": 519,
    "test_rows": 130,
    "cached_csv_sha256": "5be7b02f13f5800fddd66e2c8d834cd00ece14c552fb404def2397b31b63236b"
  },
  "model": {
    "sha256": "1bedf35d0d56c4f090cd28d2a317a10533eba0e28489d2318aebdde2d105fb83",
    "estimator": "DecisionTreeClassifier",
    "parameters": {
      "ccp_alpha": 0.0,
      "class_weight": null,
      "criterion": "gini",
      "max_depth": 10,
      "max_features": null,
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "min_samples_leaf": 1,
      "min_samples_split": 10,
      "min_weight_fraction_leaf": 0.0,
      "monotonic_cst": null,
      "random_state": 42,
      "splitter": "best"
    },
    "selection": "training-only five-fold stratified CV Macro F1",
    "classes": [
      "High",
      "Low",
      "Medium"
    ]
  },
  "environment": {
    "python": "3.14.2",
    "numpy": "2.5.4",
    "pandas": "3.1.0",
    "scikit-learn": "1.9.0",
    "joblib": "1.6.0",
    "streamlit": "1.66.1"
  }
}

Split separation, row coverage, target exclusion and model checksum: PASS

```

## Interpretation

The manifest identifies the public data and trained model exactly. All 649 original row indices are covered once across disjoint training and evaluation partitions.

## Limitations

Matching hashes establishes file consistency, not scientific validity or trusted origin. The manifest documents historical evaluation reuse.
