# Day 42 — Final ML Evaluation Review

## Objective

Audit model-selection evidence and reproduce historical metrics with an explicit baseline and uncertainty estimate.

## Executed results

```text
Project: IYB_INTERNSHIP

{
  "model_id": "1bedf35d0d56c4f0",
  "selected_model": "Decision Tree",
  "training_cv_macro_f1": 0.5236814699741081,
  "training_cv_std": 0.03894230481396425,
  "historical_test_rows": 130,
  "metrics": {
    "accuracy": 0.5384615384615384,
    "macro_precision": 0.4447496947496948,
    "macro_recall": 0.46288156288156285,
    "macro_f1": 0.44977366255144036
  },
  "majority_baseline_macro_f1": 0.2616822429906542,
  "conditional_bootstrap_macro_f1_95_percentile": [
    0.34722657869533885,
    0.5332758797065215
  ],
  "bootstrap_repeats": 500,
  "bootstrap_seed": 42,
  "class_order": [
    "Low",
    "Medium",
    "High"
  ],
  "limitations": [
    "Historical test set reused during earlier exploration; not fresh external validation.",
    "Bootstrap holds class counts fixed and conditions on this fitted model; it does not account for training or model-selection uncertainty.",
    "Small observational dataset from two Portuguese schools; no causal or educational decision validity.",
    "Class probabilities are not calibrated."
  ]
}

Evaluation and explanation reports written from verified outputs

```

## Interpretation

The saved tree, hyperparameters, training-CV ranking, split indices and all test predictions agree with the historical artifacts. The reports state measured performance, the majority baseline and bootstrap scope.

## Limitations

Repeated test inspection limits generalization claims. The interval conditions on fixed predictions and class counts and must not be presented as uncertainty for a future school population.
