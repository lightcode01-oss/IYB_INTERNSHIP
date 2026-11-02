# Training and evaluation workflow

The target is Low for G3 < 10, Medium for 10 <= G3 < 15, and High for G3 >= 15. Use the cached 649-row UCI dataset and original stratified split (random_state 42). Do not put G3 or the derived label into features. G1 and G2 are also intentionally excluded to preserve the original project’s 30-feature design.

Days 14–15 compare six pipelines and search their hyperparameter grids using five-fold stratified cross-validation on the 519 training rows. Preprocessing is inside each pipeline, so its learned categories/scaling do not use validation folds. Macro F1 is the selection metric. Day 16 selects the highest training-CV result, fits on training data and evaluates on the historical 130-row partition. Day 17 serializes the preprocessing/model pipeline and records its schema and metadata.

Files day_14_cross_validation_results.csv, day_15_tuning_results.csv, day_15_baseline_vs_tuned.csv and day_16_model_selection_report.json record the actual rerun results. day_16_error_analysis.csv contains the public dataset predictors, true labels and actual predictions. The dashboard loads these files and does not create mock metrics.

Historical test reuse in earlier model notebooks means these results are descriptive project evaluation, not pristine external validation. Repeated inspection and feature-importance analysis can influence human choices even if the final automated selection uses training CV. A future research evaluation needs a new external cohort or a predeclared nested evaluation protocol. No metric here establishes educational or causal validity.
