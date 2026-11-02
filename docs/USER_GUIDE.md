# Demonstration user guide

Start the application with `python -m streamlit run app.py`. Use public demonstration records rather than real student data. Fill every field and select Predict Performance Tier. The output is Low (G3 below 10), Medium (10–14), or High (15–20), with three uncalibrated model estimates. You never enter G3. The model cannot determine an individual student’s ability or eligibility.

Use the performance dashboard to inspect historical evaluation and the explainability page to inspect model associations. A low category is not a diagnosis or a recommendation for punitive action. The model’s historical Macro F1 is approximately 0.450; errors are common.

Prediction logging is off by default. If enabled, events remain on the local machine. Leave input-feature logging disabled. The monitoring dashboard describes usage and subjective feedback; it cannot measure accuracy without verified outcomes. Batch uploads must contain only the 30 expected predictors, with no identifiers or target columns.
