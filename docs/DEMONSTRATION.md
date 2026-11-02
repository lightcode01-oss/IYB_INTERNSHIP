# Portfolio demonstration

1. Create and activate the isolated environment using REPRODUCIBILITY.md. Start Streamlit locally.
2. Read the visible model-limit warning. Keep logging and feature logging disabled. Submit the default demonstration form and inspect the category and three model estimates.
3. Open the Batch tab and upload data/demo_input.csv. Run prediction and inspect the three output rows. The file contains public UCI predictors, not private student data.
4. Open the performance dashboard and compare the historical test Macro F1 with training CV and the majority baseline. Explain that repeated use of this small test set limits generalization claims.
5. Open the explainability page. Discuss permutation importance, native attribution and one decision path without making causal claims.
6. Optionally run the API example in API_EXAMPLE.md and compare its results with the same public records in the CSV workflow.
7. Run `python -m pytest -q` and `python scripts/verify_deployment.py` to reproduce regression and local startup checks.

The README includes real generated evaluation plots. An application screenshot could not be captured because the browser-control tool stopped when it could not verify the active URL. No screenshot, hosted URL, user feedback or educational endorsement was invented. The local server checks and app state tests are documented separately.
