# Data and model card

This project uses UCI Student Performance, ID 320, the 649-row Portuguese-language course dataset from two Portuguese secondary schools. The source provides 30 input attributes and course grades. G3 defines the target: Low below 10; Medium from 10 to below 15; High at least 15. G1 and G2 are intentionally excluded as in the existing project, and G3 and the derived category are never inputs.

Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). Citation: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). License: CC BY 4.0. This is public research data; no private student records were added.

The original split is stratified 80/20 with random_state 42: 519 training and 130 historical evaluation rows. Split indices and canonical input order are saved in artifacts/. Numeric/ordinal inputs pass through the tree preprocessor; categorical variables are one-hot encoded inside the fitted pipeline. The final tree was selected by training-only five-fold Macro F1, not by test scores.

artifacts/version_manifest.json records the current data/model hashes, exact estimator settings and environment. The original model’s reported test Macro F1 is 0.44977366255144036 and accuracy 0.5384615384615384, reproduced against every historical test prediction. Repeated reuse of this test set and earlier notebook model comparisons limit interpretation. The dataset is small, observational and context-specific; the model lacks external validation and calibrated probabilities. Demographic, family and health inputs are sensitive proxies. Use only as a learning demonstration, not to determine grades, eligibility, access or interventions.
