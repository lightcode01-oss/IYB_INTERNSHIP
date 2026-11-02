# Explainability report

Raw-column permutation importance uses the saved pipeline, Macro F1, 15 shuffles and seed 42 on the historical 130-row evaluation set. Positive values mean shuffling reduced this model score. Negative values mean shuffling improved the sample score. Neither indicates a causal effect on students.

| feature   |   importance_mean |   importance_std |   rank |
|:----------|------------------:|-----------------:|-------:|
| failures  |         0.0629285 |       0.0186386  |      1 |
| Mjob      |         0.0437143 |       0.00937186 |      2 |
| freetime  |         0.0286265 |       0.0241188  |      3 |
| romantic  |         0.022539  |       0.0124154  |      4 |
| famrel    |         0.016566  |       0.0118921  |      5 |
| Fjob      |         0.0148049 |       0.00660178 |      6 |
| absences  |         0.0143829 |       0.0123218  |      7 |
| Pstatus   |         0.0111778 |       0.00438991 |      8 |
| Medu      |         0.0107405 |       0.0150918  |      9 |
| sex       |         0.0106759 |       0.00544694 |     10 |

Native attribution tables, plots and individual tree paths are in reports/day_24/. Seed variability is in reports/explainability_validation.csv. Correlation, small sample size and test reuse limit interpretation. No educational recommendation is derived from these associations.
