# Day 25 — Explainability Validation

## Objective

Verify explanation reproducibility, conservation and uncertainty across shuffle seeds.

## Executed results

```text
Project: IYB_INTERNSHIP

Reference rows: 130 Features: 30

[2026-11-02 13:43:59,733] INFO in model_explainability: Computing permutation importance with 5 repeats (scoring='f1_macro', seed=42)...

[2026-11-02 13:44:00,916] INFO in model_explainability: Computing permutation importance with 5 repeats (scoring='f1_macro', seed=42)...

Same-seed reproducibility and attribution conservation: PASS

[2026-11-02 13:44:02,079] INFO in model_explainability: Computing permutation importance with 10 repeats (scoring='f1_macro', seed=7)...

[2026-11-02 13:44:04,272] INFO in model_explainability: Computing permutation importance with 10 repeats (scoring='f1_macro', seed=42)...

[2026-11-02 13:44:06,435] INFO in model_explainability: Computing permutation importance with 10 repeats (scoring='f1_macro', seed=99)...

           mean_drop  minimum_drop  maximum_drop  best_rank  worst_rank
feature
failures    0.067904      0.060077      0.074229          1           1
Mjob        0.038872      0.035211      0.043115          2           2
freetime    0.023437      0.012058      0.032068          3           9
romantic    0.020265      0.015831      0.024692          4           6
absences    0.017569      0.014361      0.022237          3           7
famrel      0.016433      0.014245      0.018970          6           8
studytime   0.015956      0.002600      0.026672          3          14
school      0.013340      0.003093      0.020785          5          13
Fjob        0.012166      0.009777      0.014683          7          11
Pstatus     0.010701      0.010270      0.011490          8          11

```

## Interpretation

The output identifies the leading variables and their seed sensitivity. Negative importance means shuffling improved this historical sample score; it is not evidence that a feature harms students.

## Limitations

Three seeds and ten shuffles quantify algorithmic variability, not population uncertainty. The test set was previously inspected. Importance does not imply causation.
