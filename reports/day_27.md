# Day 27 — Input Validation and Reliability

## Objective

Centralize strict validation and saved-model predictions for all serving paths.

## Executed results

```text
Project: IYB_INTERNSHIP

Model version: 1bedf35d0d56c4f0
predicted_category  model_estimated_proba_High  model_estimated_proba_Low  model_estimated_proba_Medium
            Medium                    0.000000                   0.090909                      0.909091
               Low                    0.000000                   1.000000                      0.000000
            Medium                    0.333333                   0.000000                      0.666667

Expected rejection: Missing fields: []; unexpected fields: ['G3']

..............                                                           [100%]
14 passed in 1.69s


```

## Interpretation

All serving outputs use the saved model and retain its class mapping. The regression suite checks invalid numbers, unknown categories, missing/extra columns, empty/large batches and order invariance.

## Limitations

The bounded schema intentionally rejects out-of-domain students. This validation checks data structure, not whether submitted attributes are true.
