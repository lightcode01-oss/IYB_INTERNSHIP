# Day 26 — Model Robustness Testing

## Objective

Measure prediction sensitivity while checking batch and column-order invariance.

## Executed results

```text
Project: IYB_INTERNSHIP

Column order, single/batch parity and probability normalization: PASS

  feature  changed_inputs  changed_predictions  fraction_of_all_rows
 absences             130                    4              0.030769
studytime             124                   12              0.092308
 failures             128                   63              0.484615

```

## Interpretation

Ordering and batch assertions enforce serving consistency. The sensitivity table records actual changes and can highlight brittle behavior for future review.

## Limitations

Modified records are explicit test scenarios, not new students or educational outcome evidence. No robustness score is used to choose or retrain the model.
