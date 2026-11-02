# Final model evaluation

Selected model: Decision Tree. Training CV Macro F1: 0.523681 (fold SD 0.038942). Selection was verified against the training-only tuning table.

Historical evaluation: 130 records.

|   accuracy |   macro_precision |   macro_recall |   macro_f1 |
|-----------:|------------------:|---------------:|-----------:|
|   0.538462 |           0.44475 |       0.462882 |   0.449774 |

Majority baseline Macro F1: 0.261682. Conditional stratified bootstrap 95% percentile interval: [0.34722657869533885, 0.5332758797065215].

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| Low          |    0.321429 | 0.45     |   0.375    |  20        |
| Medium       |    0.679487 | 0.630952 |   0.654321 |  84        |
| High         |    0.333333 | 0.307692 |   0.32     |  26        |
| accuracy     |    0.538462 | 0.538462 |   0.538462 |   0.538462 |
| macro avg    |    0.44475  | 0.462882 |   0.449774 | 130        |
| weighted avg |    0.55517  | 0.538462 |   0.544484 | 130        |

## Limitations

- Historical test set reused during earlier exploration; not fresh external validation.
- Bootstrap holds class counts fixed and conditions on this fitted model; it does not account for training or model-selection uncertainty.
- Small observational dataset from two Portuguese schools; no causal or educational decision validity.
- Class probabilities are not calibrated.
