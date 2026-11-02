# Day 38 — Reproducible Environment

## Objective

Capture exact dependency versions and add isolated setup and notebook rerun commands.

## Executed results

```text
Project: IYB_INTERNSHIP

No broken requirements found.

Python 3.14 and locked dependency file: PASS


{'scikit-learn': '1.9.0', 'pandas': '3.1.0', 'numpy': '2.5.4', 'nbclient': '0.11.0'}

Lock and setup scripts: PASS

```

## Interpretation

The active environment has no broken package requirements. Setup checks and script compilation passed; the exact environment is captured for independent installation.

## Limitations

The isolated installer is provided and its prerequisites are checked; a second full environment installation was not repeated in this day. Colab compatibility is documented but not claimed as tested.
