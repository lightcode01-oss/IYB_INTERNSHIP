# Day 34 — Reliable Batch Prediction

## Objective

Bound and validate CSV uploads before producing ordered predictions.

## Executed results

```text
Project: IYB_INTERNSHIP

predicted_category         model_id
            Medium 1bedf35d0d56c4f0
               Low 1bedf35d0d56c4f0
            Medium 1bedf35d0d56c4f0

.........                                                                [100%]
============================== warnings summary ===============================
tests/test_app_integration.py::test_single_student_submission
tests/test_app_integration.py::test_single_student_submission
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\streamlit\elements\lib\built_in_chart_utils.py:620: Pandas4Warning: The inplace keyword in DataFrame.reset_index is deprecated and will be removed in a future version (PDEP-8).
  See https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace for more details.
    df.reset_index(inplace=True)  # noqa: PD002

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 2 warnings in 2.72s


```

## Interpretation

The CSV workflow preserves row order and refuses the entire batch when any row is invalid. Public demo input is saved for repeatable demonstrations.

## Limitations

The 500-row limit is intended for this small local application. The workflow does not implement large-file streaming.
