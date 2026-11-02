# Day 33 — User Experience Improvements

## Objective

Make model limits, required fields and privacy choices clear in the application.

## Executed results

```text
Project: IYB_INTERNSHIP

2026-11-02 13:50:21.341 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.

2026-11-02 13:50:23.425 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

2026-11-02 13:50:23.456 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

2026-11-02 13:50:23.462 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

Visible limitations and logging-off defaults: PASS

..                                                                       [100%]
============================== warnings summary ===============================
tests/test_app_integration.py::test_single_student_submission
tests/test_app_integration.py::test_single_student_submission
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\streamlit\elements\lib\built_in_chart_utils.py:620: Pandas4Warning: The inplace keyword in DataFrame.reset_index is deprecated and will be removed in a future version (PDEP-8).
  See https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace for more details.
    df.reset_index(inplace=True)  # noqa: PD002

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 2 warnings in 2.73s


```

## Interpretation

The app presents neutral category explanations, required-field guidance and a visible restriction on educational decisions. Logging remains opt-in.

## Limitations

This review verifies app state and wording, not a formal accessibility certification or user study.
