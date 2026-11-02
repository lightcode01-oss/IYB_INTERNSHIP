# Day 43 — Final Project Testing

## Objective

Rerun application notebooks and verify the complete regression suite against the final serving code.

## Executed results

```text
Project: IYB_INTERNSHIP

Days 1–42 have successful recorded fresh-kernel execution; Days 19–42 rerun against the final application code.

........................................................................ [ 84%]
.............                                                            [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/test_app_integration.py::test_single_student_submission
tests/test_app_integration.py::test_single_student_submission
tests/test_deployment_readiness.py::TestDeploymentReadiness::test_08_streamlit_app_test_execution
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\streamlit\elements\lib\built_in_chart_utils.py:620: Pandas4Warning: The inplace keyword in DataFrame.reset_index is deprecated and will be removed in a future version (PDEP-8).
  See https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace for more details.
    df.reset_index(inplace=True)  # noqa: PD002

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
85 passed, 4 warnings in 8.50s


Notebook schema and source syntax: PASS

```

## Interpretation

Recorded fresh-kernel reruns and the final regression output provide reviewable evidence. Very large numeric inputs and ragged CSV rows now return validation errors before inference.

## Limitations

Passing tests cover the stated cases and local application behavior. They do not establish external-school generalization, production security or a hosted deployment.
