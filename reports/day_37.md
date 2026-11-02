# Day 37 — Automated Regression Testing

## Objective

Add cross-interface contract tests and a continuous-integration workflow.

## Executed results

```text
Project: IYB_INTERNSHIP

........................................................................ [ 90%]
........                                                                 [100%]
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
80 passed, 4 warnings in 8.79s


Application and source compilation: PASS

```

## Interpretation

The local full suite verifies contract parity and saved notebook execution state. The workflow is configured to repeat tests on pushes and pull requests.

## Limitations

Local success does not by itself verify the hosted workflow. Its actual status must be checked separately after GitHub receives the commit.
