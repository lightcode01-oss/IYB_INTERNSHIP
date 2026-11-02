# Final testing report

Executed command: python -m pytest -q --junitxml=reports/final_test_results.xml

```text
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

```

Scope: application form submission, API/CSV parity, strict schema, malformed inputs, payload limits, model checksum/cache identity, monitoring concurrency, real artifact loading, explanation consistency and daily notebook execution state. Local HTTP startup was separately verified on Day 39. Tests use public UCI records or explicit synthetic test cases in temporary storage.

Known upstream deprecation warnings are reported in the output. This suite is not a production penetration test or educational validation.
