# Day 31 — Application Integration

## Objective

Use one validation and prediction service for Streamlit and the optional API.

## Executed results

```text
Project: IYB_INTERNSHIP

.........................                                                [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/test_app_integration.py::test_single_student_submission
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\streamlit\elements\lib\built_in_chart_utils.py:620: Pandas4Warning: The inplace keyword in DataFrame.reset_index is deprecated and will be removed in a future version (PDEP-8).
  See https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace for more details.
    df.reset_index(inplace=True)  # noqa: PD002

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 2 warnings in 3.90s


Shared artifact identity: 1bedf35d0d56c4f0

C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa

```

## Interpretation

A real Streamlit form submission now exercises the validated service, and API parity tests use the same artifact. Invalid batch inputs are rejected before inference.

## Limitations

AppTest covers the application state and form logic; browser rendering is reviewed separately. Streamlit remains a single-process demonstration.
