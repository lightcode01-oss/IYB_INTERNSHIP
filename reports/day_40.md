# Day 40 — Performance and Reliability Review

## Objective

Avoid duplicate preprocessing and ensure the reported version identifies the cached model.

## Executed results

```text
Project: IYB_INTERNSHIP

..............                                                           [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
14 passed, 1 warning in 2.66s


 batch_rows  repeats  median_ms   p95_ms
          1       15     8.2371  8.89686
         10       15     8.4310  9.51905
        130       15    11.2154 12.28766
        500       15    16.8847 18.84903

```

## Interpretation

The complete historical split retains the same class predictions. Model loading and its identity are cached together; request inference now preprocesses once. Measured latency is saved for this environment.

## Limitations

These timings are local measurements, not service-level guarantees or a controlled before/after experiment. Restart the service to load an approved changed artifact.
