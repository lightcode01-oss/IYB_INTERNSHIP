# Day 30 — API Testing and Documentation

## Objective

Test API parity, request errors, unavailable models and the documented endpoint schema.

## Executed results

```text
Project: IYB_INTERNSHIP

.........                                                                [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 1 warning in 2.21s


/health ['get']
/predict ['post']

```

## Interpretation

The tests compare API outputs directly with the shared service and verify 422/503 error behavior. OpenAPI and a runnable public-data example document the contract.

## Limitations

In-process HTTP tests do not verify proxy configuration, TLS or external authorization. Those remain hosting concerns.
