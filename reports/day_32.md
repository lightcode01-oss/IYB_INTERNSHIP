# Day 32 — Application Security Review

## Objective

Bound request sizes, verify the model checksum and test privacy-related regressions.

## Executed results

```text
Project: IYB_INTERNSHIP

Verified SHA-256: 1bedf35d0d56c4f090cd28d2a317a10533eba0e28489d2318aebdde2d105fb83

............                                                             [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
12 passed, 1 warning in 2.27s


```

## Interpretation

The tests verify checksum mismatch handling, oversized body rejection, zero probability retention and arbitrary-field exclusion from logs.

## Limitations

This targeted review is not a penetration test or dependency vulnerability audit. The app and API lack production authorization and must remain local demonstrations.
