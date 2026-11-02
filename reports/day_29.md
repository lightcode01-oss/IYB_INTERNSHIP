# Day 29 — Prediction API

## Objective

Expose the existing validated model through local health and prediction endpoints.

## Executed results

```text
Project: IYB_INTERNSHIP

C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa

{'status': 'ready', 'model_id': '1bedf35d0d56c4f0'}
{
  "model_id": "1bedf35d0d56c4f0",
  "predictions": [
    {
      "predicted_category": "Medium",
      "model_estimated_proba_High": 0.0,
      "model_estimated_proba_Low": 0.09090909090909091,
      "model_estimated_proba_Medium": 0.9090909090909091
    }
  ]
}

Empty and target-leaking payload rejection: PASS

```

## Interpretation

The endpoints return the verified artifact identity and actual prediction probabilities. Validation errors use 422 and model availability errors use 503.

## Limitations

The TestClient check exercises the ASGI application in-process. A live HTTP startup check is scheduled for deployment verification. This API is intended for local demonstration.
