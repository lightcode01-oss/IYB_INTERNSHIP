# Day 35 — Monitoring Reliability Improvements

## Objective

Protect local event writes and add a clearly scoped prediction-distribution comparison.

## Executed results

```text
Project: IYB_INTERNSHIP

................                                                         [100%]
============================== warnings summary ===============================
..\venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\ABHINAV DASH\Documents\Codex\2026-11-02\and\work\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
16 passed, 1 warning in 2.83s


{
  "reference_count": 65,
  "current_count": 65,
  "total_variation": 0.06153846153846154,
  "reference_proportions": {
    "Low": 0.18461538461538463,
    "Medium": 0.6153846153846154,
    "High": 0.2
  },
  "current_proportions": {
    "Low": 0.24615384615384617,
    "Medium": 0.5846153846153846,
    "High": 0.16923076923076924
  },
  "interpretation": "Descriptive output shift; does not establish model degradation or accuracy."
}

437
```

## Interpretation

The concurrency test writes 40 isolated test events and verifies every event survives. The public-data window comparison records a descriptive shift without attaching an accuracy claim.

## Limitations

The CSV lock does not protect multiple processes or cloud replicas. The two windows are historical demo records, not production traffic.
