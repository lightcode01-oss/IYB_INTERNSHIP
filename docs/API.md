# Local prediction API

Install requirements-api.txt, then run `python -m uvicorn api:app --host 127.0.0.1 --port 8000` from the repository root.

`GET /health` returns readiness and the model hash prefix. `POST /predict` accepts an object with a `records` list containing 1–500 complete 30-field records. Use the feature names in artifacts/student_performance_metadata.json. Each result contains predicted_category and three model_estimated_proba_* values. The response includes model_id. Column order is irrelevant; extra columns, missing values, invalid domains and target columns are rejected with HTTP 422. Unavailable artifacts return HTTP 503.

Interactive local documentation: http://127.0.0.1:8000/docs. No real student information should be used. Authentication, access controls, TLS, rate limiting and durable operational monitoring must be designed before external hosting. No public API is deployed by this project session.
