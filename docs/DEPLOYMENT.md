# Deployment guide and actual status

Actual verified status: local Streamlit and local FastAPI only. No public deployment was created, and no hosted URL has been verified. See reports/deployment_status.json for executed checks.

Install the locked environment as described in REPRODUCIBILITY.md. Launch the UI using `python -m streamlit run app.py --server.address 127.0.0.1`. Launch the optional API using `python -m uvicorn api:app --host 127.0.0.1 --port 8000`. Run `python scripts/verify_deployment.py` to verify temporary local startup, health, frontend response and real API prediction parity; the script stops its own processes afterward.

For a future approved Streamlit Community Cloud deployment, use the existing lightcode01-oss/IYB_INTERNSHIP repository, main branch and app.py entry point with Python 3.14. Preserve CORS and XSRF protection. Run tests and review pinned package availability. Keep monitoring private and use only public demo data. Do not upload secrets or real student files. Authentication, privacy, retention and resource limits must be reviewed before external use. The API needs its own hosting process and access controls if exposed.

Rollback by deploying a previously verified commit and its matching model/checksum/environment, without rewriting Git history. Recheck health and prediction parity. Cloud filesystem logging is not durable or suitable for shared replicas; use an authenticated transactional datastore before scaling.
