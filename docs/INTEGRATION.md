# Application integration

Streamlit and FastAPI both call src.prediction.predict. The shared service validates all 30 fields, preserves feature order, loads the trusted cached pipeline and maps its class probabilities. Streamlit uses the service directly in the same process. A network call to its own API would add latency and another failure mode without improving this single-repository demonstration. External clients can use the optional API.

Batch input is validated before any prediction. Target columns and extra identifiers are rejected rather than silently dropped. The batch and single-student outputs use the same model.
