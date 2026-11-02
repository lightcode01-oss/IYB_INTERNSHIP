# Application security review

The API is a local demonstration and defaults to a loopback launch command. It rejects bodies over 1 MiB before JSON parsing and limits prediction batches to 500 records. Inputs have a strict allowlist. Error responses omit internal exception details. No request body is logged by the API.

The prediction loader verifies the repository model SHA-256 before joblib deserialization. This detects accidental artifact changes; it does not authenticate an untrusted repository. Never load uploaded pickle/joblib files. Install the pinned dependencies from the package registry, and review upstream advisories before external hosting. No vulnerability-free claim is made.

Monitoring excludes raw features by default, and its optional feature logging is restricted to known model columns. Even these attributes can be sensitive: use public demonstration data only, disable feature logging for real students, restrict filesystem access and define retention before any real use. The app has no multi-user authorization boundary. Do not expose its monitoring dashboard publicly with operational records.

Secrets, environment files, monitoring logs and batch prediction exports are ignored by Git. Public hosting, authentication, TLS, rate limiting and a privacy review remain prerequisites for external use. No deployment approval or educational validity is implied.
