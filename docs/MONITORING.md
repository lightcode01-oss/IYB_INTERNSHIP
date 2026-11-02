# Local monitoring

Logging is optional and off by default in the app. Events contain a generated event ID, timestamp, model version, predicted category, optional class probabilities and subjective feedback. Feature logging remains an explicit opt-in; do not enable it with real student data. Logs and batch exports are excluded from Git.

In-process writes, feedback updates, reads and clearing use one reentrant lock to avoid interleaved CSV writes. This protects threads in one process, not multiple processes or replicas. Use a transactional datastore before scaling. Keep monitoring private to the operator, define a retention policy and store no real student records in this demo.

compare_prediction_distribution reports total-variation distance between output class frequencies: 0 means identical frequencies and 1 means disjoint frequencies. It is a descriptive signal only. Different users, small samples or changes in input mix can change the output distribution. Neither subjective feedback nor output shift measures accuracy. Verified outcomes and a separate evaluation protocol are required for performance monitoring.
