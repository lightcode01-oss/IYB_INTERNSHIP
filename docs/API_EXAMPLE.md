# API demonstration

From the repository root, start the local API using the command in API.md. In a second Python session:

```python
import requests
from src.dataset import original_split
records = original_split()[1].head(2).to_dict("records")
response = requests.post("http://127.0.0.1:8000/predict", json={"records": records}, timeout=10)
response.raise_for_status()
print(response.json())
```

The example uses public UCI evaluation records. Do not submit personal student data. Empty batches, extra/target fields, invalid values and malformed JSON receive 422. Fix the reported fields and retry. A 503 response means the model is unavailable; check the trusted artifact and installed versions. Response probabilities are model estimates and must not be treated as calibrated confidence.
