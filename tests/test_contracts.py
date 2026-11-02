import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api import app
from src.dataset import original_split,features_target,load_dataset,ROOT
from src.prediction import predict
from src.batch import predict_csv

def test_api_batch_and_direct_predictions_match():
    data=original_split()[1].head(10)
    direct=predict(data)
    batch=predict_csv(data.to_csv(index=False).encode())
    response=TestClient(app).post('/predict',json={'records':data.to_dict('records')})
    assert response.status_code==200
    pd.testing.assert_frame_equal(direct,pd.DataFrame(response.json()['predictions']))
    assert list(batch.predicted_category)==list(direct.predicted_category)
    assert np.allclose(batch.filter(like='proba_'),direct.filter(like='proba_'))

def test_target_rule_and_manifest():
    _,y=features_target()
    expected=load_dataset().G3.apply(lambda score:'Low' if score<10 else 'Medium' if score<15 else 'High')
    assert list(y)==list(expected)
    manifest=json.loads((ROOT/'artifacts/version_manifest.json').read_text())
    assert manifest['model']['sha256']==hashlib.sha256((ROOT/'artifacts/student_performance_pipeline.joblib').read_bytes()).hexdigest()

def test_notebooks_valid_with_real_execution_state():
    import nbformat
    paths=list(ROOT.glob('Day_*.ipynb'))+list((ROOT/'notebooks').glob('Day_*.ipynb'))+list(ROOT.glob('YBI_45*.ipynb'))
    for path in paths:
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert code and all(c.execution_count is not None for c in code),path.name
        assert not any(o.output_type=='error' for c in code for o in c.outputs),path.name
