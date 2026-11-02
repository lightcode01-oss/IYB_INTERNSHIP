import hashlib
from pathlib import Path
from fastapi.testclient import TestClient
from api import app
from src.security import verify_model
import pytest

def test_tampered_artifact_rejected(tmp_path):
    model=tmp_path/'model.joblib';model.write_bytes(b'original')
    checksum=tmp_path/'model.sha256';checksum.write_text(hashlib.sha256(b'original').hexdigest())
    assert verify_model(model,checksum)==hashlib.sha256(b'original').hexdigest()
    model.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checksum'): verify_model(model,checksum)

def test_oversized_request_rejected():
    with TestClient(app) as client:
        response=client.post('/predict',content=b'x'*(1024*1024+1))
        assert response.status_code==413

def test_zero_probability_and_no_arbitrary_feature_logging(tmp_path):
    from src.monitoring import log_prediction_event,load_prediction_logs
    p=tmp_path/'events.csv'
    ok,_=log_prediction_event('Low',{'Low':1,'High':0,'Medium':0},input_features={'email':'do-not-store','G3':5,'age':17},log_features=True,log_path=p)
    assert ok
    data=load_prediction_logs(p)
    assert data.proba_High.iloc[0]==0
    assert 'feat_email' not in data and 'feat_G3' not in data
