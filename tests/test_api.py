from fastapi.testclient import TestClient
import pytest
import api
from src.dataset import original_split
from src.prediction import predict

@pytest.fixture
def client(): return TestClient(api.app)

@pytest.fixture
def records(): return original_split()[1].head(3).to_dict('records')

def test_health_and_prediction_parity(client,records):
    health=client.get('/health')
    assert health.status_code==200 and len(health.json()['model_id'])==16
    response=client.post('/predict',json={'records':records})
    assert response.status_code==200
    assert response.json()['predictions']==predict(records).to_dict('records')
    assert response.json()['model_id']==health.json()['model_id']

@pytest.mark.parametrize('payload',[{}, {'records':[]},{'records':[{}]},{'records':[1]},{'records':{},'other':1}])
def test_request_errors(client,payload):
    assert client.post('/predict',json=payload).status_code==422

def test_bad_columns_and_large_batch(client,records):
    assert client.post('/predict',json={'records':[dict(records[0],G3=19)]}).status_code==422
    assert client.post('/predict',json={'records':[dict(records[0],age=100)]}).status_code==422
    assert client.post('/predict',json={'records':records,'extra':'x'}).status_code==422
    assert client.post('/predict',json={'records':[records[0]]*501}).status_code==422
    assert client.post('/predict',content='{broken',headers={'content-type':'application/json'}).status_code==422

def test_model_failure_is_sanitized(client,records,monkeypatch):
    def fail(*args): raise RuntimeError('private filesystem detail')
    monkeypatch.setattr(api,'predict',fail)
    response=client.post('/predict',json={'records':records})
    assert response.status_code==503 and 'private' not in response.text
    monkeypatch.setattr(api,'get_pipeline',fail)
    assert client.get('/health').status_code==503

def test_openapi_documents_endpoints(client):
    paths=client.get('/openapi.json').json()['paths']
    assert '/predict' in paths and '/health' in paths
