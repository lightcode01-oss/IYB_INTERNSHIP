from src.dataset import original_split
from src.prediction import predict,get_pipeline,model_id
import src.prediction as serving
import numpy as np

def test_full_reference_prediction_parity():
    X=original_split()[1]
    assert np.array_equal(predict(X).predicted_category,get_pipeline().predict(X))
    assert get_pipeline() is get_pipeline()

def test_model_id_identifies_cached_model(tmp_path,monkeypatch):
    original=serving.MODEL_PATH
    copied=tmp_path/'model.joblib';copied.write_bytes(original.read_bytes())
    serving._load_bundle.cache_clear()
    try:
        monkeypatch.setattr(serving,'MODEL_PATH',copied)
        loaded=get_pipeline();identity=model_id()
        copied.write_bytes(b'tampered after loading')
        assert get_pipeline() is loaded
        assert model_id()==identity
    finally:
        monkeypatch.setattr(serving,'MODEL_PATH',original)
        serving._load_bundle.cache_clear()
