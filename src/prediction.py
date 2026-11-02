"""One validated prediction path shared by the app, batch workflow and API."""
from functools import lru_cache
from pathlib import Path
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from src.dataset import CANONICAL_FEATURES
from src.validation import validate_inputs

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'artifacts/student_performance_pipeline.joblib'

@lru_cache(maxsize=1)
def get_pipeline():
    # Only load the repository-owned artifact; never accept a request file path.
    model=joblib.load(MODEL_PATH)
    if list(model.feature_names_in_)!=CANONICAL_FEATURES:
        raise RuntimeError('Model input schema does not match the serving contract')
    if set(model.classes_)!={'Low','Medium','High'}:
        raise RuntimeError('Model labels do not match the serving contract')
    return model

def model_id():
    return hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16]

def predict(data):
    frame=validate_inputs(data)
    model=get_pipeline()
    labels=model.predict(frame)
    probability=model.predict_proba(frame)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1),1):
        raise RuntimeError('Model returned invalid probabilities')
    result=pd.DataFrame({'predicted_category':labels})
    for i,label in enumerate(model.classes_): result['model_estimated_proba_'+label]=probability[:,i]
    return result
