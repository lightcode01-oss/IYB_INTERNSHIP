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
def _load_bundle():
    # Only load the repository-owned artifact; never accept a request file path.
    from src.security import verify_model
    digest=verify_model(MODEL_PATH, ROOT/"artifacts/student_performance_pipeline.sha256")
    model=joblib.load(MODEL_PATH)
    if list(model.feature_names_in_)!=CANONICAL_FEATURES:
        raise RuntimeError('Model input schema does not match the serving contract')
    if set(model.classes_)!={'Low','Medium','High'}:
        raise RuntimeError('Model labels do not match the serving contract')
    return model,digest[:16]

def get_pipeline():
    return _load_bundle()[0]

def model_id():
    return _load_bundle()[1]

def predict(data):
    frame=validate_inputs(data)
    model=get_pipeline()
    probability=model.predict_proba(frame)
    labels=model.classes_[probability.argmax(axis=1)]
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1),1):
        raise RuntimeError('Model returned invalid probabilities')
    result=pd.DataFrame({'predicted_category':labels})
    for i,label in enumerate(model.classes_): result['model_estimated_proba_'+label]=probability[:,i]
    return result
