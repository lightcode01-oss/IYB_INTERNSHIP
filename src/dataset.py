"""Cache the public UCI 320 dataset and reproduce the original stratified split."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = 'https://archive.ics.uci.edu/static/public/320/data.csv'
CANONICAL_FEATURES = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences']

def load_dataset():
    path = ROOT / 'data' / 'uci_student_performance.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    df = pd.read_csv(path)
    if len(df) != 649 or 'G3' not in df or df['G3'].isna().any():
        raise ValueError('Unexpected UCI 320 dataset schema')
    return df

def features_target():
    df = load_dataset()
    y = pd.Series(np.select([df.G3 < 10, df.G3 < 15], ['Low', 'Medium'], default='High'), index=df.index, name='performance_category')
    return df[CANONICAL_FEATURES].copy(), y

def original_split():
    X, y = features_target()
    return train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

def dataset_hash():
    load_dataset()
    return hashlib.sha256((ROOT/'data'/'uci_student_performance.csv').read_bytes()).hexdigest()
