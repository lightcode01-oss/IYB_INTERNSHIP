"""Strict public input contract; no targets, identifiers or silent coercion."""
import numbers
import numpy as np
import pandas as pd
from src.dataset import CANONICAL_FEATURES

NUMERIC_BOUNDS = {'age':(15,22),'absences':(0,93),'Medu':(0,4),'Fedu':(0,4),'traveltime':(1,4),'studytime':(1,4),'failures':(0,4),'famrel':(1,5),'freetime':(1,5),'goout':(1,5),'Dalc':(1,5),'Walc':(1,5),'health':(1,5)}
CATEGORIES = {'school':['GP','MS'],'sex':['F','M'],'address':['U','R'],'famsize':['LE3','GT3'],'Pstatus':['T','A'],'Mjob':['teacher','health','services','at_home','other'],'Fjob':['teacher','health','services','at_home','other'],'reason':['home','reputation','course','other'],'guardian':['mother','father','other']}
CATEGORIES.update({k:['yes','no'] for k in ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']})
MAX_ROWS = 500

class InputValidationError(ValueError):
    """A request cannot be predicted without correcting its input."""

def validate_inputs(data):
    if not isinstance(data,pd.DataFrame):
        if isinstance(data,dict): data=[data]
        if not isinstance(data,list) or not all(isinstance(r,dict) for r in data):
            raise InputValidationError('Expected a record, list of records or DataFrame')
        data=pd.DataFrame(data)
    if not 1 <= len(data) <= MAX_ROWS:
        raise InputValidationError(f'Provide between 1 and {MAX_ROWS} records')
    if data.columns.duplicated().any():
        raise InputValidationError('Duplicate column names are not allowed')
    missing=sorted(set(CANONICAL_FEATURES)-set(data.columns))
    extra=sorted(set(data.columns)-set(CANONICAL_FEATURES),key=str)
    if missing or extra:
        raise InputValidationError(f'Missing fields: {missing}; unexpected fields: {extra}')
    result=data[CANONICAL_FEATURES].copy()
    for name,(lo,hi) in NUMERIC_BOUNDS.items():
        for pos,value in enumerate(result[name]):
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value) or value!=int(value) or not lo<=value<=hi:
                raise InputValidationError(f'Row {pos+1}, {name}: expected an integer between {lo} and {hi}')
        result[name]=result[name].astype(int)
    for name,allowed in CATEGORIES.items():
        bad=~result[name].isin(allowed)
        if bad.any():
            pos=int(np.flatnonzero(bad.to_numpy())[0])+1
            raise InputValidationError(f'Row {pos}, {name}: expected one of {allowed}')
    return result
