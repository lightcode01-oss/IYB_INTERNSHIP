import numpy as np
import pandas as pd
import pytest
from src.dataset import original_split
from src.prediction import predict,get_pipeline
from src.validation import InputValidationError,validate_inputs

@pytest.fixture
def record(): return original_split()[1].iloc[[0]].copy()

def test_prediction_parity(record):
    assert predict(record).predicted_category.iloc[0]==get_pipeline().predict(record)[0]
    assert np.isclose(predict(record).filter(like='proba_').sum(axis=1).iloc[0],1)

@pytest.mark.parametrize('field,value',[('age',True),('age',np.nan),('age',float('inf')),('age',14),('studytime',2.5),('Mjob','astronaut'),('sex',None)])
def test_invalid_values(record,field,value):
    record=record.astype(object);record.iloc[0,record.columns.get_loc(field)]=value
    with pytest.raises(InputValidationError): predict(record)

@pytest.mark.parametrize('field',['G3','performance_category','name','G1'])
def test_no_extra_fields(record,field):
    record[field]=1
    with pytest.raises(InputValidationError): predict(record)

def test_missing_and_empty(record):
    with pytest.raises(InputValidationError): predict(record.drop(columns='age'))
    with pytest.raises(InputValidationError): predict(record.iloc[:0])
    with pytest.raises(InputValidationError): predict(pd.concat([record]*501))

def test_reordered_and_duplicate(record):
    pd.testing.assert_frame_equal(predict(record),predict(record[record.columns[::-1]]))
    with pytest.raises(InputValidationError): predict(pd.concat([record,record[['age']]],axis=1))
