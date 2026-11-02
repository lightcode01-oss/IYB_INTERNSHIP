import pandas as pd
import pytest
from src.dataset import original_split
from src.data_quality import inspect_batch
from src.validation import InputValidationError

def test_quality_positions_and_missing():
    frame=original_split()[1].head(3).copy().astype(object)
    frame.iloc[1,frame.columns.get_loc('age')]=None
    frame.iloc[2,frame.columns.get_loc('school')]='BAD'
    report=inspect_batch(frame)
    assert report['valid_rows']==1 and report['invalid_rows']==2
    assert report['missing_cells']==1
    assert [r['row'] for r in report['issues']]==[2,3]
    assert 'BAD' not in str(report)

def test_duplicates_are_reported_not_deleted():
    row=original_split()[1].head(1)
    report=inspect_batch(pd.concat([row,row],ignore_index=True))
    assert report['duplicate_rows']==1 and report['valid_rows']==2

def test_empty_and_duplicate_columns():
    assert inspect_batch(pd.DataFrame())['rows']==0
    with pytest.raises(InputValidationError): inspect_batch(pd.DataFrame([[1,2]],columns=['age','age']))
