import pytest
from src.final_evaluation import review_evaluation
from src.dataset import original_split
from src.prediction import predict
from src.validation import InputValidationError
from src.batch import parse_csv

def test_evaluation_reproducibility():
    a=review_evaluation(repeats=20)
    b=review_evaluation(repeats=20)
    assert a==b
    assert a['metrics']['macro_f1']>a['majority_baseline_macro_f1']
    assert len(a['confusion_matrix'])==3

def test_very_large_integer_is_validation_error():
    record=original_split()[1].head(1).to_dict('records')[0]
    record['age']=10**100
    with pytest.raises(InputValidationError): predict(record)

def test_ragged_csv_rejected():
    text=original_split()[1].head(2).to_csv(index=False)
    lines=text.splitlines();lines[1]+=',unexpected'
    with pytest.raises(InputValidationError): parse_csv('\n'.join(lines).encode())
