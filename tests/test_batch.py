import pytest
from src.dataset import original_split
from src.batch import parse_csv,predict_csv,MAX_CSV_BYTES
from src.prediction import predict
from src.validation import InputValidationError

def test_batch_parity_and_order():
    data=original_split()[1].head(4)
    result=predict_csv(data.to_csv(index=False).encode())
    assert list(result.predicted_category)==list(predict(data).predicted_category)
    assert list(result.age)==list(data.age)
    assert len(result)==4 and result.model_id.nunique()==1

@pytest.mark.parametrize('content',[b'',b'age,age\n17,18\n',b'not-a-table',b'\xff\xfe',b'x'*(MAX_CSV_BYTES+1)])
def test_malformed_csv(content):
    with pytest.raises(InputValidationError): parse_csv(content)

def test_missing_extra_and_bad_row():
    data=original_split()[1].head(2)
    with pytest.raises(InputValidationError): parse_csv(data.assign(name='private').to_csv(index=False).encode())
    data.loc[data.index[1],'age']=100
    with pytest.raises(InputValidationError): predict_csv(data.to_csv(index=False).encode())
