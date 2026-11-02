from concurrent.futures import ThreadPoolExecutor
import pytest
from src.monitoring import log_prediction_event,load_prediction_logs,compare_prediction_distribution

def test_concurrent_events_not_lost(tmp_path):
    path=tmp_path/'events.csv'
    def log(i): return log_prediction_event('Medium',prediction_id=str(i),log_path=path)
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(log,range(40)))
    assert all(ok for ok,_ in results)
    df=load_prediction_logs(path)
    assert len(df)==40 and df.prediction_id.nunique()==40

def test_unknown_category_rejected(tmp_path):
    assert not log_prediction_event('invalid',log_path=tmp_path/'events.csv')[0]

def test_distribution_distance():
    assert compare_prediction_distribution(['Low','High'],['Low','High'])['total_variation']==0
    assert compare_prediction_distribution(['Low'],['High'])['total_variation']==1
    with pytest.raises(ValueError): compare_prediction_distribution([],['High'])
