from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def test_single_student_submission():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    button=next(b for b in app.button if b.label=='Predict Performance Tier')
    button.click().run()
    assert not app.exception
    assert app.session_state['active_prediction_category'] in {'Low','Medium','High'}

def test_app_uses_shared_service():
    source=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'serve_prediction(input_df)' in source and 'serve_prediction(df_to_predict)' in source
    assert 'pipeline.predict(' not in source
