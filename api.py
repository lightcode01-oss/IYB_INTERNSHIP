"""Local demonstration API. Bind to loopback; see docs/API.md before hosting."""
from typing import Any
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict,Field
from src.prediction import predict,get_pipeline,model_id
from src.validation import InputValidationError,MAX_ROWS

app=FastAPI(title='Student Performance Demonstration',version='1.0.0',description='Portfolio demonstration; not for educational decisions. Probabilities are uncalibrated model estimates.')

class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    records:list[dict[str,Any]]=Field(min_length=1,max_length=MAX_ROWS)

@app.get('/health')
def health():
    try:
        get_pipeline()
        return {'status':'ready','model_id':model_id()}
    except Exception:
        raise HTTPException(503,'Model unavailable') from None

@app.post('/predict')
def prediction(request:PredictionRequest):
    try:
        result=predict(request.records)
        return {'model_id':model_id(),'predictions':result.to_dict(orient='records')}
    except InputValidationError as error:
        raise HTTPException(422,str(error)) from None
    except Exception:
        raise HTTPException(503,'Prediction service unavailable') from None
