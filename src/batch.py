"""All-or-nothing public-demo CSV prediction with bounded input size."""
import csv,io
import pandas as pd
from src.prediction import predict,model_id
from src.validation import InputValidationError,validate_inputs

MAX_CSV_BYTES=1024*1024

def parse_csv(content):
    if not isinstance(content,bytes) or not content or len(content)>MAX_CSV_BYTES:
        raise InputValidationError('Provide a nonempty UTF-8 CSV no larger than 1 MiB')
    try:
        text=content.decode('utf-8-sig')
        header=next(csv.reader(io.StringIO(text)))
        if len(set(header))!=len(header): raise InputValidationError('Duplicate CSV headers are not allowed')
        for row in list(csv.reader(io.StringIO(text)))[1:]:
            if row and len(row)!=len(header):
                raise InputValidationError('Every CSV row must have the same number of fields as the header')
        frame=pd.read_csv(io.StringIO(text))
    except (UnicodeError,StopIteration,pd.errors.ParserError,pd.errors.EmptyDataError) as error:
        raise InputValidationError('CSV could not be read as a rectangular UTF-8 table') from None
    return validate_inputs(frame)

def predict_csv(content):
    frame=parse_csv(content)
    result=pd.concat([frame.reset_index(drop=True),predict(frame)],axis=1)
    result['model_id']=model_id()
    return result
