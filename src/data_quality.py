"""Report invalid row positions without exposing raw submitted feature values."""
import pandas as pd
from src.validation import validate_inputs,InputValidationError,MAX_ROWS

def inspect_batch(data):
    if not isinstance(data,pd.DataFrame):
        raise InputValidationError('Data quality checks require a DataFrame')
    if len(data)>MAX_ROWS:
        raise InputValidationError(f'Batch exceeds {MAX_ROWS} rows')
    # Structural errors apply to the entire batch.
    if data.columns.duplicated().any():
        raise InputValidationError('Duplicate column names are not allowed')
    issues=[]
    for position in range(len(data)):
        try: validate_inputs(data.iloc[[position]])
        except InputValidationError as error:
            issues.append({'row':position+1,'issue':str(error).replace('Row 1,',f'Row {position+1},')})
    return {'rows':len(data),'valid_rows':len(data)-len(issues),'invalid_rows':len(issues),'duplicate_rows':int(data.duplicated().sum()),'missing_cells':int(data.isna().sum().sum()),'issues':issues}
