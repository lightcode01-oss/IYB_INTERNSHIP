# Batch prediction

Use the Batch tab or call src.batch.predict_csv with UTF-8 CSV bytes. Upload exactly the 30 predictor columns, no more than 500 rows and no more than 1 MiB. Header order may vary. Duplicate headers, blank/nonfinite values, invalid domains, extra identifiers and target columns are rejected. Prediction is all-or-nothing; correct the reported problem and retry.

The reusable workflow returns input rows in their original order, a prediction, three class probabilities and model_id. It does not save uploads or exports. In the UI, choose Download Predictions CSV to save a result locally. Keep real student data outside the repository. The public dataset template in data/demo_input.csv is for demonstration only.
