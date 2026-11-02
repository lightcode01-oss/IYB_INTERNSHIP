"""Audit historical model selection and estimate conditional sampling uncertainty."""
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
from src.dataset import ROOT,original_split
from src.prediction import get_pipeline,model_id
from src.project_paths import result_path

LABELS=['Low','Medium','High']

def review_evaluation(repeats=500,seed=42):
    Xtrain,Xtest,ytrain,ytest=original_split()
    meta=json.loads((ROOT/'artifacts/student_performance_metadata.json').read_text())
    tuning=pd.read_csv(result_path('day_15_tuning_results.csv')).sort_values('Tuned Mean CV Macro F1',ascending=False)
    champion=tuning.iloc[0]
    assert champion.Model==meta['model_selection']['selected_model']
    assert np.isclose(champion['Tuned Mean CV Macro F1'],meta['model_selection']['training_cv_macro_f1'])
    model=get_pipeline()
    for key,value in meta['model_selection']['selected_hyperparameters'].items():
        assert model.named_steps['classifier'].get_params()[key]==value
    predicted=model.predict(Xtest)
    saved=pd.read_csv(result_path('day_16_error_analysis.csv'),index_col=0)
    assert list(saved.index)==list(Xtest.index)
    assert np.array_equal(predicted,saved.predicted_category)
    actual=ytest.to_numpy()
    metric={'accuracy':float(accuracy_score(actual,predicted)),'macro_precision':float(precision_score(actual,predicted,average='macro',zero_division=0)),'macro_recall':float(recall_score(actual,predicted,average='macro',zero_division=0)),'macro_f1':float(f1_score(actual,predicted,average='macro',zero_division=0))}
    for key,value in metric.items(): assert np.isclose(value,meta['test_metrics'][key])
    dummy=DummyClassifier(strategy='most_frequent').fit(Xtrain,ytrain)
    dummy_score=float(f1_score(actual,dummy.predict(Xtest),average='macro',labels=LABELS,zero_division=0))
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(actual==label) for label in LABELS]
    scores=[]
    for _ in range(repeats):
        indices=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        scores.append(f1_score(actual[indices],predicted[indices],average='macro',labels=LABELS,zero_division=0))
    interval=list(map(float,np.percentile(scores,[2.5,97.5])))
    return {'model_id':model_id(),'selected_model':champion.Model,'training_cv_macro_f1':float(champion['Tuned Mean CV Macro F1']),'training_cv_std':float(champion['Tuned Macro F1 Standard Deviation']),'historical_test_rows':len(ytest),'metrics':metric,'majority_baseline_macro_f1':dummy_score,'conditional_bootstrap_macro_f1_95_percentile':interval,'bootstrap_repeats':repeats,'bootstrap_seed':seed,'classification_report':classification_report(actual,predicted,labels=LABELS,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(actual,predicted,labels=LABELS).tolist(),'class_order':LABELS,'limitations':['Historical test set reused during earlier exploration; not fresh external validation.','Bootstrap holds class counts fixed and conditions on this fitted model; it does not account for training or model-selection uncertainty.','Small observational dataset from two Portuguese schools; no causal or educational decision validity.','Class probabilities are not calibrated.']}
