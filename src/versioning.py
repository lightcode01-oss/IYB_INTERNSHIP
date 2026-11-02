"""Record artifact provenance without modifying the fitted model."""
import hashlib,json,platform
from importlib.metadata import version
from pathlib import Path
from src.dataset import ROOT,original_split,load_dataset,dataset_hash
from src.prediction import get_pipeline,MODEL_PATH

def build_manifest():
    Xtrain,Xtest,ytrain,ytest=original_split()
    model=get_pipeline()
    return {'dataset':{'source':'https://archive.ics.uci.edu/dataset/320/student+performance','citation':'Cortez, P. (2008). Student Performance. UCI. DOI: 10.24432/C5TG7T','license':'CC BY 4.0','cached_csv_sha256':dataset_hash(),'rows':len(load_dataset()),'features':list(Xtrain.columns),'excluded':['G1','G2','G3','performance_category'],'train_rows':len(Xtrain),'test_rows':len(Xtest),'class_counts':ytrain.value_counts().to_dict()},'split':{'test_size':.2,'random_state':42,'stratify':'performance_category','indices_file':'artifacts/split_indices.json'},'target':{'Low':'G3 < 10','Medium':'10 <= G3 < 15','High':'G3 >= 15'},'model':{'sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),'estimator':type(model.named_steps['classifier']).__name__,'parameters':model.named_steps['classifier'].get_params(),'selection':'training-only five-fold stratified CV Macro F1','classes':list(model.classes_)},'environment':{'python':platform.python_version(),**{p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','streamlit']}},'evaluation_scope':'Historical test set already inspected in earlier days; not independent external validation.'}

def save_manifest():
    manifest=build_manifest()
    train,test,_,_=original_split()
    (ROOT/'artifacts/split_indices.json').write_text(json.dumps({'train':list(map(int,train.index)),'test':list(map(int,test.index))},indent=2)+'\n',encoding='utf-8')
    (ROOT/'artifacts/version_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest
