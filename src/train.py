import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(key,'1')
import json,hashlib,platform,time
from pathlib import Path
import numpy as np,pandas as pd,joblib,sklearn,imblearn,xgboost
from sklearn.base import clone
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .preprocessing import pipeline,SEED
from .evaluate import evaluate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/metrics'
SPECS={
 'Logistic Regression':(LogisticRegression(max_iter=2000,random_state=SEED),{'C':[.01,.1,1,10],'solver':['lbfgs','liblinear']}),
 'Random Forest':(RandomForestClassifier(random_state=SEED,n_jobs=1),{'n_estimators':[100,200],'max_depth':[None,10,20],'min_samples_leaf':[1,5],'max_features':['sqrt']}),
 'XGBoost':(XGBClassifier(random_state=SEED,eval_metric='logloss',n_jobs=1),{'n_estimators':[100,200],'max_depth':[3,6],'learning_rate':[.1,.2],'subsample':[.8,1.]}),
 'Gradient Boosting':(GradientBoostingClassifier(random_state=SEED),{'n_estimators':[100,200],'max_depth':[3,5],'learning_rate':[.05,.1,.2],'subsample':[.8,1.]}),
 'SVC':(SVC(probability=True,random_state=SEED),{'C':[.1,1,10],'gamma':['scale','auto'],'kernel':['rbf']})
}
def dump(name,obj):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def load_data():
    d=pd.read_csv(ROOT/'data/loan_approval.csv');d.columns=d.columns.str.strip()
    for c in d.select_dtypes('object'):d[c]=d[c].str.strip()
    assert set(d.loan_status.unique())=={'Approved','Rejected'}
    assert d.loan_id.is_unique and d.duplicated().sum()==0
    return d,d.drop(columns=['loan_id','loan_status']),(d.loan_status=='Approved').astype(int)
def search(model,grid):return GridSearchCV(pipeline(clone(model)),{'classifier__'+k:v for k,v in grid.items()},cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring='f1',n_jobs=4,error_score='raise',return_train_score=False)
def main():
    d,X,y=load_data();tr,te=train_test_split(np.arange(len(d)),test_size=.2,stratify=y,random_state=SEED)
    Xtr,ytr=X.iloc[tr],y.iloc[tr];Xte,yte=X.iloc[te],y.iloc[te]
    dump('dataset.json',{'applications':len(d),'originalColumns':len(d.columns),'predictors':len(X.columns),'approved':int(y.sum()),'rejected':int((1-y).sum()),'missingness':{c:{'count':int(d[c].isna().sum()),'percent':float(d[c].isna().mean()*100)} for c in d if d[c].isna().any()},'duplicates':int(d.duplicated().sum()),'uniqueIds':int(d.loan_id.nunique()),'sha256':hashlib.sha256((ROOT/'data/loan_approval.csv').read_bytes()).hexdigest(),'trainRows':len(tr),'testRows':len(te),'positiveClass':'Approved','seed':SEED})
    dump('split.json',{'trainIds':d.iloc[tr].loan_id.tolist(),'testIds':d.iloc[te].loan_id.tolist()})
    folds=[];models=list(SPECS)
    # The 854 holdout rows are excluded from every outer and inner selection fold.
    outer=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for fold,(a,b) in enumerate(outer.split(Xtr,ytr),1):
        for name,(model,grid) in SPECS.items():
            start=time.time();base=pipeline(clone(model)).fit(Xtr.iloc[a],ytr.iloc[a]);baseline=evaluate(base,Xtr.iloc[b],ytr.iloc[b]);gs=search(model,grid).fit(Xtr.iloc[a],ytr.iloc[a]);result=evaluate(gs.best_estimator_,Xtr.iloc[b],ytr.iloc[b])
            folds.append({'fold':fold,'model':name,'trainRows':len(a),'validationRows':len(b),'default':baseline,'tuned':result,'bestParameters':gs.best_params_,'innerF1':float(gs.best_score_)})
            dump('outer-folds.json',folds);print(f'Fold {fold}/5 | {name} | F1 {result["f1"]:.4f} | {time.time()-start:.1f}s',flush=True)
    summary=[]
    for name in models:
        rows=[r for r in folds if r['model']==name];scores=[r['tuned']['f1'] for r in rows]
        summary.append({'model':name,'cvMeanF1':float(np.mean(scores)),'cvStdF1':float(np.std(scores,ddof=0)),'defaultCvMeanF1':float(np.mean([r['default']['f1'] for r in rows])),'cvMeanAccuracy':float(np.mean([r['tuned']['accuracy'] for r in rows])),'foldF1':scores})
    winner=sorted(summary,key=lambda r:(-r['cvMeanF1'],r['cvStdF1']))[0]['model']
    # Persist the decision before evaluating holdout outcomes.
    dump('selection.json',{'selectedModel':winner,'criterion':'Highest mean F1 across five outer folds on the 80% training partition; lower SD breaks exact ties','summary':summary,'holdoutUsedForSelection':False,'preprocessingInsideInnerFolds':True,'samplingInsideInnerFolds':True,'outerSplits':5,'innerSplits':5,'candidates':{n:len(ParameterGrid(g)) for n,(_,g) in SPECS.items()}})
    results=[];(ROOT/'models').mkdir(exist_ok=True)
    for name,(model,grid) in SPECS.items():
        print(f'Final training | {name}',flush=True);gs=search(model,grid).fit(Xtr,ytr);result=evaluate(gs.best_estimator_,Xte,yte);default=pipeline(clone(model)).fit(Xtr,ytr)
        results.append({'model':name,**result,'defaultTest':evaluate(default,Xte,yte),'bestParameters':gs.best_params_, 'innerF1':float(gs.best_score_)})
        if name==winner:joblib.dump(gs.best_estimator_,ROOT/'models/final_pipeline.joblib')
    dump('results.json',results)
    dump('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikitLearn':sklearn.__version__,'imbalancedLearn':imblearn.__version__,'xgboost':xgboost.__version__})
    print('Selected by training-only nested validation:',winner,flush=True)
if __name__=='__main__':main()
