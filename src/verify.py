"""Integrity checks on the actual trained artefact and persisted evaluation."""
import json
import numpy as np,joblib
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from .train import ROOT,load_data
from .evaluate import evaluate
def main():
 d,X,y=load_data();tr,te=train_test_split(np.arange(len(d)),test_size=.2,stratify=y,random_state=42)
 read=lambda n:json.loads((ROOT/'outputs/metrics'/n).read_text())
 split=read('split.json');assert set(split['trainIds']).isdisjoint(split['testIds']);assert len(te)==854
 selection=read('selection.json');assert not selection['holdoutUsedForSelection'];assert selection['selectedModel']==max(selection['summary'],key=lambda r:r['cvMeanF1'])['model']
 p=joblib.load(ROOT/'models/final_pipeline.joblib');fresh=evaluate(p,X.iloc[te],y.iloc[te]);saved=next(r for r in read('results.json') if r['model']==selection['selectedModel'])
 for metric in ['accuracy','precision','recall','f1','rocAuc','averagePrecision']:assert abs(fresh[metric]-saved[metric])<1e-12
 assert fresh['confusionMatrix']==saved['confusionMatrix'];assert np.array(saved['confusionMatrix']).sum()==len(te)
 # Audit real fit behaviour: means derive only from the fitting subset; predicting
 # validation rows neither resamples them nor mutates fitted imputation statistics.
 a=tr[:900];b=tr[900:1000];fold=clone(p).fit(X.iloc[a],y.iloc[a]);num=fold.named_steps['preprocess'].named_transformers_['numeric'];cols=fold.named_steps['preprocess'].transformers_[0][2]
 np.testing.assert_allclose(num.named_steps['impute'].statistics_,X.iloc[a][cols].mean().values)
 before=num.named_steps['impute'].statistics_.copy();assert len(fold.predict(X.iloc[b]))==len(b);np.testing.assert_array_equal(before,num.named_steps['impute'].statistics_)
 assert 'loan_id' not in p.feature_names_in_ and 'loan_status' not in p.feature_names_in_
 assert read('shap.json')['maxAdditivityError']<1e-5
 print('PASS: disjoint holdout, CV selection, reload metric parity, fold-only imputation, unchanged inference rows, excluded target/ID and SHAP additivity.')
if __name__=='__main__':main()
