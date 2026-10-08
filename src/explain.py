"""SHAP explanation after selection; explain approved-class probability."""
import json
import numpy as np,joblib,shap
from sklearn.model_selection import train_test_split
from .train import ROOT,load_data,dump
from .figures import plt,save,LIME
def main():
 _,X,y=load_data();tr,te=train_test_split(np.arange(len(X)),test_size=.2,stratify=y,random_state=42);p=joblib.load(ROOT/'models/final_pipeline.joblib');pre=p.named_steps['preprocess'];m=p.named_steps['classifier'];z=pre.transform(X.iloc[te]).astype(np.float32);bg=pre.transform(X.iloc[tr].sample(100,random_state=42)).astype(np.float32);names=pre.get_feature_names_out()
 from sklearn.ensemble import RandomForestClassifier
 forest=isinstance(m,RandomForestClassifier)
 explainer=shap.TreeExplainer(m,feature_perturbation='tree_path_dependent',model_output='raw') if forest else shap.TreeExplainer(m,data=bg,feature_perturbation='interventional',model_output='probability');values=explainer.shap_values(z);base=explainer.expected_value
 if values.ndim==3:values=values[:,:,1];base=base[1]
 residual=float(np.max(np.abs(values.sum(axis=1)+base-m.predict_proba(z)[:,1])));assert residual<1e-5,residual
 mean=np.abs(values).mean(axis=0);order=np.argsort(mean)[::-1];dump('shap.json',{'method':'Tree-path-dependent Tree SHAP' if forest else 'Interventional Tree SHAP','output':'Approved-class probability','backgroundRows':0 if forest else 100,'background':'Fitted tree leaf frequencies, including SMOTE training observations' if forest else '100 seeded original training rows','explainedRows':len(z),'maxAdditivityError':residual,'features':[{'feature':str(names[i]),'meanAbsoluteShap':float(mean[i])} for i in order]})
 fig,ax=plt.subplots(figsize=(11,7));idx=order[:10][::-1];ax.barh([str(names[i]).replace('_',' ') for i in idx],mean[idx],color=LIME);ax.set_xlabel('Mean absolute contribution to approved-class probability');ax.set_title('What drives the selected model?',loc='left',fontsize=21);save(fig,'shap-summary')
if __name__=='__main__':main()
