import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve,precision_recall_curve

def evaluate(model,X,y):
    pred=model.predict(X);prob=model.predict_proba(X)[:,list(model.classes_).index(1)]
    fpr,tpr,_=roc_curve(y,prob);precision,recall,_=precision_recall_curve(y,prob)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'rocAuc':float(roc_auc_score(y,prob)),'averagePrecision':float(average_precision_score(y,prob)),'confusionMatrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},'pr':{'precision':precision.tolist(),'recall':recall.tolist()}}
