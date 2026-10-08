"""Produce portfolio evidence from persisted evaluation outputs, never typed metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'outputs/figures'
BONE='#f4f2eb';INK='#242923';LIME='#9abb29'
plt.rcParams.update({'figure.facecolor':BONE,'axes.facecolor':BONE,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlepad':22,'savefig.facecolor':BONE})
def save(fig,name):
 FIG.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 im=Image.open(FIG/(name+'.png')).convert('RGB');im.resize((1920,round(im.height*1920/im.width))).save(FIG/(name+'.webp'),quality=90)
 im.thumbnail((960,960));im.save(FIG/(name+'.960.webp'),quality=87)
def main():
 selection=json.loads((ROOT/'outputs/metrics/selection.json').read_text());results=json.loads((ROOT/'outputs/metrics/results.json').read_text());cv=selection['summary'];winner=selection['selectedModel'];selected=next(r for r in results if r['model']==winner)
 names=[r['model'] for r in results];colors=[LIME if n==winner else '#646d61' for n in names]
 fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');cols=['Model','Accuracy','Precision','Recall','F1','ROC AUC','Avg precision'];keys=['accuracy','precision','recall','f1','rocAuc','averagePrecision'];rows=[[r['model']]+[f'{r[k]:.4f}' for k in keys] for r in results]
 table=ax.table(cellText=rows,colLabels=cols,cellLoc='center',loc='center',colWidths=[.23]+[.125]*6);table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2.6)
 for (r,c),cell in table.get_celld().items():
  cell.set_edgecolor('#d5d8cc');cell.set_facecolor(INK if r==0 else ('#e1edb7' if rows[r-1][0]==winner else BONE));cell.set_text_props(color='white' if r==0 else INK)
 ax.set_title('Five models. One reserved holdout.',loc='left',fontsize=22);ax.text(0,.04,f'854 applications · Approved = positive class · selected by training-only nested CV: {winner}',transform=ax.transAxes,fontsize=11);save(fig,'model-comparison')
 fig,ax=plt.subplots(figsize=(11,6));y=np.arange(len(cv));ax.barh(y,[r['cvMeanF1'] for r in cv],xerr=[r['cvStdF1'] for r in cv],color=colors,capsize=5);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(.8,1);ax.set_xlabel('Mean F1 ± population SD across five outer folds');ax.set_title('Model selection stays inside the training partition',loc='left',fontsize=18)
 for i,r in enumerate(cv):ax.text(.805,i,f"{r['cvMeanF1']:.4f} ± {r['cvStdF1']:.4f}",va='center',color=INK if r['model']==winner else 'white',weight='bold')
 save(fig,'cross-validation')
 fig,ax=plt.subplots(figsize=(8,6));matrix=np.array(selected['confusionMatrix']);ax.imshow(matrix,cmap='Greens');ax.set_xticks([0,1],['Predicted Rejected','Predicted Approved']);ax.set_yticks([0,1],['Actual Rejected','Actual Approved']);labels=[['True rejection','False approval'],['False rejection','True approval']]
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{matrix[i,j]}\n{labels[i][j]}',ha='center',va='center',fontsize=15,color='white' if matrix[i,j]>matrix.max()*.75 else INK)
 ax.set_title(f'{winner} · where the errors occur',loc='left',fontsize=18);save(fig,'confusion-matrix')
 for kind,name,title,xlabel,ylabel in [('roc','roc-curves','Ranking outcomes across thresholds','False positive rate','True positive rate'),('pr','pr-curves','Precision and recall across thresholds','Recall','Precision')]:
  fig,ax=plt.subplots(figsize=(10,7))
  for r,col in zip(results,colors):
   curve=r[kind];x=curve['fpr'] if kind=='roc' else curve['recall'];y=curve['tpr'] if kind=='roc' else curve['precision'];metric=r['rocAuc'] if kind=='roc' else r['averagePrecision'];ax.plot(x,y,label=f"{r['model']} · {metric:.4f}",color='#637714' if r['model']==winner else col,linewidth=2.5 if r['model']==winner else 1.5,linestyle='-' if r['model']==winner else ['--',':','-.','--',':'][names.index(r['model'])])
  if kind=='roc':ax.plot([0,1],[0,1],color='#aaa',linestyle=':',label='Chance ranking')
  else:ax.axhline(matrix[1].sum()/matrix.sum(),color='#aaa',linestyle=':',label='Holdout approval prevalence')
  ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02));ax.set_title(title,loc='left',fontsize=19);ax.legend(loc='lower right' if kind=='roc' else 'lower left',frameon=False);save(fig,name)
 fig,ax=plt.subplots(figsize=(11,6));x=np.arange(5);ax.bar(x-.18,[r['defaultCvMeanF1'] for r in cv],.36,label='Default',color='#646d61');ax.bar(x+.18,[r['cvMeanF1'] for r in cv],.36,label='Tuned',color=LIME);ax.set_xticks(x,names,rotation=12);ax.set_ylim(.8,1);ax.set_ylabel('Mean outer-fold F1');ax.set_title('Does tuning improve generalisation?',loc='left',fontsize=19);ax.legend(frameon=False);save(fig,'tuning')
if __name__=='__main__':main()
