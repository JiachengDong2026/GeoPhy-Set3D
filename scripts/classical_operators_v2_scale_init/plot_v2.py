"""Post-hoc descriptive curves, no model selection or configuration changes."""
import csv
from scale_context_base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for col,arm in enumerate(['deeponet','fno']):
  root=ROOT/arm/'pilot';initial=read(root/'epoch0.json');rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
  epochs=[0]+[r['epoch'] for r in rows]
  val=[initial['val']['measurement']['heldout']['relative_l1']]+[r['val']['measurement']['heldout']['relative_l1'] for r in rows]
  loss=[initial['train']['loss']['total']]+[r['train']['loss']['total'] for r in rows]
  means=[initial['val']['output']['mean']]+[r['val']['output']['mean'] for r in rows]
  axes[0,col].plot(epochs,val,label='Val H relative L1');axes[0,col].axhline(.99,color='red',ls='--',label='gate 0.99');axes[0,col].axhline(1,color='gray',ls=':',label='zero field')
  axes[0,col].set(title=arm,xlabel='epoch',ylabel='relative L1');axes[0,col].legend(fontsize=8)
  axes[1,col].semilogy(epochs,means,label='Val field mean');axes[1,col].axhline(1e-8,color='red',ls='--',label='mean gate');axes[1,col].set(xlabel='epoch',ylabel='mass fraction');axes[1,col].legend(fontsize=8)
 fig.savefig(ART/'pilot_curves.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
