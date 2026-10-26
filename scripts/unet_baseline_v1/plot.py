"""Post-run descriptive pilot curves; no model/configuration selection."""
from unet_paths import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 root=ROOT/'unet/pilot';e0=read(root/'epoch0.json');rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
 epochs=[0]+[r['epoch'] for r in rows]
 h=[e0['val']['measurement']['heldout']['relative_l1']]+[r['val']['measurement']['heldout']['relative_l1'] for r in rows]
 loss=[e0['train']['loss']['total']]+[r['train']['loss']['total'] for r in rows]
 means=[e0['val']['output']['mean']]+[r['val']['output']['mean'] for r in rows]
 fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
 axes[0].plot(epochs,h);axes[0].axhline(.99,color='red',ls='--',label='gate 0.99');axes[0].axhline(1,color='gray',ls=':',label='zero field');axes[0].set(ylabel='Val H relative L1');axes[0].legend(fontsize=8)
 axes[1].plot(epochs,loss);axes[1].set(ylabel='train total objective')
 axes[2].semilogy(epochs,means);axes[2].axhline(1e-8,color='red',ls='--',label='mean gate');axes[2].set(ylabel='Val mean (mass fraction)');axes[2].legend(fontsize=8)
 for ax in axes:ax.set(xlabel='epoch')
 fig.savefig(ART/'pilot_curves.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
