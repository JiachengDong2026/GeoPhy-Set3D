"""Plot only frozen measurement-only logs, never load model/data/GT."""
from context import *
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for j,arm in enumerate(['deeponet','fno']):
        p=ROOT/arm/'pilot';h=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()];initial=read(p/'epoch0.json')
        epochs=[0]+[r['epoch'] for r in h]
        for split in ['train','val']:
            y=[initial[split]['measurement']['heldout']['relative_l1']]+[r[split]['measurement']['heldout']['relative_l1'] for r in h]
            axes[0,j].plot(epochs,y,label=split)
        axes[0,j].axhline(1,color='black',linestyle='--',linewidth=1,label='zero field')
        axes[0,j].axhline(.99,color='red',linestyle=':',linewidth=1,label='gate <=0.99')
        axes[0,j].set_yscale('log');axes[0,j].set_ylabel('Held-out measurement Relative L1');axes[0,j].set_title(arm+' (pilot only)');axes[0,j].legend(fontsize=8)
        for key in ['mean','std']:
            y=[initial['val']['output'][key]]+[r['val']['output'][key] for r in h]
            axes[1,j].plot(epochs,y,label=key)
        axes[1,j].set_yscale('log');axes[1,j].set_ylabel('Val field amplitude');axes[1,j].set_xlabel('Completed epoch');axes[1,j].legend()
    fig.savefig(ART/'pilot_measurement_curves.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
