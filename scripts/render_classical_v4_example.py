#!/usr/bin/env python3
"""Preregistered first case/draw, visual comparison from completed saved volumes."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,yaml
from run_classical_v4 import verify,verify_selection,digest,write


def main():
    cfg=yaml.safe_load((ROOT/'configs/classical_v4_v1.yaml').read_text());out=Path(cfg['output']);verify(cfg,out);selected=verify_selection(out)['selected']
    hist=Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/clustered_v4_one_swap/case_images')
    paths={}
    for condition in cfg['validation_conditions']:
        for method in ['SART','Tikhonov','TV']:
            candidate=selected[method]['measurement_only']['candidate'];paths[method,condition]=out/'workers'/method/'volumes/evaluate'/f'{method}_c{candidate}_val0_d0_{condition}.npy'
        for model in ['SelfSup','Supervised']:paths[model,condition]=hist/f'{model}_val0_{condition}.npz'
    allowed={p.resolve() for p in paths.values()};gtpath=hist/'SelfSup_val0_Original.npz';allowed.add(gtpath.resolve());touched=[]
    active=True
    def audit(event,args):
        if not active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix in ('.npy','.npz'):
            if p not in allowed:raise PermissionError('Only frozen first-case saved rendering inputs')
            touched.append(str(p))
    sys.addaudithook(audit)
    with np.load(gtpath) as z:gt=z['GT']
    fields={}
    for (model,condition),p in paths.items():
        if p.suffix=='.npy':fields[model,condition]=np.load(p)
        else:
            with np.load(p) as z:fields[model,condition]=z[condition]
    active=False
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=['GT','SART','Tikhonov','TV','SelfSup','Supervised'];fig,axes=plt.subplots(3,6,figsize=(15,8));maxv=float(gt.max())
    for i,condition in enumerate(cfg['validation_conditions']):
        for j,name in enumerate(names):
            c=gt if name=='GT' else fields[name,condition]
            ax=axes[i,j];image=ax.imshow(c.max(1).T,origin='lower',extent=(0,.6,0,.6),vmin=0,vmax=maxv,cmap='magma',aspect='equal')
            ax.set_title(f'{name}: peak={c.max():.3g}',fontsize=9)
            if j==0:ax.set_ylabel(condition+'\nZ (m)')
            if i==2:ax.set_xlabel('X (m)')
    fig.suptitle('Frozen Val case0 / draw0: max over Y; C mass fraction; GT display scale only')
    fig.subplots_adjust(right=.93,wspace=.2,hspace=.3);fig.colorbar(image,cax=fig.add_axes([.95,.15,.015,.65]),label='CH4 mass fraction')
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/reconstruction_example.png',dpi=170);plt.close(fig)
    write(out/'render_audit.json',dict(case_index=0,draw=0,selection='original figure case0; not selected by reconstruction error',GT_used_for_solver=False,GT_scale_for_display_only=True,projection='max over Y, XYZ to XZ',touched=touched,input_sha256={str(p):digest(p) for p in sorted(allowed)}))
    print('Fixed first-case visual comparison saved')
if __name__=='__main__':main()
