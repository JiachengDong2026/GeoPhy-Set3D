#!/usr/bin/env python3
"""Prespecified case-level summaries, exploratory associations and fixed figures."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import (
    METRICS, AMPLITUDES, GROUPS, aggregate_cases, holm, paired_summary)
from geophy_set3d.training.utils import write_json
from diagnose_selfsup_v1_c0c1 import configuration, verify, read_rows, modes_complete, ARTIFACT

COLORS=['#0072B2','#009E73','#E69F00','#D55E00','#CC79A7']


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def corr(x,y,rank=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if rank:x=rankdata(x,axis=-1);y=rankdata(y,axis=-1)
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
    return np.divide((x*y).sum(-1),den,out=np.full(np.broadcast_shapes(den.shape,np.shape((x*y).sum(-1))),np.nan),where=den>0)


def correlations(c,normal,case,indices):
    desc=pd.DataFrame(json.loads((Path(c['paths']['output'])/'geometry_layouts.json').read_text())['descriptors']).set_index('geometry_id')
    rows=[]
    layouts=normal.drop_duplicates('geometry_id').sort_values(['group','draw'])[['geometry_id','group']]
    ids=layouts['geometry_id'].tolist();groups=layouts['group'].to_numpy()
    case_ids=case.sort_values('case_index').drop_duplicates('case_id')['case_id'].tolist()
    targets=['delta_relative_l1','delta_rms','delta_context_ratio']
    scopes={'all_G1_G5':np.ones(25,bool),'G1_G4':groups!='G5',**{g:groups==g for g in GROUPS}}
    for yname in targets:
        y=normal.pivot(index='case_id',columns='geometry_id',values=yname).loc[case_ids,ids].to_numpy(float)
        boot=np.einsum('bi,ij->bj',np.stack([np.bincount(a,minlength=36)/36 for a in indices]),y)
        for xname in c['geometry_descriptors']:
            x=desc.loc[ids,xname].to_numpy(float)
            for scope,mask in scopes.items():
                for ranked in [False,True]:
                    coefficient=float(corr(x[mask],y.mean(0)[mask],ranked))
                    samples=corr(x[mask],boot[:,mask],ranked)
                    valid=samples[np.isfinite(samples)]
                    ci=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None]
                    rows.append(dict(kind='geometry_to_layout_mean',scope=scope,x=xname,y=yname,correlation='Spearman' if ranked else 'Pearson',coefficient=coefficient,ci_low=ci[0],ci_high=ci[1],unit='fixed layout mean; case bootstrap',layout_count=int(mask.sum()),case_count=36,p_value=None))
    scopes2={g:case[case.group==g] for g in GROUPS}
    scopes2['G1_G4_case_macro']=case[case.group!='G5'].groupby(['case_id','case_index'],sort=True).mean(numeric_only=True).reset_index()
    scopes2['all_G1_G5_case_macro']=case.groupby(['case_id','case_index'],sort=True).mean(numeric_only=True).reset_index()
    for scope,part in scopes2.items():
        part=part.sort_values('case_index')
        for xname in AMPLITUDES:
            x=part[xname].to_numpy(float)
            for yname in ['delta_relative_l1','delta_observed_relative_l1','delta_heldout_relative_l1']:
                y=part[yname].to_numpy(float)
                for ranked in [False,True]:
                    coefficient=float(corr(x,y,ranked));samples=corr(x[indices],y[indices],ranked)
                    valid=samples[np.isfinite(samples)];ci=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None]
                    rows.append(dict(kind='amplitude_to_case_delta',scope=scope,x=xname,y=yname,correlation='Spearman' if ranked else 'Pearson',coefficient=coefficient,ci_low=ci[0],ci_high=ci[1],unit='36 case means/macro; case bootstrap',layout_count=5 if scope in GROUPS else 20 if scope.startswith('G1_G4') else 25,case_count=36,p_value=None))
    return pd.DataFrame(finite_json(rows))


def discordance(frame,level,eps,indices=None):
    rows=[]
    for group,part in frame.groupby('group'):
        for target in ['relative_l1','heldout_relative_l1']:
            mask=(part.delta_observed_relative_l1 < -eps) & (part['delta_'+target] > eps)
            selected=part.loc[mask,'delta_'+target]
            item=dict(level=level,group=group,event='input_improved_'+target+'_worsened',count=int(mask.sum()),total=len(part),frequency=float(mask.mean()),
                      target_mean_worsening=float(selected.mean()) if len(selected) else None,target_max_worsening=float(selected.max()) if len(selected) else None,
                      input_mean_improvement=float(-part.loc[mask,'delta_observed_relative_l1'].mean()) if len(selected) else None)
            if indices is not None:
                # For raw rows, resample entire cases, never independently resample draws.
                counts=part.assign(event=mask.astype(float)).groupby('case_index').event.mean().sort_index().to_numpy()
                item['frequency_case_ci']=np.quantile(counts[indices].mean(1),[.025,.975]).tolist()
            rows.append(item)
    return rows


def figures(c,out,normal,case,paired):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    x=np.arange(5);before=[];after=[]
    for g in GROUPS:
        p=case[case.group==g];before.append(p.C0_relative_l1.mean());after.append(p.C1_relative_l1.mean())
    ax.bar(x-.18,before,.36,label='C0');ax.bar(x+.18,after,.36,label='C1')
    ax.set(xticks=x,xticklabels=GROUPS,ylabel='3D relative L1',title='Frozen SelfSup v1: 36 cases; five layouts averaged per case');ax.legend()
    save(fig,'group_c0_c1_l1')
    matrix=case.pivot(index='case_index',columns='group',values='delta_relative_l1').loc[range(36),GROUPS].to_numpy();limit=float(np.abs(matrix).max())
    fig,ax=plt.subplots(figsize=(7,10),layout='constrained');im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
    ax.set(xticks=range(5),xticklabels=GROUPS,yticks=range(36),yticklabels=range(36),xlabel='Geometry group',ylabel='Original Val case index',title='C1 - C0 relative L1 (negative = improvement)')
    fig.colorbar(im,ax=ax,label='Case/group delta relative L1');save(fig,'case_group_delta_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,name,label in zip(axes,['observed','heldout'],['Input','Common H4']):
        for g,color in zip(GROUPS,COLORS):
            p=case[case.group==g];ax.scatter(p['delta_'+name+'_relative_l1'],p.delta_relative_l1,color=color,label=g,s=18,alpha=.7)
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set(xlabel=f'{label} measurement delta L1 (C1-C0)',ylabel='3D delta relative L1',title='Each point: one case/group mean')
    axes[1].legend();save(fig,'measurement_vs_voxel_delta')
    desc=pd.DataFrame(json.loads((out/'geometry_layouts.json').read_text())['descriptors']).set_index('geometry_id')
    means=normal.groupby('geometry_id').mean(numeric_only=True).join(desc[['center_direction_score','coarse_effective_rank']],rsuffix='_descriptor')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for g,color in zip(GROUPS,COLORS):
        ids=normal[normal.group==g].geometry_id.unique();p=means.loc[ids]
        axes[0].scatter(p.center_direction_score_descriptor,p.delta_context_ratio,color=color,label=g)
        axes[1].scatter(p.coarse_effective_rank,p.delta_relative_l1,color=color)
        q=case[case.group==g];axes[2].scatter(q.delta_context_ratio,q.delta_relative_l1,color=color,alpha=.7,s=18)
    axes[0].set(xlabel='Original center direction score',ylabel='delta RMS / context0 RMS',title='25 fixed layout means')
    axes[1].set(xlabel='Original coarse 8-cubed effective rank',ylabel='3D delta relative L1',title='25 fixed layout means')
    axes[2].set(xlabel='delta RMS / context0 RMS',ylabel='3D delta relative L1',title='36 case means per group')
    axes[0].legend();save(fig,'geometry_amplitude_delta')
    # Fixed original case indices/groups, both prespecified views; no selection by error.
    records=[json.loads(l) for l in (Path(c['paths']['dataset'])/'source_index.jsonl').read_text().splitlines()];records=[r for r in records if r['split']=='val']
    from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard
    with DiagnosticReadGuard(c['paths']['dataset'],[],out,'gt') as audit:
        for index in c['representative_case_indices']:
            gt=np.load(Path(c['paths']['dataset'])/records[index]['gt_volume_64_path'],allow_pickle=False)
            data=[]
            for g in c['representative_groups']:
                with np.load(out/'case_images'/f'normal_case{index}_V4-{g}-d0.npz',allow_pickle=False) as a:data.append((g,a['C0'],a['C1']))
            maximum=max(float(gt.max()),*(float(v.max()) for _,a,b in data for v in [a,b]))
            delta_max=max(float(np.abs(b-a).max()) for _,a,b in data);delta_max=max(delta_max,1e-12)
            for view in ['slice','ymean']:
                def project(v):return (v[:,c['slice_y_index'],:].T if view=='slice' else v.mean(axis=1).T)
                # Shared scale of displayed fields, rather than the unprojected voxel peak.
                maximum=max(float(project(gt).max()),*(float(project(v).max()) for _,a,b in data for v in [a,b]))
                delta_max=max(max(float(np.abs(project(b-a)).max()) for _,a,b in data),1e-12)
                fig,axes=plt.subplots(3,4,figsize=(13,10),layout='constrained')
                for row,(g,a,b) in enumerate(data):
                    for col,v in enumerate([gt,a,b,b-a]):
                        ax=axes[row,col]
                        if col==3:im=ax.imshow(project(v),origin='lower',extent=(0,.6,0,.6),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-delta_max,vcenter=0,vmax=delta_max))
                        else:con=ax.imshow(project(v),origin='lower',extent=(0,.6,0,.6),cmap='viridis',vmin=0,vmax=maximum)
                        ax.set(xlabel='x (m)',ylabel='z (m)',aspect='equal')
                        if row==0:ax.set_title(['GT64 (posthoc only)','C0','C1','Signed correction C1-C0'][col])
                        if col==0:ax.text(.02,.98,g,transform=ax.transAxes,va='top',color='white',bbox=dict(facecolor='black',alpha=.5))
                fig.colorbar(con,ax=axes[:,:3],shrink=.7,label='CH4 mass fraction (shared)')
                fig.colorbar(im,ax=axes[:,3],shrink=.7,label='Signed CH4 mass-fraction change')
                title=f'fixed y-index {c["slice_y_index"]}, y = {c["slice_y_index"]*.6/63:.5f} m' if view=='slice' else 'mean along height y (full volume)'
                fig.suptitle(f'Original Val index {index}; draw0; {title}\nXYZ volume; y is height; geometry groups fixed before inference')
                save(fig,f'fixed_case{index}_{view}')
        write_json(out/'figure_gt_access.json',audit.summary())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(REPO/'configs/diagnose_selfsup_v1_c0c1.yaml'))
    args=parser.parse_args();c,out=configuration(args.config);verify(c,out);start=time.perf_counter()
    modes_complete(c,out,['normal','zero','signflip','wrong_pairing'])
    assert json.loads((out/'normal_reproduction.json').read_text())['passed']
    for mode in ['normal','zero','signflip','wrong_pairing']:
        assert digest(out/f'{mode}_evaluated.jsonl')==json.loads((out/f'{mode}_gt_complete.json').read_text())['rows_sha256']
    normal=pd.DataFrame(read_rows(out/'normal_evaluated.jsonl'));zero=pd.DataFrame(read_rows(out/'zero_evaluated.jsonl'))
    assert len(normal)==len(zero)==900
    case=aggregate_cases(normal.to_dict('records'));zc=aggregate_cases(zero.to_dict('records'))
    assert len(case)==len(zc)==180
    indices=np.random.default_rng(c['bootstrap_seed']).integers(0,36,(c['bootstrap_samples'],36))
    pairs=[]
    for group in GROUPS:
        a=case[case.group==group].sort_values('case_index');z=zc[zc.group==group].sort_values('case_index')
        assert a.case_id.tolist()==z.case_id.tolist()
        for metric in METRICS:
            pairs.append(dict(group=group,comparison='C1-C0',metric=metric,**paired_summary(a['C0_'+metric],a['C1_'+metric],indices,c['tie_epsilon'])))
            pairs.append(dict(group=group,comparison='zero-normal_C1',metric=metric,**paired_summary(a['C1_'+metric],z['C1_'+metric],indices,c['tie_epsilon'])))
    paired=pd.DataFrame(pairs);primary=(paired.comparison=='C1-C0')&(paired.metric=='relative_l1')
    for mask in [primary,~primary]:paired.loc[mask,'holm_p']=holm(paired.loc[mask,'wilcoxon_p'].to_numpy())
    paired['p_family']=np.where(primary,'five_primary_groups','all_secondary')
    joined=zero.merge(normal[['case_id','geometry_id',*['C1_'+m for m in METRICS]]],on=['case_id','geometry_id'],suffixes=('','_normal'),validate='one_to_one')
    for metric in METRICS:joined['zero_minus_normal_'+metric]=joined['C1_'+metric]-joined['C1_'+metric+'_normal']
    shortcuts=[]
    for mode in ['signflip','wrong_pairing']:
        frame=pd.DataFrame(read_rows(out/f'{mode}_evaluated.jsonl'));assert len(frame)==9
        frame=frame.merge(normal[['case_id','geometry_id',*['C1_'+m for m in METRICS]]],on=['case_id','geometry_id'],suffixes=('','_normal'),validate='one_to_one')
        for metric in METRICS:frame['shortcut_minus_normal_'+metric]=frame['C1_'+metric]-frame['C1_'+metric+'_normal']
        shortcuts.append(frame)
    shortcut=pd.concat(shortcuts,ignore_index=True)
    correlation=correlations(c,normal,case,indices)
    disagreements=discordance(normal,'case_layout_descriptive',c['tie_epsilon'],indices)+discordance(case,'case_group',c['tie_epsilon'],indices)
    amplitude=case.groupby('group')[AMPLITUDES+['ratio_undefined','signed_residual_rms_ppm_m']].mean()
    summary=dict(bootstrap_seed=c['bootstrap_seed'],bootstrap_samples=c['bootstrap_samples'],case_count=36,
                 case_group_count=180,normal_rows=900,zero_rows=900,shortcut_rows=18,
                 primary=paired[primary].to_dict('records'),paired_metrics=paired.to_dict('records'),
                 discordance=disagreements,amplitudes=amplitude.reset_index().to_dict('records'),
                 ratio_undefined_count=int(normal.ratio_undefined.sum()),
                 shortcut_summary=[dict(mode=m,group=g,rows=len(part),**{k:float(part[k].mean()) for k in ['shortcut_minus_normal_relative_l1','shortcut_minus_normal_observed_relative_l1','shortcut_minus_normal_heldout_relative_l1','output_vs_normal_mae']}) for (m,g),part in shortcut.groupby(['mode','group'])],
                 normalization_scale_ppm_m=32284.894580078173,metric_units='C: dimensionless CH4 mass fraction; Y: ppm*m; relative errors: dimensionless; integral: voxel-sum proxy, not physical gas mass',
                 statistical_limits='CI pointwise; five primary Holm tests; secondary one Holm family; correlations exploratory, no p-values; cases/layouts/seeds are not independent replications')
    # Avoid duplicate selected columns in amplitude summaries.
    summary=finite_json(summary)
    write_json(out/'statistics.json',summary)
    for name,frame in [('paired_900.csv',normal),('case_group_180.csv',case),('zero_paired_900.csv',joined),('zero_case_group_180.csv',zc),('shortcut_18.csv',shortcut),('paired_statistics.csv',paired),('correlations.csv',correlation),('amplitude_case_group.csv',case[['case_id','case_index','group',*AMPLITUDES]]),('geometry_correction_900.csv',normal[['case_id','case_index','group','draw','geometry_id',*AMPLITUDES,'delta_relative_l1','delta_observed_relative_l1','delta_heldout_relative_l1','local_observability_mean']])]:frame.to_csv(out/name,index=False)
    write_json(out/'correlations.json',finite_json(correlation.to_dict('records')))
    figures(c,out,normal,case,paired)
    write_json(out/'report_runtime.json',dict(elapsed_seconds=time.perf_counter()-start))
    for path in list(out.glob('*.csv'))+[out/'statistics.json',out/'correlations.json',out/'normal_reproduction.json',out/'report_runtime.json',out/'figure_gt_access.json']:
        shutil.copy2(path,ARTIFACT/path.name)
    shutil.copytree(out/'figures',ARTIFACT/'figures',dirs_exist_ok=True)
    print(json.dumps(summary['primary'],indent=2));print('Report data and fixed figures complete',flush=True)

if __name__=='__main__':main()
