#!/usr/bin/env python3
"""Reuse frozen B0/B1 and evaluate only the new B2 under the same validation protocol."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
import yaml
from geophy_set3d.models import build_model
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.build_utils import source_snapshot,digest
from geophy_set3d.evaluation.protocols import evaluation_dataset,evaluate,load_model
from geophy_set3d.evaluation.analysis import aggregate_draws,write_csv
from geophy_set3d.training.utils import load_config,seed_everything,move_batch,permute_views,write_json
from geophy_set3d.training.comparison import assert_fair_configs,assert_fair_b1_b2

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_b1_b2_300')
REFERENCE=ROOT.parent/'b0_vs_b1_300'
LABELS=['B0-300','B1-GCA-300','B2-GCSI-300']


def prepare(configs):
    assert_fair_configs(configs[0],configs[1]);assert_fair_b1_b2(configs[1],configs[2])
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(configs=configs,reference_summary=str(REFERENCE/'summary.json'),
        reference_summary_sha256=digest(REFERENCE/'summary.json'),train_seed=20261011,
        validation_seed=20261011,validation_epoch=0,validation_views=4,
        model_selection='minimum fixed Val-A-V4 total supervised loss',epochs=300,
        evaluation_seed=20261011,sparse_views=[3,4,6],draw_ids=list(range(5)),
        clustered_views=4,clustered_draw_ids=list(range(5)),spread_views=4,spread_draw_ids=[0],
        test_used=False,pool_b_evaluation_only=True,retrain_b0_b1=False,
        pairwise_geometry_bias=False,interaction='one per-voxel view-attention block; d32/h4/FFN64',
        diagnostics_val_indices=[0,18,35],
        engineering_go=dict(max_relative_main_l1_worsening=.02,clear_sparse_l1_absolute_gain=.005,
                            shortcut_min_ratio=1.2,fp32_set_max_difference=1e-5),
        engineering_go_note='2% relative main-metric tolerance and 0.5pp sparse gain are simple operational references, not statistical thresholds; retain raw deltas')
    target=ROOT/'effective_protocol.yaml'
    if target.exists() and yaml.safe_load(target.read_text())!=protocol:
        raise ValueError('B2 effective protocol changed')
    if not target.exists():target.write_text(yaml.safe_dump(protocol,sort_keys=False))
    inventory=ROOT/'inventory_before.json'
    if not inventory.exists():
        base=ROOT.parent.parent;dataset=source_snapshot(configs[2]['dataset']['root'])
        if dataset!=json.loads((REFERENCE/'inventory_before.json').read_text())['dataset']:
            raise AssertionError('Frozen dataset differs from completed B0/B1 baseline')
        directories=[REFERENCE,base/'evaluations/supervised_b0_generalization_v1']
        directories += [base/k/name for k in ('logs','checkpoints','figures')
                        for name in ('supervised_b0','supervised_b0_300','supervised_b1_gca_300','supervised_b1_gca_300_overfit')]
        files=[p for p in (REPO/'src/geophy_set3d/models').glob('*.py')
               if p.name not in ('__init__.py','set_interaction.py','geolift_gcsi_res3d.py')]
        files += [REPO/'configs/train_supervised_b0.yaml',REPO/'configs/train_supervised_b0_300.yaml',
                  REPO/'configs/train_supervised_b1_300.yaml',*(REPO/'docs/reports').glob('supervised_b[01]*.md')]
        write_json(inventory,dict(dataset=dataset,
            existing_artifacts={str(f.relative_to(base)):digest(f) for directory in directories for f in directory.rglob('*') if f.is_file()},
            protected_code={str(f.relative_to(REPO)):digest(f) for f in files},
            frozen_reference_summary_sha256=digest(REFERENCE/'summary.json')))
    manifest=ROOT/'training_source_manifest.json'
    if not manifest.exists():
        sources=[*(REPO/'src/geophy_set3d/models').glob('*.py'),*(REPO/'src/geophy_set3d/training').glob('*.py'),
                 REPO/'scripts/train_supervised.py',REPO/'scripts/overfit_supervised_b2.py',REPO/'configs/train_supervised_b2_300.yaml']
        write_json(manifest,dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            sha256={str(p.relative_to(REPO)):digest(p) for p in sources}))
    return protocol


def integrity(config):
    before=json.loads((ROOT/'inventory_before.json').read_text());base=ROOT.parent.parent
    sources=json.loads((ROOT/'training_source_manifest.json').read_text())['sha256']
    current=source_snapshot(config['dataset']['root'])
    result=dict(dataset_unchanged=current==before['dataset'],dataset_files=len(current),
        b0_b1_artifacts_unchanged=all(digest(base/p)==sha for p,sha in before['existing_artifacts'].items()),
        b0_b1_model_config_report_unchanged=all(digest(REPO/p)==sha for p,sha in before['protected_code'].items()),
        numerical_training_sources_unchanged=all(digest(REPO/p)==sha for p,sha in sources.items()),
        test_model_evaluation=False,pool_b_training=False)
    result['passed']=all(result[k] for k in ('dataset_unchanged','b0_b1_artifacts_unchanged',
                                          'b0_b1_model_config_report_unchanged','numerical_training_sources_unchanged'))
    write_json(ROOT/'integrity_summary.json',result)
    if not result['passed']:raise AssertionError('Protected artifacts changed')
    return result


def interaction_diagnostics(model,config,output):
    examples=[];finite=True;masked=True;pool_masked=True;sum_error=0.;pool_sum_error=0.
    permutation=0.;padding=0.;equivariance=0.;nonuniform=0.
    for selection in ('spread','clustered'):
        ds=evaluation_dataset(config,views=4,selection=selection,draw=0)
        for index in (0,len(ds)//2,len(ds)-1):
            sample=ds[index];batch=move_batch(projection_pool_collate([sample]),config['training']['device'])
            with torch.no_grad():
                prediction,diag=model(batch,True)
                a,m,w=diag['interaction_attention'],diag['visibility'],diag['attention_weights']
                pair=m[:,None,:,None]&m[:,None,None,:]
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(w).all())
                masked &= bool((a[~pair.expand_as(a)]==0).all())
                pool_masked &= bool((w[~m]==0).all())
                sum_error=max(sum_error,float((a.sum(3)-m[:,None].float()).abs().max()))
                pool_sum_error=max(pool_sum_error,float((w.sum(1)-m.any(1).float()).abs().max()))
                uniform=pair.float()/m.sum(1)[:,None,None,None].clamp_min(1)
                difference=float((a-uniform).abs().max());nonuniform=max(nonuniform,difference)
                average=a.sum((4,5,6))/m.sum((2,3,4))[:,None,:,None].clamp_min(1)
                ids=sample['input_station_indices'].tolist()
                examples.append(dict(case_id=sample['case_id'],case_index=index,selection=selection,
                    station_ids=ids,station_azimuth_deg=ds.geometry['station_azimuth_deg'][ids].tolist(),
                    attention_matrix_mean_by_head=average[0].cpu().tolist(),
                    attention_matrix_mean=average.mean(1)[0].cpu().tolist(),
                    max_deviation_from_uniform=difference,
                    pooling_mean_weights=w.mean((2,3,4))[0].cpu().tolist()))
                p=torch.tensor([3,1,0,2],device=a.device)
                other,changed=model(permute_views(batch,p),True)
                permutation=max(permutation,float((prediction-other).abs().max()))
                equivariance=max(equivariance,float((diag['context_features'][:,p]-changed['context_features']).abs().max()))
                padded=dict(batch)
                for key,value in batch.items():
                    if isinstance(value,torch.Tensor) and (key.startswith('input_') or key=='view_mask'):
                        padded[key]=torch.cat((value,torch.zeros_like(value[:,:1])),1)
                padded['input_projection'][:,-1]=float('nan')
                other,changed=model(padded,True);padding=max(padding,float((prediction-other).abs().max()))
                masked &= bool((changed['interaction_attention'][:,:,-1]==0).all() and
                               (changed['interaction_attention'][:,:,:,-1]==0).all())
                pool_masked &= bool((changed['attention_weights'][:,-1]==0).all())
    # Directly exercise empty voxel sets and the requested close/close/different controlled geometry.
    device=config['training']['device'];torch.manual_seed(20261011)
    features=torch.randn(1,3,32,2,2,2,device=device)
    geometry=torch.zeros(1,3,4,2,2,2,device=device)
    geometry[:,:,2]=.92;geometry[0,:,3]=torch.tensor([10.,11.,100.],device=device)[:,None,None,None]/180
    visibility=torch.ones(1,3,2,2,2,dtype=torch.bool,device=device);view_mask=torch.ones(1,3,dtype=torch.bool,device=device)
    with torch.no_grad():
        context,a=model.interaction(features,geometry,visibility,view_mask,True)
        redundancy=dict(station_azimuth_deg=[10.,11.,100.],finite=bool(torch.isfinite(a).all()),
            attention_matrix_mean=a.mean((1,4,5,6))[0].cpu().tolist(),
            max_deviation_from_uniform=float((a-1/3).abs().max()),
            interpretation='Functional finite/nonuniform check; no prescribed preferred view or scientific acceptance threshold')
        visibility.zero_();context,a=model.interaction(features,geometry,visibility,view_mask,True)
        fused,w=model.fusion(context,geometry,visibility,view_mask)
        all_invalid=bool((context==0).all() and (fused==0).all() and (a==0).all() and (w==0).all())
    result=dict(attention_finite=finite,interaction_mask_correct=masked,pooling_mask_correct=pool_masked,
        interaction_weight_sum_max_error=sum_error,pooling_weight_sum_max_error=pool_sum_error,
        all_invalid_safe=all_invalid,nonuniform=nonuniform>1e-5,max_deviation_from_uniform=nonuniform,
        prediction_permutation_max_difference=permutation,padding_max_difference=padding,
        context_permutation_equivariance_max_difference=equivariance,examples=examples,redundancy_sanity=redundancy)
    if not all((finite,masked,pool_masked,all_invalid,redundancy['finite'])) or max(permutation,padding)>1e-5 or max(sum_error,pool_sum_error)>1e-6:
        raise AssertionError('B2 mask/finite/set invariance diagnostics failed')
    write_json(output/'interaction_diagnostics.json',result);return result


def pairing(path):
    return [(r['case_id'],r['selected_station_ids'],r['station_azimuth_deg'])
            for r in map(json.loads,path.read_text().splitlines())]


def improvement_table(models):
    rows=[]
    for before,after in [(LABELS[0],LABELS[1]),(LABELS[1],LABELS[2]),(LABELS[0],LABELS[2])]:
        for condition in ('Val-A-V4','Val-B-V4','V3','clustered-V4'):
            def metrics(label):
                m=models[label];return (m['evaluations'] if condition.startswith('Val-') else m['sparse'])[condition]['mean_per_case']
            a,b=metrics(before),metrics(after)
            for metric in ('relative_l1','relative_l2'):
                gain=a[metric]-b[metric]
                rows.append(dict(comparison=f'{after} vs {before}',condition=condition,metric=metric,
                    absolute_improvement=gain,improvement_percentage_points=100*gain,relative_improvement=gain/a[metric]))
    return rows


def make_figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,condition in zip(axes,('Val-A-V4','Val-B-V4')):
        for i,label in enumerate(LABELS):
            m=summary['models'][label]['evaluations'][condition]['mean_per_case']
            ax.bar(np.arange(2)+i*.25,[m['relative_l1'],m['relative_l2']],.25,label=label)
        ax.set_xticks(np.arange(2)+.25,['Relative L1','Relative L2']);ax.set_title(condition);ax.legend(fontsize=8)
    fig.savefig(directory/'b0_b1_b2_main_metrics.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for i,label in enumerate(LABELS):
        ax.bar(np.arange(3)+i*.25,[summary['models'][label]['sparse'][k]['mean_per_case']['relative_l1'] for k in ('V3','V4','V6')],.25,label=label)
    ax.set_xticks(np.arange(3)+.25,['V3','V4','V6']);ax.set_ylabel('Mean per-case relative L1');ax.legend()
    fig.savefig(directory/'b0_b1_b2_sparse_views.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.bar(LABELS,[summary['models'][label]['sparse']['clustered-V4']['mean_per_case']['relative_l1'] for label in LABELS])
    ax.set(ylabel='Mean per-case relative L1',title='Val-A clustered V4: five deterministic draws')
    fig.savefig(directory/'b0_b1_b2_clustered.png',dpi=150);plt.close(fig)
    examples=summary['interaction']['examples'];indices=sorted({e['case_index'] for e in examples})
    fig,axes=plt.subplots(3,2,figsize=(10,10),constrained_layout=True)
    for e in examples:
        ax=axes[indices.index(e['case_index']),0 if e['selection']=='spread' else 1]
        img=ax.imshow(e['attention_matrix_mean'],vmin=0,vmax=1,cmap='viridis')
        labels=[f"{sid}\n{az:.1f}°" for sid,az in zip(e['station_ids'],e['station_azimuth_deg'])]
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_yticks(range(4),labels,fontsize=8)
        ax.set(xlabel='Key view',ylabel='Query view',title=f"Val {e['case_index']}: {e['selection']}")
        fig.colorbar(img,ax=ax,shrink=.7)
    fig.savefig(directory/'set_interaction_example.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--device',default='cuda:3');args=parser.parse_args()
    configs=[load_config(f'configs/train_supervised_b{i}_300.yaml') for i in range(3)]
    protocol=prepare(configs)
    if args.prepare:print('B2 comparison protocol and protected inventory frozen');return
    seed_everything(20261011);torch.set_num_threads(2)
    reference=json.loads((REFERENCE/'summary.json').read_text())
    if not reference['completed']:raise ValueError('B0/B1 reference is incomplete')
    models=copy.deepcopy(reference['models'])
    for label in LABELS[:2]:
        if digest(models[label]['checkpoint'])!=models[label]['checkpoint_sha256']:raise AssertionError('Frozen B0/B1 checkpoint changed')
    config=configs[2];name=config['experiment']['name'];base=Path(config['experiment']['output_root']);logs=base/'logs'/name
    training=json.loads((logs/'training_summary.json').read_text())
    if training['completed_epochs']!=300 or training['stopping_reason']!='max_epochs':raise ValueError('B2 must finish all 300 epochs')
    checkpoint=base/'checkpoints'/name/'best.pt';checkpoint_hash=digest(checkpoint)
    model,ckpt=load_model(checkpoint,args.device)
    if ckpt['config']!=config:raise ValueError('B2 checkpoint config changed')
    config=copy.deepcopy(config);config['training']['device']=args.device
    output=ROOT/name;output.mkdir(exist_ok=True)
    result=dict(logs=str(logs),checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        best_epoch=ckpt['epoch']+1,best_val_loss=ckpt['best_val_metric'],parameter_count=sum(p.numel() for p in model.parameters()),
        interaction_parameter_count=sum(p.numel() for p in model.interaction.parameters()),evaluations={},sparse={})
    selections=json.loads((logs/'validation_station_ids.json').read_text())
    for label in LABELS[:2]:
        if selections!=json.loads((Path(models[label]['logs'])/'validation_station_ids.json').read_text()):raise AssertionError('B2 checkpoint-selection views differ')
    for pool,condition in [('train','Val-A-V4'),('ood','Val-B-V4')]:
        ds=evaluation_dataset(config,pool=pool)
        s,_=evaluate(model,ds,config,output/condition,condition=condition);result['evaluations'][condition]=s
    for metric in ('mae','rmse','relative_l1','relative_l2'):
        actual=result['evaluations']['Val-A-V4']['mean_per_case'][metric];expected=training['best_val_metrics']['mean_per_case'][metric]
        if abs(actual-expected)>1e-7+1e-5*abs(expected):raise AssertionError('B2 best validation metrics failed reproduction')
    for shortcut in ('wrong_geometry','zero_measurement'):
        ds=evaluation_dataset(config);s,_=evaluate(model,ds,config,output/shortcut,condition='Val-A-V4',shortcut=shortcut)
        result['evaluations'][shortcut]=s
    for views,selection,key in [(v,'random',f'V{v}') for v in (3,4,6)]+[(4,'clustered','clustered-V4')]:
        summaries=[];rows=[]
        for draw in range(5):
            ds=evaluation_dataset(config,views=views,selection=selection,draw=draw)
            s,r=evaluate(model,ds,config,output/key/f'draw{draw}',condition=key,draw_id=draw);summaries.append(s);rows.extend(r)
        result['sparse'][key]=aggregate_draws(summaries,rows)
    interaction=interaction_diagnostics(model,config,output);models[LABELS[2]]=result
    ds=evaluation_dataset(config,views=4,selection='spread',draw=0)
    result['spread'],_=evaluate(model,ds,config,output/'spread-V4',condition='spread-V4')
    del model
    # Spread is the only optional new condition for the existing baselines; all other metrics are reused.
    for label,cfg in zip(LABELS[:2],configs[:2]):
        baseline,_=load_model(models[label]['checkpoint'],args.device);cfg=copy.deepcopy(cfg);cfg['training']['device']=args.device
        out=ROOT/cfg['experiment']['name'];out.mkdir(exist_ok=True)
        ds=evaluation_dataset(cfg,views=4,selection='spread',draw=0)
        models[label]['spread'],_=evaluate(baseline,ds,cfg,out/'spread-V4',condition='spread-V4');del baseline
    for path in output.rglob('per_case_metrics.jsonl'):
        suffix=path.relative_to(output)
        for cfg in configs[:2]:
            source=(ROOT if str(suffix).startswith('spread-V4') else REFERENCE)/cfg['experiment']['name']/suffix
            if pairing(path)!=pairing(source):raise AssertionError(f'Station/case pairing differs: {suffix}')
    b1,b2=models[LABELS[1]],models[LABELS[2]]
    a1,a2=[m['evaluations']['Val-A-V4']['mean_per_case']['relative_l1'] for m in (b1,b2)]
    b1l,b2l=[m['evaluations']['Val-B-V4']['mean_per_case']['relative_l1'] for m in (b1,b2)]
    go=protocol['engineering_go'];normal=a2
    criteria=dict(val_a_not_materially_worse=a2<=a1*(1+go['max_relative_main_l1_worsening']),
        val_b_not_materially_worse=b2l<=b1l*(1+go['max_relative_main_l1_worsening']),
        v3_or_clustered_clear_gain=any(b1['sparse'][key]['mean_per_case']['relative_l1']-b2['sparse'][key]['mean_per_case']['relative_l1']>=go['clear_sparse_l1_absolute_gain'] for key in ('V3','clustered-V4')),
        shortcuts_normal=all(b2['evaluations'][key]['mean_per_case']['relative_l1']>normal*go['shortcut_min_ratio'] for key in ('wrong_geometry','zero_measurement')),
        mask_permutation_correct=interaction['interaction_mask_correct'] and interaction['pooling_mask_correct'] and interaction['all_invalid_safe'])
    improvements=improvement_table(models);check=integrity(configs[2])
    summary=dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol=protocol,models=models,interaction=interaction,improvements=improvements,
        go_criteria=criteria,recommend_b3=all(criteria.values()),retain_b2=all(criteria.values()),
        integrity=check,b0_b1_retrained=False,actual_case_station_pairings_identical=True,
        single_training_seed=True,test_used=False,pool_b_evaluation_only=True)
    if digest(checkpoint)!=checkpoint_hash:raise AssertionError('B2 best checkpoint changed during evaluation')
    main_rows=[];sparse_rows=[]
    for label,m in models.items():
        for condition in ('Val-A-V4','Val-B-V4'):
            main_rows.append(dict(model=label,condition=condition,best_epoch=m['best_epoch'],best_val_loss=m['best_val_loss'],**m['evaluations'][condition]['mean_per_case']))
        for condition in ('V3','V4','V6','clustered-V4'):
            sparse_rows.append(dict(model=label,condition=condition,draws=5,**m['sparse'][condition]['mean_per_case']))
    write_json(ROOT/'summary.json',summary);write_csv(ROOT/'comparison.csv',main_rows);write_csv(ROOT/'sparse_views.csv',sparse_rows)
    write_csv(ROOT/'clustered_v4.csv',[r for r in sparse_rows if r['condition']=='clustered-V4'])
    write_csv(ROOT/'improvements.csv',improvements)
    write_csv(ROOT/'spread_v4.csv',[dict(model=k,**m['spread']['mean_per_case']) for k,m in models.items()])
    make_figures(summary)
    from geophy_set3d.evaluation.b2_report import render_b2_report
    render_b2_report(ROOT,REPO)


if __name__=='__main__':main()
