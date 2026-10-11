#!/usr/bin/env python3
"""Frozen-best-checkpoint validation comparison; never load test or train Pool B."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
import yaml
from geophy_set3d.models.attention_fusion import normalized_attention_entropy
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.build_utils import digest, source_snapshot
from geophy_set3d.evaluation.protocols import evaluation_dataset, evaluate, load_model
from geophy_set3d.evaluation.analysis import aggregate_draws, write_csv
from geophy_set3d.evaluation.comparison_report import render_report
from geophy_set3d.training.comparison import assert_fair_configs
from geophy_set3d.training.utils import load_config, move_batch, permute_views, seed_everything, write_json

ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_vs_b1_300')
CONFIGS = ['configs/train_supervised_b0_300.yaml', 'configs/train_supervised_b1_300.yaml']
LABELS = ['B0-300', 'B1-GCA-300']


def effective_protocol(configs):
    assert_fair_configs(*configs)
    return dict(configs=configs, model_selection='minimum fixed Val-A-V4 total supervised loss',
                epoch_numbering='summary/report: 1-based; checkpoints/logs: 0-based',
                validation_selection_seed=20261011, validation_epoch=0,
                evaluation_seed=20261011, sparse_views=[3, 4, 6], draw_ids=list(range(5)),
                clustered_views=4, clustered_draw_ids=list(range(5)),
                test_used=False, pool_b_evaluation_only=True,
                geometry_descriptor=['endpoint-normalized alpha', 'endpoint-normalized beta',
                                     'distance / frozen max scanner radius', 'azimuth / frozen max azimuth'],
                entropy='mean H/log(number visible), voxels with >=2 visible views only',
                loss='SmoothL1(beta=1) + 0.1 * mean per-case relative L1')


def freeze_protocol(configs):
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = effective_protocol(configs)
    target = ROOT/'effective_protocol.yaml'
    if target.exists() and yaml.safe_load(target.read_text()) != protocol:
        raise ValueError('Effective comparison protocol changed')
    if not target.exists():
        target.write_text(yaml.safe_dump(protocol, sort_keys=False))
    inventory=ROOT/'inventory_before.json'
    if not inventory.exists():
        base=Path(configs[0]['experiment']['output_root'])
        directories=[base/k/'supervised_b0' for k in ('logs','checkpoints','figures')]
        directories.append(base/'evaluations/supervised_b0_generalization_v1')
        write_json(inventory, dict(dataset=source_snapshot(configs[0]['dataset']['root']),
            historical={str(p.relative_to(base)):digest(p) for directory in directories
                        for p in directory.rglob('*') if p.is_file()},
            b0_sources={str(p):digest(p) for p in Path('src/geophy_set3d/models').glob('*.py')
                        if p.name not in ('__init__.py','attention_fusion.py','geolift_gca_res3d.py')},
            original_config=digest('configs/train_supervised_b0.yaml')))
    return protocol


def attention_diagnostics(model, config, output):
    examples = []
    finite, invalid_zero, sum_error, permutation_error, padding_error = True, True, 0., 0., 0.
    entropy_sum, entropy_voxels = 0., 0
    # Three predefined val cases, two coverage settings; no selection by performance.
    for selection in ('spread', 'clustered'):
        ds = evaluation_dataset(config, selection=selection, views=4, draw=0)
        for index in (0, len(ds)//2, len(ds)-1):
            sample = ds[index]
            batch = move_batch(projection_pool_collate([sample]), config['training']['device'])
            with torch.no_grad():
                prediction, diagnostics = model(batch, return_attention=True)
                w, m = diagnostics['attention_weights'], diagnostics['visibility']
                finite &= bool(torch.isfinite(w).all())
                invalid_zero &= bool((w[~m] == 0).all())
                sum_error = max(sum_error, float((w.sum(1)-m.any(1).float()).abs().max()))
                eligible = int((m.sum(1)>1).sum())
                entropy_sum += float(normalized_attention_entropy(w, m))*eligible
                entropy_voxels += eligible
                ids = sample['input_station_indices'].tolist()
                # Domain mean is unconditional; visible mean accounts for support differences.
                examples.append(dict(case_id=sample['case_id'], case_index=index, selection=selection,
                    selected_station_ids=ids, station_azimuth_deg=ds.geometry['station_azimuth_deg'][ids].tolist(),
                    mean_attention_weight=w.mean((2,3,4))[0].cpu().tolist(),
                    visible_mean_attention_weight=(w.sum((2,3,4))/m.sum((2,3,4)).clamp_min(1))[0].cpu().tolist(),
                    normalized_entropy=float(normalized_attention_entropy(w,m))))
                other = model(permute_views(batch, torch.tensor([3,1,0,2], device=w.device)))
                permutation_error = max(permutation_error, float((prediction-other).abs().max()))
                padded = dict(batch)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and (key.startswith('input_') or key=='view_mask'):
                        padding = torch.zeros_like(value[:,:1])
                        padded[key] = torch.cat((value, padding),1)
                padded['input_projection'][:,-1] = float('nan')
                padded_prediction, padded_diag = model(padded, return_attention=True)
                padding_error = max(padding_error, float((prediction-padded_prediction).abs().max()))
                invalid_zero &= bool((padded_diag['attention_weights'][:,-1] == 0).all())
    summary = dict(finite=finite, invalid_padded_weight_zero=invalid_zero, weight_sum_max_error=sum_error,
        normalized_entropy=entropy_sum/max(entropy_voxels,1), entropy_eligible_voxels=entropy_voxels,
        permutation_max_difference=permutation_error, padding_max_difference=padding_error, examples=examples)
    if not finite or not invalid_zero or sum_error>1e-6 or max(permutation_error,padding_error)>2e-6:
        raise AssertionError('Frozen B1 attention/set invariance diagnostic failed')
    write_json(output/'attention_diagnostics.json', summary)
    return summary


def check_integrity(config):
    before = json.loads((ROOT/'inventory_before.json').read_text())
    dataset = source_snapshot(config['dataset']['root'])
    base = Path(config['experiment']['output_root'])
    historical = {relative: digest(base/relative) for relative in before['historical']}
    unchanged_sources = {relative:digest(relative) for relative in before['b0_sources']}
    result = dict(dataset_unchanged=dataset==before['dataset'],
        historical_results_unchanged=historical==before['historical'],
        b0_model_sources_unchanged=unchanged_sources==before['b0_sources'],
        b0_150_config_unchanged=digest('configs/train_supervised_b0.yaml')==before['original_config'],
        dataset_files=len(dataset), test_model_evaluation=False, pool_b_training=False)
    result['passed'] = all(result[k] for k in ('dataset_unchanged','historical_results_unchanged',
                                              'b0_model_sources_unchanged','b0_150_config_unchanged'))
    write_json(ROOT/'integrity_summary.json', result)
    if not result['passed']:
        raise AssertionError('Frozen artifacts changed')
    return result


def make_figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=ROOT/'figures'; directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for label,model in summary['models'].items():
        for split,style in [('train','--'),('val','-')]:
            rows=[json.loads(line) for line in Path(model['logs'],'metrics.jsonl').read_text().splitlines()]
            for ax,metric in zip(axes,('relative_l1','relative_l2')):
                ax.plot([r['epoch']+1 for r in rows], [r[split]['metrics']['mean_per_case'][metric] for r in rows],
                        style,label=f'{label} {split}')
                ax.set(xlabel='Epoch',ylabel=metric,yscale='log'); ax.legend(fontsize=8)
    fig.savefig(directory/'training_curves_b0_b1.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,pool in zip(axes,('Val-A-V4','Val-B-V4')):
        for index,label in enumerate(LABELS):
            metrics=summary['models'][label]['evaluations'][pool]['mean_per_case']
            ax.bar(np.arange(2)+index*.35,[metrics['relative_l1'],metrics['relative_l2']],.35,label=label)
        ax.set_xticks(np.arange(2)+.175,['Relative L1','Relative L2']);ax.set_title(pool);ax.legend()
    fig.savefig(directory/'b0_vs_b1_main_metrics.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    conditions=['V3','V4','V6','clustered-V4']
    for index,label in enumerate(LABELS):
        values=[summary['models'][label]['sparse'][c]['mean_per_case']['relative_l1'] for c in conditions]
        ax.bar(np.arange(4)+index*.35,values,.35,label=label)
    ax.set_xticks(np.arange(4)+.175,conditions);ax.set_ylabel('Mean per-case relative L1');ax.legend()
    fig.savefig(directory/'b0_vs_b1_sparse_views.png',dpi=150);plt.close(fig)
    examples=summary['attention']['examples']
    fig,axes=plt.subplots(3,2,figsize=(10,9),constrained_layout=True)
    example_indices=sorted({ex['case_index'] for ex in examples})
    for ex in examples:
        i=example_indices.index(ex['case_index'])
        ax=axes[i,0 if ex['selection']=='spread' else 1]
        ax.scatter(ex['station_azimuth_deg'],ex['mean_attention_weight'])
        for az,w,sid in zip(ex['station_azimuth_deg'],ex['mean_attention_weight'],ex['selected_station_ids']):
            ax.annotate(str(sid),(az,w),fontsize=8)
        ax.set(xlabel='Station azimuth (degrees)',ylabel='Domain mean attention weight',
               title=f"Val case {ex['case_index']}: {ex['selection']}",xlim=(0,180),ylim=(0,1))
    fig.savefig(directory/'attention_example.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--device',default='cuda:3')
    args=parser.parse_args()
    configs=[load_config(path) for path in CONFIGS]
    protocol=freeze_protocol(configs)
    if args.prepare:
        print('Frozen comparison protocol saved');return
    seed_everything(20261011);torch.set_num_threads(2)
    models={};main_table=[];sparse_table=[];selections={};checkpoint_hashes={}
    for label,config in zip(LABELS,configs):
        name=config['experiment']['name'];base=Path(config['experiment']['output_root'])
        logs=base/'logs'/name;ckpt_path=base/'checkpoints'/name/'best.pt'
        training=json.loads((logs/'training_summary.json').read_text())
        if not training['completed'] or training['completed_epochs']!=300 or training['stopping_reason']!='max_epochs':
            raise RuntimeError('Both official runs must finish all 300 epochs before comparison')
        model,ckpt=load_model(ckpt_path,args.device)
        if ckpt['config']!=config or ckpt['seed']!=20261011:
            raise ValueError('Frozen checkpoint configuration differs')
        checkpoint_hashes[label]=digest(ckpt_path)
        selections[label]=json.loads((logs/'validation_station_ids.json').read_text())
        config=dict(config);config['training']=dict(config['training'],device=args.device)
        output=ROOT/name;output.mkdir(exist_ok=True)
        result=dict(logs=str(logs),checkpoint=str(ckpt_path),checkpoint_sha256=checkpoint_hashes[label],
                    best_epoch=ckpt['epoch']+1,best_val_loss=ckpt['best_val_metric'],
                    parameter_count=sum(p.numel() for p in model.parameters()),evaluations={},sparse={})
        for pool,condition in [('train','Val-A-V4'),('ood','Val-B-V4')]:
            ds=evaluation_dataset(config,pool=pool)
            if pool=='train' and {ds[i]['case_id']:ds[i]['input_station_indices'].tolist() for i in range(len(ds))}!=selections[label]:
                raise AssertionError('Evaluation does not reproduce checkpoint-selection views')
            s,_=evaluate(model,ds,config,output/condition,condition=condition)
            result['evaluations'][condition]=s
            main_table.append(dict(model=label,condition=condition,best_epoch=result['best_epoch'],best_val_loss=result['best_val_loss'],**s['mean_per_case']))
        # Verify batch-1 read-only evaluation reproduces best training validation metrics.
        for metric in ('mae','rmse','relative_l1','relative_l2'):
            expected=training['best_val_metrics']['mean_per_case'][metric]
            actual=result['evaluations']['Val-A-V4']['mean_per_case'][metric]
            if abs(actual-expected)>1e-7+1e-5*abs(expected):
                raise AssertionError('Best validation metrics failed reproduction')
        ds=evaluation_dataset(config)
        for shortcut in ('wrong_geometry','zero_measurement'):
            s,_=evaluate(model,ds,config,output/shortcut,condition='Val-A-V4',shortcut=shortcut)
            result['evaluations'][shortcut]=s
        for views,selection,key in [(v,'random',f'V{v}') for v in (3,4,6)]+[(4,'clustered','clustered-V4')]:
            summaries=[];rows=[]
            for draw in range(5):
                ds=evaluation_dataset(config,views=views,selection=selection,draw=draw)
                s,r=evaluate(model,ds,config,output/key/f'draw{draw}',condition=key,draw_id=draw)
                summaries.append(s);rows.extend(r)
            agg=aggregate_draws(summaries,rows);result['sparse'][key]=agg
            sparse_table.append(dict(model=label,condition=key,draws=5,**agg['mean_per_case']))
        if label=='B1-GCA-300':
            attention=attention_diagnostics(model,config,output)
            result['fusion_parameter_count']=sum(p.numel() for p in model.fusion.parameters())
        models[label]=result
        print(label,'Val-A',result['evaluations']['Val-A-V4']['mean_per_case'],flush=True)
        del model
    if selections[LABELS[0]]!=selections[LABELS[1]]:
        raise AssertionError('B0/B1 validation views differ')
    # Check actual saved case/draw pairings, including Pool B, rather than relying on seeds alone.
    for relative in (ROOT/configs[0]['experiment']['name']).rglob('per_case_metrics.jsonl'):
        suffix=relative.relative_to(ROOT/configs[0]['experiment']['name'])
        counterpart=ROOT/configs[1]['experiment']['name']/suffix
        def pairing(path):
            return [(r['case_id'],r['selected_station_ids'],r['station_azimuth_deg'])
                    for r in map(json.loads,path.read_text().splitlines())]
        if pairing(relative)!=pairing(counterpart):
            raise AssertionError(f'B0/B1 case/station pairing differs: {suffix}')
    b0,b1=[models[label] for label in LABELS]
    a0,a1=[m['evaluations']['Val-A-V4']['mean_per_case'] for m in (b0,b1)]
    gain=a0['relative_l1']-a1['relative_l1']
    criteria=dict(val_a_l1_improved=gain>0,val_a_l2_not_worse=a1['relative_l2']<=a0['relative_l2'],
        val_b_l1_not_worse=b1['evaluations']['Val-B-V4']['mean_per_case']['relative_l1']<=b0['evaluations']['Val-B-V4']['mean_per_case']['relative_l1'],
        v3_not_worse=b1['sparse']['V3']['mean_per_case']['relative_l1']<=b0['sparse']['V3']['mean_per_case']['relative_l1'],
        measurement_geometry_used=all(b1['evaluations'][key]['mean_per_case']['relative_l1']>1.2*a1['relative_l1'] for key in ('wrong_geometry','zero_measurement')))
    integrity=check_integrity(configs[0])
    summary=dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol=protocol,models=models,attention=attention,integrity=integrity,
        val_a_l1_absolute_improvement=gain,val_a_l1_improvement_percentage_points=100*gain,
        val_a_l1_relative_improvement=gain/a0['relative_l1'],criteria=criteria,
        engineering_criteria_note='Conservative no-worsening checks; shortcut >1.2x normal follows prior sanity convention; no statistical significance claim',
        b1_gain_strength='meaningful' if gain>=.01 else 'weak' if gain<.005 else 'modest',
        recommend_b2=gain>=.01 and all(criteria.values()),
        single_training_seed=True,test_used=False,pool_b_evaluation_only=True,
        actual_evaluation_case_station_pairings_identical=True)
    for label in LABELS:
        if digest(models[label]['checkpoint'])!=checkpoint_hashes[label]:
            raise AssertionError('Best checkpoint changed during evaluation')
    write_json(ROOT/'summary.json',summary)
    write_csv(ROOT/'comparison.csv',main_table);write_csv(ROOT/'sparse_views.csv',sparse_table)
    make_figures(summary)
    render_report(ROOT, Path(__file__).resolve().parents[1])
    print(json.dumps(dict(gain=gain,criteria=criteria,recommend_b2=summary['recommend_b2']),indent=2))


if __name__=='__main__':main()
