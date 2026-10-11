"""Finish seed/horizon evaluation and emit explicitly defined go/no-go checks."""
import json
from pathlib import Path
import numpy as np
from geophy_set3d.data.build_utils import digest,source_snapshot
from geophy_set3d.training.utils import write_json
from .protocols import (load_protocol,require_reference,freeze_reference,load_model,evaluation_dataset,evaluate,SOURCE,REPO)
from .analysis import write_csv,read_jsonl,METRICS
from .figures import multiseed_figure,horizon_figure


def integrity_check(protocol,config,root):
    before=json.loads((root/'inventory_before.json').read_text())
    old=Path(config['experiment']['output_root'])
    after=dict(dataset=source_snapshot(config['dataset']['root']),source=source_snapshot(SOURCE),
               old_experiment={kind:source_snapshot(old/kind/'supervised_b0') for kind in ('logs','checkpoints','figures')})
    differences={}
    for key in ('dataset','source'):
        differences[key]=[p for p in sorted(set(before[key])|set(after[key])) if before[key].get(p)!=after[key].get(p)]
    differences['old_experiment']={k:[p for p in sorted(set(before['old_experiment'][k])|set(after['old_experiment'][k]))
                                    if before['old_experiment'][k].get(p)!=after['old_experiment'][k].get(p)] for k in before['old_experiment']}
    reference=json.loads((root/'reference.json').read_text())
    model_unchanged=all(digest(REPO/path)==sha for path,sha in reference['model_source_hashes'].items())
    hashes_unchanged=all(digest(Path(config['dataset']['root'])/path)==sha for path,sha in reference['dataset_hashes'].items())
    summary=dict(passed=before==after and model_unchanged and hashes_unchanged,
                 dataset_file_count=len(after['dataset']),source_file_count=len(after['source']),
                 old_experiment_unchanged=before['old_experiment']==after['old_experiment'],
                 checkpoint_unchanged=digest(protocol['reference_checkpoint'])==reference['checkpoint_sha256'],
                 model_source_unchanged=model_unchanged,dataset_hashes_unchanged=hashes_unchanged,changes=differences)
    write_json(root/'inventory_after.json',after);write_json(root/'integrity_summary.json',summary)
    if not summary['passed']:raise AssertionError('Frozen source/dataset/reference/model changed')
    return summary


def summarize_training(protocol,config,root):
    seed_results=[];shortcuts={}
    for seed in protocol['training_seeds']:
        if seed==protocol['reference_seed']:
            checkpoint=Path(protocol['reference_checkpoint'])
        else:
            directory=root/'multiseed'/f'supervised_b0_seed{seed}'
            if not (directory/'logs/training_summary.json').exists():raise RuntimeError(f'Seed {seed} training incomplete')
            checkpoint=directory/'checkpoints/best.pt'
        model,ckpt=load_model(checkpoint,config['training']['device'])
        if ckpt['seed'] != seed:
            raise ValueError('Checkpoint training seed does not match the registered run')
        for section in ('dataset','train','validation','normalization','model','loss','optimizer','loader'):
            if ckpt['config'][section] != config[section]:
                raise ValueError(f'Seed run changed frozen {section} settings')
        expected_training = dict(config['training'], seed=seed)
        if ckpt['config']['training'] != expected_training or ckpt['scheduler']['T_max'] != 150:
            raise ValueError('Multi-seed run changed the frozen 150-epoch training protocol')
        if seed != protocol['reference_seed'] and ckpt['validation_selection_seed'] != protocol['reference_seed']:
            raise ValueError('Seed run changed the validation station subset protocol')
        results={}
        for pool,label in (('train','A'),('ood','B')):
            ds=evaluation_dataset(config,pool=pool,views=4,evaluation_seed=protocol['reference_seed'])
            output=root/'multiseed'/f'seed{seed}_evaluation'/f'Val-{label}-V4'
            result,_=evaluate(model,ds,config,output,condition=f'seed{seed}-Val-{label}-V4')
            results[label]=result
        row=dict(seed=seed,best_epoch=ckpt['epoch']+1,checkpoint_sha256=digest(checkpoint),
                 **{f'val_{label.lower()}_{metric}':results[label]['mean_per_case'][metric] for label in ('A','B') for metric in METRICS})
        seed_results.append(row)
        if seed!=protocol['reference_seed']:
            ds=evaluation_dataset(config,views=4,evaluation_seed=protocol['reference_seed'])
            shortcuts[str(seed)]={'normal':results['A']}
            for shortcut in ('zero_measurement','wrong_geometry'):
                result,_=evaluate(model,ds,config,root/'multiseed'/f'seed{seed}_evaluation'/shortcut,
                                  condition=f'seed{seed}-{shortcut}',shortcut=shortcut)
                shortcuts[str(seed)][shortcut]=result
        print('SEED_EVALUATED',seed,row['val_a_relative_l1'],row['val_b_relative_l1'],flush=True)
        del model
    mean={k:float(np.mean([r[k] for r in seed_results])) for k in seed_results[0] if k not in ('seed','checkpoint_sha256')}
    std={k:float(np.std([r[k] for r in seed_results],ddof=1)) for k in mean}
    csv_rows=[{k:v for k,v in r.items() if k!='checkpoint_sha256'} for r in seed_results]
    csv_rows += [dict(seed='mean',**mean),dict(seed='std_ddof1',**std)]
    write_csv(root/'multiseed_metrics.csv',csv_rows)
    write_json(root/'multiseed/summary.json',dict(seeds=seed_results,mean=mean,std=std,std_definition='sample std across three training seeds; same reference val station IDs'))
    write_json(root/'multiseed/shortcut_summary.json',shortcuts);multiseed_figure(seed_results,root/'figures')
    directory=root/'horizon_probe/supervised_b0_horizon_probe'
    if not (directory/'logs/training_summary.json').exists():raise RuntimeError('Horizon training incomplete')
    model,ckpt=load_model(directory/'checkpoints/best.pt',config['training']['device'])
    expected_training = dict(config['training'], max_epochs=protocol['horizon_probe']['max_epochs'])
    if ckpt['config']['training'] != expected_training or ckpt['scheduler']['T_max'] != 220:
        raise ValueError('Horizon diagnostic changed settings beyond the registered horizon')
    ds=evaluation_dataset(config,views=4,evaluation_seed=protocol['reference_seed'])
    result,_=evaluate(model,ds,config,root/'horizon_probe/evaluation',condition='Val-A-V4-horizon220')
    original=json.loads((root/'plume_generalization/Val-A-V4/metrics.json').read_text())
    improvement=original['mean_per_case']['relative_l1']-result['mean_per_case']['relative_l1']
    horizon=dict(strategy=protocol['horizon_probe']['strategy'],best_epoch=ckpt['epoch']+1,
                 training_summary=json.loads((directory/'logs/training_summary.json').read_text()),reference=original,diagnostic=result,
                 absolute_relative_l1_improvement=improvement,percentage_point_improvement=improvement*100,
                 meaningful_improvement=improvement>=protocol['horizon_probe']['meaningful_absolute_relative_l1_improvement'],
                 learning_rate_trajectory_changed_from_epoch_zero=True,reference_replaced=False,
                 checkpoint_sha256=digest(directory/'checkpoints/best.pt'))
    write_json(root/'horizon_probe/summary.json',horizon)
    reference_rows=read_jsonl(Path(config['experiment']['output_root'])/'logs/supervised_b0/metrics.jsonl')
    diagnostic_rows=read_jsonl(directory/'logs/metrics.jsonl')
    horizon_figure(reference_rows,diagnostic_rows,root/'figures')
    return dict(seeds=seed_results,mean=mean,std=std),shortcuts,horizon


def final_summary(protocol,config,root,seed_results,shortcuts,horizon):
    view=json.loads((root/'view_count/summary.json').read_text());coverage=json.loads((root/'angular_coverage/summary.json').read_text())
    criteria=protocol['b1_criteria'];mean,std=seed_results['mean'],seed_results['std']
    checks=dict(multiseed_stable=std['val_a_relative_l1']/mean['val_a_relative_l1']<=criteria['seed_relative_l1_cv_max'],
                pool_b_not_catastrophic=mean['val_b_relative_l1']<=criteria['pool_b_relative_l1_max'] and mean['val_b_relative_l1']/mean['val_a_relative_l1']-1<=criteria['pool_b_degradation_max'],
                v3_to_v6_improves=all(view[f'{pool}_V6']['mean_per_case']['relative_l1']<view[f'{pool}_V3']['mean_per_case']['relative_l1'] for pool in protocol['pools']),
                shortcuts_replicated=all(result[shortcut]['mean_per_case']['relative_l1']/result['normal']['mean_per_case']['relative_l1']>=criteria['shortcut_min_relative_l1_ratio'] for result in shortcuts.values() for shortcut in ('zero_measurement','wrong_geometry')),
                horizon_not_limiting=horizon['absolute_relative_l1_improvement']<criteria['horizon_absolute_relative_l1_improvement_max'])
    integrity=integrity_check(protocol,config,root)
    summary=dict(reference_reproduction=json.loads((root/'reference_reproduction.json').read_text()),
        val_a_v4=json.loads((root/'plume_generalization/Val-A-V4/metrics.json').read_text()),
        val_b_v4=json.loads((root/'geometry_generalization/Val-B-V4/metrics.json').read_text()),
        degradation=json.loads((root/'geometry_generalization/degradation.json').read_text()),
        view_count=view,angular_coverage=coverage,multiseed=seed_results,shortcuts=shortcuts,horizon_probe=horizon,
        b1_checks=checks,b1_recommendation='GO to B1' if all(checks.values()) else 'NO-GO; resolve failed baseline characterization checks first',
        integrity=integrity,test_set_used=False,pool_b_training=False,
        pool_b_definition='Pool B consists of independently sampled unseen station poses from the same defined TDLAS acquisition family.')
    write_json(root/'summary.json',summary);return summary
