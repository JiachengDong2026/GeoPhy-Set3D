"""Frozen-weight reference, view-count, coverage and condition experiments."""
import json
from pathlib import Path
import numpy as np
import torch

from geophy_set3d.data.projection_pool_dataset import artifact_path
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import write_json, save_slices, move_batch
from .protocols import evaluation_dataset, evaluate
from .analysis import aggregate_draws, write_csv, metric_degradation, plume_descriptors, correlation, read_jsonl
from .stratification import stratify
from .figures import view_count_figures, coverage_figures, pool_figure, condition_figures


def geometry_generalization(protocol,config,root,model,a):
    ds=evaluation_dataset(config,pool='ood',views=4,evaluation_seed=protocol['reference_seed'])
    b,rows=evaluate(model,ds,config,root/'geometry_generalization/Val-B-V4',condition='Val-B-V4')
    write_json(root/'geometry_generalization/degradation.json',metric_degradation(a,b))
    pool_figure(a,b,root/'figures');return b


def view_count(protocol,config,root,model):
    table=[];aggregates={}
    for pool in protocol['pools']:
        for views in protocol['view_counts']:
            summaries=[];rows=[]
            for draw in protocol['draw_ids']:
                ds=evaluation_dataset(config,pool=pool,views=views,selection='random',draw=draw,evaluation_seed=protocol['evaluation_seed'])
                condition=f"Val-{'A' if pool=='train' else 'B'}-V{views}-draw{draw}"
                s,r=evaluate(model,ds,config,root/'view_count'/condition,condition=condition,draw_id=draw)
                summaries.append(s);rows.extend(r)
            agg=aggregate_draws(summaries,rows);key=f'{pool}_V{views}';aggregates[key]=agg
            table.append(dict(condition='Val-A' if pool=='train' else 'Val-B',pool=pool,views=views,
                              cardinality_extrapolation=views not in protocol['trained_cardinality'],draw_count=len(summaries),
                              **{'mean_'+k:v for k,v in agg['mean_per_case'].items()},
                              **{'draw_std_'+k:v for k,v in agg['std_across_draw_means'].items()},
                              **{'pooled_'+k:v for k,v in agg['pooled'].items()}))
            print('VIEW_COUNT',key,agg['mean_per_case']['relative_l1'],flush=True)
    write_csv(root/'view_count_metrics.csv',table);write_json(root/'view_count/summary.json',aggregates)
    view_count_figures(table,root/'figures');return table


def angular_coverage(protocol,config,root,model):
    table=[];individual=[];aggregates={}
    for views in protocol['angular_coverage']['views']:
        for selection in ('spread','random','clustered'):
            summaries=[];rows=[]
            draws=[0] if selection=='spread' else protocol['draw_ids']
            for draw in draws:
                ds=evaluation_dataset(config,views=views,selection=selection,draw=draw,evaluation_seed=protocol['evaluation_seed'])
                condition=f'Val-A-V{views}-{selection}-draw{draw}'
                s,r=evaluate(model,ds,config,root/'angular_coverage'/condition,condition=condition,draw_id=draw)
                for row in r: row['protocol']=selection
                summaries.append(s);rows.extend(r)
            agg=aggregate_draws(summaries,rows);aggregates[f'V{views}_{selection}']=agg;individual.extend(rows)
            table.append(dict(views=views,protocol=selection,draw_count=len(draws),
                              mean_azimuth_span_deg=float(np.mean([r['azimuth_span_deg'] for r in rows])),
                              **{'mean_'+k:v for k,v in agg['mean_per_case'].items()},
                              **{'draw_std_'+k:v for k,v in agg['std_across_draw_means'].items()}))
            print('COVERAGE',views,selection,agg['mean_per_case']['relative_l1'],flush=True)
    write_csv(root/'angular_coverage_metrics.csv',table)
    (root/'angular_coverage/per_case_per_draw_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in individual))
    write_json(root/'angular_coverage/summary.json',aggregates)
    correlations={str(v):correlation([r for r in individual if r['views']==v],'azimuth_span_deg','relative_l1') for v in (4,6)}
    write_json(root/'angular_coverage/correlations.json',correlations)
    coverage_figures(table,individual,root/'figures');return table


def condition_analysis(protocol,config,root,model):
    enhanced={};rankings={}
    for pool,label,folder in (('train','A','plume_generalization'),('ood','B','geometry_generalization')):
        rows=read_jsonl(root/folder/f'Val-{label}-V4/per_case_metrics.jsonl')
        ds=evaluation_dataset(config,pool=pool,views=4,evaluation_seed=protocol['reference_seed'])
        by_id={r['case_id']:i for i,r in enumerate(ds.records)}
        for row in rows:
            record=ds.records[by_id[row['case_id']]]
            g=np.load(artifact_path(ds.root,record['gt_volume_64_path']),mmap_mode='r')
            row.update(plume_descriptors(g,protocol['plume_active_threshold']))
        ranked=sorted(rows,key=lambda r:r['relative_l1'])
        rankings[label]=dict(best_5=ranked[:5],worst_10=ranked[-10:][::-1])
        enhanced[label]=rows
        figdir=root/'figures/worst_cases'/f'Pool_{label}';figdir.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            for rank,row in enumerate(rankings[label]['worst_10'],1):
                batch=move_batch(projection_pool_collate([ds[by_id[row['case_id']]]]),config['training']['device'])
                save_slices(figdir/f'worst_{rank:02d}.png',batch['gt_volume_64'],model(batch),row['case_id'])
        (root/'condition_stratification'/f'Val-{label}-V4_enriched.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(root/'condition_stratification/rankings.json',rankings)
    groups,continuous=stratify(enhanced['A'],protocol['small_n_threshold'])
    write_csv(root/'condition_stratification/condition_metrics.csv',groups)
    write_json(root/'condition_stratification/continuous_condition_correlations.json',continuous)
    descriptors=('gt_max_concentration','gt_mean_concentration','nonzero_voxel_fraction','active_voxel_fraction',
                 'gt_integrated_concentration_proxy','concentration_center_x_m','concentration_center_y_m','concentration_center_z_m')
    correlations={pool:[correlation(rows,x,y) for x in descriptors for y in ('relative_l1','relative_l2')] for pool,rows in enhanced.items()}
    write_json(root/'condition_stratification/plume_correlations.json',correlations)
    condition_figures(groups,enhanced['A'],root/'figures');return groups
