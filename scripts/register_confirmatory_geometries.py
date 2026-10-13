#!/usr/bin/env python3
"""Geometry-only confirmation, new unordered subsets and common H4 per draw."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard,keyed_rng
from geophy_set3d.evaluation.geometry_strata import population
from geophy_set3d.evaluation.geometry_sensitivity import group_masks,circular_span,spatial_descriptors
from geophy_set3d.evaluation.geometry_catalog import GeometryCatalog
from geophy_set3d.training.utils import write_json
from run_geometry_augmentation_formal import configuration

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');a=ap.parse_args();c,out=configuration(a.config);dest=out/'confirmatory_geometries'
    assert json.loads((out/'registration/control_reuse_audit.json').read_text())['CONTROL_REUSE']=='REJECTED'
    if (dest/'evaluation_records.json').exists():print('Confirmation already registered; not regenerated');return
    dest.mkdir(exist_ok=True);old=json.loads((Path(c['paths']['sensitivity'])/'geometry_layouts.json').read_text());excluded={tuple(sorted(r['station_ids'])) for r in old['layouts']}
    with ClusteredEvaluationReadGuard(c['paths']['dataset'],[],out,geometry_only=True) as guard:
        g=dict(np.load(Path(c['paths']['dataset'])/'geometry/train_geometry.npz'));ids,scores,pop=population(g['station_origin_m'],4,c['confirmatory_seed'])
        spans=circular_span(g['station_azimuth_deg'][ids]);masks=group_masks(scores,spans,old['boundaries'],5.)
        rng=np.random.default_rng(c['confirmatory_seed']);layouts=[]
        for group,mask in masks.items():
            allowed=np.asarray([i for i in np.flatnonzero(mask) if tuple(sorted(ids[i])) not in excluded]);assert len(allowed)>=c['confirmatory_draws']
            for draw,j in enumerate(rng.choice(allowed,c['confirmatory_draws'],replace=False)):
                layouts.append(dict(group=group,draw=draw,geometry_id=f'confirm-V4-{group}-d{draw}',station_ids=rng.permutation(ids[j]).tolist(),center_direction_score=float(scores[j]),span_deg=float(spans[j]),eligible_excluding_development=int(len(allowed))))
        assert len({tuple(sorted(r['station_ids'])) for r in layouts})==50
        cat=GeometryCatalog(g,16,96);descriptors=[dict(cat.describe(r['station_ids']),**spatial_descriptors(g['station_origin_m'][r['station_ids']],g['station_azimuth_deg'][r['station_ids']]),geometry_id=r['geometry_id'],group=r['group'],draw=r['draw']) for r in layouts]
        cases=[r for r in map(json.loads,(Path(c['paths']['dataset'])/'source_index.jsonl').read_text().splitlines()) if r['split']=='val'];assert len(cases)==36
        confirm=[]
        for i,case in enumerate(cases):
            for draw in range(c['confirmatory_draws']):
                union={s for r in layouts if r['draw']==draw for s in r['station_ids']};available=sorted(set(range(64))-union)
                h=sorted(keyed_rng(c['heldout_seed'],draw,case['case_id'],'formal-confirm-H4').choice(available,4,replace=False).tolist())
                for r in layouts:
                    if r['draw']==draw:confirm.append(dict(r,case_id=case['case_id'],case_index=i,H_station_ids=h,section='confirmatory',V=4,stratum=r['group']))
        development=[dict(r,section='development',V=4,stratum=r['group']) for r in json.loads((Path(c['paths']['sensitivity'])/'station_subsets.json').read_text())]
        historical=json.loads((Path(c['paths']['historical_strata'])/'station_subsets.json').read_text())['records']
        natural=[dict(r,group='natural') for r in historical if r['section']=='natural' and r['V']==4];assert len(natural)==180
        records=natural+development+confirm;assert len(records)==2880
        write_json(dest/'geometry_layouts.json',dict(layouts=layouts,descriptors=descriptors,boundaries=old['boundaries'],seed=c['confirmatory_seed'],heldout_seed=c['heldout_seed'],excluded_development_subsets=len(excluded),G5_span_max_deg=5,training_hard_span_max_deg=10,population=pop))
        write_json(dest/'evaluation_records.json',records);write_json(dest/'access_audit.json',guard.summary())
    print('50 new layouts, 1800 confirmation rows; total2880/arm; no GT/model/measurement reads',flush=True)
if __name__=='__main__':main()
