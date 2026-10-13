#!/usr/bin/env python3
"""Freeze geometry-only v2 populations, draws, held-out policies and provenance."""
import json
import sys
import subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
import numpy as np
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.evaluation.geometry_strata import population, frozen_subsets
from geophy_set3d.evaluation.geometry_catalog import GeometryCatalog
from geophy_set3d.evaluation.clustered_one_swap import ClusteredEvaluationReadGuard, keyed_rng
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.training.utils import write_json
from diagnose_clustered_one_swap import write_csv

CONFIG = REPO/'configs/evaluate_geometry_strata_v2.yaml'


def main():
    cfg = yaml.safe_load(CONFIG.read_text()); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p=json.loads((out/'protocol.json').read_text())
        assert p['config']==cfg
        assert all(digest(REPO/k)==v for k,v in p['source_sha256'].items())
        assert all(digest(out/k)==v for k,v in p['registration_sha256'].items())
        print('Existing complete registration verified; no regeneration'); return
    data = Path(cfg['dataset']); geompath=data/'geometry/train_geometry.npz'
    with ClusteredEvaluationReadGuard(data, [], out, geometry_only=True) as audit:
        geometry=dict(np.load(geompath, allow_pickle=False))
        origins=geometry['station_origin_m']; assert len(origins)==64
        assert np.all((origins[:,1]>=.25)&(origins[:,1]<=.35))
        records=[r for r in map(json.loads,(data/'source_index.jsonl').read_text().splitlines()) if r['split']=='val']
        assert len(records)==36
        bounds={}; selected=[]
        for views in cfg['views']:
            ids,scores,stats=population(origins,views,cfg['population_seed'],cfg['monte_carlo_count'])
            stats.update(source_geometry_sha256=digest(geompath), source_code_sha256=digest(REPO/'src/geophy_set3d/evaluation/geometry_strata.py'))
            bounds[str(views)]=stats
            selected.extend(frozen_subsets(ids,scores,stats['boundaries'],cfg['subset_seed'],views,cfg['draws']))
            print(f'V{views}: {len(ids)} combinations, boundaries={stats["boundaries"]}',flush=True)
        write_json(out/'boundaries.json',dict(name=cfg['name'],station_count=len(origins),V=bounds,boundary_only_access='origins only; no K96 coverage, measurements, GT or model'))
        write_json(out/'boundary_access_audit.json',audit.summary())
        # Coverage is computed only AFTER all boundary populations are frozen.
        catalog=GeometryCatalog(geometry,cfg['coverage_resolution'],cfg['samples'])
        station_records=[]; descriptors={}
        oldroot=Path('/data/djc/GeoPhy-Set3D/outputs/diagnostics/clustered_v4_one_swap')
        old=json.loads((oldroot/'station_protocol.json').read_text())['records']
        oldlookup={(r['case_id'],r['draw']):r for r in old}
        for i,case in enumerate(records):
            for s in selected:
                station_records.append(dict(case_id=case['case_id'],case_index=i,section='controlled',**s))
            for views in cfg['views']:
                for draw in range(cfg['draws']):
                    sampler=EvaluationViewSampler(geometry['station_azimuth_deg'],views,draw,'train',cfg['natural_seed'],'random')
                    ids=sampler.select(case['case_id']).tolist()
                    station_records.append(dict(case_id=case['case_id'],case_index=i,section='natural',V=views,stratum='natural',draw=draw,
                        station_ids=ids,geometry_id=f'natural-V{views}-{case["case_id"]}-d{draw}'))
            for draw in range(cfg['draws']):
                r=oldlookup[case['case_id'],draw]
                station_records.append(dict(case_id=case['case_id'],case_index=i,section='clustered',V=4,stratum='clustered',draw=draw,
                    station_ids=r['Original'],geometry_id=f'clustered-V4-d{draw}',H_station_ids=r['H']))
        # Same H4 per case and V across low/middle/high and draws whenever practical:
        # outside the union of the 15 controlled inputs. Natural H is keyed per input.
        controlled_h={}
        for case in records:
            for views in cfg['views']:
                excluded={j for s in selected if s['V']==views for j in s['station_ids']}
                allowed=sorted(set(range(len(origins)))-excluded)
                assert len(allowed)>=cfg['heldout_views']
                controlled_h[case['case_id'],views]=sorted(keyed_rng(cfg['heldout_seed'],views,case['case_id'],'controlled-common-H4').choice(allowed,4,replace=False).tolist())
        for r in station_records:
            if 'H_station_ids' not in r:
                if r['section']=='controlled':r['H_station_ids']=controlled_h[r['case_id'],r['V']]
                else:
                    allowed=sorted(set(range(len(origins)))-set(r['station_ids']))
                    r['H_station_ids']=sorted(keyed_rng(cfg['heldout_seed'],r['draw'],r['geometry_id'],'natural-H4').choice(allowed,4,replace=False).tolist())
            assert not set(r['station_ids'])&set(r['H_station_ids'])
            key=tuple(r['station_ids'])
            if key not in descriptors:descriptors[key]=catalog.describe(key)
        write_json(out/'station_subsets.json',dict(controlled_draws=selected,records=station_records))
        write_json(out/'geometry_descriptors.json',[dict(descriptor_id=i,**v) for i,v in enumerate(descriptors.values())])
        write_csv(out/'geometry_descriptors.csv',[dict(descriptor_id=i,**v) for i,v in enumerate(descriptors.values())])
        assert audit.counts['checkpoint_opens']==audit.counts['val_gt64_opens']==audit.counts['val_pool_a_projection_opens']==0
        write_json(out/'registration_access_audit.json',audit.summary())
    sources=sorted(list((REPO/'src').rglob('*.py'))+[CONFIG,Path(__file__),REPO/'scripts/evaluate_geometry_strata_frozen.py',REPO/'tests/test_geometry_strata_v2.py'])
    modelcfg=yaml.safe_load((REPO/'configs/diagnose_clustered_one_swap.yaml').read_text())['models']
    history={str(oldroot/p):digest(oldroot/p) for p in ['station_protocol.json','per_case_draw.json','original_reproduction.json','protocol.json']}
    for name,m in modelcfg.items():
        for views in cfg['views']:
            for draw in range(cfg['draws']):
                p=Path(m['historical_root']).parent/f'V{views}/draw{draw}/C1/per_case_metrics.jsonl';history[str(p)]=digest(p)
    frozen=['boundaries.json','station_subsets.json','geometry_descriptors.json','geometry_descriptors.csv']
    protocol=dict(status='REGISTERED BEFORE CHECKPOINT LOAD OR INFERENCE',config=cfg,models=modelcfg,
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256={str(p.relative_to(REPO)):digest(p) for p in sources},
        registration_sha256={p:digest(out/p) for p in frozen},historical_sha256=history,
        statistics=dict(case_count=36,draws=5,bootstrap_samples=10000,seed=cfg['bootstrap_seed'],unit='case; average five draws before case means',
            correlation='per V/model, 15 controlled geometries: mean error over cases by geometry ID; Spearman with case-bootstrap 95%CI; no p-values',
            gate='At least one model must satisfy all three V: low>middle>high, low-high CI lower>0, at least one gate descriptor rho<=-0.30. Association CI reported descriptively, no extra gate. Conservative stable-ordering operationalization.'),
        heldout='Controlled common H4 per case/V outside union of 15 inputs; natural keyed disjoint H4; clustered historical H4. H never network input.',
        reproduction_tolerance=dict(absolute=1e-8,relative=1e-5),historical_source_change='Authorized geometry_catalog axis/descriptor fix only; old full-src manifests are preserved and not revalidated against new src.',
        GT_policy='Only Val GT64 evaluation after protocol commit; no Pool B, test, GT128 or optimizer.',
        inventory_before_sha256=digest(out/'inventory_before.json'))
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    archive=REPO/'docs/reports/artifacts/geometry_strata_v2';archive.mkdir(parents=True,exist_ok=True)
    import shutil
    for p in frozen+['protocol.json','protocol.sha256','boundary_access_audit.json','registration_access_audit.json']:
        shutil.copy2(out/p,archive/p)
    print(f'Registered {len(station_records)} paired acquisition records; geometry descriptors={len(descriptors)}',flush=True)

if __name__=='__main__':main()
