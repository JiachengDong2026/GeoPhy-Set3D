#!/usr/bin/env python3
"""Generate and freeze geometry-only V3/4/6 catalog. No expanded evaluation."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,yaml
from geophy_set3d.evaluation.geometry_catalog import GeometryCatalog
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler


def main():
    cfg=yaml.safe_load((ROOT/'configs/evaluate_geometry_strata_v1.yaml').read_text());out=Path(cfg['output'])
    if (out/'freeze.json').exists():
        freeze=json.loads((out/'freeze.json').read_text())
        for name,sha in freeze['sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
        print('Existing geometry catalog verified; no overwrite');return
    out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(cfg['dataset'])/'geometry/train_geometry.npz') as a:g={k:a[k] for k in a.files}
    catalog=GeometryCatalog(g,cfg['coverage_resolution'],cfg['samples']);rng=np.random.default_rng(cfg['seed'])
    records=[];boundaries={}
    for v in cfg['views']:
        combinations=[rng.choice(len(g['station_origin_m']),v,replace=False).tolist() for _ in range(cfg['pregenerated_combinations_per_V'])]
        descriptions=[catalog.describe(ids) for ids in combinations]
        b=np.quantile([r['center_direction_score'] for r in descriptions],[1/3,2/3]);boundaries[str(v)]=b.tolist()
        for index,row in enumerate(descriptions):
            records.append(dict(section='controlled_geometry_strata',combination_index=index,stratum=['low','middle','high'][int(np.searchsorted(b,row['center_direction_score'],side='right'))],**row))
        for draw in range(5):
            s=EvaluationViewSampler(g['station_azimuth_deg'],v,draw,'train',20261011,'clustered')
            records.append(dict(section='extreme_clustered_pressure',draw=draw,**catalog.describe(s.select('geometry-only'))))
    # Natural random follows frozen seed and case identity; metadata only, no npy reads.
    sources=[json.loads(l) for l in (Path(cfg['dataset'])/'source_index.jsonl').read_text().splitlines() if l.strip()]
    for v in cfg['views']:
        for draw in range(5):
            s=EvaluationViewSampler(g['station_azimuth_deg'],v,draw,'train',20261011,'random')
            for r in sources:
                if r['split']=='val':records.append(dict(section='natural_random',case_id=r['case_id'],draw=draw,**catalog.describe(s.select(r['case_id']))))
    original=json.loads((ROOT/'docs/reports/artifacts/clustered_v4_one_swap/station_protocol.json').read_text())['records']
    paired=[]
    for r in original:
        paired.append(dict(case_id=r['case_id'],draw=r['draw'],conditions={condition:catalog.describe(r[condition]) for condition in ['Original','Targeted','Random','H']}))
    for name,value in [('catalog.json',records),('paired_v4_geometry.json',paired),('boundaries.json',dict(boundaries=boundaries,rule=cfg['stratification'],config=cfg,models_run=False))]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    freeze=dict(sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['catalog.json','paired_v4_geometry.json','boundaries.json']},measurements_read=False,GT_read=False,models_run=False)
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(f'Frozen {len(records)} geometry entries, {len(paired)} paired V4 rows; expanded matrix not run')
if __name__=='__main__':main()
