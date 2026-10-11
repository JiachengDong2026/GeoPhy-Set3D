"""Manifest-driven, staged dataset generation and acceptance/freeze workflow."""
import argparse
import collections
import copy
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
from scipy.ndimage import map_coordinates
import yaml
from geophy_set3d.data.build_utils import (timestamp, digest, canonical_hash, atomic_text,
    write_json, write_jsonl, save_npy, save_npz, source_snapshot)
from geophy_set3d.geometry.tdlas import make_geometry
from geophy_set3d.physics.projectors import ReferenceProjector
from geophy_set3d.data.validation import geometry_qa, projector_validation, make_preview

SOURCE = Path('/data/djc/GeoPhy-Set3D/data/raw/Fluent/static_final_frame_partial_v1')
OUTPUT = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
META_FIELDS = ['geometry_id', 'source_family', 'condition_id', 'pressure_pa',
               'wind_speed_m_s', 'active_hole_count', 'source_mode', 'role']


def read_json(path):
    return json.loads(Path(path).read_text())


def load_records(config):
    records = [json.loads(line) for line in (SOURCE/'manifest.jsonl').read_text().splitlines() if line.strip()]
    assert len({r['sample_id'] for r in records}) == len(records), 'Duplicate sample_id'
    accepted, rejected, audits = [], [], []
    for r in records:
        try:
            sample_id=r['sample_id']
            assert sample_id not in ['.','..'] and '/' not in sample_id and '\\' not in sample_id
            for field, directory in [('target_path','targets'),('metadata_path','metadata')]:
                p=(SOURCE/r[field]).resolve()
                assert p.is_relative_to((SOURCE/directory).resolve()) and p.is_file(), field
            meta=read_json(SOURCE/r['metadata_path'])
            for field in META_FIELDS:
                if r.get(field) is None:
                    r[field]=meta.get(field)
            assert r.get('frame_index',meta.get('frame_index')) == 60
            assert r.get('physical_time_s',meta.get('physical_time_s')) == 2.0
            assert r.get('target_axes') == ['x','y','z']
            assert r.get('target_quantity') == 'methane_mass_fraction'
            assert r.get('domain_bounds_m') == dict(x=[0,.6],y=[0,.6],z=[0,.6])
            v=np.load(SOURCE/r['target_path'],mmap_mode='r',allow_pickle=False)
            assert v.shape==(128,128,128) and v.dtype==np.float32
            assert np.isfinite(v).all()
            minimum,maximum=float(v.min()),float(v.max())
            tolerance=config['gt']['range_tolerance']
            assert minimum>=-tolerance and maximum<=1+tolerance, 'Physical range violation'
            clipped=int(np.count_nonzero((v<0)|(v>1)))
            r['case_id']=r.get('case_id') or sample_id
            r['condition_meta']={field:r.get(field) for field in META_FIELDS}
            r['clipped_voxels']=clipped
            audits.append(dict(sample_id=sample_id, shape=list(v.shape),dtype=str(v.dtype),
                               minimum=minimum,maximum=maximum,finite=True,clipped_voxels=clipped,
                               max_underflow=max(0,-minimum),max_overflow=max(0,maximum-1)))
            accepted.append(r)
        except Exception as exc:
            rejected.append(dict(sample_id=r.get('sample_id'),reason=repr(exc)))
    assert accepted, 'No admissible cases'
    audit=dict(passed=True,total_manifest_records=len(records),accepted_count=len(accepted),
               rejected_count=len(rejected),rejected=rejected,gt_checks=audits,
               source_audit_evidence='Source README attests prior audits; manifest records original audit paths. '
                   'Windows audit paths are provenance strings, not accessible server audit files. '
                   'All available GT/frame/schema checks were independently rerun here.',
               source_mode_policy='Missing source_mode is null; no physical metadata inferred from sample filenames.',
               case_id_policy='Use explicit case_id if present; otherwise manifest sample_id identifies this final-frame case.')
    return accepted,audit


def case_splits(records,config):
    """Group identities/reruns and balance available metadata marginals."""
    n=len(records);parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(records):
        keys=[('case',r['case_id']),('trajectory',r.get('source_run_dir')),
              ('condition',(r.get('geometry_id'),r.get('condition_id'))),
              ('gt_hash',digest(SOURCE/r['target_path']))]
        for kind,value in keys:
            if value is None or (kind=='condition' and None in value):continue
            key=(kind,value)
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    groups=collections.defaultdict(list)
    for i in range(n):groups[find(i)].append(i)
    groups=list(groups.values())
    fields=['geometry_id','source_family','active_hole_count','wind_speed_m_s','pressure_pa','source_mode']
    feature_keys=sorted({(f,str(r.get(f))) for f in fields for r in records})
    features=np.array([[int(str(r.get(f))==v) for f,v in feature_keys] for r in records])
    gf=np.array([features[g].sum(0) for g in groups])
    size=np.array([len(g) for g in groups])
    forced=np.array([any(records[i]['role']=='retained_method_validation_evidence' for i in g) for g in groups])
    ratio=np.array([config['split'][x] for x in ['train','val','test']])
    targets=np.floor(ratio*n).astype(int)
    for j in np.argsort(-(ratio*n-targets))[:n-targets.sum()]:targets[j]+=1
    rng=np.random.default_rng(config['split']['seed'])
    assignment=np.full(len(groups),-1);assignment[forced]=0
    counts=np.array([size[forced].sum(),0,0])
    for i in rng.permutation(np.where(~forced)[0]):
        remaining=targets-counts
        possible=np.where(remaining>=size[i])[0]
        assert len(possible),'Grouped cases cannot fit requested split counts'
        probabilities=np.maximum(remaining[possible],0).astype(float);probabilities/=probabilities.sum()
        choice=rng.choice(possible,p=probabilities)
        assignment[i]=choice;counts[choice]+=size[i]
    desired=ratio[:,None]*features.sum(0)
    weights=1/np.maximum(desired,1)
    totals=np.array([gf[assignment==s].sum(0) for s in range(3)],float)
    score=lambda t:float(np.sum((t-desired)**2*weights))
    best=score(totals)
    movable=np.where(~forced)[0]
    for _ in range(20000):
        a,b=rng.choice(movable,size=2,replace=False)
        sa,sb=assignment[a],assignment[b]
        if sa==sb or size[a]!=size[b]:continue
        candidate=totals.copy();candidate[sa]+=gf[b]-gf[a];candidate[sb]+=gf[a]-gf[b]
        value=score(candidate)
        if value<best-1e-9:
            assignment[a],assignment[b]=sb,sa;totals=candidate;best=value
    names=['train','val','test']
    for g,s in zip(groups,assignment):
        for i in g:records[i]['split']=names[s]
    summary=dict(seed=config['split']['seed'],method='Identity/rerun/hash grouping; fixed-count seeded assignment then metadata-marginal swap balancing',
                 requested_ratios=dict(zip(names,ratio.tolist())),counts=dict(zip(names,counts.tolist())),
                 group_count=len(groups),retained_evidence_policy='train_only',balance_score=best,
                 distributions={s:{f:dict(collections.Counter(str(r.get(f)) for r in records if r['split']==s)) for f in fields+['role']} for s in names})
    return records,summary


def leakage_audit(records,path):
    problems=[]
    for kind in ['case_id','source_run_dir','rerun_condition','gt_sha256']:
        split_map=collections.defaultdict(set)
        for r in records:
            value=(r.get('geometry_id'),r.get('condition_id')) if kind=='rerun_condition' else r.get(kind)
            if value is not None:split_map[str(value)].add(r['split'])
        problems.extend(dict(kind=kind,key=k,splits=sorted(v)) for k,v in split_map.items() if len(v)>1)
    retained=[r for r in records if r['role']=='retained_method_validation_evidence']
    assert all(r['split']=='train' for r in retained)
    assert not problems,problems
    result=dict(passed=True,overlap_count=0,problems=problems,
                retained_evidence_count=len(retained),retained_evidence_in_final_test=0,
                projector_and_debug_validation_uses_training_cases_only=True,
                trajectory_identity='manifest source_run_dir; Windows path retained as identity, never scanned',
                rerun_identity='geometry_id + condition_id')
    write_json(path,result);return result


def prepare(config):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'generation_config.yaml').exists():
        assert yaml.safe_load((OUTPUT/'generation_config.yaml').read_text())==config,'Config changed: upgrade dataset version'
    else:atomic_text(OUTPUT/'generation_config.yaml',yaml.safe_dump(config,sort_keys=False))
    before=source_snapshot(SOURCE)
    old=OUTPUT/'audits/source_fingerprint_before.json'
    if old.exists():assert read_json(old)==before,'Source changed since build started'
    else:write_json(old,before)
    records,source_audit=load_records(config)
    records,summary=case_splits(records,config)
    for r in records:
        r['gt_sha256']=before[r['target_path']]['sha256']
    existing=OUTPUT/'splits/split_summary.json'
    if existing.exists():assert read_json(existing)==summary,'Frozen split changed'
    for split in ['train','val','test']:
        atomic_text(OUTPUT/f'splits/{split}_cases.txt',''.join(r['case_id']+'\n' for r in records if r['split']==split))
    write_json(existing,summary)
    write_json(OUTPUT/'audits/source_validation.json',source_audit)
    leakage_audit(records,OUTPUT/'audits/split_leakage_audit.json')
    # Publish the derived manifest before projecting any volume. It is updated
    # to complete only after the full file/measurement audit succeeds.
    if not (OUTPUT/'source_index.jsonl').exists():
        write_jsonl(OUTPUT/'source_index.jsonl',[
            dict(sample_id=r['sample_id'],case_id=r['case_id'],split=r['split'],
                 source_target_path=str(SOURCE/r['target_path']),
                 source_metadata_path=str(SOURCE/r['metadata_path']),
                 condition_meta=r['condition_meta'],source_gt_sha256=r['gt_sha256'],
                 train_projection_path=f'samples/{r["case_id"]}/projections_train.npy',
                 ood_projection_path=f'samples/{r["case_id"]}/projections_ood.npy',
                 generation_status='pending') for r in records])
    return records,before


def load_gt(r):
    v=np.load(SOURCE/r['target_path'],mmap_mode='r',allow_pickle=False)
    return np.clip(v,0,1) if r['clipped_voxels'] else v


def choose_debug_records(records):
    selected=[];families=set()
    for r in records:
        if r['split']=='train' and r['geometry_id'] not in families:
            selected.append(r);families.add(r['geometry_id'])
        if len(selected)==5:break
    assert len(selected)==5
    return selected


def check_array(a,g,count):
    assert a.shape==(count,64,64) and a.dtype==np.float32
    assert np.isfinite(a).all() and np.all(a>=0)
    assert np.count_nonzero(a[~g['valid_ray']])==0
    return dict(minimum=float(a.min()),maximum=float(a.max()),mean=float(a.mean(dtype=np.float64)),
                nonzero_ratio=float(np.count_nonzero(a)/a.size),nan_count=0,inf_count=0,negative_count=0)


def build_sample(record,projector,train,ood,root,resume):
    destination=root/'samples'/record['case_id']
    if resume and (destination/'sample_meta.json').exists():
        try:
            meta=read_json(destination/'sample_meta.json')
            assert meta['source_gt_sha256']==record['gt_sha256']
            assert meta['reference_samples']==projector.config['reference_samples']
            for name,g in [('train',train),('ood',ood)]:
                p=destination/f'projections_{name}.npy'
                assert digest(p)==meta['hashes'][p.name]
                check_array(np.load(p,mmap_mode='r',allow_pickle=False),g,len(g['station_origin_m']))
            p=destination/'gt_volume_64.npy'
            assert digest(p)==meta['hashes'][p.name]
            v=np.load(p,mmap_mode='r',allow_pickle=False)
            assert v.shape==(64,64,64) and v.dtype==np.float32 and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
            assert (destination/'frame_manifest.jsonl').is_file()
            return meta,True
        except (AssertionError,OSError,ValueError,KeyError):
            logging.warning('Existing sample failed resume validation; regenerating %s',record['case_id'])
    stage=root/'samples'/('.'+record['case_id']+'.tmp')
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        gt=load_gt(record)
        results={}
        for name,g in [('train',train),('ood',ood)]:
            value=projector(gt,g)
            results[name]=check_array(value,g,len(g['station_origin_m']))
            save_npy(stage/f'projections_{name}.npy',value)
        grid=np.meshgrid(*([np.linspace(0,127,64)]*3),indexing='ij')
        gt64=map_coordinates(gt,np.array(grid),order=1,mode='nearest',prefilter=False).astype(np.float32)
        save_npy(stage/'gt_volume_64.npy',gt64)
        hashes={p.name:digest(p) for p in stage.glob('*.npy')}
        meta=dict(sample_id=record['sample_id'],case_id=record['case_id'],split=record['split'],
                  frame_index=60,physical_time_s=2.0,gt_path=str(SOURCE/record['target_path']),
                  gt_volume_64_path='gt_volume_64.npy',source_metadata_path=str(SOURCE/record['metadata_path']),
                  source_gt_sha256=record['gt_sha256'],condition_meta=record['condition_meta'],
                  clipped_voxels=record['clipped_voxels'],gt_shape=[128]*3,gt_axes=['x','y','z'],
                  projection_unit='ppm_m',projection_dtype='float32',projection_shape=[64,64],
                  train_geometry_path='../../geometry/train_geometry.npz',ood_geometry_path='../../geometry/ood_geometry.npz',
                  reference_samples=projector.config['reference_samples'],
                  train_projection_shape=[len(train['station_origin_m']),64,64],
                  ood_projection_shape=[len(ood['station_origin_m']),64,64],
                  projection_statistics=results,hashes=hashes,generation_status='complete',
                  downsample_method='Trilinear resampling at linspace(0,0.6,64); endpoint-aligned point grid',
                  generated_at=timestamp())
        write_json(stage/'sample_meta.json',meta)
        write_jsonl(stage/'frame_manifest.jsonl',[dict(frame_index=60,physical_time_s=2.0,
            gt_path=meta['gt_path'],gt_shape=[128]*3,projection_shape=[64,64],projection_unit='ppm_m',
            train_projection_path='projections_train.npy',ood_projection_path='projections_ood.npy',
            gt_volume_64_path='gt_volume_64.npy')])
        if destination.exists():
            # Old incomplete output is preserved until replacement is ready.
            backup=destination.with_name('.'+destination.name+'.backup')
            assert not backup.exists(),'Interrupted replacement requires inspection'
            destination.rename(backup);stage.rename(destination);shutil.rmtree(backup)
        else:stage.rename(destination)
        return meta,False
    except Exception:
        if stage.exists():shutil.rmtree(stage)
        raise


def build_pools(records,config,root,resume):
    root.mkdir(parents=True,exist_ok=True)
    train=make_geometry(config['geometry'],config['geometry']['train_count'],config['geometry']['train_seed'])
    ood=make_geometry(config['geometry'],config['geometry']['ood_count'],config['geometry']['ood_seed'])
    for name,g in [('train',train),('ood',ood)]:
        p=root/f'geometry/{name}_geometry.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as existing:
                assert set(existing.files)==set(g)
                for k in g:np.testing.assert_array_equal(existing[k],g[k])
        else:save_npz(p,g)
    canonical=make_geometry(config['geometry'],6,0,azimuth=[0,36,72,108,144,180])
    if not (root/'geometry/canonical_fixed_geometry.npz').exists():
        save_npz(root/'geometry/canonical_fixed_geometry.npz',canonical)
    write_json(root/'geometry/geometry_profile.json',dict(**config['geometry'],
        formula='d(alpha,beta)=cos(beta)*cos(alpha)*forward + cos(beta)*sin(alpha)*right + sin(beta)*up',
        point_grid='linspace(0,0.6,128); XYZ',oddity='64 even beam samples do not include angle zero; center_direction is stored separately'))
    geometry_qa(train,ood,root/'audits/geometry_audit.json')
    projector=ReferenceProjector(config['projector'],OUTPUT/'cache')
    metas=[];failures=[]
    start=time.monotonic()
    for i,r in enumerate(records):
        try:
            meta,skipped=build_sample(r,projector,train,ood,root,resume)
            metas.append(meta)
            logging.info('%d/%d %s %s elapsed %.1fs',i+1,len(records),'resumed' if skipped else 'generated',r['sample_id'],time.monotonic()-start)
        except Exception as exc:
            logging.exception('Failed %s',r['case_id'])
            failures.append(dict(case_id=r['case_id'],frame_index=60,exception=repr(exc),
                                 geometry_chunk='see traceback in generation log',timestamp=timestamp()))
            write_json(root/'audits/generation_failures.json',failures)
    write_json(root/'audits/generation_failures.json',failures)
    assert not failures, f'{len(failures)} projection failures; see generation_failures.json'
    r=records[0];preview=make_preview(train,ood,load_gt(r),
        np.load(root/'samples'/r['case_id']/'projections_train.npy'),root/'audits/figures/geometry_and_projection.png')
    return train,ood,metas,projector


def audit_pool(records,root,train,ood):
    metas=[];missing=[];hashes={};minimum=float('inf');maximum=0
    for r in records:
        folder=root/'samples'/r['case_id']
        meta=read_json(folder/'sample_meta.json');metas.append(meta)
        assert meta['case_id']==r['case_id'] and meta['split']==r['split'] and meta['source_gt_sha256']==r['gt_sha256']
        for name,g in [('train',train),('ood',ood)]:
            p=folder/f'projections_{name}.npy'
            assert p.is_file()
            a=np.load(p,mmap_mode='r',allow_pickle=False)
            stats=check_array(a,g,len(g['station_origin_m']))
            minimum=min(minimum,stats['minimum']);maximum=max(maximum,stats['maximum'])
            assert digest(p)==meta['hashes'][p.name]
        p=folder/'gt_volume_64.npy';v=np.load(p,mmap_mode='r',allow_pickle=False)
        assert v.shape==(64,64,64) and v.dtype==np.float32 and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
        assert digest(p)==meta['hashes'][p.name]
        for p in folder.iterdir():hashes[p.relative_to(root).as_posix()]=digest(p)
        frames=[json.loads(line) for line in (folder/'frame_manifest.jsonl').read_text().splitlines()]
        assert len(frames)==1 and frames[0]['frame_index']==60 and frames[0]['physical_time_s']==2.0
    actual={p.name for p in (root/'samples').iterdir() if p.is_dir() and not p.name.startswith('.')}
    assert actual=={r['case_id'] for r in records},'Manifest/file membership mismatch'
    for p in (root/'geometry').iterdir():hashes[p.relative_to(root).as_posix()]=digest(p)
    write_json(root/'audits/artifact_hashes.json',hashes)
    result=dict(passed=True,total_accepted_cases=len(records),total_final_frame_samples=len(records),
                split_case_count=dict(collections.Counter(r['split'] for r in records)),
                generated_projection_arrays=len(records)*2,generated_station_maps=len(records)*(len(train['station_origin_m'])+len(ood['station_origin_m'])),
                failed_final_frame_samples=0,nan_count=0,inf_count=0,negative_measurement_count=0,
                gt_range_violations=0,gt_clipped_voxels=sum(r['clipped_voxels'] for r in records),
                projection_min=minimum,projection_max=maximum,missing_file_count=0,
                invalid_ray_nonzero_count=0,valid_ray_statistics={s:dict(fraction=float(g['valid_ray'].mean()),
                    maximum_path_length_m=float(g['path_length_m'].max())) for s,g in [('train',train),('ood',ood)]},
                hash_summary=dict(algorithm='sha256',artifact_count=len(hashes),inventory_sha256=canonical_hash(hashes)))
    write_json(root/'audits/projection_pool_audit.json',result)
    index=[]
    for r in records:
        index.append(dict(sample_id=r['sample_id'],case_id=r['case_id'],split=r['split'],
            source_target_path=str(SOURCE/r['target_path']),source_metadata_path=str(SOURCE/r['metadata_path']),
            condition_meta=r['condition_meta'],source_gt_sha256=r['gt_sha256'],
            train_projection_path=f'samples/{r["case_id"]}/projections_train.npy',
            ood_projection_path=f'samples/{r["case_id"]}/projections_ood.npy',
            gt_volume_64_path=f'samples/{r["case_id"]}/gt_volume_64.npy',generation_status='complete'))
    write_jsonl(root/'source_index.jsonl',index)
    return result


def train_statistics(records,root,train):
    selected=[r for r in records if r['split']=='train']
    def summarize(values,percentiles):
        quantiles=np.percentile(values,percentiles,method='linear')
        return dict(mean=float(np.mean(values,dtype=np.float64)),std=float(np.std(values,dtype=np.float64)),
                    minimum=float(values.min()),maximum=float(values.max()),count=int(values.size),
                    **{'p'+str(p):float(q) for p,q in zip(percentiles,quantiles)})
    projection=np.concatenate([np.load(root/'samples'/r['case_id']/'projections_train.npy').reshape(-1) for r in selected])
    projection_stats=summarize(projection,[50,90,95,99,99.5])
    valid=np.concatenate([np.load(root/'samples'/r['case_id']/'projections_train.npy')[train['valid_ray']] for r in selected])
    valid_stats=summarize(valid,[50,90,95,99,99.5])
    del valid,projection
    gt=np.concatenate([np.asarray(load_gt(r)).reshape(-1) for r in selected])
    gt_stats=summarize(gt,[95,99]);del gt
    result=dict(split='train',case_count=len(selected),geometry_pool='A only; Pool B excluded',
        quantiles='exact full-data linear quantiles, not sampled estimates',projection=projection_stats,
        projection_valid_rays=valid_stats,gt=gt_stats,projection_unit='ppm_m',gt_quantity='CH4 mass fraction',
        normalization=dict(Y_ref_ppm_m=valid_stats['p99.5'],method='fixed scale: Y/Y_ref or log1p(Y/Y_ref)',
            per_projection_max_normalization=False),case_ids=[r['case_id'] for r in selected])
    assert result['normalization']['Y_ref_ppm_m']>0
    write_json(root/'train_statistics.json',result);return result


def read_only_audit(before,root):
    after=source_snapshot(SOURCE)
    assert before==after,'Source inventory/mtime/content changed'
    report=dict(passed=True,source_root=str(SOURCE),source_file_count=len(before),
                changed_file_count=0,created_source_file_count=0,removed_source_file_count=0,
                source_before_sha256=canonical_hash(before),source_after_sha256=canonical_hash(after),
                inputs_policy='Historical input arrays were never read; only size/mtime/inode inventoried',
                source_content_hash_policy='All targets, metadata, README, summary and manifest hashed before and after; inputs stat only')
    write_json(root/'audits/source_read_only_audit.json',report);return report


def code_hashes():
    repo=Path(__file__).resolve().parents[3]
    return {str(p.relative_to(repo)):digest(p) for parent in [repo/'src',repo/'scripts',repo/'configs/dataset_generation_v1.yaml']
            for p in ([parent] if parent.is_file() else sorted(parent.rglob('*')))
            if p.is_file() and p.suffix in ['.py','.cpp','.yaml']}


def run(config_path,phase,resume,audit_only):
    config=yaml.safe_load(Path(config_path).read_text())
    assert Path(config['dataset']['source_root'])==SOURCE and Path(config['dataset']['output_root'])==OUTPUT
    assert config['dataset']['source_read_only'] and config['dataset']['ignore_existing_inputs']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    log_dir=OUTPUT/'logs';log_dir.mkdir(exist_ok=True)
    log_file=log_dir/(timestamp().replace(':','-')+'_'+phase+'.log')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(log_file)],force=True)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info('Starting phase=%s resume=%s',phase,resume)
    records,before=prepare(config)
    debug_records=choose_debug_records(records)
    logging.info('Admitted %d records, split=%s',len(records),dict(collections.Counter(r['split'] for r in records)))
    if phase=='debug':
        phase_config=copy.deepcopy(config)
        phase_config['geometry']['train_count']=config['debug']['train_count']
        phase_config['geometry']['ood_count']=config['debug']['ood_count']
        phase_config['projector']['reference_samples']=config['debug']['reference_samples']
        root=OUTPUT/'debug';root.mkdir(exist_ok=True)
        atomic_text(root/'generation_config.yaml',yaml.safe_dump(phase_config,sort_keys=False))
        train,ood,metas,projector=build_pools(debug_records,phase_config,root,resume)
        logging.info('Running analytical, convergence and independent-projector validation')
        projector_validation(projector,config,debug_records,SOURCE,root/'audits/projector_validation.json')
        audit_pool(debug_records,root,train,ood)
        read_only_audit(before,root)
        write_json(root/'debug_acceptance.json',dict(passed=True,sample_count=5,train_station_count=16,
            ood_station_count=8,reference_samples=128,validation_cases='train_only',
            config_sha256=canonical_hash(config),code_sha256=canonical_hash(code_hashes()),
            source_sha256=canonical_hash(before),completed_at=timestamp(),
            visual_review_status='pending',preview='audits/figures/geometry_and_projection.png'))
        logging.info('Debug QA passed. Visual preview must be reviewed before MVP generation.')
        return
    acceptance=read_json(OUTPUT/'debug/debug_acceptance.json')
    assert acceptance['passed'] and acceptance['visual_review_status']=='passed','Debug visual review required'
    assert acceptance['config_sha256']==canonical_hash(config) and acceptance['code_sha256']==canonical_hash(code_hashes()),'Debug validation stale after code/config change'
    assert acceptance['source_sha256']==canonical_hash(before)
    if audit_only:
        with np.load(OUTPUT/'geometry/train_geometry.npz') as f:train=dict(f)
        with np.load(OUTPUT/'geometry/ood_geometry.npz') as f:ood=dict(f)
        geometry_qa(train,ood,OUTPUT/'audits/geometry_audit.json')
    else:
        train,ood,metas,projector=build_pools(records,config,OUTPUT,resume)
    shutil.copyfile(OUTPUT/'debug/audits/projector_validation.json',OUTPUT/'audits/projector_validation.json')
    report=audit_pool(records,OUTPUT,train,ood)
    logging.info('Full pool audit passed; computing exact train-only statistics')
    statistics=train_statistics(records,OUTPUT,train)
    read_only_audit(before,OUTPUT)
    hashes=read_json(OUTPUT/'audits/artifact_hashes.json')
    build=dict(dataset_name=config['dataset']['name'],status='frozen',phase='MVP',source_dataset=SOURCE.name,
        source_gt_quantity='CH4 mass fraction',source_gt_shape=[128]*3,domain_bounds_m=config['gt']['bounds_m'],
        projection_quantity='path-integrated CH4 mole fraction',projection_unit='ppm_m',projection_shape=[64,64],
        geometry_profile=config['geometry']['profile'],reference_projector_version=config['projector']['version'],
        split_version='case_split_v1',train_geometry_count=64,ood_geometry_count=32,reference_samples=256,
        sample_count=len(records),split_counts=report['split_case_count'],failed_count=0,
        complete_effective_config=config,config_sha256=canonical_hash(config),
        source_manifest_sha256=digest(SOURCE/'manifest.jsonl'),source_index_sha256=digest(OUTPUT/'source_index.jsonl'),
        source_inventory_sha256=canonical_hash(before),split_hashes={p.name:digest(p) for p in (OUTPUT/'splits').iterdir()},
        geometry_hashes={p.name:digest(p) for p in (OUTPUT/'geometry').iterdir()},
        projector_config_sha256=canonical_hash(config['projector']),code_hashes=code_hashes(),
        artifact_inventory_sha256=canonical_hash(hashes),statistics_sha256=digest(OUTPUT/'train_statistics.json'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        frozen_at=timestamp(),source_audit_limit='Original Windows CFD audit files were not supplied; see source_validation.json for provenance and independent checks.')
    write_json(OUTPUT/'dataset_build_manifest.json',build)
    atomic_text(OUTPUT/'README.md',f'''# GeoPhy-Set3D Projection Pool v1 (MVP)

Frozen: {build['frozen_at']}. {len(records)} final-frame CFD samples; splits {report['split_case_count']}.
Frame 60 / 2.0 s only. Source GT is read-only CH4 mass fraction [128,128,128], axes XYZ,
endpoint-aligned coordinates linspace(0,0.6,128). Original 8-view inputs are ignored.

TDLAS stations: horizontal XZ half ring; radius 5.5–6.5 m; height 0.25–0.35 m.
Each station has 64 vertical × 64 horizontal independent beams (±4° elevation, ±5° yaw).
Row indices increase upward; column indices increase in station-local right direction.
Directions use spherical yaw/elevation. Pool A: 64 stations, Pool B: 32 independently seeded stations.
Pool B must never be used for training. Geometry arrays use float64; measurements use float32 ppm·m.
Invalid-ray measurements are zero; use geometry valid_ray masks in losses.

Reference: trilinear interpolation of mass fraction, then mass-to-mole conversion (16.043/28.965),
256 endpoint samples, trapezoidal quadrature, float64 accumulation, ×10^6; integration only inside [0,0.6]^3.
No offline noise or per-sample normalization. Sample metadata references original GT; no full GT copies.
gt_volume_64.npy is endpoint-aligned trilinear resampling to linspace(0,0.6,64), not a cell average.

Index: source_index.jsonl. Each samples/<case_id>/ contains projections_train.npy [64,64,64],
projections_ood.npy [32,64,64], gt_volume_64.npy, sample_meta.json, frame_manifest.jsonl.
geometry/ contains shared true geometry, valid masks, and an optional canonical 6-station geometry.
Canonical measurements are not precomputed. Online view combinations/noise/calibration are not materialized.

Normalization scale: train-only valid-ray p99.5 = {statistics['normalization']['Y_ref_ppm_m']:.9g} ppm·m.
All percentile statistics are exact. Source fields absent from manifest/metadata remain null.
Three retained method-validation cases are confined to training. Debug/projector tests use training cases only.
Original Windows CFD audit paths are provenance; original audit files were not supplied on this server.
Independent GT, geometry, numerical physics, completeness, leakage and source-immutability checks passed.

Reproduce/resume with /scratch/djc/envs/gas-torch/bin/python scripts/build_projection_pool.py --phase mvp --resume
from /home/djc/projects/GeoPhy-Set3D. For checks use --phase mvp --audit-only.
See generation_config.yaml, dataset_build_manifest.json, splits/, audits/, logs/ and debug/.
''')
    logging.info('FROZEN %d cases, %d station maps, max %.9g ppm m',len(records),report['generated_station_maps'],report['projection_max'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/dataset_generation_v1.yaml')
    parser.add_argument('--phase',choices=['debug','mvp'],required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    run(args.config,args.phase,args.resume,args.audit_only)
