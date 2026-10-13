#!/usr/bin/env python3
"""Independent registered calibration/evaluation, resumable per reconstruction."""
import sys,os,json,time,hashlib,datetime,argparse,resource,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch,yaml
from geophy_set3d.data.projection_pool_dataset import load_source_index
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.evaluation.clustered_one_swap import keyed_rng
from geophy_set3d.reconstruction.classical import RayMatrix,geometry_tensor,project,sart,variational,mole_to_mass,mass_to_mole
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.data.build_utils import digest
from diagnose_clustered_one_swap import inventory,checkpoint_provenance,snapshot
CFG=ROOT/'configs/classical_v4_v1.yaml'
METHODS=['SART','Tikhonov','TV']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def numeric_sources():
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.cpp'))+[CFG,Path(__file__),ROOT/'tests/test_classical_reconstruction.py',ROOT/'scripts/register_geometry_catalog.py',ROOT/'configs/evaluate_geometry_strata_v1.yaml']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


class ReadGuard:
    def __init__(self,cfg,root,phase,allowed_cases=()):
        self.data=Path(cfg['dataset']).resolve();self.out=root.resolve();self.phase=phase;self.active=False
        rows=load_source_index(self.data);chosen={r['case_id'] for r in allowed_cases}
        self.allowed={}
        for r in rows:
            if (phase=='calibrate' and r['split']=='train' and r['case_id'] in chosen) or (phase=='evaluate' and r['split']=='val'):
                self.allowed[(self.data/r['train_projection_path']).resolve()]='projection_'+r['split']
                self.allowed[(self.data/r['gt_volume_64_path']).resolve()]='GT64_'+r['split']
        self.raw={Path(r['source_target_path']).resolve() for r in rows}
        old=yaml.safe_load((ROOT/'configs/diagnose_clustered_one_swap.yaml').read_text())
        self.ckpt={Path(m['checkpoint']).resolve() for m in old['models'].values()}
        self.counts={};self.touched=set();self.forbidden=0
    def audit(self,event,args):
        if not self.active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=any(c in mode for c in 'wa+') if isinstance(mode,str) else bool(flags&(os.O_WRONLY|os.O_RDWR))
        reason=None;label=None
        if writing:
            if p.is_relative_to(self.data) or p in self.raw or p.suffix in ('.pt','.pth','.ckpt'):reason='Frozen dataset/checkpoint write'
        elif p in self.raw:reason='GT128 forbidden'
        elif p.suffix in ('.pt','.pth','.ckpt'):
            if p not in self.ckpt or self.phase!='register':reason='Unregistered checkpoint access'
            else:label='registered_checkpoint'
        elif p.suffix=='.npy':
            if p in self.allowed:label=self.allowed[p]
            elif p.is_relative_to(self.out/'volumes'):label='own_reconstruction'
            else:reason='Only phase-selected train/Val Pool A and GT64 allowed; no Pool B/test'
        elif p.suffix=='.npz':
            if p==self.data/'geometry/train_geometry.npz':label='Pool_A_geometry'
            else:reason='Only Pool A geometry allowed'
        if reason:self.forbidden+=1;raise PermissionError(reason+': '+str(p))
        if label:self.counts[label]=self.counts.get(label,0)+1;self.touched.add(str(p))
    def __enter__(self):self.active=True;sys.addaudithook(self.audit);return self
    def __exit__(self,*args):
        self.active=False
        with (self.out/'access_sessions.jsonl').open('a') as f:f.write(json.dumps(dict(phase=self.phase,failed=args[0] is not None,counts=self.counts,forbidden_attempts=self.forbidden,touched=sorted(self.touched)))+'\n')


def original_cfg():return yaml.safe_load((ROOT/'configs/diagnose_clustered_one_swap.yaml').read_text())


def register(cfg,out):
    if (out/'protocol.json').exists():verify(cfg,out);print('Existing registration verified');return
    out.mkdir(parents=True,exist_ok=True)
    rows=load_source_index(cfg['dataset']);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    selected=[train[i] for i in np.rint(np.linspace(0,len(train)-1,cfg['calibration_count'])).astype(int)]
    with ReadGuard(cfg,out,'register') as guard:
        with np.load(Path(cfg['dataset'])/'geometry/train_geometry.npz') as z:g={k:z[k] for k in z.files}
        old=original_cfg();models=checkpoint_provenance(old)
        oldroot=Path(old['output']);oldprotocol=json.loads((oldroot/'protocol.json').read_text())
        assert digest(oldroot/'protocol.json')==(oldroot/'protocol.sha256').read_text().strip()
        assert digest(oldroot/'station_protocol.json')==oldprotocol['station_protocol_sha256']
        assert json.loads((oldroot/'original_reproduction.json').read_text())['passed']
        assert all(json.loads((oldroot/'integrity.json').read_text()).get(k) for k in ['checkpoint_hashes_unchanged','dataset_metadata_unchanged','historical_records_hashes_unchanged'])
        records=json.loads((oldroot/'station_protocol.json').read_text())['records'];network=json.loads((oldroot/'per_case_draw.json').read_text())
        assert len(records)==180 and len(network)==1080
        bykey={(r['case_id'],r['draw']):r for r in records}
        for r in network:
            origin=bykey[r['case_id'],r['draw']]
            assert r['input_station_ids']==origin[r['condition']] and r['H_station_ids']==origin['H']
            assert r['checkpoint_sha256']==old['models'][r['model']]['sha256']
        calibration=[]
        for index,row in enumerate(selected):
            for layout in cfg['calibration_geometries']:
                selection='random' if layout=='ordinary_random_V4' else 'clustered'
                ids=EvaluationViewSampler(g['station_azimuth_deg'],4,0,'train',cfg['calibration_seed'],selection).select(row['case_id']).tolist()
                h=keyed_rng(cfg['calibration_H4_seed'],0,row['case_id'],layout).choice(sorted(set(range(64))-set(ids)),4,replace=False).tolist()
                calibration.append(dict(case_id=row['case_id'],calibration_index=index,layout=layout,input_station_ids=ids,H_station_ids=h))
        write(out/'calibration_stations.json',calibration);write(out/'station_protocol.json',records)
        write(out/'network_historical_rows.json',network)
        write(out/'inventory_before.json',dict(base=inventory(old),original_clustered=snapshot(oldroot)))
        write(out/'protected_tracked_hashes.json',{p:digest(ROOT/p) for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if p!='README.md' and (ROOT/p).is_file()})
        protocol=dict(registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=cfg,calibration_cases=selected,
            val_cases=[r['case_id'] for r in val],source_sha256=numeric_sources(),models=models,
            old_protocol_sha256=digest(oldroot/'protocol.json'),old_station_sha256=digest(oldroot/'station_protocol.json'),
            old_results_sha256=digest(oldroot/'per_case_draw.json'),old_original_reproduction_sha256=digest(oldroot/'original_reproduction.json'),
            local_files_sha256={n:digest(out/n) for n in ['calibration_stations.json','station_protocol.json','network_historical_rows.json']},
            geometry_freeze_sha256=digest(out/'geometry/freeze.json'),train_updates=0,
            classical_calibration_budget=dict(reconstructions=12*2*3*3,SART_max_sweeps=150,variational_max_iterations=200,candidates_per_method=3),
            stopping='fixed maximum; input-objective relative change tolerance 1e-5 for 8 successive iterations after minimum; no GT/H iterate selection',
            selection='GT-assisted train relative L1 and measurement-only train H4 relative L1 separately, averaged over all 12 cases and both layouts; exact tie candidate index',
            evaluation_GT_policy='Read GT64 and H only after reconstruction is saved. H never enters solver. No Val tuning, trajectory best-GT selection, Pool B/test/GT128.',
            solver_status='objective_stable is limited numerical stop, not proof of optimality; budget_exhausted and line_search_failed reported without Val retuning',
            runtime='64 station sparse matrices constructed cold once per process; per-case cache lookup and total cold station cost separately recorded; variational calls frozen Pdiff without persistent grid cache')
        write(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    print('Registered protocol',digest(out/'protocol.json'),'216 calibration reconstructions before Val evaluation',flush=True)


def verify(cfg,out):
    p=json.loads((out/'protocol.json').read_text());assert p['config']==cfg
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for file,sha in p['source_sha256'].items():assert digest(ROOT/file)==sha,file
    for file,sha in p['local_files_sha256'].items():assert digest(out/file)==sha,file
    assert digest(out/'geometry/freeze.json')==p['geometry_freeze_sha256']
    old=original_cfg();oldroot=Path(old['output'])
    assert digest(oldroot/'per_case_draw.json')==p['old_results_sha256']
    return p


class Runtime:
    def __init__(self,cfg,out,phase):
        self.cfg=cfg;self.out=out;self.phase=phase;self.device=cfg['device'];torch.set_num_threads(2)
        with np.load(Path(cfg['dataset'])/'geometry/train_geometry.npz') as z:self.g={k:z[k] for k in z.files}
        rows=load_source_index(cfg['dataset']);self.rows={r['case_id']:r for r in rows}
        self.cache={};self.costs=[]
    def geometry(self,ids):
        return geometry_tensor({k:v[ids] for k,v in self.g.items() if k not in ['horizontal_angle_deg','vertical_angle_deg']},self.device)
    def blocks(self,ids):
        for station in ids:
            if station not in self.cache:
                a=RayMatrix({k:v[station:station+1] for k,v in self.g.items() if k not in ['horizontal_angle_deg','vertical_angle_deg']},device=self.device)
                self.cache[station]=a;self.costs.append(dict(station_id=station,build_seconds=a.build_seconds,bytes=a.bytes,nnz=a.nnz))
        return [self.cache[i] for i in ids]
    def sync(self):torch.cuda.synchronize(self.device)
    def reconstruct(self,entry,method,config,key):
        done=self.out/'results'/self.phase/(key+'.json')
        if done.exists():
            row=json.loads(done.read_text());assert row['solver_config']==config;return row
        volume=self.out/'volumes'/self.phase/(key+'.npy')
        checkpoint=self.out/'resume'/self.phase/(key+'.json')
        # Complete volumes are a reconstruction boundary; after a reporting interruption resume evaluation only.
        if checkpoint.exists():
            row=json.loads(checkpoint.read_text());assert row['solver_config']==config and volume.exists()
            c=torch.as_tensor(np.load(volume),device=self.device)
        else:
            ids=entry['input_station_ids'];g=self.geometry(ids)
            record=self.rows[entry['case_id']];p=np.load(Path(self.cfg['dataset'])/record['train_projection_path'],mmap_mode='r')
            y=torch.as_tensor(np.array(p[ids]),device=self.device)
            self.sync();start=time.perf_counter();before=len(self.costs)
            blocks=self.blocks(ids) if method=='SART' else None
            self.sync();build=time.perf_counter()-start
            torch.cuda.reset_peak_memory_stats(self.device);self.sync();start=time.perf_counter()
            if method=='SART':chi,stats=sart(blocks,y,config)
            else:c,stats=variational(g,y,method,config,self.cfg['normalization_scale_ppm_m'])
            self.sync();iteration=time.perf_counter()-start;start=time.perf_counter()
            if method=='SART':c=mole_to_mass(chi)
            self.sync();conversion=time.perf_counter()-start
            if not torch.isfinite(c).all() or float(c.min())<0 or float(c.max())>1:raise FloatingPointError('Invalid C')
            row=dict(entry,method=method,solver_config=config,solver_stats={k:v for k,v in stats.items() if k!='curve'},
                operator_setup_seconds=build,iteration_seconds=iteration,conversion_seconds=conversion,
                cold_new_station_build_seconds=sum(s['build_seconds'] for s in self.costs[before:]),
                equivalent_cold_geometry_seconds=sum(self.cache[i].build_seconds for i in ids) if blocks else 0,
                reused_station_cache=bool(blocks and before==len(self.costs)),
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                volume_sha256=None,curve_file=str(self.out/'curves'/self.phase/(key+'.json')))
            write(row['curve_file'],stats['curve']);volume.parent.mkdir(parents=True,exist_ok=True)
            tmp=volume.with_suffix('.tmp.npy');np.save(tmp,c.detach().cpu().numpy());tmp.replace(volume)
            row['volume_sha256']=digest(volume);write(checkpoint,row)
        # Evaluation measurements and GT are inaccessible to the reconstruction function above.
        self.sync();start=time.perf_counter();record=self.rows[entry['case_id']]
        p=np.load(Path(self.cfg['dataset'])/record['train_projection_path'],mmap_mode='r')
        gt=torch.as_tensor(np.array(np.load(Path(self.cfg['dataset'])/record['gt_volume_64_path'],mmap_mode='r')),device=self.device)
        metric=VolumeMetrics();metric.update(c[None,None],gt[None,None],[entry['case_id']]);row.update({k:v for k,v in metric.rows[0].items() if k!='case_id'})
        with torch.no_grad():
            for label,ids in [('observed',entry['input_station_ids']),('heldout',entry['H_station_ids'])]:
                g=self.geometry(ids);y=torch.as_tensor(np.array(p[ids]),device=self.device);pred=project(c,g)
                row.update({label+'_'+k:v for k,v in relative_projection_metrics(pred,y,g['valid_ray']).items()})
            # Discretization difference on final C, separately from reconstruction error.
            if method=='SART':
                ids=entry['input_station_ids'];g=self.geometry(ids);pred=project(c,g);blocks=self.blocks(ids)
                delta=torch.cat([a.forward(mass_to_mole(c))-a.measurement(pred[j]) for j,a in enumerate(blocks)])
                pd=pred[g['valid_ray']]
                row['discretization_relative_l1']=float(delta.abs().sum()/pd.abs().sum().clamp_min(1e-6))
                row['discretization_relative_l2']=float((delta.square().sum()/pd.square().sum().clamp_min(1e-12)).sqrt())
            row['output_max']=float(c.max());row['output_mean']=float(c.mean())
        self.sync();row['evaluation_seconds']=time.perf_counter()-start
        write(done,row)
        write(self.out/('operator_build_'+self.phase+'.json'),dict(stations=self.costs,total_build_seconds=sum(s['build_seconds'] for s in self.costs),persistent_disk_cache=False,geometry_cache_reused=True))
        return row


def calibrate(cfg,out):
    protocol=verify(cfg,out)
    if (out/'selected_configs.json').exists():
        verify_selection(out);print('Calibration complete; frozen configs verified, no repeat');return
    entries=json.loads((out/'calibration_stations.json').read_text())
    with ReadGuard(cfg,out,'calibrate',protocol['calibration_cases']):
        runtime=Runtime(cfg,out,'calibrate');rows=[]
        for method in METHODS:
            for candidate,params in enumerate(cfg[method]['candidates']):
                config=dict(cfg[method]['fixed'],**params)
                for entry in entries:
                    key=f'{method}_c{candidate}_train{entry["calibration_index"]}_{entry["layout"]}'
                    row=runtime.reconstruct(entry,method,config,key);rows.append(dict(row,candidate=candidate))
                group=rows[-len(entries):]
                print(method,candidate,'train L1',np.mean([r['relative_l1'] for r in group]),'H L1',np.mean([r['heldout_relative_l1'] for r in group]),'seconds',sum(r['iteration_seconds'] for r in group),flush=True)
        scores=[];selected={}
        for method in METHODS:
            group=[r for r in rows if r['method']==method];selected[method]={}
            for candidate in range(3):
                sub=[r for r in group if r['candidate']==candidate]
                scores.append(dict(method=method,candidate=candidate,train_relative_l1=float(np.mean([r['relative_l1'] for r in sub])),train_H4_relative_l1=float(np.mean([r['heldout_relative_l1'] for r in sub])),config=dict(cfg[method]['fixed'],**cfg[method]['candidates'][candidate]),statuses={s:sum(r['solver_stats']['status']==s for r in sub) for s in sorted({r['solver_stats']['status'] for r in sub})}))
            for identity,metric in [('GT_assisted','train_relative_l1'),('measurement_only','train_H4_relative_l1')]:
                best=min([r for r in scores if r['method']==method],key=lambda r:(r[metric],r['candidate']))
                selected[method][identity]=dict(candidate=best['candidate'],config=best['config'],selection_metric=metric)
        write(out/'calibration_rows.json',rows);write(out/'calibration_scores.json',scores)
        write(out/'selected_configs.json',dict(selected=selected,source_sha256=protocol['source_sha256'],protocol_sha256=digest(out/'protocol.json'),budget_actual_reconstructions=len(rows),Val_read=False,calibration_rows_sha256=digest(out/'calibration_rows.json')))
        (out/'selected_configs.sha256').write_text(digest(out/'selected_configs.json')+'\n')
    print('Two selection identities frozen',selected,flush=True)


def verify_selection(out):
    assert digest(out/'selected_configs.json')==(out/'selected_configs.sha256').read_text().strip()
    s=json.loads((out/'selected_configs.json').read_text());assert digest(out/'calibration_rows.json')==s['calibration_rows_sha256'];return s


def evaluate(cfg,out):
    verify(cfg,out);selected=verify_selection(out)['selected'];records=json.loads((out/'station_protocol.json').read_text());rows=[]
    if (out/'evaluation_complete.json').exists():
        final=json.loads((out/'evaluation_complete.json').read_text());assert digest(out/'classical_rows.json')==final['rows_sha256'];print('Evaluation complete and verified; no repeat');return
    with ReadGuard(cfg,out,'evaluate'):
        runtime=Runtime(cfg,out,'evaluate')
        for method in METHODS:
            candidates=sorted(set(v['candidate'] for v in selected[method].values()))
            for candidate in candidates:
                identities=[identity for identity,s in selected[method].items() if s['candidate']==candidate]
                config=selected[method][identities[0]]['config']
                for record in records:
                    for condition in cfg['validation_conditions']:
                        entry=dict(case_id=record['case_id'],case_index=record['case_index'],draw=record['draw'],condition=condition,input_station_ids=record[condition],H_station_ids=record['H'])
                        key=f'{method}_c{candidate}_val{record["case_index"]}_d{record["draw"]}_{condition}'
                        row=runtime.reconstruct(entry,method,config,key);rows.append(dict(row,candidate=candidate,selection_identities=identities))
                    if record['case_index']==35:print(method,candidate,'draw',record['draw'],'complete, rows',len(rows),flush=True)
        write(out/'classical_rows.json',rows)
        write(out/'evaluation_complete.json',dict(reconstructions=len(rows),rows_sha256=digest(out/'classical_rows.json'),selected_configs_sha256=digest(out/'selected_configs.json'),completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    before=json.loads((out/'inventory_before.json').read_text());after=dict(base=inventory(original_cfg()),original_clustered=snapshot(Path(original_cfg()['output'])));assert before==after,'Historical/data metadata changed'
    hashes=json.loads((out/'protected_tracked_hashes.json').read_text());assert all(digest(ROOT/f)==sha for f,sha in hashes.items())
    write(out/'integrity.json',dict(dataset_and_historical_inventory_unchanged=True,protected_tracked_files_unchanged=True,projector_unchanged=True,training_guard_unchanged=True,GT128_PoolB_test_used=False,new_network_training=False,unfinished_old_experiments_resumed=False))
    print('Evaluation and integrity complete',len(rows),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','calibrate','evaluate']);args=a.parse_args()
    cfg=yaml.safe_load(CFG.read_text());out=Path(cfg['output']);torch.set_num_threads(2)
    globals()[args.stage](cfg,out)
if __name__=='__main__':main()
