#!/usr/bin/env python3
"""One RVTC gate, then fresh random 300-epoch GT-free training; no tuning."""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.rich_measurement import RichMeasurementDataset, rich_measurement_collate
from geophy_set3d.evaluation.measurement_diagnostics import set_properties
from geophy_set3d.training.selfsup_v2 import (validate_selfsup_config, initialize_fresh_model, make_dataset,
    make_loader, run_epoch, selection_metric, zero_volume_baseline, move_rich_batch, V2ReadGuard,
    audit_is_clean, output_collapsed, teacher_gradient_diagnostic, SELECTION)
from geophy_set3d.training.utils import (load_config, seed_everything, directories, write_json,
    atomic_checkpoint, environment, rng_state, restore_rng)

ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/selfsup_mrgr_v2')


def station_protocol(dataset):
    return {dataset[i]['A']['case_id']: {key: dataset[i][key]['input_station_indices'].tolist()
            for key in ('A', 'B', 'H', 'T')} for i in range(len(dataset))}


def metadata_snapshot(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
            inode=p.stat().st_ino) for p in sorted(root.rglob('*')) if p.is_file()}


def prepare(config):
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'effective_protocol.yaml').exists():
        raise FileExistsError('v2 protocol already frozen')
    protocol = dict(method='SelfSup-MRGR v2; Rich-View Teacher Consistency (working name)',
        config=config, model='GeoLift-OARI-MRGR-Res3D', parameter_count=332070, architecture_changed=False,
        train_views=dict(A=[3,5], B=[3,5], H=2, T=[6,10]), validation_views=dict(A=4,B=4,H=4,T=8),
        teacher='same network; T = A union B; T excludes H; only consistency target detached',
        selection_metric=SELECTION, fresh_initialization=True, test_used=False,
        projector=dict(samples=96,view_chunk=2,all_rays=True,unchanged=True),
        gate=dict(cases=8,epochs=80,station_epoch=0,v1_frozen_final_heldout_l1=.520501,
                  criteria=['finite losses', 'teacher heldout below zero-volume baseline',
                            'both sparse branches improve from initialization',
                            'student-teacher mean loss decreases: first10 vs last10 training epochs', 'no output collapse',
                            'permutation safe', 'padding safe', 'all-invalid safe',
                            'teacher stop-gradient and measurement gradients correct', 'zero forbidden reads']),
        posthoc_boundary='300 epochs completed; sparse measurement-only best selected; SHA256 frozen; measurement evaluations precede GT',
        decision_references=dict(val_a_absolute_l1_improvement_min=.003,val_b_relative_worsening_max=.03,
                                 v3_l1_reference=.115,clustered_l1_reference=.42),
        formal_go='Val-A improvement >= .003; Val-B relative worsening <= 3%; V3 and V6 improve; clustered L1 <= .42',
        research_boundary='Training and model selection are volumetric-GT-free; method development has inspected v1 post-hoc GT.')
    (ROOT / 'effective_protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    # No GT or supervised checkpoint is opened even for a hash before completion.
    base = Path(config['experiment']['output_root'])
    historical = {kind: metadata_snapshot(base/kind) for kind in ('logs','checkpoints','figures','comparisons')}
    historical['comparisons'] = {k:v for k,v in historical['comparisons'].items() if not k.startswith('selfsup_mrgr_v2/')}
    protected = {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(REPO.rglob('*'))
                 if p.is_file() and '.git' not in p.parts and p.suffix in ('.py','.cpp','.yaml','.md')
                 and not p.relative_to(REPO).as_posix().startswith(('outputs/','data/'))}
    rows = [json.loads(line) for line in (Path(config['dataset']['root'])/'source_index.jsonl').read_text().splitlines()]
    raw = {row['source_target_path']: dict(size=Path(row['source_target_path']).stat().st_size,
            mtime_ns=Path(row['source_target_path']).stat().st_mtime_ns,
            inode=Path(row['source_target_path']).stat().st_ino) for row in rows}
    write_json(ROOT / 'inventory_before.json', dict(dataset=metadata_snapshot(config['dataset']['root']),
               source_gt128_metadata=raw, existing_artifacts=historical, protected_code=protected,
               gt_and_checkpoint_content_not_opened=True))
    print('v2 protocol and metadata-only inventory frozen', flush=True)


def verify_sources():
    manifest = json.loads((ROOT / 'training_source_manifest.json').read_text())
    if not all(digest(REPO/p) == sha for p, sha in manifest['sha256'].items()):
        raise RuntimeError('Numerical protocol changed after source freeze')
    return manifest


def smoke(config):
    target = ROOT / 'memory_smoke.json'
    if target.exists():
        raise FileExistsError('Memory smoke already recorded')
    paths = directories(config, '_smoke')
    guard = V2ReadGuard(config['dataset']['root'], paths['checkpoints'])
    with guard:
        seed_everything(config['training']['seed']); torch.set_num_threads(2)
        model = initialize_fresh_model(config)
        assert sum(p.numel() for p in model.parameters()) == 332070
        ds = RichMeasurementDataset(config['dataset']['root'], split='train', a_range=(5,5), b_range=(5,5),
                                    hold_views=2, base_seed=config['training']['seed'], normalization=config['normalization']['method'])
        optimizer = torch.optim.AdamW(model.parameters(), **{k:config['optimizer'][k] for k in ('lr','weight_decay')})
        torch.cuda.reset_peak_memory_stats(config['training']['device'])
        result = run_epoch(model, make_loader(Subset(ds, range(2)), config), config, optimizer)
        record = dict(passed=True, worst_case_views=dict(A=5,B=5,H=2,T=10), batch_size=1, gradient_accumulation=2,
                      measurement_only=result, gt_access=guard.summary(),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']))
        assert audit_is_clean(guard.summary())
        write_json(target, record)
        print(json.dumps(record, indent=2), flush=True)


def gate(config):
    verify_sources()
    if not json.loads((ROOT / 'memory_smoke.json').read_text())['passed']:
        raise RuntimeError('Run maximum-view memory smoke first')
    paths = directories(config, '_gate'); target = ROOT / 'gate_summary.json'
    # Reserve the single allowed experiment before any optimization.
    with (ROOT / 'gate_started.json').open('x') as f:
        json.dump(dict(started_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), epochs=80), f)
    guard = V2ReadGuard(config['dataset']['root'], paths['checkpoints'])
    with guard:
        seed_everything(config['training']['seed']); torch.set_num_threads(2)
        model = initialize_fresh_model(config); ds = make_dataset(config); ds.set_epoch(0)
        subset = Subset(ds, range(8)); loader = make_loader(subset, config)
        protocol = station_protocol(subset)
        optimizer = torch.optim.AdamW(model.parameters(), **{k:config['optimizer'][k] for k in ('lr','weight_decay')})
        baseline = zero_volume_baseline(loader, config); initial = run_epoch(model, loader, config)
        torch.cuda.reset_peak_memory_stats(config['training']['device']); began = time.perf_counter(); rows = []
        for epoch in range(80):
            tr = run_epoch(model, loader, config, optimizer)
            rows.append(dict(epoch=epoch,train=tr))
            with (paths['logs']/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(rows[-1], allow_nan=False)+'\n')
            write_json(paths['logs']/'gt_access_audit.json', guard.summary())
            print(f"v2 gate {epoch+1}/80: sparseH={selection_metric(tr):.6f} teacherH={tr['measurement']['teacher_heldout']['relative_l1']:.6f} D={tr['loss']['teacher']:.6f}", flush=True)
        final = run_epoch(model, loader, config)
        assert station_protocol(subset) == protocol
        assert initial['case_station_digest'] == final['case_station_digest']
        batch = move_rich_batch(rich_measurement_collate([ds[0]]), config['training']['device'])
        properties = {key:set_properties(model,batch[key]) for key in ('A','B','T')}
        gradients = teacher_gradient_diagnostic(model, batch)
        collapsed = output_collapsed(final['output']) or output_collapsed(final['teacher_output'])
        criteria = dict(all_losses_finite=all(all(torch.isfinite(torch.tensor(v)) for v in row['train']['loss'].values()) for row in rows),
            teacher_better_than_zero=final['measurement']['teacher_heldout']['relative_l1'] < baseline['relative_l1'],
            sparse_A_improved=final['measurement']['A_heldout']['relative_l1'] < initial['measurement']['A_heldout']['relative_l1'],
            sparse_B_improved=final['measurement']['B_heldout']['relative_l1'] < initial['measurement']['B_heldout']['relative_l1'],
            student_teacher_loss_decreased=sum(row['train']['loss']['teacher'] for row in rows[-10:]) < sum(row['train']['loss']['teacher'] for row in rows[:10]),
            output_not_collapsed=not collapsed, set_properties=all(p['passed'] for p in properties.values()),
            teacher_gradients=gradients['passed'], gt_free=audit_is_clean(guard.summary()))
        summary = dict(passed=all(criteria.values()), criteria=criteria, epochs=80, case_count=8,
            station_protocol=protocol, station_epoch=0, initialization='fresh random; no checkpoints',
            initial=initial, final=final, zero_volume_baseline=baseline, output_collapsed=collapsed,
            set_properties=properties, teacher_gradients=gradients, gt_access=guard.summary(),
            v1_frozen_gate_heldout_final_l1=.520501, v1_retrained=False, formal_comparison=False,
            student_teacher_decrease=dict(first10_mean=sum(r['train']['loss']['teacher'] for r in rows[:10])/10,
                                         last10_mean=sum(r['train']['loss']['teacher'] for r in rows[-10:])/10),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']),
            mean_epoch_seconds=(time.perf_counter()-began)/80, volumetric_metrics_computed=False)
        atomic_checkpoint(paths['checkpoints']/'gate.pt', dict(model=model.state_dict(), config=config, gate=summary))
        (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(paths['logs']/'gate_summary.json', summary); write_json(target, summary)
        print(json.dumps(summary, indent=2), flush=True)


def formal(config, resume=None):
    manifest = verify_sources()
    paths = directories(config)
    gate_summary = json.loads((ROOT/'gate_summary.json').read_text())
    if not gate_summary['passed']:
        raise RuntimeError('v2 gate failed; no formal training, GT evaluation or objective tuning')
    if load_config(directories(config,'_gate')['logs']/'effective_config.yaml') != config:
        raise ValueError('Formal protocol differs from gate')
    if (paths['checkpoints']/'last.pt').exists() and not resume:
        raise FileExistsError('Formal run exists; resume only its exact last.pt')
    if resume and Path(resume).resolve() != (paths['checkpoints']/'last.pt').resolve():
        raise ValueError('Resume only this v2 run last.pt')
    guard = V2ReadGuard(config['dataset']['root'], paths['checkpoints'])
    with guard:
        seed_everything(config['training']['seed']); torch.set_num_threads(2)
        model = initialize_fresh_model(config)
        assert sum(p.numel() for p in model.parameters()) == 332070
        optimizer = torch.optim.AdamW(model.parameters(), **{k:config['optimizer'][k] for k in ('lr','weight_decay')})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        generator = torch.Generator().manual_seed(config['training']['seed'])
        train, val = make_dataset(config), make_dataset(config, validation=True)
        assert (len(train),len(val)) == (166,36)
        train_loader = make_loader(train, config, True, generator); val_loader = make_loader(val, config)
        protocol = station_protocol(val); station_path = paths['logs']/'validation_station_ids.json'
        if station_path.exists() and json.loads(station_path.read_text()) != protocol:
            raise ValueError('Frozen validation stations changed')
        write_json(station_path, protocol)
        start=0; best=float('inf'); rows=[]; env=environment(config); prior_audit = None
        if resume:
            ckpt=torch.load(resume,map_location=config['training']['device'],weights_only=False)
            if ckpt['config']!=config or ckpt['training_signal']!='TDLAS measurements only' or ckpt['source_manifest']!=manifest:
                raise ValueError('Invalid v2 resume provenance')
            model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
            restore_rng(ckpt['rng'],generator); start=ckpt['epoch']+1; best=ckpt['best_val_metric']
            rows=[json.loads(line) for line in (paths['logs']/'metrics.jsonl').read_text().splitlines() if json.loads(line)['epoch']<start]
            (paths['logs']/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            prior_audit = ckpt['gt_access']
            if not audit_is_clean(prior_audit): raise RuntimeError('Resume contains forbidden reads')
        env.update(initialization='fresh random; no v1/supervised/gate checkpoint',volumetric_gt_used=False)
        write_json(paths['logs']/'environment.json',env)
        (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write_json(paths['logs']/'data_protocol.json',dict(train_cases=166,val_cases=36,train_pool='A',validation_pool='A',
            volumetric_gt_used=False,parameter_count=332070,selection_metric=SELECTION))
        for epoch in range(start,300):
            train.set_epoch(epoch); val.set_epoch(0)
            torch.cuda.reset_peak_memory_stats(config['training']['device']); began=time.perf_counter(); lr=optimizer.param_groups[0]['lr']
            tr=run_epoch(model,train_loader,config,optimizer); va=run_epoch(model,val_loader,config)
            scheduler.step(); score=selection_metric(va); improved=score<best; best=min(best,score)
            row=dict(epoch=epoch,train=tr,val=va,learning_rate=lr,seconds=time.perf_counter()-began,
                     gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(config['training']['device']))
            rows.append(row)
            with (paths['logs']/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            audit = guard.summary()
            if not audit_is_clean(audit): raise AssertionError('GT-free protocol violated')
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                best_val_metric=best,config=config,rng=rng_state(generator),git_commit=env['git_commit'],source_manifest=manifest,
                training_signal='TDLAS measurements only',volumetric_gt_used=False,selection_metric=SELECTION,
                initialization='fresh random',gt_access=audit,prior_resume_audit=prior_audit)
            atomic_checkpoint(paths['checkpoints']/'last.pt',payload)
            if improved: atomic_checkpoint(paths['checkpoints']/'best.pt',payload)
            write_json(paths['logs']/'gt_access_audit.json',audit)
            print(f"v2 {epoch+1}/300: trainH={selection_metric(tr):.6f} valH={score:.6f} teacherH={va['measurement']['teacher_heldout']['relative_l1']:.6f} best={best:.6f}",flush=True)
        assert len(rows)==300 and [row['epoch'] for row in rows]==list(range(300))
        best_row=min(rows,key=lambda row:selection_metric(row['val']))
        summary=dict(completed=True,completed_epochs=300,best_epoch=best_row['epoch'],best_val_heldout_relative_l1=best,
            best_validation=best_row['val'],selection_metric=SELECTION,prior_resume_audit=prior_audit,
            mean_epoch_seconds=sum(r['seconds'] for r in rows)/len(rows),peak_gpu_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in rows),
            gt_access=guard.summary(),stopping_reason='max_epochs',volumetric_gt_used=False)
        write_json(paths['logs']/'training_summary.json',summary)
    checkpoint=paths['checkpoints']/'best.pt'
    write_json(ROOT/'frozen_checkpoint.json',dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),best_epoch=summary['best_epoch']+1,
        selection_metric=SELECTION,completed_epochs=300,volumetric_gt_used=False,gt_posthoc_evaluation_started=False,
        training_source_commit=manifest['commit'],frozen_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/train_selfsup_oari_mrgr_v2.yaml')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--smoke',action='store_true'); mode.add_argument('--gate',action='store_true')
    parser.add_argument('--resume'); args=parser.parse_args()
    config=load_config(args.config); validate_selfsup_config(config)
    if args.prepare: prepare(config)
    elif args.smoke: smoke(config)
    elif args.gate: gate(config)
    else: formal(config,args.resume)


if __name__=='__main__': main()
