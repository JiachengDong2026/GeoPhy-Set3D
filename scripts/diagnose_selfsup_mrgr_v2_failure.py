#!/usr/bin/env python3
"""Independent measurement-only RVTC failure diagnosis; never train or change weights."""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest
from geophy_set3d.models import build_model
from geophy_set3d.training.selfsup_v2 import make_dataset, audit_is_clean, validate_selfsup_config
from geophy_set3d.training.utils import load_config, seed_everything, write_json
from geophy_set3d.evaluation.rvtc_failure_diagnostics import (DiagnosticReadGuard, history_review,
    unseen_station_ids, fixed_sample, collate_sample, measurement_rows, paired_summary,
    gradient_diagnostic, vector_geometry)


def file_hashes(root):
    root = Path(root)
    return {str(p.resolve()): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def metadata_digest(root):
    root = Path(root)
    rows = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
            for p in sorted(root.rglob('*')) if p.is_file()]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('No actual results; do not write invented rows')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/diagnose_selfsup_mrgr_v2_failure.yaml')
    args = parser.parse_args(); cfg = yaml.safe_load((REPO / args.config).read_text())
    output = Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    if (output / 'diagnostic_started.json').exists():
        raise FileExistsError('Use a new output directory for a diagnostic replay; preserve this evidence')
    base = REPO / cfg['historical_artifacts']; training = load_config(REPO / cfg['training_config'])
    validate_selfsup_config(training)
    if cfg['validation'] != dict(split='val', geometry_pool='train', a_views=[4,4], b_views=[4,4], heldout_views=4, seed=20261011, epoch=0):
        raise ValueError('Freeze Val-A A4/B4/H4 at epoch0 using the existing sampler')
    if cfg['gradients']['clip'] or cfg['gradients']['updates']:
        raise ValueError('No clipping or parameter updates are allowed')
    checkpoint = Path(cfg['checkpoint']).resolve()
    if not checkpoint.is_file():
        write_json(output / 'missing_inputs.json', dict(checkpoint=str(checkpoint), status='missing; no substitute or training'))
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != 'gate.pt' or checkpoint.parent.name != 'selfsup_oari_mrgr_v2_rvtc_gate':
        raise ValueError('Only the actual v2 gate checkpoint is allowed')
    if output.resolve().is_relative_to(Path(training['dataset']['root']).resolve()):
        raise ValueError('Never write to the dataset')
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    write_json(output / 'diagnostic_started.json', dict(started_at=now, review_base_commit=cfg['review_base_commit'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), optimizer_updates_allowed=False))
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_files = [Path(__file__), REPO / 'src/geophy_set3d/evaluation/rvtc_failure_diagnostics.py', REPO / args.config]
    write_json(output / 'diagnostic_source_manifest.json', dict(sha256={p.relative_to(REPO).as_posix():digest(p) for p in source_files}))
    historical, curve = history_review(base)
    write_json(output / 'historical_recalculation.json', historical); write_csv(output / 'historical_epoch_metrics.csv', curve)
    guard = DiagnosticReadGuard(training['dataset']['root'], checkpoint)
    try:
        with guard:
            seed_everything(cfg['seed']); torch.set_num_threads(2)
            # All actual experiment reads, including checkpoint hashes, happen under the audit.
            original = Path(training['experiment']['output_root']) / 'comparisons/selfsup_mrgr_v2'
            gate_logs = checkpoint.parents[2] / 'logs/selfsup_oari_mrgr_v2_rvtc_gate'
            historical_hashes = dict(file_hashes(base), **file_hashes(original), **file_hashes(gate_logs))
            dataset_before = metadata_digest(training['dataset']['root'])
            checkpoint_before = digest(checkpoint)
            saved = torch.load(checkpoint, map_location=cfg['device'], weights_only=False)
            gate = json.loads((base / 'gate_summary.json').read_text())
            if saved['config'] != training or saved['gate'] != gate or gate['passed']:
                raise ValueError('Gate/config provenance mismatch or original failure altered')
            frozen_sources = json.loads((base / 'training_source_manifest.json').read_text())['sha256']
            if not all(digest(REPO / p) == sha for p, sha in frozen_sources.items()):
                raise ValueError('Original numerical implementation changed')
            model = build_model(saved['config']).to(cfg['device']); model.load_state_dict(saved['model']); model.eval()
            assert sum(p.numel() for p in model.parameters()) == 332070
            state_before = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            train_ds, val_ds = make_dataset(training), make_dataset(training, validation=True)
            assert len(val_ds) == 36
            assert set(gate['station_protocol']).isdisjoint({row['case_id'] for row in val_ds.records})
            work = []; protocol_rows = []
            for cohort, ds, indices in (('gate', train_ds, range(8)), ('Val-A', val_ds, range(36))):
                for index in indices:
                    case_id = ds.records[index]['case_id']
                    if cohort == 'gate':
                        protocol = gate['station_protocol'][case_id]
                        sampled = ds[index]
                        assert {key:sampled[key]['input_station_indices'].tolist() for key in ('A','B','H','T')} == protocol
                        u = unseen_station_ids(ds.station_count, protocol, case_id, cfg['seed'], cfg['unseen_stations']['count'])
                    else:
                        sampled = ds[index]
                        protocol = {key:sampled[key]['input_station_indices'].tolist() for key in ('A','B','H','T')}
                        u = None
                    row = dict(cohort=cohort, case_id=case_id, dataset_index=index, pool='A', split=ds.split,
                        station_epoch=0, ABH_seed=training['training']['seed'], U_seed=cfg['seed'] if u is not None else None,
                        **protocol, U=u)
                    protocol_rows.append(row); work.append((cohort, ds, index, protocol, u))
            write_json(output / 'case_station_protocol.json', dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_before,
                selection='all gate8 and all Val-A36; no metric-based case or station selection',cases=protocol_rows))
            metrics = []; gradients = []
            aggregate = {}
            for step, (cohort, ds, index, protocol, u) in enumerate(work):
                sample = fixed_sample(ds, index, protocol, u); batch = collate_sample(sample, cfg['device'])
                with torch.no_grad():
                    # Only A/B/T enter the network; H and U remain projector targets.
                    volumes = tuple(model(batch[key]) for key in ('A','B','T'))
                    if not all(torch.isfinite(c).all() for c in volumes): raise FloatingPointError('Nonfinite forward')
                    rows = measurement_rows(volumes, batch, ('H','U') if u is not None else ('H',), cohort)
                metrics.extend(rows); del volumes
                result, vectors = gradient_diagnostic(model, batch)
                gradients.append(dict(cohort=cohort,case_id=sample['A']['case_id'],**result))
                if cohort not in aggregate:
                    aggregate[cohort] = {key:torch.zeros_like(v) for key,v in vectors.items()}
                for key, value in vectors.items():aggregate[cohort][key] += value
                if any(not torch.equal(state_before[key], value.detach().cpu()) for key,value in model.state_dict().items()):
                    raise AssertionError('Parameter or buffer mutated during diagnosis')
                print(f"diagnostic {step+1}/44 {cohort}: {sample['A']['case_id']} teacherΔH={rows[0]['teacher_minus_student_relative_l1']:.6f} cosine(M,D)={result['cosine_measurement_consistency']:.4f}",flush=True)
                del batch, sample, vectors
            write_csv(output / 'measurement_per_case.csv',metrics); write_csv(output / 'gradient_per_case.csv',gradients)
            groups = {}
            for cohort, domain in (('gate','H'),('gate','U'),('Val-A','H')):
                selected = [row for row in metrics if row['cohort']==cohort and row['domain']==domain]
                groups[cohort+'_'+domain] = paired_summary(selected,cfg['bootstrap']['samples'],cfg['bootstrap']['seed'])
            gradient_summary = {}
            for cohort in ('gate','Val-A'):
                selected = [row for row in gradients if row['cohort']==cohort]
                mean_vectors = {key:value/len(selected) for key,value in aggregate[cohort].items()}
                aggregate_geometry = vector_geometry(mean_vectors)
                fields = [key for key in aggregate_geometry if key.startswith('cosine_') or key.endswith('_norm')]
                descriptive = {}
                for key in fields:
                    values = np.array([row[key] for row in selected if row[key] is not None])
                    descriptive[key] = dict(mean=float(values.mean()),median=float(np.median(values)),min=float(values.min()),max=float(values.max()),
                        negative_fraction=float(np.mean(values<0)) if key.startswith('cosine_') else None)
                gradient_summary[cohort] = dict(case_count=len(selected),weighted_unclipped=True,
                    cohort_mean_loss_gradient=aggregate_geometry,per_case_statistics=descriptive,
                    all_teacher_detach_checks_passed=all(row['teacher_consistency_output_gradient_zero'] for row in selected),
                    all_student_direction_checks_passed=all(row['student_A_gradient_dot_error']>0 and row['student_B_gradient_dot_error']>0 for row in selected))
            gate_h = [row for row in metrics if row['cohort']=='gate' and row['domain']=='H']
            replay = dict(student_H_l1=float(np.mean([row['student_relative_l1'] for row in gate_h])),
                teacher_H_l1=float(np.mean([row['T_relative_l1'] for row in gate_h])),
                teacher_discrepancy=float(np.mean([.5*(row['D_A_T']+row['D_B_T']) for row in gate_h])),
                absolute_voxel_disagreement_mean=float(np.mean([.5*(row['absolute_voxel_disagreement_A_T']+row['absolute_voxel_disagreement_B_T']) for row in gate_h])),
                teacher_denominator_mean=float(np.mean([row['teacher_denominator'] for row in gate_h])))
            expected = gate['final']
            for value, target in ((replay['student_H_l1'],expected['measurement']['heldout']['relative_l1']),
                                  (replay['teacher_H_l1'],expected['measurement']['teacher_heldout']['relative_l1']),
                                  (replay['teacher_discrepancy'],expected['loss']['teacher'])):
                if not np.isclose(value,target,rtol=1e-5,atol=1e-7):raise AssertionError('Final checkpoint replay disagrees with historical fixed evaluation')
            integrity = dict(checkpoint_sha256_unchanged=digest(checkpoint)==checkpoint_before,
                parameters_and_buffers_unchanged=all(torch.equal(state_before[key],value.detach().cpu()) for key,value in model.state_dict().items()),
                historical_files_unchanged=all(digest(path)==sha for path,sha in historical_hashes.items()),
                dataset_metadata_unchanged=metadata_digest(training['dataset']['root'])==dataset_before,
                original_numerical_sources_unchanged=all(digest(REPO/p)==sha for p,sha in frozen_sources.items()),
                original_no_go_retained=not saved['gate']['passed'],optimizer_created=False,optimizer_steps=0,
                gradient_clipping_applied=False,parameter_grad_accumulation=False)
            if not all(value for key,value in integrity.items() if key.endswith('_unchanged') or key.endswith('_retained')):
                raise AssertionError('Immutable diagnostic boundary violated')
            audit = guard.summary(); assert audit_is_clean(audit)
            summary = dict(completed=True,review_base_commit=cfg['review_base_commit'],completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_before,parameter_count=332070,
                history=historical,final_checkpoint_replay=replay,measurement=groups,gradients=gradient_summary,
                integrity=integrity,gt_access=audit,independent_measurement_not_voxel_evidence=True,
                initial_and_intermediate_checkpoints_missing=True)
            write_json(output / 'summary.json',summary); write_json(output / 'historical_file_hashes.json',historical_hashes)
            write_json(output / 'integrity.json',integrity); write_json(output / 'gt_access_audit.json',audit)
            print(json.dumps(dict(measurement=groups,gradients=gradient_summary,integrity=integrity),indent=2),flush=True)
    except BaseException:
        write_json(output / 'gt_access_audit.json',guard.summary())
        raise


if __name__ == '__main__':
    main()
