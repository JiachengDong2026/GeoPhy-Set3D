#!/usr/bin/env python3
"""One fixed-input, GT-free C0 adapter continuation; independent finite protocol."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Subset

from geophy_set3d.data.build_utils import digest
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.evaluation.geometry_stratified_sampler import SamplerTrainingReadGuard, PlannedABHSampler
from geophy_set3d.evaluation.selfsup_c0c1_diagnostic import DiagnosticReadGuard, METRICS, paired_summary
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.models import build_model
from geophy_set3d.models.geolift_oari_mrgr_c0_adapter import GeoLiftOARIMRGRC0Adapter
from geophy_set3d.models.geolift_oari_mrgr_res3d import project_input
from geophy_set3d.training.deep_comparison import tensor_identity, check_prediction
from geophy_set3d.training.metrics import VolumeMetrics
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.selfsup import make_dataset, make_loader, run_epoch, selection_metric, validate_selfsup_config
from geophy_set3d.training.utils import seed_everything, write_json, atomic_checkpoint, move_batch, rng_state, restore_rng
from diagnose_clustered_one_swap import dataset, target_gt

CONFIG = REPO / 'configs/ours_c0_adapter_matched_v1.yaml'
ARCHIVE = REPO / 'docs/reports/artifacts/ours_c0_adapter_matched_v1'
ARMS = ('Control', 'Treatment')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def configuration():
    c = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load((REPO / c['base_config']).read_text())
    validate_selfsup_config(base)
    if c['epochs'] != 20 or c['scheduler'] != dict(name='cosine', T_max=20):
        raise ValueError('Only the preregistered 20-epoch experiment is supported')
    return c, base, Path(c['output'])


def run_config(c, base):
    cfg = copy.deepcopy(base)
    cfg['training'].update(device=c['device'], max_epochs=c['epochs'])
    cfg['loader'].update(num_workers=0, persistent_workers=False)
    cfg['optimizer'] = c['optimizer']
    cfg['scheduler'] = c['scheduler']
    return cfg


class ContinuationReadGuard(SamplerTrainingReadGuard):
    """Whitelist exactly the registered GT-free v1 checkpoint for continuation."""
    def __init__(self, dataset_root, checkpoint_root, output, source):
        super().__init__(dataset_root, checkpoint_root, output)
        self.source = Path(source).resolve()
        self.source_reads = 0

    def _audit(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path == self.source:
                mode, flags = args[1:3]
                writing = any(k in mode for k in 'wa+') if isinstance(mode, str) else bool(flags & 3)
                if writing:
                    raise PermissionError('Original checkpoint is read-only')
                self.source_reads += 1
                return
        super()._audit(event, args)

    def summary(self):
        return dict(super().summary(), registered_gt_free_source_checkpoint_reads=self.source_reads)


def archive(out, names):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = ARCHIVE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / name, target)


def build_arm(arm, base, state, device):
    if arm == 'Control':
        model = build_model(base)
        model.load_state_dict(state)
    else:
        model = GeoLiftOARIMRGRC0Adapter(**base['model']['gca'], **base['model']['set_interaction'])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {f'c0_adapter.body.{i}.{k}' for i in (0, 2) for k in ('weight', 'bias')} or unexpected:
            raise ValueError('Only the new adapter may be absent from the source state')
    for key, value in state.items():
        if not torch.equal(value.cpu(), model.state_dict()[key].cpu()):
            raise ValueError('Shared parameter/buffer differs: ' + key)
    return model.to(device)


def source_checkpoint(c, base):
    if digest(c['source_checkpoint']) != c['source_sha256']:
        raise ValueError('Original source checkpoint changed')
    ck = torch.load(c['source_checkpoint'], map_location='cpu', weights_only=False)
    if ck['config'] != base or ck['volumetric_gt_used'] or ck['training_signal'] != 'TDLAS measurements only':
        raise ValueError('Expected the original GT-free v1 best checkpoint')
    return ck


def verify(c, out):
    p = json.loads((out / 'protocol.json').read_text())
    if p['config'] != c or digest(out / 'protocol.json') != (out / 'protocol.sha256').read_text().strip():
        raise ValueError('Registration changed')
    sources = dict(p['source_sha256'])
    amendment_path = out / 'preflight_amendment.json'
    if amendment_path.exists():
        assert digest(amendment_path) == (out / 'preflight_amendment.sha256').read_text().strip()
        amendment = json.loads(amendment_path.read_text())
        assert amendment['protocol_sha256'] == digest(out / 'protocol.json')
        assert amendment['optimizer_steps_at_amendment'] == 0
        allowed = {'scripts/run_ours_c0_adapter_matched.py', 'tests/test_ours_c0_adapter.py'}
        assert set(amendment['source_changes']) <= allowed
        for name, change in amendment['source_changes'].items():
            assert change['before'] == sources[name]
            sources[name] = change['after']
    for name, sha in sources.items():
        if digest(REPO / name) != sha:
            raise ValueError('Registered source changed: ' + name)
    for name, sha in p['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('Registered input changed: ' + name)
    return p


def register(c, base, out):
    if (out / 'protocol.json').exists():
        verify(c, out)
        print('Existing registration verified; no overwrite', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = run_config(c, base)
    with ContinuationReadGuard(base['dataset']['root'], out / 'registration_checkpoints', out, c['source_checkpoint']) as guard:
        ck = source_checkpoint(c, base)
        train = make_dataset(cfg)
        val = make_dataset(cfg, validation=True)
        assert len(train) == 166 and len(val) == 36
        orders, plans = [], []
        generator = torch.Generator().manual_seed(c['seed'])
        for epoch in range(c['epochs']):
            orders.append(torch.randperm(len(train), generator=generator).tolist())
            rows = []
            for index, r in enumerate(train.records):
                a, b, h = train.sampler.sample(train.station_count, epoch=epoch, sample_index=index, case_id=r['case_id'])
                rows.append(dict(case_id=r['case_id'], A=a.tolist(), B=b.tolist(), H=h.tolist()))
            plans.append(rows)
        write_json(out / 'station_plan.json', plans)
        write_json(out / 'case_orders.json', orders)
        initial, params = {}, {}
        for arm in ARMS:
            seed_everything(c['seed'])
            model = build_arm(arm, base, ck['model'], 'cpu')
            initial[arm] = state_hash(model.state_dict())
            params[arm] = sum(p.numel() for p in model.parameters())
        assert guard.summary()['projection_open_count'] == guard.summary()['gt_open_attempts'] == 0
        write_json(out / 'registration_access.json', guard.summary())
    paths = [CONFIG, REPO / c['base_config'], Path(__file__), REPO / 'docs/ours_c0_adapter_matched_v1_protocol_zh.md',
             REPO / 'tests/test_ours_c0_adapter.py', REPO / 'scripts/diagnose_clustered_one_swap.py']
    for folder in ('data', 'physics', 'models'):
        paths += [x for x in (REPO / 'src/geophy_set3d' / folder).rglob('*.py') if 'senseiver' not in x.name]
    paths += [REPO / 'src/geophy_set3d' / x for x in (
        'training/selfsup.py', 'training/utils.py', 'training/metrics.py', 'training/gt_read_guard.py',
        'training/deep_comparison.py', 'training/rvtc_matched.py',
        'evaluation/geometry_stratified_sampler.py', 'evaluation/selfsup_c0c1_diagnostic.py',
        'evaluation/measurement_consistency.py', 'evaluation/clustered_one_swap.py')]
    inputs = [Path(c['source_checkpoint']), REPO / c['reference'], REPO / c['layouts'],
              out / 'station_plan.json', out / 'case_orders.json']
    root = Path(base['dataset']['root'])
    inputs += [root / 'source_index.jsonl', root / 'train_statistics.json', root / 'geometry/train_geometry.npz']
    p = dict(registered_at=now(), status='REGISTERED BEFORE OPTIMIZATION', config=c, base_config=base,
             initialization='Same frozen GT-free v1 best; fresh optimizer; zero adapter',
             initial_state_sha256=initial, parameter_count=params, train_cases=166, val_cases=36,
             epochs_per_arm=20, optimizer_steps_per_arm=1660,
             source_sha256={str(x.relative_to(REPO)): digest(x) for x in paths},
             input_sha256={str(x): digest(x) for x in inputs},
             only_intervention='C0 fused-feature adapter using existing local directional descriptor',
             station_plan_shared=True, case_orders_shared=True, volumetric_training_labels=0,
             selector='original natural Val mean held-out relative L1; include epoch0',
             posthoc='Both training complete, selectors replayed, all C0/C1 predictions sealed before GT64',
             limitations='One seed, limited continuation, previously studied Val; not equal budget to fresh baselines')
    write_json(out / 'protocol.json', p)
    (out / 'protocol.sha256').write_text(digest(out / 'protocol.json') + '\n')
    archive(out, ['protocol.json', 'protocol.sha256', 'registration_access.json', 'case_orders.json'])
    print('Registered fixed-input matched20; parameters', params, flush=True)


def smoke(c, base, out):
    verify(c, out)
    if (out / 'smoke.json').exists():
        print('Smoke already complete', flush=True)
        return
    torch.set_num_threads(2)
    cfg = run_config(c, base)
    with ContinuationReadGuard(base['dataset']['root'], out / 'smoke_checkpoints', out, c['source_checkpoint']) as guard:
        ck = source_checkpoint(c, base)
        ds = make_dataset(cfg)
        raw = next(iter(make_loader(Subset(ds, [0, 1]), cfg)))
        batch = move_batch(raw['A'], c['device'])
        seed_everything(c['seed'])
        control = build_arm('Control', base, ck['model'], c['device']).eval()
        treatment = build_arm('Treatment', base, ck['model'], c['device']).eval()
        with torch.no_grad():
            a = control(batch, return_stages=True)
            b = treatment(batch, return_stages=True)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        del a, b, control
        from geophy_set3d.training.selfsup import move_dual_batch, objective
        treatment.train()
        before = state_hash(treatment.state_dict())
        loss, _, _, _ = objective(treatment, move_dual_batch(raw, c['device']))
        loss.backward()
        grads = {n: float(p.grad.abs().sum()) for n, p in treatment.named_parameters() if n.startswith('c0_adapter')}
        assert all(torch.isfinite(p.grad).all() for p in treatment.parameters() if p.grad is not None)
        assert grads['c0_adapter.body.2.weight'] > 0
        assert state_hash(treatment.state_dict()) == before
        write_json(out / 'smoke.json', dict(at=now(), zero_init_output_exact=True, backward_finite=True,
                   adapter_gradient_abs_sum=grads, optimizer_steps=0, parameters_unchanged=True, access=guard.summary()))
    archive(out, ['smoke.json'])
    print('Real-data equality and backward smoke passed; 0 optimizer steps', flush=True)


def train(c, base, out, arm):
    p = verify(c, out)
    assert json.loads((out / 'smoke.json').read_text())['zero_init_output_exact']
    root = out / arm
    root.mkdir(exist_ok=True)
    ckdir = root / 'checkpoints'
    ckdir.mkdir(exist_ok=True)
    if (root / 'complete.json').exists():
        print(arm, 'already completed; not retraining', flush=True)
        return
    cfg = run_config(c, base)
    with ContinuationReadGuard(base['dataset']['root'], ckdir, out, c['source_checkpoint']) as guard:
        seed_everything(c['seed'])
        torch.set_num_threads(2)
        ck = source_checkpoint(c, base)
        model = build_arm(arm, base, ck['model'], c['device'])
        assert state_hash(model.state_dict()) == p['initial_state_sha256'][arm]
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['optimizer']['lr'], weight_decay=c['optimizer']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'])
        ds = make_dataset(cfg)
        ds.sampler = PlannedABHSampler(json.loads((out / 'station_plan.json').read_text()))
        val = make_dataset(cfg, validation=True)
        orders = json.loads((out / 'case_orders.json').read_text())
        vl = make_loader(val, cfg)
        generator = torch.Generator().manual_seed(c['seed'])
        seed_everything(c['seed'])
        counter = dict(steps=0)
        optimizer.register_step_post_hook(lambda *args: counter.update(steps=counter['steps'] + 1))
        last = ckdir / 'last.pt'
        start, best, best_epoch = 0, float('inf'), 0
        if last.exists():
            saved = torch.load(last, map_location=c['device'], weights_only=False)
            assert saved['arm'] == arm and saved['protocol_sha256'] == digest(out / 'protocol.json')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'], generator)
            start, best, best_epoch = saved['completed_epochs'], saved['best'], saved['best_epoch']
            counter['steps'] = saved['optimizer_steps']
            rows = [json.loads(x) for x in (root / 'metrics.jsonl').read_text().splitlines() if json.loads(x)['epoch'] <= start]
            (root / 'metrics.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))

        def checkpoint(epoch):
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        completed_epochs=epoch, best=best, best_epoch=best_epoch, optimizer_steps=counter['steps'],
                        arm=arm, config=cfg, protocol_sha256=digest(out / 'protocol.json'), rng=rng_state(generator),
                        volumetric_gt_used=False, source_sha256=c['source_sha256'])

        if start == 0 and not last.exists():
            initial = run_epoch(model, vl, cfg)
            best = selection_metric(initial)
            atomic_checkpoint(ckdir / 'best.pt', checkpoint(0))
            atomic_checkpoint(last, checkpoint(0))
            with (root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=0, val=initial, optimizer_steps=0)) + '\n')
        began = time.perf_counter()
        for epoch in range(start, c['epochs']):
            ds.set_epoch(epoch)
            loader = make_loader(Subset(ds, orders[epoch]), cfg, generator=generator)
            t = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(c['device'])
            tr = run_epoch(model, loader, cfg, optimizer)
            va = run_epoch(model, vl, cfg)
            assert tr['sample_count'] == 166 and va['sample_count'] == 36
            assert counter['steps'] == 83 * (epoch + 1)
            score = selection_metric(va)
            improved = score < best
            if improved:
                best, best_epoch = score, epoch + 1
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            row = dict(epoch=epoch + 1, train=tr, val=va, optimizer_steps=counter['steps'], learning_rate=lr,
                       seconds=time.perf_counter() - t, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(c['device']))
            atomic_checkpoint(last, checkpoint(epoch + 1))
            if improved:
                atomic_checkpoint(ckdir / 'best.pt', checkpoint(epoch + 1))
            with (root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            write_json(root / 'access.json', guard.summary())
            print(f'{arm} {epoch+1}/20 steps={counter["steps"]} ValH_L1={score:.7f} best={best:.7f} seconds={row["seconds"]:.1f}', flush=True)
        assert counter['steps'] == 1660
        write_json(root / 'complete.json', dict(at=now(), epochs=20, optimizer_steps=counter['steps'],
                   best_epoch=best_epoch, best_val_heldout_relative_l1=best, checkpoint=str(ckdir / 'best.pt'),
                   best_sha256=digest(ckdir / 'best.pt'), last_sha256=digest(last),
                   resumed_process_training_seconds=time.perf_counter() - began, access=guard.summary()))
    archive(out, [f'{arm}/complete.json', f'{arm}/metrics.jsonl', f'{arm}/access.json'])
    verify(c, out)


def freeze(c, out):
    verify(c, out)
    completed = {a: json.loads((out / a / 'complete.json').read_text()) for a in ARMS}
    for a, value in completed.items():
        assert value['epochs'] == 20 and value['optimizer_steps'] == 1660
        assert digest(value['checkpoint']) == value['best_sha256']
    rows = {a: [json.loads(x) for x in (out / a / 'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for i in range(1, 21):
        assert rows['Control'][i]['train']['case_station_digest'] == rows['Treatment'][i]['train']['case_station_digest']
        assert rows['Control'][i]['val']['case_station_digest'] == rows['Treatment'][i]['val']['case_station_digest']
    value = dict(at=now(), completed=completed, matched_train_and_val_station_digest=True,
                 protocol_sha256=digest(out / 'protocol.json'))
    if not (out / 'frozen_best.json').exists():
        write_json(out / 'frozen_best.json', value)
    archive(out, ['frozen_best.json'])
    return completed


def infer(c, base, out):
    frozen = freeze(c, out)
    cfg = run_config(c, base)
    records = json.loads((REPO / c['layouts']).read_text())
    assert len(records) == 900
    reference = pd.read_csv(REPO / c['reference']).set_index(['case_id', 'group', 'draw'])
    torch.set_num_threads(2)
    for arm in ARMS:
        root = out / arm
        (root / 'predictions').mkdir(exist_ok=True)
        (root / 'prediction_rows').mkdir(exist_ok=True)
        ckpath = frozen[arm]['checkpoint']
        with DiagnosticReadGuard(base['dataset']['root'], [ckpath], out, 'inference') as guard:
            ck = torch.load(ckpath, map_location='cpu', weights_only=False)
            assert ck['arm'] == arm and not ck['volumetric_gt_used']
            model = build_model(base) if arm == 'Control' else GeoLiftOARIMRGRC0Adapter(**base['model']['gca'], **base['model']['set_interaction'])
            model.load_state_dict(ck['model'])
            model.to(c['device']).eval()
            before = state_hash(model.state_dict())
            replay = run_epoch(model, make_loader(make_dataset(cfg, validation=True), cfg), cfg)
            assert np.isclose(selection_metric(replay), frozen[arm]['best_val_heldout_relative_l1'], atol=1e-8, rtol=1e-5)
            write_json(root / 'selector_replay.json', dict(passed=True, score=selection_metric(replay)))
            ds = dataset(dict(dataset=base['dataset']['root'], original_seed=c['seed']))
            with torch.no_grad():
                for index in range(36):
                    source, projections = ds._projections(index)
                    for r in [x for x in records if x['case_index'] == index]:
                        token = f'case{index}_{r["geometry_id"]}'
                        rowpath = root / 'prediction_rows' / (token + '.json')
                        if rowpath.exists():
                            oldrow = json.loads(rowpath.read_text())
                            assert oldrow['record'] == r and oldrow['checkpoint_sha256'] == frozen[arm]['best_sha256']
                            assert digest(oldrow['prediction_file']) == oldrow['prediction_sha256']
                            continue
                        assert source['case_id'] == r['case_id']
                        empty = np.empty(0, dtype=np.int64)
                        b = move_batch(projection_pool_collate([ds._measurement_sample(source, projections, np.asarray(r['station_ids']), empty)]), c['device'])
                        h = move_batch(projection_pool_collate([ds._measurement_sample(source, projections, np.asarray(r['H_station_ids']), empty)]), c['device'])
                        _, visible = model.lifting.angular_grid(b)
                        visible = visible.reshape(1, 4, 32, 32, 32)
                        identity = dict(input_identity_sha256=tensor_identity(b['input_projection_raw'], b['input_station_indices'], b['input_station_origin_m'], b['input_path_length_m']),
                                        input_mask_sha256=tensor_identity(b['view_mask'], b['input_valid_ray'], visible),
                                        H_mask_sha256=tensor_identity(h['view_mask'], h['input_valid_ray']),
                                        complete_input_sha256=tensor_identity(*[b[k] for k in sorted(b) if isinstance(b[k], torch.Tensor)]),
                                        complete_H_sha256=tensor_identity(*[h[k] for k in sorted(h) if isinstance(h[k], torch.Tensor)]))
                        old = reference.loc[(r['case_id'], r['group'], r['draw'])]
                        for key in ('input_identity_sha256', 'input_mask_sha256', 'H_mask_sha256'):
                            assert old[key] == identity[key], key
                        stages = model(b, return_stages=True)
                        measures = {}
                        for stage in ('C0', 'C1'):
                            pred = check_prediction(stages[stage], b)
                            for label, target in [('observed', b), ('heldout', h)]:
                                projected = project_input(pred, target)
                                valid = target['input_valid_ray'][0, :, 0].bool() & target['view_mask'][0, :, None, None]
                                m = relative_projection_metrics(projected[0, :, 0], target['input_projection_raw'][0, :, 0], valid)
                                measures.update({f'{stage}_{label}_{k}': v for k, v in m.items()})
                        file = root / 'predictions' / (token + '.npz')
                        temp = file.with_suffix('.tmp')
                        with temp.open('wb') as f:
                            np.savez(f, C0=stages['C0'][0, 0].cpu().numpy(), C1=stages['C1'][0, 0].cpu().numpy())
                        temp.replace(file)
                        write_json(rowpath, dict(record=r, **identity, **measures, prediction_file=str(file),
                                   prediction_sha256=digest(file), checkpoint_sha256=frozen[arm]['best_sha256']))
                    print(f'{arm}: {index+1}/36 cases saved', flush=True)
            assert state_hash(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
            assert guard.summary()['forbidden_attempts'] == 0
            write_json(root / 'inference_access.json', guard.summary())
        del model, ck
        torch.cuda.empty_cache()
    manifests = {}
    for arm in ARMS:
        manifests[arm] = {}
        for r in records:
            token = f'case{r["case_index"]}_{r["geometry_id"]}'
            path = out / arm / 'prediction_rows' / (token + '.json')
            row = json.loads(path.read_text())
            assert digest(row['prediction_file']) == row['prediction_sha256']
            manifests[arm][token] = digest(path)
            if arm == 'Treatment':
                other = json.loads((out / 'Control' / 'prediction_rows' / (token + '.json')).read_text())
                for key in ('complete_input_sha256', 'complete_H_sha256'):
                    assert row[key] == other[key]
    write_json(out / 'predictions_complete.json', dict(at=now(), rows_per_arm=900,
               all_C0_C1_saved_before_GT=True, paired_complete_input_hashes_match=True, row_sha256=manifests))
    archive(out, ['predictions_complete.json'] + [f'{a}/{x}' for a in ARMS for x in ('selector_replay.json', 'inference_access.json')])


def gt(c, base, out):
    verify(c, out)
    sealed = json.loads((out / 'predictions_complete.json').read_text())
    assert sealed['all_C0_C1_saved_before_GT'] and sealed['paired_complete_input_hashes_match']
    if (out / 'gt_complete.json').exists():
        print('GT evaluation already complete; no repeat', flush=True)
        return
    records = json.loads((REPO / c['layouts']).read_text())
    saved = {a: [] for a in ARMS}
    for arm in ARMS:
        for r in records:
            token = f'case{r["case_index"]}_{r["geometry_id"]}'
            path = out / arm / 'prediction_rows' / (token + '.json')
            assert digest(path) == sealed['row_sha256'][arm][token]
            row = json.loads(path.read_text())
            assert digest(row['prediction_file']) == row['prediction_sha256']
            saved[arm].append(row)
    torch.set_num_threads(2)
    rows = []
    write_json(out / 'first_posthoc_gt_boundary.json', dict(at=now(), freeze_sha256=digest(out / 'frozen_best.json'),
               prediction_manifest_sha256=digest(out / 'predictions_complete.json'), epochs_per_arm=20, steps_per_arm=1660))
    with DiagnosticReadGuard(base['dataset']['root'], [], out, 'gt') as guard:
        ds = dataset(dict(dataset=base['dataset']['root'], original_seed=c['seed']))
        for index in range(36):
            target = target_gt(ds, index)
            for arm in ARMS:
                for row in [x for x in saved[arm] if x['record']['case_index'] == index]:
                    with np.load(row['prediction_file'], allow_pickle=False) as file:
                        values = {}
                        for stage in ('C0', 'C1'):
                            prediction = torch.from_numpy(file[stage].copy())[None, None]
                            metric = VolumeMetrics()
                            metric.update(prediction, target, [row['record']['case_id']])
                            values.update({stage + '_' + k: v for k, v in metric.rows[0].items() if k != 'case_id'})
                    rows.append(dict(**row['record'], arm=arm, **values,
                                     **{k: v for k, v in row.items() if k.startswith(('C0_', 'C1_', 'complete_'))}))
            print(f'Saved-output GT64: {index+1}/36', flush=True)
        assert guard.summary()['forbidden_attempts'] == 0
        write_json(out / 'gt_access.json', guard.summary())
    frame = pd.DataFrame(rows)
    assert len(frame) == 1800
    frame.to_csv(out / 'metrics_1800.csv', index=False)
    write_json(out / 'gt_complete.json', dict(at=now(), rows=1800, saved_output_only=True,
               metrics_sha256=digest(out / 'metrics_1800.csv')))
    archive(out, ['metrics_1800.csv', 'gt_access.json', 'gt_complete.json', 'first_posthoc_gt_boundary.json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'smoke', 'train', 'infer', 'gt'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    c, base, out = configuration()
    if args.phase == 'train':
        if args.arm is None:
            parser.error('--arm required for train')
        train(c, base, out, args.arm)
    else:
        globals()[args.phase](c, base, out)


if __name__ == '__main__':
    main()
