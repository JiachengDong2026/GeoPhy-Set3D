#!/usr/bin/env python3
"""Bounded, independent mechanism diagnostics; never a formal model selection.

No frozen implementation is edited. Short runs cannot unlock volume labels.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest, write_json, timestamp
from geophy_set3d.training.gt_read_guard import GTReadGuard
from plan_a_budget_v4 import build_budget_model
from plan_a_runtime_candidates import configure_backend, configure_model

BASE = ROOT / 'outputs/research/plan_a_mechanisms_v1'
FORMAL = ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011'
PROTOCOL = 'scheme_a_mechanisms_short_v1'
ARMS = ('reference', 'learned_feedback', 'oari_initializer')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tensor_hash(value):
    h = hashlib.sha256()
    for name, tensor in sorted(value.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_arm(config, runtime=True):
    model = build_budget_model(config, runtime=False)
    arm = config['mechanism_arm']
    if arm == 'learned_feedback':
        model.variant = 'A-CONTROL'
    elif arm == 'oari_initializer':
        from geophy_set3d.models.geolift_oari_res3d import GeoLiftOARIRes3D
        initializer = GeoLiftOARIRes3D(d_model=32, num_heads=4, num_blocks=1, ffn_channels=64)
        initializer.encoder.load_state_dict(model.initializer.encoder.state_dict())
        initializer.decoder.load_state_dict(model.initializer.decoder.state_dict())
        model.initializer = initializer
        model.variant = 'A3'
    elif arm != 'reference':
        raise ValueError(arm)
    return configure_model(model, config['execution_runtime']) if runtime else model


def register():
    from geophy_set3d.evaluation.research import frozen_record
    frozen = frozen_record(FORMAL)
    BASE.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load((FORMAL / 'config_resolved.yaml').read_text())
    settings = []
    for arm in ARMS:
        c = copy.deepcopy(template)
        c.update(protocol=PROTOCOL, status='executable', stage='gate', mechanism_arm=arm)
        c['experiment'] = dict(name=arm, output_root=str(BASE / 'short_runs'))
        c['dataset'].update(train_limit=16, val_limit=8)
        c['training'].update(max_epochs=20, device='cuda:0')
        c['scheduler']['T_max'] = 20
        c['budget'] = dict(max_updates=160, max_seconds=300)
        torch.manual_seed(c['training']['seed'])
        model = build_arm(c, runtime=False)
        sd = model.state_dict()
        common = {k: v for k, v in sd.items() if k.startswith(('initializer.encoder.', 'initializer.decoder.', 'prior.', 'step_embedding.')) or k == 'step_logits'}
        settings.append(dict(arm=arm, parameters=sum(p.numel() for p in model.parameters()),
                             common_initialization_sha256=tensor_hash(common),
                             full_initialization_sha256=tensor_hash(sd)))
        (BASE / f'{arm}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    assert len({s['common_initialization_sha256'] for s in settings}) == 1
    assert settings[0]['parameters'] == settings[1]['parameters'] == 488919
    assert settings[0]['full_initialization_sha256'] == settings[1]['full_initialization_sha256']
    write_json(BASE / 'registration.json', dict(
        registered_at=timestamp(), frozen_checkpoint_sha256=frozen['sha256'],
        script_sha256=digest(Path(__file__)), settings=settings,
        training='fresh random; 16 train / 8 val; 20 epochs / 160 updates per arm; <=300 s each',
        common='same initial encoder/decoder/prior/steps, sampling, shuffle, Q96, loss, AdamW and 20-epoch cosine',
        screening_only=True, short_run_gt_evaluation_allowed=False,
        diagnostic='frozen formal A: 36 legacy V4 + 36 clustered V4 draw0; save all C0/Z1/C1/.../Z4/C4 before independent GT score',
        frozen_sources_changed=False, final_test_access_allowed=False))
    print(json.dumps(settings), flush=True)


def train(arm):
    # Private module globals; frozen validation/training entrypoints stay intact.
    path = ROOT / 'src/geophy_set3d/training/research_selfsup.py'
    spec = importlib.util.spec_from_file_location('geophy_set3d.training._mechanism_probe', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original_validate, original_provenance = trainer.validate_research_config, trainer._provenance
    trainer.PROTOCOL = PROTOCOL
    config = yaml.safe_load((BASE / f'{arm}.yaml').read_text())

    def validate(c):
        if c['mechanism_arm'] != arm or c['stage'] != 'gate' or c['training']['max_epochs'] != 20:
            raise ValueError('Short experimental arm changed')
        return original_validate(c)

    def provenance(c, train_data, val_data):
        result = original_provenance(c, train_data, val_data)
        for name in ('scripts/probe_plan_a_mechanisms.py', 'scripts/plan_a_budget_v4.py', 'scripts/plan_a_runtime_candidates.py'):
            result['code'][name] = digest(ROOT / name)
        result['source_sha256'] = trainer._canonical(result['code'])
        result['mechanism_arm'] = arm
        result['screening_only'] = True
        return result

    def build(c):
        model = build_arm(c)
        expected = next(s for s in json.loads((BASE / 'registration.json').read_text())['settings'] if s['arm'] == arm)
        assert tensor_hash(model.state_dict()) == expected['full_initialization_sha256']
        return model

    import geophy_set3d.models.set_unrolled3d as models
    models.build_set_unrolled_model = build
    original_seed = trainer.seed_everything
    def seed(value):
        original_seed(value)
        configure_backend(config['execution_runtime'])
    trainer.seed_everything = seed
    trainer.validate_research_config = validate
    trainer._provenance = provenance
    trainer.train_research(config)


def stage_predict(device):
    from geophy_set3d.evaluation.research import frozen_record
    from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
    from geophy_set3d.data.collate import projection_pool_collate
    from geophy_set3d.training.utils import move_batch
    from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
    from geophy_set3d.evaluation.research import EvaluationWithHoldout
    from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
    directory = BASE / 'stages'
    directory.mkdir(exist_ok=False)
    frozen = frozen_record(FORMAL)
    prior = json.loads((FORMAL / 'evaluation/frozen_model_test/registration.json').read_text())
    settings = [s for s in prior['settings'] if s['name'] in ('Val-A-V4', 'clustered-V4')]
    config = yaml.safe_load((FORMAL / 'config_resolved.yaml').read_text())
    configure_backend(config['execution_runtime'])
    model = build_budget_model(config).to(device)
    dataset = MeasurementOnlyPoolDataset(config['dataset']['root'], split='val', geometry_pool='train',
        evaluation_only=True, min_input_views=4, max_input_views=4, hold_views=0,
        base_seed=20261011, normalization='linear_p99_5', statistics_file='train_statistics.json')
    manifest = []
    with GTReadGuard(config['dataset']['root'], FORMAL / 'checkpoints') as guard:
        checkpoint = torch.load(frozen['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model']); model.eval()
        for setting in settings:
            class Replay(EvaluationViewSampler):
                def sample(self, station_count, *, epoch, sample_index, case_id):
                    return np.array(setting['input_ids'][case_id], dtype=np.int64), np.empty(0, dtype=np.int64)
            dataset.sampler = EvaluationWithHoldout(Replay(dataset.geometry['station_azimuth_deg'], 4, 0, 'train', 20261011, 'random'))
            for index in range(len(dataset)):
                batch = move_batch(projection_pool_collate([dataset[index]]), device)
                inputs = {k:v for k,v in batch.items() if not k.startswith('hold_')}
                heldout = {k:v for k,v in batch.items() if not k.startswith(('input_', 'hold_')) and k!='view_mask'}
                heldout.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k!='hold_view_mask'})
                heldout['view_mask'] = batch['hold_view_mask']
                captured = []
                handle = model.prior.register_forward_pre_hook(lambda module, args: captured.append(args[0][:,1:2].detach().clone()))
                with torch.no_grad():
                    result = model(inputs, return_diagnostics=True)
                    handle.remove()
                    states = [('C0', result['C0'])]
                    for k in range(4):
                        states.extend([(f'Z{k+1}', captured[k]), (f'C{k+1}', result['steps'][k]['volume'])])
                    measurements = {}
                    for name, field in states:
                        measurements[name] = {}
                        for domain, target in [('observed', inputs), ('heldout', heldout)]:
                            pred = model.operator(field, target)
                            valid = target['input_valid_ray'].bool() & target['view_mask'][:,:,None,None,None]
                            measurements[name][domain] = relative_projection_metrics(pred[0], target['input_projection_raw'][0], valid[0])
                path = directory / f"{setting['name']}_{index:03d}.npz"
                np.savez_compressed(path, **{name:field[0,0].cpu().numpy() for name,field in states})
                manifest.append(dict(case_id=batch['case_id'][0], condition=setting['name'],
                    prediction=path.name, sha256=digest(path), measurement=measurements,
                    input_ids=setting['input_ids'][batch['case_id'][0]],
                    heldout_ids=batch['hold_station_indices'][0].tolist()))
            print('Saved nine states for 36 cases:', setting['name'], flush=True)
    (directory / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    write_json(directory / 'boundary.json', dict(cases=72, states_per_case=9,
        checkpoint_sha256=frozen['sha256'], gt_opened=False, all_states_saved_before_gt=True,
        dataset_root=config['dataset']['root'], manifest_sha256=digest(directory / 'manifest.jsonl'),
        audit=guard.summary(), generated_at=timestamp()))


def stage_score():
    from geophy_set3d.evaluation.research import frozen_record, volume_metrics
    directory = BASE / 'stages'
    boundary = json.loads((directory / 'boundary.json').read_text())
    assert frozen_record(FORMAL)['sha256'] == boundary['checkpoint_sha256']
    assert digest(directory / 'manifest.jsonl') == boundary['manifest_sha256']
    manifest = rows(directory / 'manifest.jsonl')
    source = {r['case_id']:r for r in rows(Path(boundary['dataset_root']) / 'source_index.jsonl')}
    assert len(manifest)==72 and len({(r['condition'],r['case_id']) for r in manifest})==72
    for row in manifest:
        assert source[row['case_id']]['split']=='val'
        assert digest(directory / row['prediction'])==row['sha256']
    scored=[]
    for row in manifest:
        target=np.load(Path(boundary['dataset_root']) / source[row['case_id']]['gt_volume_64_path'],allow_pickle=False)
        with np.load(directory / row['prediction'],allow_pickle=False) as saved:
            scored.append(dict(row, volume={name:volume_metrics(saved[name],target) for name in saved.files}))
    (directory / 'per_case_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    summary=[]
    for condition in ('Val-A-V4','clustered-V4'):
        group=[r for r in scored if r['condition']==condition]
        for name in group[0]['volume']:
            summary.append(dict(condition=condition,state=name,
                relative_l1=float(np.mean([r['volume'][name]['relative_l1'] for r in group])),
                observed_l1=float(np.mean([r['measurement'][name]['observed']['relative_l1'] for r in group])),
                heldout_l1=float(np.mean([r['measurement'][name]['heldout']['relative_l1'] for r in group]))))
    write_json(directory / 'summary.json',dict(results=summary,development_diagnostic_only=True,
        gt_for_training=False,gt_for_selection=False,final_test_evaluated=False))
    print(json.dumps(summary),flush=True)


def run_short():
    result=[]
    for arm in ARMS:
        began=time.monotonic()
        with (BASE / f'{arm}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__)), 'train','--arm',arm],stdout=log,stderr=subprocess.STDOUT)
            try:
                returncode=proc.wait(timeout=360)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
                returncode=proc.returncode
        result.append(dict(arm=arm,returncode=returncode,elapsed_seconds=time.monotonic()-began))
        write_json(BASE / 'execution.json',dict(arms=result))
        print(json.dumps(result[-1]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','train','stage-predict','stage-score','run-short'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.command=='register': register()
    elif args.command=='train': train(args.arm)
    elif args.command=='stage-predict': stage_predict(args.device)
    elif args.command=='stage-score': stage_score()
    else: run_short()
