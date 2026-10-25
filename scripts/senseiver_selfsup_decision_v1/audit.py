"""Read-only historical identity audit; no dataset tensors or GPU are read."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from geophy_set3d.training.utils import seed_everything
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.ray_senseiver import build_comparison_model
from geophy_set3d.models.baseline_senseiver_v1 import RaySenseiverV1

ROOT = Path('/scratch/djc/GeoPhy-Set3D/senseiver_selfsup_decision_v1')
ART = PROJECT / 'docs/reports/artifacts/senseiver_selfsup_decision_v1'
HIST = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/ray_senseiver_selfsup_v1')
CK = Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/ray_senseiver_selfsup_v1')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def main():
    ROOT.mkdir(exist_ok=False)
    ART.mkdir(parents=True, exist_ok=False)
    c = yaml.safe_load((PROJECT / 'configs/train_ray_senseiver_selfsup_v1.yaml').read_text())
    arc = read(HIST / 'early_termination_archive.json')
    env = read(HIST / 'environment.json')
    protocol = read(HIST / 'protocol.json')
    manifest = read(HIST / 'training_source_manifest.json')['sha256']
    history = [json.loads(s) for s in (HIST / 'metrics.jsonl').read_text().splitlines()]
    assert len(history) == 113 and [r['epoch'] for r in history] == list(range(113))
    assert all(r['optimizer_steps'] == 83 * (i + 1) and r['optimizer_steps_this_epoch'] == 83 for i, r in enumerate(history))
    assert all(r['train']['sample_count'] == 166 and r['val']['sample_count'] == 36 and not r['train']['gt_used'] and not r['val']['gt_used'] for r in history)
    assert c == arc['config'] == protocol['config']
    source_comparison = {p: dict(expected=h, actual=sha(PROJECT/p)) for p, h in manifest.items()}
    assert all(v['expected'] == v['actual'] for v in source_comparison.values())
    metadata = {p: dict(expected=h, actual=sha(p)) for p, h in protocol['metadata_sha256'].items()}
    assert all(v['expected'] == v['actual'] for v in metadata.values())
    assert sha(HIST/'protocol.json') == arc['protocol_sha256'] == (HIST/'protocol.sha256').read_text().strip()
    assert sha(HIST/'training_source_manifest.json') == arc['source_manifest_sha256']
    checkpoints = {}
    for key in ('best', 'last'):
        path = CK / (key + '.pt')
        digest = sha(path)
        assert digest == arc[key]['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 112 and saved['optimizer_steps'] == 9379
        assert saved['config'] == c and not saved['volumetric_gt_used']
        assert saved['source_manifest_sha256'] == arc['source_manifest_sha256']
        steps = sorted({int(v['step']) for v in saved['optimizer']['state'].values()})
        assert steps == [9379] and saved['scheduler']['last_epoch'] == 113
        assert saved['rng'] and all(torch.isfinite(t).all() for t in saved['model'].values())
        checkpoints[key] = dict(path=str(path), sha256=digest, completed_epochs=113, updates=9379,
            optimizer_state_steps=steps, scheduler_last_epoch=113, model_state_sha256=state_hash(saved['model']),
            initial_state_sha256=saved['initial_state_sha256'], best_val_metric=saved['best_val_metric'],
            rng_keys=list(saved['rng']), training_signal=saved['training_signal'], selection_metric=saved['selection_metric'])
    seed_everything(c['training']['seed'])
    fresh = build_comparison_model(c, 'cpu')
    initial = state_hash(fresh.state_dict())
    seed_everything(c['training']['seed'])
    wrapper = RaySenseiverV1(**c['runtime'])
    assert state_hash(wrapper.state_dict()) == initial == env['initial_state_sha256']
    assert sum(p.numel() for p in fresh.parameters()) == 254913
    fields = ('dataset','normalization','loss','train','validation','optimizer','scheduler','training')
    ds = yaml.safe_load((PROJECT/'configs/train_deepsets_selfsup_v1.yaml').read_text())
    plain = read('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/main_anchor.json')['model']['config']
    equivalence = {name: {k: c[k] == other[k] for k in fields} for name, other in [('GeoLift-DeepSets-Sum', ds), ('Plain-final', plain)]}
    assert all(all(v.values()) for v in equivalence.values())
    records = [json.loads(s) for s in (Path(c['dataset']['root'])/'source_index.jsonl').read_text().splitlines()]
    counts = {s: sum(r['split'] == s for r in records) for s in ('train','val','test')}
    assert counts['train'] == 166 and counts['val'] == 36
    paths = subprocess.check_output(['git','ls-files','-z'], cwd=PROJECT).decode().split('\0')
    protected = {p: sha(PROJECT/p) for p in paths if p and (PROJECT/p).is_file()}
    write(ROOT/'protected_tracked_files.json', protected)
    supervised_path = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/ray_senseiver_supervised_v1/science/checkpoints/best.pt')
    supervised_sha = sha(supervised_path)
    assert supervised_sha == 'eca3419f19b103feeb14840741d14a369110a970c55f7d7d746b727848d63d9f'
    queue = Path('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/queue_status.json')
    evidence = {str(p): sha(p) for p in HIST.iterdir() if p.is_file()}
    evidence[str(supervised_path)] = supervised_sha
    evidence[str(queue)] = sha(queue)
    for p in [PROJECT/'configs/train_ray_senseiver_selfsup_v1.yaml', PROJECT/'configs/baselines/senseiver/model_v1.json', PROJECT/'docs/senseiver_selfsup_decision_v1_protocol_zh.md', *Path(__file__).parent.glob('*.py')]:
        evidence[str(p)] = sha(p)
    result = dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        outcome='NO_GO_EXISTING_SAME_PROTOCOL_FAILED', new_training_updates=0,
        history=dict(completed_epochs=113, actual_updates=9379, planned_epochs=300, planned_updates=24900,
            all_epochs_consecutive=True, each_epoch_updates=83, all_val_H_L1_above_one=all(r['val']['measurement']['heldout']['relative_l1']>1 for r in history),
            best_epoch_1based=1+min(history,key=lambda r:r['val']['measurement']['heldout']['relative_l1'])['epoch'],
            original_exit=arc['status'], stopped_at=arc['stopped_at'], stop_reason=arc['reason'],
            epoch0_full_val_available=False, missing_gate_evidence='no full-Val epoch0, no two consecutive passing conditional checks'),
        config=c, source_comparison=source_comparison, metadata=metadata, checkpoints=checkpoints,
        source_files_checked=len(manifest), source_changes=0, initial_state_sha256=initial,
        wrapper_initial_state_exact_match=True, parameters=254913, common_training_fields_equal=equivalence,
        split_counts=counts, historical_access=read(HIST/'training_access_audit.json'),
        archive_embedded_access=arc['access'], supervised=dict(path=str(supervised_path),sha256=supervised_sha,
            paradigm='supervised-only; no network loaded', complete_epochs=187, partial_next_updates=59, updates=15580, val_nmse=.28681096),
        queue_status_before=read(queue), evidence_sha256=evidence)
    write(ART/'history_audit.json', result)
    with (ART/'training_curve.csv').open('w') as f:
        names=['epoch','updates','train_loss','val_loss','train_H_L1','val_H_L1','output_mean','output_std','output_case_difference','gradient_max','postclip_gradient_max','learning_rate','seconds','peak_allocated_bytes']
        w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader()
        for r in history:
            w.writerow(dict(epoch=r['epoch']+1,updates=r['optimizer_steps'],train_loss=r['train']['loss']['total'],val_loss=r['val']['loss']['total'],train_H_L1=r['train']['measurement']['heldout']['relative_l1'],val_H_L1=r['val']['measurement']['heldout']['relative_l1'],output_mean=r['val']['output']['mean'],output_std=r['val']['output']['std'],output_case_difference=r['val']['output']['case_to_case_mean_absolute_difference'],gradient_max=r['train']['gradient_norm_max'],postclip_gradient_max=r['post_clip_gradient_norm_max'],learning_rate=r['learning_rate'],seconds=r['seconds'],peak_allocated_bytes=r['gpu_peak_allocated_bytes']))
    print(json.dumps(dict(outcome=result['outcome'], source_files=len(manifest), initial=initial, epochs=len(history), updates=9379)))

if __name__ == '__main__':
    main()
