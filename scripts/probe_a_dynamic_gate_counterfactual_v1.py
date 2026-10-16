#!/usr/bin/env python3
"""Post hoc, measurement-only last-checkpoint gamma=1 counterfactual.

No training, checkpoint edits, GT reads, or formal model selection. A repeated
normal-gate pass also measures numerical repeat variation at the same weights.
"""
import copy
import json
from pathlib import Path
import time

import torch
import yaml
import probe_a_dynamic_steps_v1 as p
from geophy_set3d.data.build_utils import digest, timestamp, write_json
from geophy_set3d.training.gt_read_guard import GTReadGuard


def main():
    registration = json.loads((p.BASE / 'registration.json').read_text())
    assert registration['script_sha256'] == digest(Path(p.__file__))
    run = p.BASE / 'runs/bounded_dynamic'
    checkpoint = run / 'checkpoints/last.pt'
    target = p.BASE / 'posthoc_gate_counterfactual'
    target.mkdir(exist_ok=False)
    c = yaml.safe_load((run / 'config_resolved.yaml').read_text())
    t = p.private_trainer(c)
    t.seed_everything(c['training']['seed'])
    torch.set_num_threads(2)
    checksum = digest(checkpoint)
    write_json(target / 'registration.json', dict(registered_at=timestamp(), posthoc=True,
        script_sha256=digest(Path(__file__)), checkpoint_sha256=checksum,
        intervention='same last weights: normal gate, a_k=0 (gamma=1), repeated restored normal gate',
        train_allowed=False, gt_allowed=False, final_test_access_allowed=False,
        selection_allowed=False, cases=36, layout='fixed registered dual validation A/B/H epoch0'))
    began = time.monotonic()
    with GTReadGuard(c['dataset']['root'], run / 'checkpoints') as guard:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['global_step'] == registration['planned_updates_per_arm']
        model = p.build_arm(c).to(c['training']['device'])
        model.load_state_dict(saved['model'])
        strengths = model.dynamic_strength.detach().clone()
        data = t.make_research_dataset(c, validation=True)
        data.set_epoch(0)
        loader = t.make_research_loader(data, c, generator=torch.Generator().manual_seed(0))
        results = {}
        for setting in ('normal', 'gamma_one', 'normal_repeat'):
            with torch.no_grad():
                model.dynamic_strength.copy_(torch.zeros_like(strengths) if setting == 'gamma_one' else strengths)
            value = t.run_research_epoch(model, loader, c, model.operator)
            assert value['complete_epoch'] and value['sample_count'] == 36
            write_json(target / f'{setting}.json', value)
            results[setting] = dict(heldout_relative_l1=value['measurement']['heldout']['relative_l1'],
                loss=value['loss']['total'], gates=value['dynamic_gate'], seconds=value['seconds'])
        audit = guard.summary()
    assert digest(checkpoint) == checksum
    a, b, repeat = [results[k]['heldout_relative_l1'] for k in ('normal', 'gamma_one', 'normal_repeat')]
    summary = dict(posthoc=True, screening_only=True, no_training=True, gt_audit=audit,
        checkpoint_preserved=True, checkpoint_sha256=checksum, results=results,
        gamma_one_minus_normal_H=b-a, gamma_one_relative_change=b/a-1,
        normal_repeat_minus_normal_H=repeat-a, elapsed_seconds=time.monotonic()-began,
        interpretation='Inference-only gate intervention; does not remove training coadaptation or establish causality for between-arm H differences.')
    write_json(target / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
