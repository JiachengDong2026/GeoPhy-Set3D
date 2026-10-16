#!/usr/bin/env python3
"""Independent A2 bounded-step screening; never freezes a formal model.

Only the last two physical displacements change. No GT or held-out observation
enters their gate. Frozen source files and existing experiment manifests stay
untouched. Both arms use the same 300-epoch cosine prefix and full 166/36 data.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from geophy_set3d.data.build_utils import digest, timestamp, write_json
from geophy_set3d.training.gt_read_guard import GTReadGuard
from plan_a_budget_v4 import build_budget_model
from plan_a_runtime_candidates import configure_backend, configure_model

BASE = ROOT / 'outputs/research/a_dynamic_steps_v1'
FORMAL = ROOT / 'outputs/research/scheme_a_budget_v4_wide_seed20261011'
PROTOCOL = 'a_dynamic_steps_screening_v1'
ARMS = ('reference', 'bounded_dynamic')
RELATIVE_SCALE = .25


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def changed_forward():
    """Exact private copy of the frozen forward except one explicit statement.

    rho = ||clamp(c-tau*u)-c|| / (.25*||c||+1e-8).
    gamma = 2*sigmoid(-a[k-2]*rho); a is zero-initialized and trainable.
    Rho is detached; gamma is FP32, in (0,2), initially exactly 1.
    First two steps and prior signal are exact.
    """
    import geophy_set3d.models.set_unrolled3d as module
    source = textwrap.dedent(inspect.getsource(module.SetUnrolled3D.forward))
    old = '            z_unclipped = c - tau[k] * update'
    new = '''            z_unclipped = c - tau[k] * update
            if k >= 2:
                proposed = z_unclipped.clamp(0, 1) - c
                relative = (self._norm(proposed) / (.25 * self._norm(c) + 1e-8)).detach().float()
                bounded_gate = 2 * torch.sigmoid(-self.dynamic_strength[k-2] * relative)
                self._dynamic_records.append((k, bounded_gate.detach(), relative))
                # Keep the un-clipped proposal: at initialization gamma=1 is
                # exactly the reference both forward and backward through clip.
                z_unclipped = c - bounded_gate.reshape(-1, 1, 1, 1, 1) * tau[k] * update'''
    assert source.count(old) == 1
    namespace = {}
    exec(compile(source.replace(old, new), str(Path(__file__)) + ':private_forward', 'exec'), module.__dict__, namespace)
    return namespace['forward']


def build_arm(config, runtime=True):
    model = build_budget_model(config, runtime=False)
    if config['screening_arm'] == 'bounded_dynamic':
        model.dynamic_strength = torch.nn.Parameter(torch.zeros(2))
        model._dynamic_records = []
        model.forward = types.MethodType(changed_forward(), model)
    elif config['screening_arm'] != 'reference':
        raise ValueError('Unknown arm')
    return configure_model(model, config['execution_runtime']) if runtime else model


def template(arm):
    c = yaml.safe_load((FORMAL / 'config_resolved.yaml').read_text())
    c.update(status='executable', protocol=PROTOCOL, stage='screening', screening_arm=arm)
    c['experiment'] = dict(name=arm, output_root=str(BASE / 'runs'))
    c['training']['device'] = 'cuda:0'
    c['training']['max_epochs'] = c['scheduler']['T_max'] = 300
    c.pop('acceptance', None)
    c['entrypoint_required'] = str(Path(__file__).relative_to(ROOT))
    return c


def register(epochs):
    if not 1 <= epochs <= 20:
        raise ValueError('Screening budget must be 1..20 complete epochs')
    BASE.mkdir(parents=True, exist_ok=False)
    settings = []
    for arm in ARMS:
        c = template(arm)
        torch.manual_seed(c['training']['seed'])
        m = build_arm(c, runtime=False)
        settings.append(dict(arm=arm, parameters=sum(p.numel() for p in m.parameters()),
                             common_initialization_sha256=tensor_hash({k:v for k,v in m.state_dict().items() if k != 'dynamic_strength'}),
                             full_initialization_sha256=tensor_hash(m.state_dict())))
        (BASE / f'{arm}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    assert len({s['common_initialization_sha256'] for s in settings}) == 1
    assert [s['parameters'] for s in settings] == [488919, 488921]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src').rglob('*.py'))}
    write_json(BASE / 'registration.json', dict(registered_at=timestamp(), script_sha256=digest(Path(__file__)),
        planned_completed_epochs=epochs, updates_per_epoch=83, planned_updates_per_arm=epochs * 83,
        full_training_schedule_epochs=300, screening_only=True, formal_training_completed=False,
        train_cases=166, measurement_validation_cases=36, gt_read_allowed=False,
        parameters=settings, frozen_source_hashes=sources,
        intervention='Only k=3,4: rho=detach(norm(clamp(C-tau*U,0,1)-C)/(.25*norm(C)+1e-8)); gamma=2*sigmoid(-a_k*rho), a_k starts at 0; Z=clamp(C-gamma*tau*U,0,1). Gate FP32 in (0,2), initial gamma exactly 1. Full gradient through correction and trainable strength, rho detached; unchanged prior_signal.',
        preregistered_scale=RELATIVE_SCALE, model_selection='heldout projection relative L1 only',
        stopping='same complete epoch count in both arms; <=20; no 300-epoch launch; no GT model selection'))
    print(json.dumps(settings), flush=True)


def private_trainer(config):
    path = ROOT / 'src/geophy_set3d/training/research_selfsup.py'
    spec = importlib.util.spec_from_file_location('geophy_set3d.training._a_dynamic_screening', path)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    original_validate, original_provenance = t.validate_research_config, t._provenance
    t.PROTOCOL = PROTOCOL

    def validate(c):
        if c['stage'] != 'screening' or c['screening_arm'] not in ARMS:
            raise ValueError('Independent screening only')
        if c['dataset']['train_limit'] is not None or c['dataset']['val_limit'] is not None:
            raise ValueError('Full 166/36 membership required')
        checked = copy.deepcopy(c)
        checked['stage'] = 'formal'
        return original_validate(checked)

    def provenance(c, train, val):
        result = original_provenance(c, train, val)
        for name in ('scripts/probe_a_dynamic_steps_v1.py', 'scripts/plan_a_budget_v4.py', 'scripts/plan_a_runtime_candidates.py'):
            result['code'][name] = digest(ROOT / name)
        result.update(source_sha256=t._canonical(result['code']), screening_arm=c['screening_arm'],
                      screening_only=True, formal_training_completed=False)
        return result

    def build(c):
        model = build_arm(c)
        expected = next(s for s in json.loads((BASE / 'registration.json').read_text())['parameters'] if s['arm'] == c['screening_arm'])
        assert tensor_hash(model.state_dict()) == expected['full_initialization_sha256']
        return model

    import geophy_set3d.models.set_unrolled3d as models
    models.build_set_unrolled_model = build
    original_seed = t.seed_everything
    def seed(value):
        original_seed(value)
        configure_backend(config['execution_runtime'])
    t.seed_everything, t.validate_research_config, t._provenance = seed, validate, provenance
    original_epoch = t.run_research_epoch
    def epoch(model, *args, **kwargs):
        if hasattr(model, '_dynamic_records'):
            model._dynamic_records.clear()
        result = original_epoch(model, *args, **kwargs)
        if hasattr(model, '_dynamic_records'):
            summaries = []
            for k in (2, 3):
                records = [v for v in model._dynamic_records if v[0] == k]
                summary = dict(step=k+1, a_k=float(model.dynamic_strength[k-2].detach().cpu()))
                for name, idx in (('gamma', 1), ('rho', 2)):
                    values = torch.cat([v[idx].flatten() for v in records]).float().cpu()
                    assert bool(torch.isfinite(values).all())
                    summary[name] = dict(count=values.numel(), min=float(values.min()), max=float(values.max()),
                        mean=float(values.mean()), p10=float(values.quantile(.1)), p50=float(values.quantile(.5)), p90=float(values.quantile(.9)))
                summaries.append(summary)
            result['dynamic_gate'] = summaries
            model._dynamic_records.clear()
        return result
    t.run_research_epoch = epoch
    return t


def bench():
    """One real two-case optimizer update per arm; no use of heldout/GT gate input."""
    from torch.utils.data import Subset
    from geophy_set3d.training import research_selfsup as t
    result = []
    c = template('reference')
    torch.set_num_threads(2)
    with GTReadGuard(c['dataset']['root'], BASE / 'unused_bench_checkpoints') as guard:
        data = t.make_research_dataset(c)
        first = next(iter(t.make_research_loader(Subset(data, range(1)), c)))
        first = t.move_dual_batch(first, 'cuda:0')['A']
        initial_outputs = []
        for arm in ARMS:
            c['screening_arm'] = arm
            t.seed_everything(c['training']['seed']); configure_backend(c['execution_runtime'])
            model = build_arm(c).to('cuda:0').eval()
            with torch.no_grad(): initial_outputs.append(model(first).detach().cpu())
            del model
            torch.cuda.empty_cache()
        initial_max_difference = float((initial_outputs[0]-initial_outputs[1]).abs().max())
        print(json.dumps(dict(initial_forward_max_abs_difference=initial_max_difference)), file=sys.stderr, flush=True)
        assert initial_max_difference <= 1e-6, initial_max_difference
        for arm in ARMS:
            c['screening_arm'] = arm
            t.seed_everything(c['training']['seed'])
            configure_backend(c['execution_runtime'])
            model = build_arm(c).to('cuda:0')
            loader = t.make_research_loader(Subset(data, range(4)), c, generator=torch.Generator().manual_seed(0))
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            output = t.run_research_epoch(model, loader, c, model.operator, optimizer)
            torch.cuda.synchronize()
            result.append(dict(arm=arm, seconds=time.monotonic()-began,
                update_count=output['optimizer_updates'], max_gradient_norm=output['gradient_norm_max'],
                finite_gradients=bool(np.isfinite(output['gradient_norm_max'])),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), train_H=output['measurement']['heldout']['relative_l1']))
            del model, optimizer, loader
            torch.cuda.empty_cache()
        audit = guard.summary()
    value = dict(bench=result, initial_forward_max_abs_difference=initial_max_difference,
        initial_forward_exactly_equal=initial_max_difference == 0, initial_forward_equal_at_atol_1e_6=initial_max_difference <= 1e-6,
        gt_read_audit=audit, gpu=torch.cuda.get_device_name(), generated_at=timestamp())
    if BASE.exists():
        write_json(BASE / 'bench.json', value)
    print(json.dumps(value), flush=True)


def train(arm):
    registration = json.loads((BASE / 'registration.json').read_text())
    assert registration['script_sha256'] == digest(Path(__file__))
    c = yaml.safe_load((BASE / f'{arm}.yaml').read_text())
    t = private_trainer(c)
    began = time.monotonic()
    run = t.train_research(c, stop_after_epochs=registration['planned_completed_epochs'])
    metrics = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()]
    audit = json.loads((run / 'gt_read_audit.json').read_text())
    write_json(run / 'screening_summary.json', dict(completed_screening=True, completed_formal_training=False,
        completed_epochs=len(metrics), optimizer_updates=metrics[-1]['global_step'], elapsed_seconds=time.monotonic()-began,
        best_H=min(m['selection_metric'] for m in metrics), final_H=metrics[-1]['selection_metric'],
        gt_read_audit=audit, checkpoint_not_formally_frozen=True))


def report():
    registration = json.loads((BASE / 'registration.json').read_text())
    assert all(digest(ROOT / name) == value for name, value in registration['frozen_source_hashes'].items())
    arms = {}
    for arm in ARMS:
        run = BASE / 'runs' / arm
        arms[arm] = dict(summary=json.loads((run / 'screening_summary.json').read_text()),
            metrics=[json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()],
            sampling_manifest_sha256=digest(run / 'sampling_manifest.jsonl'))
    assert len({a['sampling_manifest_sha256'] for a in arms.values()}) == 1
    assert len({a['summary']['optimizer_updates'] for a in arms.values()}) == 1
    # Loader RNG equality confirms matching shuffle sequence in addition to view manifests.
    last = [torch.load(BASE / 'runs' / arm / 'checkpoints/last.pt', map_location='cpu', weights_only=False) for arm in ARMS]
    assert torch.equal(last[0]['rng']['loader'], last[1]['rng']['loader'])
    assert last[0]['scheduler'] == last[1]['scheduler']
    for value in arms.values():
        assert not value['summary']['gt_read_audit']['gt_open_attempts']
        assert all(np.isfinite(m['train']['gradient_norm_max']) for m in value['metrics'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, value in arms.items():
        m = value['metrics']; x = [r['epoch']+1 for r in m]
        axes[0].plot(x, [r['train']['loss']['total'] for r in m], label=arm)
        axes[1].plot(x, [100*r['selection_metric'] for r in m], label=arm)
    axes[0].set_ylabel('Training total loss'); axes[1].set_ylabel('Heldout projection relative L1 (%)')
    for ax in axes:
        ax.set_xlabel('Epoch (300-epoch cosine prefix)'); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(BASE / 'curves.png', dpi=160); plt.close(fig)
    summary = dict(screening_only=True, frozen_sources_preserved=True, identical_initial_weights=True,
        identical_sampling_manifests=True, identical_final_loader_rng=True, identical_scheduler=True,
        gt_read_allowed=False, results={k:v['summary'] for k,v in arms.items()})
    write_json(BASE / 'summary.json', summary)
    lines = ['# A 有界动态步长匹配短程筛查', '', '本实验不是正式 300 轮训练，也不构成三维精度结论。全程仅使用投影，未读取体素 GT。', '',
        '两臂从完全相同随机权重开始，使用全部 166 个训练案例、36 个验证案例、相同视角采样与 shuffle，以及 300 轮 cosine 的相同前缀；每轮 83 次更新。', '',
        '唯一干预为后两步物理位移的有界动态强度：`rho=detach(||clamp(C−tau*U,0,1)−C||/(0.25||C||+1e−8))`，`gamma=2*sigmoid(−a_k*rho)`，`Z=clamp(C−gamma*tau*U,0,1)`。保留物理校正与可学习强度梯度、原 prior_signal，rho 统计量不反传，gamma 在 FP32 计算，前两步不变。两个新增标量零初始化，因此初始 gamma 精确为 1，继承 reference 的前向与共有参数梯度；学习时 gamma 在 (0,2) 内，可衰减或增强。0.25 是预注册固定值，未搜索。每轮 metrics 含 gamma/rho 分布与 a_k。', '',
        '| Arm | Epochs | Updates | Best H | Final H | Runtime | Parameters |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, value in arms.items():
        s = value['summary']; parameters = next(v['parameters'] for v in registration['parameters'] if v['arm'] == arm); lines.append(f"| {arm} | {s['completed_epochs']} | {s['optimizer_updates']} | {100*s['best_H']:.4f}% | {100*s['final_H']:.4f}% | {s['elapsed_seconds']/60:.2f} min | {parameters} |")
    relative = arms['bounded_dynamic']['summary']['best_H']/arms['reference']['summary']['best_H']-1
    lines.extend(['', f'动态臂相对 reference 的最佳 H 变化为 {100*relative:+.2f}%。这只反映短程投影泛化，尚不能证明普通/困难布局的三维收益。', '',
        '已核对：完整初始化哈希相同、视角采样 manifest 相同、最终 loader RNG 与 scheduler 相同、梯度有限、全部冻结 src 文件哈希不变、两臂 GT 读取次数均为 0。两臂均未产生正式 freeze.json。', '',
        '![曲线](../../outputs/research/a_dynamic_steps_v1/curves.png)', '',
        '证据：`outputs/research/a_dynamic_steps_v1/{registration.json,summary.json,runs/*/metrics.jsonl,runs/*/gt_read_audit.json}`。', '',
        '局限：单种子、早期学习阶段；验证 H 的改善未必对应三维误差改善；未测独立最终测试集。继续训练必须先综合冻结缩放诊断和同期其他候选。'])
    (ROOT / 'docs/reports/a_dynamic_steps_v1_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'bench', 'train', 'report'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    if a.action == 'register': register(a.epochs)
    elif a.action == 'bench': bench()
    elif a.action == 'train': train(a.arm)
    else: report()


if __name__ == '__main__':
    main()
