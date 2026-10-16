#!/usr/bin/env python3
"""Read saved scalars only; never constructs a model or runs a projector."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
METRICS = ['relative_l1', 'relative_l2', 'mae', 'rmse', 'peak_value_absolute_error',
           'integrated_concentration_relative_error', 'observed_relative_l1',
           'observed_relative_l2', 'heldout_relative_l1', 'heldout_relative_l2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stat(before, after, indices):
    before, after = np.asarray(before), np.asarray(after)
    delta = after - before
    lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
    return dict(before=float(before.mean()), after=float(after.mean()), delta=float(delta.mean()),
                ci_low=float(lo), ci_high=float(hi), ratio=float(after.mean() / before.mean()),
                reduction=float(1 - after.mean() / before.mean()),
                wins=int((delta < -1e-8).sum()), losses=int((delta > 1e-8).sum()),
                ties=int((np.abs(delta) <= 1e-8).sum()))


def main():
    c = yaml.safe_load((REPO / 'configs/ours_c0_adapter_matched_v1.yaml').read_text())
    out = Path(c['output'])
    complete = json.loads((out / 'gt_complete.json').read_text())
    assert sha(out / 'metrics_1800.csv') == complete['metrics_sha256']
    manifest = dict(source='scripts/report_ours_c0_adapter_matched.py', sha256=sha(__file__),
                    scalar_only=True, model_forwards=0, projector_calls=0)
    mf = out / 'report_executor_manifest.json'
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest
    else:
        dump(mf, manifest)
    f = pd.read_csv(out / 'metrics_1800.csv')
    old = pd.read_csv(REPO / c['reference'])
    assert len(f) == 1800 and len(old) == 900
    identity = ['case_id', 'case_index', 'group', 'draw', 'geometry_id']
    assert not f.duplicated(['arm', *identity]).any() and not old.duplicated(identity).any()
    columns = [s + '_' + m for s in ('C0', 'C1') for m in METRICS]
    old = old[identity + columns].copy()
    old['arm'] = 'Original'
    allrows = pd.concat([old, f[identity + ['arm'] + columns]], ignore_index=True)
    assert len(allrows.groupby(['arm', 'case_id', 'group']).size()) == 540
    assert (allrows.groupby(['arm', 'case_id', 'group']).size() == 5).all()
    cases = allrows.groupby(['arm', 'case_id', 'group'], as_index=False)[columns].mean()
    cases.to_csv(out / 'case_group_540.csv', index=False)
    rng = np.random.default_rng(c['bootstrap']['seed'])
    indices = rng.integers(0, 36, (c['bootstrap']['samples'], 36))
    groups = [f'G{i}' for i in range(1, 6)]
    results = []
    for stage in ('C0', 'C1'):
        for group in groups + ['ordinary_macro']:
            sub = cases[cases['group'] != 'G5'] if group == 'ordinary_macro' else cases[cases['group'] == group]
            means = sub.groupby(['arm', 'case_id'])[columns].mean()
            for before, after in [('Control', 'Treatment'), ('Original', 'Treatment'), ('Original', 'Control')]:
                a = means.loc[before].sort_index()
                b = means.loc[after].sort_index()
                assert list(a.index) == list(b.index) and len(a) == 36
                for metric in METRICS:
                    results.append(dict(stage=stage, group=group, metric=metric, comparison=after + '-' + before,
                                        **stat(a[stage + '_' + metric], b[stage + '_' + metric], indices)))
    stats = pd.DataFrame(results)
    stats.to_csv(out / 'paired_statistics.csv', index=False)
    tails = []
    for (arm, group), part in cases.groupby(['arm', 'group']):
        for stage in ('C0', 'C1'):
            values = part[stage + '_relative_l1']
            tails.append(dict(arm=arm, group=group, stage=stage, mean=float(values.mean()),
                              median=float(values.median()), p90=float(values.quantile(.9)),
                              p95=float(values.quantile(.95)), maximum=float(values.max()),
                              worst_case=part.loc[values.idxmax(), 'case_id']))
    pd.DataFrame(tails).to_csv(out / 'tails.csv', index=False)
    checks = {}
    for baseline in ('Control', 'Original'):
        def get(group, metric):
            return stats[(stats.stage == 'C1') & (stats.group == group) &
                         (stats.metric == metric) & (stats.comparison == 'Treatment-' + baseline)].iloc[0]
        hard = get('G5', 'relative_l1')
        ordinary = get('ordinary_macro', 'relative_l1')
        h = get('G5', 'heldout_relative_l1')
        checks[baseline] = dict(g5_at_least_5pct=bool(hard['reduction'] >= c['gate']['g5_relative_reduction']),
                               g5_ci_below_zero=bool(hard['ci_high'] < 0),
                               ordinary_within_3pct=bool(ordinary['ratio'] <= c['gate']['ordinary_max_ratio']),
                               g5_heldout_within_3pct=bool(h['ratio'] <= c['gate']['heldout_max_ratio']))
    decision = 'GO_LIMITED_CONTINUATION' if all(all(v.values()) for v in checks.values()) else 'NO_GO'
    dump(out / 'decision.json', dict(decision=decision, checks=checks,
         meaning='Only this fixed-input single-seed continuation, not a final paper claim',
         protocol_sha256=sha(out / 'protocol.json')))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(5)
    colors = dict(Original='#7a7a7a', Control='#286fb4', Treatment='#d36a29')
    for arm in ('Original', 'Control', 'Treatment'):
        part = cases[cases.arm == arm].groupby('group').mean(numeric_only=True).reindex(groups)
        for ax, stage in zip(axes, ('C0', 'C1')):
            ax.plot(x, part[stage + '_relative_l1'], 'o-', label=arm, color=colors[arm])
            ax.set_xticks(x, groups)
            ax.set_title(stage + ': fixed identical V4 inputs')
            ax.set_ylabel('3D relative L1 (lower is better)')
            ax.legend()
    fig.savefig(out / 'group_errors.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm in ('Control', 'Treatment'):
        rows = [json.loads(x) for x in (out / arm / 'metrics.jsonl').read_text().splitlines()]
        axes[0].plot([x['epoch'] for x in rows], [x['val']['measurement']['heldout']['relative_l1'] for x in rows], label=arm)
        trainrows = [x for x in rows if 'train' in x]
        axes[1].plot([x['epoch'] for x in trainrows], [x['train']['loss']['total'] for x in trainrows], label=arm)
    for ax, label in zip(axes, ('Natural Val H relative L1', 'Training objective')):
        ax.set_xlabel('Continuation epoch')
        ax.set_ylabel(label)
        ax.legend()
    fig.savefig(out / 'training_curves.png', dpi=160)
    plt.close(fig)
    summary = stats[(stats.stage == 'C1') & (stats.metric == 'relative_l1') &
                    (stats.comparison == 'Treatment-Control')].set_index('group')
    frozen = json.loads((out / 'frozen_best.json').read_text())['completed']
    lines = [
        '# Ours C0局部方向描述适配：固定输入有限对照', '',
        f'**判定：{decision}。** 两组均实际完成20轮/1,660次更新，从同一无三维标签v1权重继续训练；没有调整视角分布。', '',
        '新增C0适配器1,152参数（332,070→333,222，+0.347%），只将已有局部方向描述直接提供给C0融合特征。末层零初始化；真实数据C0/C1及反馈输出逐元素一致，实际原测量loss反传梯度有限非零。原模型/loss/projector保持原文件，独立实现以零适配输出逐stage等价验证。', '',
        '## 固定观测条件与预算', '',
        '两臂原166 train/36 Val、原natural sampler epoch0–19、相同case顺序及A/B/H站位；所有20轮训练与Val的case/station digest相同。评价复用原900条V4布局，含全部36case、每组5draw、相同H4。两臂完整输入tensor与H hash逐条相同，并与原冻结评价核验measurement/geometry/mask身份。没有新观测、监督初始化或三维标签训练。', '',
        'AdamW lr2e−5/wd1e−4、cosine20、FP32、batch2、clip1，原loss全部权重不变。两组均重置optimizer/scheduler；epoch0也参与自然Val held-out L1选模。不是fresh300轮比较；不能据此次继续训练宣称超过所有经典/深度学习基线。', '',
        '| arm | epochs / steps | best continuation epoch | natural Val selector |',
        '| --- | --- | --- | --- |']
    for arm in ('Control', 'Treatment'):
        v = frozen[arm]
        lines.append(f'| {arm} | 20 / 1660 | {v["best_epoch"]} | {v["best_val_heldout_relative_l1"]:.8f} |')
    lines += ['', '## C1主终点：Treatment与同期Control', '',
              '先每case/group平均5layout，以36case配对bootstrap10,000次。Δ=Treatment−Control；区间为逐项95%，无多重覆盖或跨seed保证。', '',
              '| group | Control L1 | Treatment L1 | Δ | Δ95%CI | 相对降低 | 胜/负/平 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for group in groups + ['ordinary_macro']:
        r = summary.loc[group]
        lines.append(f'| {group} | {r["before"]:.6f} | {r["after"]:.6f} | {r["delta"]:+.6f} | [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}] | {100*r["reduction"]:.2f}% | {r["wins"]}/{r["losses"]}/{r["ties"]} |')
    lines += ['', '## 原冻结v1与模块增量分开', '',
              '| comparison | group | before L1 | after L1 | Δ95%CI |',
              '| --- | --- | --- | --- | --- |']
    for comparison in ('Treatment-Original', 'Control-Original'):
        for group in ('ordinary_macro', 'G5'):
            r = stats[(stats.stage == 'C1') & (stats.metric == 'relative_l1') &
                      (stats.comparison == comparison) & (stats.group == group)].iloc[0]
            lines.append(f'| {comparison} | {group} | {r["before"]:.6f} | {r["after"]:.6f} | [{r["ci_low"]:+.6f}, {r["ci_high"]:+.6f}] |')
    lines += ['', '全部C0/C1、10项指标与三组配对见 [paired_statistics.csv](artifacts/ours_c0_adapter_matched_v1/paired_statistics.csv)；case组均值与最坏案例见 [case_group_540.csv](artifacts/ours_c0_adapter_matched_v1/case_group_540.csv)、[tails.csv](artifacts/ours_c0_adapter_matched_v1/tails.csv)。L2/peak和H指标不因与L1不同向而删除。', '',
              '![固定输入C0/C1误差](artifacts/ours_c0_adapter_matched_v1/group_errors.png)', '',
              '![训练与选模](artifacts/ours_c0_adapter_matched_v1/training_curves.png)', '',
              '## 预登记判据与结论边界', '',
              '要求Treatment相对同期Control和原冻结v1均：G5 L1至少下降5%且差值CI上界<0；普通macro L1、G5 H L1均不恶化超过3%。全部检查保留：', '',
              '```json', json.dumps(checks, indent=2), '```', '',
              '单seed、有限继续训练、已研究过的Val只支持这次探索；不构成最终test、新物理域或完整三维标签依赖曲线。未改gate、采样、loss或继续搜索第二模块；保留原v1作为基准。', '',
              '## 工程事件与访问', '',
              '首次新增单元测试误用了缺少ray_direction/t_near的简化fixture，3项错误发生在旧MRGR投影调用；修复为完整measurement-only fixture后5项新测试通过。原MRGR/selfsup共37项原测试在首次运行通过。原失败日志保留，预更新修订记录保留原登记及源码hash链，科学模型/输入/预算未改。', '',
              '真实smoke仅forward/backward，0 optimizer steps，权重不变；正式两臂均重新加载唯一登记的GT-free源。历史DiagnosticReadGuard只允许旧case_images目录，本轮独立评价入口额外精确白名单1,800个自身预测NPZ路径，未修改历史guard；执行器hash在推理/GT前封存。', '',
              '两组训练结束、best冻结且selector重放通过后，先保存并封存全部1,800份C0/C1，再由独立进程读36 Val GT64评价。训练/推理GT64/128、Pool B/test禁止；GT评价不加载checkpoint/测量。实际计数与异常session见归档access及complete，不将旧报告的测试结果冒称本轮重跑。', '',
              '[协议](../ours_c0_adapter_matched_v1_protocol_zh.md)；[完整归档](artifacts/ours_c0_adapter_matched_v1/)。']
    report = REPO / 'docs/reports/ours_c0_adapter_matched_v1_zh.md'
    report.write_text('\n'.join(lines) + '\n')
    archive = REPO / 'docs/reports/artifacts/ours_c0_adapter_matched_v1'
    import shutil
    for name in ['paired_statistics.csv', 'case_group_540.csv', 'tails.csv', 'decision.json',
                 'group_errors.png', 'training_curves.png', 'report_executor_manifest.json']:
        shutil.copy2(out / name, archive / name)
    print(json.dumps(dict(decision=decision, checks=checks,
                         main=summary.loc[['ordinary_macro', 'G5']].reset_index().to_dict('records')), indent=2))


if __name__ == '__main__':
    main()
