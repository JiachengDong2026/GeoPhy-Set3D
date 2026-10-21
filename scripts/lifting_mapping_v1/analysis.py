"""GT64 scoring only after global seal; paired case statistics and report."""
from runtime import *
import csv
import gzip
import math
import shutil


def csv_write(path, records):
    assert not path.exists(), path
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(records)


def independent_metrics(p, g):
    """Independent array formulas; no call to repository metric functions."""
    axis = np.linspace(0, .6, 64)
    def center(a):
        return np.array([np.dot(a.sum(axis=tuple(j for j in range(3) if j != i)), axis) / a.sum() for i in range(3)])
    peak = g.max(); pm = p >= .5 * peak; gm = g >= .5 * peak
    return dict(relative_l1=float(np.abs(p-g).sum()/max(np.abs(g).sum(), 1e-6)),
                relative_l2=float(np.sqrt(((p-g)**2).sum()/max((g*g).sum(), 1e-12))),
                dice50=float(2*np.count_nonzero(pm & gm)/(np.count_nonzero(pm)+np.count_nonzero(gm))),
                absolute_peak_ratio_error=float(abs(p.max()/peak-1)),
                centroid_distance_mm=float(np.linalg.norm(center(p)-center(g))*1000) if p.sum() > 0 else None)


def canonical_metrics(p, g):
    from geophy_set3d.evaluation.research import volume_metrics
    import evaluation
    v = volume_metrics(p, g); s = evaluation.safe_shape(p, g)
    return dict(relative_l1=v['relative_l1'], relative_l2=v['relative_l2'], dice50=s['dice_0.5'],
                absolute_peak_ratio_error=abs(s['peak_ratio']-1), centroid_distance_mm=s['centroid_distance_mm'])


def scoring():
    seal = verified(BASE / 'all_predictions_sealed.json'); verify_hashes(seal['manifest_hashes'])
    p = verified(BASE / 'protocol.json'); verify_hashes(p['sources'])
    verify_hashes(verified(BASE / 'data_identity.json'))
    # Confirm GT64 metadata has not changed since the official final lock, without opening GT128.
    lock = verified(OUT / 'final_lock.json')
    for info in lock['gt64_stat_only'].values():
        stat = Path(info['path']).stat()
        assert stat.st_size == info['size'] and stat.st_mtime_ns == info['mtime_ns']
    cases = verified(BASE / 'cases.json'); source = {r['case_id']: r for r in rows(DATA / 'source_index.jsonl')}
    metrics_rows = []; paired = []; sensitivity = []; error = {}; old_error = {}; gt_hashes = {}; first_gt = None
    old_metrics = {(r['case_id'], r['condition'], r['draw'], r['stage']): r
                   for r in rows(OUT / 'evaluation/test/metrics.jsonl') if r['method'] == 'MRGR'}
    baseline = verified(BASE / 'baseline_manifest.json')
    new = []
    for li in range(5):
        root = BASE / 'predictions' / f'layout{li:02d}'
        new.extend((root, r) for r in rows(root / 'manifest.jsonl'))
    assert len(new) == 525
    with Guard(scoring=True) as guard:
        for ci, case in enumerate(cases):
            gt_path = DATA / source[case]['gt_volume_64_path']
            if first_gt is None:
                first_gt = now()
            g = np.load(gt_path, allow_pickle=False).squeeze().astype(np.float64)
            assert g.shape == (64,)*3 and np.isfinite(g).all() and g.min() >= 0 and g.max() <= 1 and g.sum() > 0
            gt_hashes[str(gt_path)] = sha(gt_path)
            references = {}
            for row in (r for r in baseline if r['case_id'] == case):
                li = row['layout_index']; refs = reference(row); references[li] = refs
                for stage in ('C0', 'C1'):
                    pred = refs[stage].astype(np.float64)
                    m = canonical_metrics(pred, g); ind = independent_metrics(pred, g)
                    for key in METRICS:
                        if m[key] is None:
                            assert ind[key] is None
                        else:
                            assert math.isclose(m[key], ind[key], abs_tol=1e-9, rel_tol=1e-8)
                            error[key] = max(error.get(key, 0), abs(m[key]-ind[key]))
                    archived = old_metrics[case, row['condition'], 0, stage]
                    old = dict(relative_l1=archived['volume']['relative_l1'], relative_l2=archived['volume']['relative_l2'],
                               dice50=archived['morphology']['dice_0.5'], absolute_peak_ratio_error=archived['morphology']['absolute_peak_ratio_error'],
                               centroid_distance_mm=archived['morphology']['centroid_distance_mm'])
                    for k in METRICS:
                        assert (m[k] is None and old[k] is None) or math.isclose(m[k], old[k], abs_tol=1e-9, rel_tol=1e-8), (case, k, m[k], old[k])
                        if m[k] is not None:
                            old_error[k] = max(old_error.get(k, 0), abs(m[k]-old[k]))
                    metrics_rows.append(dict(case_id=case, case_index=ci, layout_index=li, condition=row['condition'], draw=0,
                        intervention='baseline', stage=stage, prediction_path=row['path'], prediction_sha256=row['sha256'],
                        checkpoint_sha256=EXPECTED, **m))
            for root, row in ((root, r) for root, r in new if r['case_id'] == case):
                path = root / row['prediction']; assert sha(path) == row['sha256']
                li = row['layout_index']
                with np.load(path, allow_pickle=False) as z:
                    for stage in ('C0', 'C1'):
                        pred = z[stage].astype(np.float64); m = canonical_metrics(pred, g); ind = independent_metrics(pred, g)
                        for key in METRICS:
                            if m[key] is None:
                                assert ind[key] is None
                            else:
                                assert math.isclose(m[key], ind[key], abs_tol=1e-9, rel_tol=1e-8)
                                error[key] = max(error.get(key, 0), abs(m[key]-ind[key]))
                        meta = dict(case_id=case, case_index=ci, layout_index=li, condition=row['condition'], draw=0,
                                    intervention=row['intervention'], stage=stage)
                        metrics_rows.append(dict(meta, prediction_path=str(path), prediction_sha256=row['sha256'], checkpoint_sha256=EXPECTED, **m))
                        b = next(r for r in metrics_rows if r['case_id'] == case and r['layout_index'] == li and r['stage'] == stage and r['intervention'] == 'baseline')
                        delta = {k: None if m[k] is None or b[k] is None else m[k]-b[k] for k in METRICS}
                        paired.append(dict(meta, **{f'baseline_{k}': b[k] for k in METRICS},
                                           **{f'intervention_{k}': m[k] for k in METRICS}, **{f'delta_{k}': delta[k] for k in METRICS},
                                           delta_relative_l1_percentage_points=delta['relative_l1']*100))
                        d = differences(pred, references[li][stage])
                        for k in d:
                            assert math.isclose(d[k], row['prediction_difference'][stage][k], abs_tol=1e-12, rel_tol=1e-10)
                        sensitivity.append(dict(meta, **d))
            print(f'SCORED AND INDEPENDENTLY VERIFIED case {ci+1}/35', flush=True)
        assert not guard.denied
    assert len(metrics_rows) == 1400 and len(paired) == len(sensitivity) == 1050
    csv_write(BASE / 'per_case_metrics.csv', metrics_rows)
    csv_write(BASE / 'paired_case_metrics.csv', paired)
    csv_write(BASE / 'prediction_changes.csv', sensitivity)
    freeze(BASE / 'metric_verification.json', dict(at=now(), passed=True, new_fields=1050, baseline_fields=350,
        independent_metrics=5, metric_max_absolute_disagreement=error, archived_baseline_max_disagreement=old_error,
        historical_metrics_sha256=sha(OUT / 'evaluation/test/metrics.jsonl'), audit=guard.summary(),
        first_GT64_access=first_gt, seal_time=seal['at'], GT64_hashes=gt_hashes, GT128_read=False,
        files={str(BASE / n): sha(BASE / n) for n in ('per_case_metrics.csv', 'paired_case_metrics.csv', 'prediction_changes.csv')}))
    return metrics_rows, paired, sensitivity


def summarize(metrics_rows, paired, sensitivity):
    cases = verified(BASE / 'cases.json')
    idx = np.random.default_rng(SEED).integers(0, 35, size=(20000, 35))
    panels = []; case_rows = []
    groups = list(CONDITIONS) + ['ALL_5_LAYOUTS']
    for stage in ('C0', 'C1'):
        for metric in METRICS:
            for group in groups:
                for mode in (*MODES, 'mean_3_interventions'):
                    deltas = []; baselines = []; changed = []
                    for case in cases:
                        selected = [r for r in paired if r['case_id'] == case and r['stage'] == stage and
                                    (group == 'ALL_5_LAYOUTS' or r['condition'] == group) and
                                    (mode == 'mean_3_interventions' or r['intervention'] == mode)]
                        assert len(selected) == (5 if group == 'ALL_5_LAYOUTS' else 1) * (3 if mode == 'mean_3_interventions' else 1)
                        values = [r[f'delta_{metric}'] for r in selected]
                        # Undefined auxiliary metrics remain present, never silently removed.
                        if any(v is None for v in values):
                            delta = base = value = None
                        else:
                            by_layout = {}
                            for r in selected:
                                by_layout.setdefault(r['condition'], []).append(r)
                            delta = float(np.mean([np.mean([r[f'delta_{metric}'] for r in rr]) for rr in by_layout.values()]))
                            base = float(np.mean([np.mean([r[f'baseline_{metric}'] for r in rr]) for rr in by_layout.values()]))
                            value = float(np.mean([np.mean([r[f'intervention_{metric}'] for r in rr]) for rr in by_layout.values()]))
                        deltas.append(delta); baselines.append(base); changed.append(value)
                        case_rows.append(dict(case_id=case, stage=stage, metric=metric, condition=group, intervention=mode,
                                              delta=delta, baseline=base, changed=value,
                                              delta_percentage_points=delta*100 if metric == 'relative_l1' and delta is not None else None))
                    defined = all(d is not None for d in deltas)
                    a = np.asarray(deltas, dtype=float) if defined else None
                    ci = np.quantile(a[idx].mean(1), [.025, .975]).tolist() if defined else None
                    primary = stage == 'C1' and metric == 'relative_l1' and group == 'ALL_5_LAYOUTS' and mode == 'mean_3_interventions'
                    panels.append(dict(stage=stage, metric=metric, condition=group, intervention=mode, primary=primary,
                        exploratory=True, n_cases=35, defined_cases=sum(d is not None for d in deltas),
                        baseline_mean=float(np.mean(baselines)) if defined else None,
                        intervention_mean=float(np.mean(changed)) if defined else None,
                        mean_delta=float(a.mean()) if defined else None, ci95=ci,
                        median_delta=float(np.median(a)) if defined else None,
                        min_delta=float(a.min()) if defined else None, max_delta=float(a.max()) if defined else None,
                        increased_cases=int((a > 0).sum()) if defined else None, decreased_cases=int((a < 0).sum()) if defined else None,
                        mean_delta_percentage_points=float(a.mean()*100) if defined and metric == 'relative_l1' else None,
                        ci95_percentage_points=[v*100 for v in ci] if defined and metric == 'relative_l1' else None,
                        case_deltas=deltas, bootstrap_seed=SEED, resamples=20000))
    sens = []
    for stage in ('C0', 'C1'):
        for group in groups:
            for mode in (*MODES, 'mean_3_interventions'):
                for key in ('mean_abs', 'relative_change'):
                    values = []
                    for case in cases:
                        selected = [r[key] for r in sensitivity if r['case_id'] == case and r['stage'] == stage and
                                    (group == 'ALL_5_LAYOUTS' or r['condition'] == group) and
                                    (mode == 'mean_3_interventions' or r['intervention'] == mode)]
                        values.append(float(np.mean(selected)))
                    a = np.array(values)
                    sens.append(dict(stage=stage, condition=group, intervention=mode, metric=key,
                                     mean=float(a.mean()), median=float(np.median(a)),
                                     ci95=np.quantile(a[idx].mean(1), [.025, .975]).tolist(), case_values=values, n_cases=35))
    csv_write(BASE / 'case_aggregates.csv', case_rows)
    csv_write(BASE / 'paired_statistics.csv', [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in panels])
    freeze(BASE / 'statistics.json', dict(at=now(), primary=next(r for r in panels if r['primary']),
        panels=panels, prediction_changes=sens, n_cases=35, seed=SEED, bootstrap_resamples=20000,
        aggregation='3 interventions within case/layout, then5 layouts within case, finally35 cases', independent_confirmation=False))


def figures_and_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = verified(BASE / 'statistics.json')
    def panel(stage, group):
        return next(r for r in s['panels'] if r['stage'] == stage and r['condition'] == group and
                    r['metric'] == 'relative_l1' and r['intervention'] == 'mean_3_interventions')
    groups = [*CONDITIONS, 'ALL_5_LAYOUTS']
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, stage in ((-.12, 'C0'), (.12, 'C1')):
        pp = [panel(stage, c) for c in groups]; y = np.array([r['mean_delta']*100 for r in pp])
        bounds = np.array([r['ci95'] for r in pp])*100
        ax.errorbar(np.arange(6)+offset, y, yerr=np.maximum(np.stack((y-bounds[:, 0], bounds[:, 1]-y)), 0), fmt='o', capsize=3, label=stage)
    ax.axhline(0, color='gray', lw=1); ax.set_xticks(np.arange(6), groups, rotation=20)
    ax.set_ylabel('L1 change, percentage points (flip minus correct)'); ax.legend()
    ax.set_title('Paper MRGR epoch278; draw0; mean3 flips; paired35-case bootstrap95%CI')
    fig.savefig(BASE / 'lifting_l1_changes.png', dpi=180); fig.savefig(BASE / 'lifting_l1_changes.pdf'); plt.close(fig)
    check = verified(BASE / 'engineering_checks.json'); metric = verified(BASE / 'metric_verification.json'); ind = verified(BASE / 'independent_statistics_verification.json')
    p = s['primary']
    lines = ['# 正式MRGR：GeometryLifting空间映射利用诊断', '', f'完成时间（北京时间）：{now()}。', '',
        '**模型固定为论文正式MRGR：epoch278、344678参数，保留GCA融合；未使用UniformFusion epoch285。**',
        f'检查点SHA256：`{EXPECTED}`。通过官方`module_models_v2.model_for(config, "multiscale", device)`构造，沿正式evaluation加载流程严格核验权重、配置、轮次、协议和参数量。', '',
        '## 预登记范围与数据边界', '',
        '复用正式测试35案例。每类布局只使用历史登记draw0：Val-A-V4、spread-V4、V3、V6、clustered-V4，共175个案例/布局组合。V3/V6/聚簇历史其余draw未纳入本轮，不能把本表当作五draw均值。此选择在当前GT64访问前固定，以严格执行35×5×3=525次正式前向。',
        '历史595个预测文件全部核验哈希，所需175组C0/C1基准直接复用。工程阶段仅3个固定案例/布局各运行identity和3种反转，共12次前向；正式阶段525次前向，同时保存1050个C0/C1体积。全部新增预测封存后才读取GT64，GT128始终禁止。',
        '本轮为已使用测试集上的探索性机制诊断，不是独立测试确认。无训练、反向传播、优化器、调参、选轮或模型替换。', '',
        '## 干预隔离与工程检查', '',
        '仅在模型实例第一次GeometryLifting.forward中clone采样网格，对最后一维水平、垂直或两分量取负。原始angular_grid方法及其输出从未修改；visibility逐元素保持相同。grid_sample仍使用bilinear、zeros padding、align_corners=True，特征、通道、shape和dtype保持。',
        '第二次反馈GeometryLifting调用原始绑定forward；工程阶段另外独立grid_sample核对其输出逐元素一致。每次前向检查几何描述符、射线关系和可观测性描述符与不运行基准网络的几何参考完全相同；输入哈希、所有参数/缓冲区状态哈希保持。C0/C1均有限且在[0,1]。',
        f'工程检查通过。identity对历史基准最大绝对差：{check["max_identity_absolute"]:.9g}；最大相对平均绝对差：{check["max_identity_relative"]:.9g}。阈值分别为1e-5和1e-4。',
        '注意：第二次反馈的空间映射不变，但第一次干预改变C0，因此预测测量、残差值和最终C1允许随之改变；这是干预的下游效应。', '',
        '## 主要终点与逐布局结果', '',
        '差值为干预误差−正确映射误差，正值表示恶化。先平均同案例/布局3种反转，再在同案例内平均5种布局，最后以35案例为单位。固定seed20261022，配对bootstrap20000次，报告未校正95%探索性案例区间；三种反转及布局不作为独立样本。',
        f'预登记主要终点：跨布局C1相对L1平均变化 **{p["mean_delta"]:+.6f}（绝对比例）**，即 **{p["mean_delta"]*100:+.4f}个百分点**；95%区间[{p["ci95"][0]:+.6f}, {p["ci95"][1]:+.6f}]，即[{p["ci95"][0]*100:+.4f}, {p["ci95"][1]*100:+.4f}]个百分点。案例中位数变化{p["median_delta"]*100:+.4f}个百分点，误差增加案例{p["increased_cases"]}/35。', '',
        '| 布局（均draw0） | 阶段 | 正确L1% | 干预均值L1% | 差值比例 | 差值百分点及95%区间 | 案例中位差百分点 |',
        '|---|---|---:|---:|---:|---|---:|']
    for group in groups:
        for stage in ('C0', 'C1'):
            r = panel(stage, group)
            lines.append(f'| {group} | {stage} | {r["baseline_mean"]*100:.4f} | {r["intervention_mean"]*100:.4f} | {r["mean_delta"]:+.6f} | {r["mean_delta"]*100:+.4f} [{r["ci95"][0]*100:+.4f}, {r["ci95"][1]*100:+.4f}] | {r["median_delta"]*100:+.4f} |')
    lines += ['', '## 三种干预分别报告（探索性）', '', '| 布局 | 干预 | C1 L1差百分点 | 95%案例区间 |', '|---|---|---:|---|']
    for group in groups:
        for mode in MODES:
            r = next(r for r in s['panels'] if r['condition'] == group and r['intervention'] == mode and r['stage'] == 'C1' and r['metric'] == 'relative_l1')
            lines.append(f'| {group} | {mode} | {r["mean_delta"]*100:+.4f} | [{r["ci95"][0]*100:+.4f}, {r["ci95"][1]*100:+.4f}] |')
    lines += ['', '## 机制与辅助指标（探索性）', '', '| 布局 | C1 L2差 | Dice50差 | 峰比绝对误差差 | 质心距离差mm | 预测体平均绝对变化 | 预测体相对变化 |', '|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        vals = [next(r for r in s['panels'] if r['condition'] == group and r['intervention'] == 'mean_3_interventions' and r['stage'] == 'C1' and r['metric'] == key)['mean_delta'] for key in METRICS[1:]]
        changes = [next(r for r in s['prediction_changes'] if r['condition'] == group and r['intervention'] == 'mean_3_interventions' and r['stage'] == 'C1' and r['metric'] == key)['mean'] for key in ('mean_abs', 'relative_change')]
        lines.append('| '+group+' | '+' | '.join('未定义' if v is None else f'{v:+.6g}' for v in [*vals, *changes])+' |')
    conclusion = ('结果支持当前冻结模型依赖正确的二维到三维空间映射：在其余已检查几何通路不变时，预登记反转导致跨布局C1重建误差增加。' if p['ci95'][0] > 0 else
                  '跨布局主要终点区间跨零或方向非恶化；当前诊断不能支持正确空间映射的平均重建收益，仍应结合逐布局完整结果。')
    lines += ['', '## 解释、异常与局限', '', conclusion,
        '该结论仅针对当前冻结权重、所选draw0布局和3种确定性反转。反转使特征与仍然正确的几何描述符/物理射线不再对应，可能产生分布偏移；不能把干预差值等同为重新训练删除GeometryLifting的净收益，也不能单独证明学到了可泛化几何规律。',
        '所有案例和负变化均保留，未按结果追加干预或选择布局。正式525次前向全部完成；工程门槛、数值范围、输入/权重/掩码/其他几何通路审计均通过，无数值失败。工程及正式预测均封存；GT128访问为0。',
        f'全部1400个基准/干预体积的5项指标经独立NumPy公式核对；350个基准体积同时与历史正式指标核对。{ind["panels_checked"]}组配对统计及{ind["sensitivity_panels_checked"]}组预测变化统计经独立分组与重采样权重核验。',
        '资源调度复用仓库UUID锁、间隔15秒两次空闲检查、锁FD继承和主机内存/磁盘门槛，串行使用空闲GPU，未终止其他任务。原始资源快照、任务日志和每次前向GPU UUID均保留。', '',
        '![配对L1变化](artifacts/mrgr_lifting_mapping_v1/lifting_l1_changes.png)', '',
        '[不可变协议](artifacts/mrgr_lifting_mapping_v1/protocol.json) · [工程检查](artifacts/mrgr_lifting_mapping_v1/engineering_checks.json) · [基准清单](artifacts/mrgr_lifting_mapping_v1/baseline_manifest.json) · [逐案例指标](artifacts/mrgr_lifting_mapping_v1/per_case_metrics.csv) · [逐案例配对差值](artifacts/mrgr_lifting_mapping_v1/paired_case_metrics.csv) · [案例聚合](artifacts/mrgr_lifting_mapping_v1/case_aggregates.csv) · [完整统计](artifacts/mrgr_lifting_mapping_v1/statistics.json) · [独立统计核验](artifacts/mrgr_lifting_mapping_v1/independent_statistics_verification.json) · [验收](artifacts/mrgr_lifting_mapping_v1/acceptance.json)。', '',
        f'原始预测及日志目录：`{BASE}`。大型数组保留服务器，不覆盖历史输出。',
        '复现入口：`scripts/lifting_mapping_v1/diagnostic.py register`及`controller`。已存在目录/产物不覆盖，不自动重试失败任务。']
    assert not REPORT.exists(); REPORT.write_text('\n'.join(lines)+'\n')


def export():
    ARTIFACT.mkdir(exist_ok=False)
    exported = {}
    for path in [*BASE.glob('*.json'), *BASE.glob('*.csv'), *BASE.glob('*.png'), *BASE.glob('*.pdf')]:
        if path.name == 'status.json':
            continue
        target = ARTIFACT / path.name; shutil.copy2(path, target)
        side = path.with_suffix(path.suffix+'.sha256')
        if side.exists():
            shutil.copy2(side, target.with_suffix(target.suffix+'.sha256'))
        exported[str(target.relative_to(PROJECT))] = dict(source=str(path), sha256=sha(target))
    for label, root in [('engineering', BASE / 'engineering'), *[(f'layout{li:02d}', BASE / 'predictions' / f'layout{li:02d}') for li in range(5)]]:
        target = ARTIFACT / 'prediction_manifests' / f'{label}.jsonl.gz'; target.parent.mkdir(exist_ok=True)
        raw = (root / 'manifest.jsonl').read_bytes()
        with target.open('xb') as f:
            with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as stream:
                stream.write(raw)
        assert gzip.decompress(target.read_bytes()) == raw
        if label != 'engineering':
            shutil.copy2(root / 'complete.json', target.with_name(label+'_complete.json'))
        exported[str(target.relative_to(PROJECT))] = dict(source=str(root / 'manifest.jsonl'), source_sha256=sha(root / 'manifest.jsonl'), sha256=sha(target), transform='lossless gzip')
    resource = dict(at=now(), tasks=[read(p) for p in (BASE / 'scheduling').glob('*.json')],
                    raw_snapshots=str(BASE / 'scheduling/resources.jsonl'), external_processes_signalled=False)
    freeze(ARTIFACT / 'resource_audit.json', resource)
    freeze(ARTIFACT / 'export_manifest.json', dict(at=now(), files=exported, predictions_on_scratch=True))


def main():
    backend()
    assert not (BASE / 'acceptance.json').exists()
    data, paired, sensitivity = scoring()
    summarize(data, paired, sensitivity)
    subprocess.run([sys.executable, str(HERE / 'verify_statistics.py')], cwd=PROJECT, check=True)
    figures_and_report()
    p = verified(BASE / 'protocol.json'); verify_hashes(p['sources'])
    assert sha(method()['checkpoint']) == EXPECTED
    metric = verified(BASE / 'metric_verification.json'); ind = verified(BASE / 'independent_statistics_verification.json')
    assert metric['passed'] and ind['passed']
    files = [BASE / n for n in ('protocol.json', 'reuse_audit.json', 'engineering_checks.json', 'all_predictions_sealed.json',
                               'metric_verification.json', 'statistics.json', 'independent_statistics_verification.json',
                               'per_case_metrics.csv', 'paired_case_metrics.csv', 'case_aggregates.csv', 'prediction_changes.csv')]
    freeze(BASE / 'acceptance.json', dict(at=now(), completed=True, epoch=278, parameters=344678, checkpoint_sha256=EXPECTED,
        cases=35, layouts=5, draw=0, formal_model_forwards=525, engineering_model_forwards=12,
        new_formal_fields=1050, reused_baseline_fields=350, training_runs=0, backward_calls=0, optimizer_updates=0,
        engineering_passed=True, formal_predictions_sealed_before_GT64=True, GT128_read=False,
        finite_valid_range_all_outputs=True, other_geometry_unchanged=True, feedback_mapping_unchanged=True,
        model_state_unchanged=True, independent_metrics_verified=True, independent_statistics_verified=True,
        supplementary_used_test=True, independent_confirmation=False, primary=verified(BASE / 'statistics.json')['primary'],
        anomalies=[], evidence={str(path): sha(path) for path in [*files, REPORT]}))
    export()
    import re
    for link in re.findall(r'\]\(([^)]+)\)', REPORT.read_text()):
        assert (REPORT.parent / link).exists(), link
    print('ACCEPTED paper MRGR lifting mapping diagnostic', flush=True)


if __name__ == '__main__':
    main()
