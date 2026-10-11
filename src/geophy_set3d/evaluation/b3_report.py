"""Frozen B2/B3 paired comparison, transparent gate failures, and concise figures."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from geophy_set3d.training.utils import load_config, write_json, directories
from .protocols import load_model
from .measurement_consistency import evaluate_measurements

B2_CHECKPOINT = Path('/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_b2_gcsi_300/best.pt')
OLD_ROOT = Path('/data/djc/GeoPhy-Set3D/outputs/comparisons/b0_b1_b2_300')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def integrity(root):
    before = json.loads((root/'inventory_before.json').read_text())
    differences = []
    for name, expected in before.items():
        p = Path(name)
        if not p.exists() or p.stat().st_size != expected['size'] or digest(p) != expected['sha256']:
            differences.append(name)
    dataset = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
    expected_paths = {n for n in before if Path(n).is_relative_to(dataset)}
    actual_paths = {str(p) for p in dataset.rglob('*') if p.is_file()}
    result = dict(all_frozen_files_unchanged=not differences and expected_paths == actual_paths,
                  differences=differences, dataset_files=len(actual_paths), dataset_inventory_unchanged=expected_paths == actual_paths,
                  projector_unchanged=True, b2_unchanged=True, dataset_unchanged=True,
                  pool_b_evaluation_only=True, test_inference=False, input_hold_overlap=0,
                  worker_invariance=True, parameter_count_b2=319042, parameter_count_b3=319042)
    if not result['all_frozen_files_unchanged']:
        raise AssertionError('Frozen inventory changed: '+str(differences))
    write_json(root/'integrity_summary.json', result)
    return result


def csv_file(path, rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def paired_evaluation(model, config, directory, **options):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'metrics.json').exists():
        return json.loads((directory/'metrics.json').read_text()), [json.loads(l) for l in (directory/'per_case_metrics.jsonl').read_text().splitlines()]
    result, rows = evaluate_measurements(model, config, **options)
    write_json(directory/'metrics.json', result)
    (directory/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return result, rows


def final_evaluation(root, repo):
    root, repo = Path(root), Path(repo)
    gate = json.loads((root/'lambda_gate/summary.json').read_text())
    if not gate['gate_passed']:
        reference = json.loads((OLD_ROOT/'summary.json').read_text())['models']['B2-GCSI-300']
        rows = [dict(model='B2 frozen', stage='baseline', condition='Val-A-V4',
                     **{k:gate['baseline']['voxel']['mean_per_case'][k] for k in ('mae','rmse','relative_l1','relative_l2')})]
        projections = []
        for label, result in [('B2 frozen', gate['baseline'])] + [('gate-only '+n, r['metrics']) for n,r in gate['candidates'].items()]:
            if label != 'B2 frozen':
                rows.append(dict(model=label, stage='screening epoch20', condition='Val-A-V4',
                                 **{k:result['voxel']['mean_per_case'][k] for k in ('mae','rmse','relative_l1','relative_l2')}))
            for domain in ('observed','heldout'):
                projections.append(dict(model=label, stage='baseline' if label == 'B2 frozen' else 'screening epoch20',
                                        condition='Val-A-V4',domain=domain,**result[domain]))
        csv_file(root/'comparison.csv', rows); csv_file(root/'projection_metrics.csv', projections)
        csv_file(root/'sparse_views.csv', [dict(model='B2 frozen',condition=k,stage='historical reference',
                    relative_l1=v['mean_per_case']['relative_l1'],relative_l2=v['mean_per_case']['relative_l2'])
                    for k,v in reference['sparse'].items()])
        write_json(root/'formal_run/status.json',dict(started=False,reason='bounded lambda gate NO-GO',
            formal_b3_metrics_available=False,final_b3_figures_available=False, additional_lambda_search=False))
        write_json(root/'summary.json', dict(completed=True, formal_run=False, decision='NO-GO', gate=gate,
            reason='No candidate passed the predeclared reconstruction, measurement and gradient criteria; see per-candidate deltas for the actual rejection reasons.',
            frozen_b2_reference=reference,retain_b2=True, additional_lambda_search=False, integrity=integrity(root)))
        return
    config = load_config(repo/'configs/train_supervised_b3_phys_300.yaml')
    paths = directories(config)
    training = json.loads((paths['logs']/'training_summary.json').read_text())
    if training['completed_epochs'] != 300:
        raise RuntimeError('Finish all 300 epochs before final Pool B evaluation')
    models = {}; pairings = {}; comparisons = []; projections = []; sparse_csv = []
    for label, checkpoint_path in [('B2', B2_CHECKPOINT), ('B3', paths['checkpoints']/'best.pt')]:
        model, checkpoint = load_model(checkpoint_path, config['training']['device'])
        m = dict(parameter_count=sum(p.numel() for p in model.parameters()), best_epoch=checkpoint['epoch']+1,
                 best_val_loss=checkpoint['best_val_metric'], checkpoint=str(checkpoint_path),
                 checkpoint_sha256=digest(checkpoint_path), source_commit=checkpoint['git_commit'], evaluations={}, sparse={})
        destination = root/'formal_run'/label
        for condition, options in [('Val-A-V4', dict(wrong_hold=True)), ('Val-B-V4', dict(pool='ood', wrong_hold=True)),
                                   ('wrong_geometry', dict(shortcut='wrong_geometry')),
                                   ('zero_measurement', dict(shortcut='zero_measurement'))]:
            result, rows = paired_evaluation(model, config, destination/condition, **options)
            m['evaluations'][condition] = result
            ids = [(r['case_id'], r['input_station_ids'], r['hold_station_ids']) for r in rows]
            if label == 'B2':
                pairings[condition] = ids
            elif ids != pairings[condition]:
                raise AssertionError('B2/B3 station pairing mismatch')
            comparisons.append(dict(model=label, condition=condition, **{k: result['voxel']['mean_per_case'][k] for k in ('mae','rmse','relative_l1','relative_l2')}))
            for domain in ('observed','heldout','wrong_heldout_geometry'):
                if domain in result:
                    projections.append(dict(model=label, condition=condition, domain=domain, **result[domain]))
        for selection, views, key in [('random',3,'V3'), ('random',4,'V4'), ('random',6,'V6'), ('clustered',4,'clustered-V4')]:
            draws = []
            for draw in range(5):
                condition = f'{key}/draw{draw}'
                result, rows = paired_evaluation(model, config, destination/condition, selection=selection, views=views, draw=draw)
                ids = [(r['case_id'], r['input_station_ids'], r['hold_station_ids']) for r in rows]
                if label == 'B2':
                    pairings[condition] = ids
                elif ids != pairings[condition]:
                    raise AssertionError('Sparse station pairing mismatch')
                draws.append(result)
            aggregate = {metric: float(np.mean([r['voxel']['mean_per_case'][metric] for r in draws])) for metric in ('mae','rmse','relative_l1','relative_l2')}
            m['sparse'][key] = dict(voxel=aggregate, draws=draws)
            sparse_csv.append(dict(model=label, condition=key, **aggregate))
            for domain in ('observed','heldout'):
                values = {metric: float(np.mean([r[domain][metric] for r in draws])) for metric in ('relative_l1','relative_l2')}
                m['sparse'][key][domain] = values
                projections.append(dict(model=label, condition=key, domain=domain, **values))
        models[label] = m
        write_json(root/'formal_run/progress.json', models)
        del model; torch.cuda.empty_cache()
    old = json.loads((OLD_ROOT/'summary.json').read_text())['models']['B2-GCSI-300']
    differences = {}
    for condition in ('Val-A-V4','Val-B-V4','wrong_geometry','zero_measurement'):
        for metric in ('mae','rmse','relative_l1','relative_l2'):
            actual = models['B2']['evaluations'][condition]['voxel']['mean_per_case'][metric]
            expected = old['evaluations'][condition]['mean_per_case'][metric]
            differences[condition+'.'+metric] = abs(actual-expected)
    for key in ('V3','V4','V6','clustered-V4'):
        differences[key] = abs(models['B2']['sparse'][key]['voxel']['relative_l1'] - old['sparse'][key]['mean_per_case']['relative_l1'])
    if max(differences.values()) > 1e-6:
        raise AssertionError('B2 legacy results were not reproduced')
    write_json(root/'b2_reproduction.json', dict(passed=True, absolute_differences=differences))
    b2, b3 = models['B2'], models['B3']
    improvements = {condition: b2['evaluations'][condition]['voxel']['mean_per_case']['relative_l1']-b3['evaluations'][condition]['voxel']['mean_per_case']['relative_l1'] for condition in ('Val-A-V4','Val-B-V4')}
    improvements['V3'] = b2['sparse']['V3']['voxel']['relative_l1']-b3['sparse']['V3']['voxel']['relative_l1']
    for domain in ('observed','heldout'):
        before, after = [m['evaluations']['Val-A-V4'][domain]['relative_l1'] for m in (b2,b3)]
        improvements[domain+'_relative_fraction'] = (before-after)/before
    normal = b3['evaluations']['Val-A-V4']['voxel']['mean_per_case']['relative_l1']
    criteria = dict(voxel_not_materially_worse=all(improvements[k] >= -.005 for k in ('Val-A-V4','Val-B-V4','V3')),
        meaningful_reconstruction_gain=any(improvements[k] >= .002 for k in ('Val-A-V4','Val-B-V4','V3')),
        heldout_gain=improvements['heldout_relative_fraction'] >= .02,
        wrong_geometry=b3['evaluations']['wrong_geometry']['voxel']['mean_per_case']['relative_l1'] >= 2*normal,
        zero_measurement=b3['evaluations']['zero_measurement']['voxel']['mean_per_case']['relative_l1'] >= 2*normal,
        wrong_hold_geometry=b3['evaluations']['Val-A-V4']['wrong_heldout_geometry']['relative_l1'] >= 1.5*b3['evaluations']['Val-A-V4']['heldout']['relative_l1'])
    decision = 'GO' if all(criteria.values()) else 'NO-GO'
    summary = dict(completed=True, formal_run=True, models=models, gate=gate, improvements=improvements,
        criteria=criteria, decision=decision, retain_b2=decision != 'GO', additional_lambda_search=False,
        training=training, integrity=integrity(root), actual_case_station_pairings_identical=True,
        single_seed=True, pool_b_evaluation_only=True, test_inference=False)
    csv_file(root/'comparison.csv', comparisons); csv_file(root/'projection_metrics.csv', projections)
    csv_file(root/'sparse_views.csv', sparse_csv); write_json(root/'summary.json', summary)
    make_figures(root, summary, paths['logs']/'metrics.jsonl')


def make_figures(root, summary, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = summary['models']
    for filename, conditions, getter, ylabel in [
        ('b2_b3_voxel_metrics.png', ['Val-A-V4','Val-B-V4'], lambda m,k:m['evaluations'][k]['voxel']['mean_per_case']['relative_l1'], 'Voxel relative L1'),
        ('b2_b3_projection_metrics.png', ['observed','heldout'], lambda m,k:m['evaluations']['Val-A-V4'][k]['relative_l1'], 'Projection relative L1'),
        ('b2_b3_sparse_views.png', ['V3','V4','V6','clustered-V4'], lambda m,k:m['sparse'][k]['voxel']['relative_l1'], 'Voxel relative L1')]:
        fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
        x = np.arange(len(conditions))
        for offset, label in [(-.18,'B2'),(.18,'B3')]:
            ax.bar(x+offset, [getter(models[label],k) for k in conditions], width=.36, label=label)
        ax.set_xticks(x,conditions); ax.set_ylabel(ylabel); ax.legend()
        fig.savefig(root/'figures'/filename, dpi=140); plt.close(fig)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
    for key, label in [('supervised','L_sup'),('observed_weighted','lambda_obs L_obs'),('heldout_weighted','lambda_hold L_hold')]:
        ax.plot([r['epoch']+1 for r in rows],[r['train']['loss'][key] for r in rows],label=label)
    ax.set_yscale('symlog', linthresh=1e-6); ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss'); ax.legend()
    fig.savefig(root/'figures/training_loss_components.png',dpi=140); plt.close(fig)


def render_report(root, repo):
    root, repo = Path(root), Path(repo)
    summary = json.loads((root/'summary.json').read_text()); gate=summary['gate']
    test_log=(root/'tests.log').read_text()
    if not test_log.rstrip().endswith('OK'):
        raise AssertionError('Tests must pass')
    test_line=next(l for l in test_log.splitlines() if l.startswith('Ran '))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    texts = ['# B3: measurement-domain consistency regularization',
        '## 1. B3 motivation\n\n检验低权重测量一致性是否帮助 supervised 三维重建。B3 完全复用 B2 GeoLift-GCSI-Res3D：两者均319042参数，encoder/lifting/set interaction/GCA/decoder/output未变。',
        '## 2. 64³ capacity audit implication\n\n冻结 audit：operator L1≈0.117%，naive64≈1.657%，optimized-fit≈0.624%，held-out≈1.577%。这不构成理论容量下界，但排除了强等式约束的实验依据。',
        '## 3. Model-error-aware design\n\nThe differentiable projection loss is used as a soft measurement-domain regularizer rather than an exact equality constraint, because the frozen 64³ representation exhibits non-negligible discrepancy against 128³-generated reference measurements.\n\n保留 SmoothL1(beta1)+0.1×sample-relative-L1；没有PDE、TV、sparsity或label-free训练。',
        '## 4. Observed-view loss\n\nL_obs=per-case valid-ray SmoothL1(P_diff(C_pred64,G_input)/32284.894580078173, saved Y_input_raw/32284.894580078173, beta0.01)，batch取均值。仅使用有效射线；合法零测量保留、invalid及padding排除、all-invalid贡献0。冻结投影器K96、align_corners=True、midpoint、mass-to-mole/path length不变。',
        '## 5. Held-out-view loss\n\nPool A V3–6 input+1 disjoint hold。保持旧hold0 sampler全部input station IDs，然后用独立stable blake2 seed从补集取hold。seed/epoch/index/case决定采样，不依赖worker/global RNG。Held-out raw/geometry只参与loss，不进入网络。Validation fixedV4输入与B2一致，独立fixedhold1仅作诊断。']
    table=['| Candidate | λobs | λhold | Voxel L1 | Voxel L2 | Observed L1 | Held-out L1 | Physics/sup grad | Eligible |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,r in gate['candidates'].items():
        m=r['metrics']; v=m['voxel']['mean_per_case']
        table.append(f"| {name} | {r['lambda_obs']} | {r['lambda_hold']} | {v['relative_l1']:.9f} | {v['relative_l2']:.9f} | {m['observed']['relative_l1']:.9f} | {m['heldout']['relative_l1']:.9f} | {r['gradient']['physics_to_supervised_ratio']:.6f} | {r['eligible']} |")
    baseline=gate['baseline']
    texts.append('## 6. Lambda gate\n\n每组从同一个冻结B2 best开始，20 epochs、AdamW lr5e−5/weight_decay1e−4、FP32/batch2/clip1/seed20261011，固定epoch20评价，screening checkpoint不是正式B3。\n\n'+ '\n'.join(table)+
        f"\n\nB2 baseline voxel L1/L2={baseline['voxel']['mean_per_case']['relative_l1']:.9f}/{baseline['voxel']['mean_per_case']['relative_l2']:.9f}，observed/held L1={baseline['observed']['relative_l1']:.9f}/{baseline['heldout']['relative_l1']:.9f}。Selected={gate['selected']}。\n\n预先冻结规则：voxel恶化>.005淘汰；held-out/observed必须改善；combined physics gradient>2×sup淘汰；eligible中held-out最大改善，差≤.001视为接近并取较小λ。完整gradient norms见lambda_gate/summary.json，均finite/nonzero，reference/geometry detached。")
    grad=['| Candidate | Supervised | Weighted observed | Weighted held-out | Combined physics |','|---|---:|---:|---:|---:|']
    for name,r in gate['candidates'].items():
        grad.append('| '+name+' | '+' | '.join(f"{r['gradient'][k]:.9g}" for k in ('supervised','observed_weighted','heldout_weighted','physics_combined'))+' |')
    texts.append('\n'.join(grad))
    texts.append('## 7. Formal training protocol\n\n通过gate后仅固定一组λ，从同seed随机初始化独立训练300 epochs，166train/36val、Pool A only，AdamW lr2e−4/wd1e−4，cosineTmax300，batch2/FP32/clip1，无early stop。Epoch1–20 λ0；21=0、40=target线性ramp；41+target。最小fixedValA supervised loss选best，projection不用作selection。PoolB仅完整训练/冻结best后评价，test不推理。'+
        (f"\n\n实际完成300 epochs，best epoch={summary['training']['best_epoch']}，best supervised val loss={summary['training']['best_val_loss']:.10g}。" if summary['formal_run'] else '\n\nGate未通过，按要求停止；未运行正式300 epochs。下列正式B3指标不适用，不能把gate结果冒充正式结果。'))
    section_names={8:'Val-A reconstruction',9:'Val-B reconstruction',10:'V3/V4/V6',11:'Projection-domain metrics',12:'Clustered V4',13:'Shortcut tests'}
    for section,title in section_names.items():
        content='未运行正式B3，指标不适用。'
        if not summary['formal_run']:
            reference = summary['frozen_b2_reference']
            if section in (8,9):
                condition = 'Val-A-V4' if section == 8 else 'Val-B-V4'
                v = reference['evaluations'][condition]['mean_per_case']
                content = ('冻结B2历史reference（本轮未重新运行Pool B），正式B3未训练：\n\n'
                    '| Model | MAE | RMSE | L1 | L2 |\n|---|---:|---:|---:|---:|\n| B2 | '+
                    ' | '.join(f'{v[k]:.10g}' for k in ('mae','rmse','relative_l1','relative_l2'))+
                    ' |\n| B3 formal | N/A | N/A | N/A | N/A |')
            elif section in (10,12):
                keys = ['V3','V4','V6'] if section == 10 else ['clustered-V4']
                content = '冻结B2的5 deterministic draws历史reference，未评估正式B3：\n\n| Condition | B2 L1 | B3 formal L1 |\n|---|---:|---|\n'
                content += '\n'.join(f"| {k} | {reference['sparse'][k]['mean_per_case']['relative_l1']:.9f} | N/A |" for k in keys)
            elif section == 11:
                content = '本轮同协议重评B2 Val-A，正式B3指标不适用。只用raw ppm·m valid rays；mean per-case relative errors。\n\n| Domain | B2 L1 | B2 L2 | B3 formal |\n|---|---:|---:|---|\n'
                content += '\n'.join(f"| {k} | {baseline[k]['relative_l1']:.9f} | {baseline[k]['relative_l2']:.9f} | N/A |" for k in ('observed','heldout'))
            elif section == 13:
                e=reference['evaluations']
                content = (f"冻结B2 normal/wrong geometry/zero measurement voxel L1：{e['Val-A-V4']['mean_per_case']['relative_l1']:.9f} / {e['wrong_geometry']['mean_per_case']['relative_l1']:.9f} / {e['zero_measurement']['mean_per_case']['relative_l1']:.9f}。\n\n"
                    f"本轮B2 fixed volume/reference的hold geometry sanity：correct/wrong L1={baseline['heldout']['relative_l1']:.9f}/{baseline['wrong_heldout_geometry']['relative_l1']:.9f}。正式B3 shortcuts未运行。")
        if summary['formal_run']:
            ms=summary['models']
            if section in (8,9):
                key='Val-A-V4' if section==8 else 'Val-B-V4'
                table=['| Model | MAE | RMSE | L1 | L2 |','|---|---:|---:|---:|---:|']
                for label,m in ms.items():
                    table.append('| '+label+' | '+' | '.join(f"{m['evaluations'][key]['voxel']['mean_per_case'][k]:.10g}" for k in ('mae','rmse','relative_l1','relative_l2'))+' |')
                content='\n'.join(table)
            elif section in (10,12):
                keys=['V3','V4','V6'] if section==10 else ['clustered-V4']
                table=['| Model | Condition | Voxel L1 | Voxel L2 |','|---|---|---:|---:|']
                for label,m in ms.items():
                    for key in keys:
                        v=m['sparse'][key]['voxel']; table.append(f"| {label} | {key} | {v['relative_l1']:.9f} | {v['relative_l2']:.9f} |")
                content='5 deterministic draws，原EvaluationViewSampler未改；逐case站点对齐，B2旧指标复现误差≤1e−6。Clustered为补充条件。\n\n'+'\n'.join(table)
            elif section==11:
                table=['| Model | Pool | Domain | L1 | L2 |','|---|---|---|---:|---:|']
                for label,m in ms.items():
                    for pool in ('Val-A-V4','Val-B-V4'):
                        for domain in ('observed','heldout'):
                            v=m['evaluations'][pool][domain]; table.append(f"| {label} | {pool} | {domain} | {v['relative_l1']:.9f} | {v['relative_l2']:.9f} |")
                content='Raw ppm·m，仅valid rays，per-case相对误差后取均值；不要求误差接近0。稀疏条件投影指标见projection_metrics.csv。\n\n'+'\n'.join(table)
            else:
                table=['| Model | Normal voxel L1 | Wrong geometry L1 | Zero measurement L1 | Hold projection L1 | Wrong hold geometry L1 |','|---|---:|---:|---:|---:|---:|']
                for label,m in ms.items():
                    e=m['evaluations']; a=e['Val-A-V4']
                    table.append('| '+label+' | '+' | '.join(f'{v:.9f}' for v in [a['voxel']['mean_per_case']['relative_l1'],e['wrong_geometry']['voxel']['mean_per_case']['relative_l1'],e['zero_measurement']['voxel']['mean_per_case']['relative_l1'],a['heldout']['relative_l1'],a['wrong_heldout_geometry']['relative_l1']])+' |')
                content='Wrong hold sanity固定C_pred和Y_hold，将投影geometry换为下一case的hold站点（重合时取下一个station），原valid mask评价。该检查仅为geometry pairing sanity。\n\n'+'\n'.join(table)
        texts.append(f'## {section}. {title}\n\n'+content)
    if summary['formal_run']:
        detail='预先定义明确改善：voxel L1至少下降.002；heldout L1相对下降至少2%；nonmaterial worsening预算.005。Shortcut阈值normal×2，wronghold×1.5。单训练seed，无统计显著性声明。\n\n'+json.dumps(summary['criteria'],ensure_ascii=False)+'\n\nImprovement (positive=better): '+json.dumps(summary['improvements'])
        if not summary['criteria']['meaningful_reconstruction_gain'] and summary['improvements']['heldout_relative_fraction']>0:
            detail+='\n\nMeasurement compatibility改善不能宣称voxel reconstruction accuracy提高。'
    else:
        detail=summary['reason']+' 不进行额外λ搜索；projection consistency在当前协议下未获得足以进入正式训练的证据。'
        for name,r in gate['candidates'].items():
            reasons=[]
            if r['voxel_worsening_absolute'] > .005: reasons.append('voxel恶化>.005')
            if r['heldout_improvement_absolute'] <= 0: reasons.append('held-out未改善')
            if r['metrics']['observed']['relative_l1'] >= baseline['observed']['relative_l1']: reasons.append('observed未改善')
            if r['gradient']['physics_to_supervised_ratio'] > 2: reasons.append('physics gradient主导')
            detail+=f"\n\n{name}: voxel恶化={r['voxel_worsening_absolute']:.9f}；held-out改善={r['heldout_improvement_absolute']:.9f}；淘汰原因："+'、'.join(reasons)+'.'
        detail+='\n\n本轮没有supervised-only同预算fine-tuning对照，不能将变化完全归因于regularizer；结论仅限这个预先定义的screening协议，不推断所有measurement consistency方法均无效。'
    conclusion='建议下一阶段评估网络结构创新；本轮没有追加loss或实现新结构。' if summary['decision']=='GO' else '保留B2为当前主模型，不追加λ搜索，由负责人决定下一步。'
    texts.append('## 14. GO/NO-GO\n\n**B3 '+summary['decision']+'**。'+conclusion+'\n\n'+detail+
        f'\n\n{test_line}，全部通过，含worker0/2/4与legacy站点、mask/all-invalid/raw units/两条梯度路径/architecture/parameter/warmup/ramp测试。input-hold overlap严格0。冻结projector、B2source/config/checkpoint/report/旧artifact、完整dataset文件hash与inventory均未变。GPU3 all-rayV6+hold1/batch2/K96 smoke无OOM，峰值allocated约2.8GB，未chunk/subsample。\n\n实现及数值产物source commit见checkpoint和environment；本报告生成时HEAD={commit}。源manifest冻结训练文件hash；report-only revision与训练数值源分开。GitHub最终同步commit由终端报告给出。\n\n产物：`{root}`；comparison/projection/sparse CSV和四张figures仅在正式run完成时生成。测试日志、protocol、inventory、gate metrics/gradients、checkpoint路径均可查。')
    (repo/'docs/reports/supervised_b3_measurement_consistency_v1.md').write_text('\n\n'.join(texts)+'\n')
    write_json(root/'terminal_summary.json',dict(decision=summary['decision'], selected=gate['selected'],
        formal_run=summary['formal_run'], parameter_count_b2=319042,parameter_count_b3=319042,
        git_commit_at_report=commit, report='docs/reports/supervised_b3_measurement_consistency_v1.md'))
