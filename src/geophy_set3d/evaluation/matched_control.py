"""Paired comparison against one matched zero-weight continuation, without reruns."""
import copy
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from geophy_set3d.training.matched_control import ROOT, B2_CHECKPOINT, CANDIDATES, digest
from geophy_set3d.training.utils import load_config, write_json
from .protocols import load_model
from .measurement_consistency import evaluate_measurements

RULES = dict(voxel_worsening_absolute_max=.001, observed_worsening_absolute_max=.001,
             meaningful_heldout_improvement_relative_min=.01, close_heldout_absolute=.001,
             tie_priority=list(CANDIDATES), case_equality_tolerance=1e-12,
             final_checkpoint_epoch=20, formal_training_authorized=False,
             pool_b_evaluation_authorized=False, test_inference_authorized=False)


def prepare(repo):
    repo=Path(repo); output=ROOT/'lambda_gate/control'; output.mkdir(parents=True,exist_ok=True)
    target=output/'inventory_before.json'
    if target.exists():
        if yaml.safe_load((output/'effective_protocol.yaml').read_text()) != RULES:
            raise AssertionError('Control decision thresholds changed')
        return
    # Retain every existing dataset/B2 artifact already inventoried by the original experiment.
    paths=[Path(p) for p in json.loads((ROOT/'inventory_before.json').read_text())]
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','c4e1ef4'],cwd=repo,text=True).splitlines()
    report=repo/'docs/reports/supervised_b3_measurement_consistency_v1.md'
    paths += [repo/n for n in names if repo/n != report]
    oldoutput=Path('/data/djc/GeoPhy-Set3D/outputs')
    for name in CANDIDATES:
        for d in [ROOT/'lambda_gate'/name, oldoutput/'logs'/f'b3_lambda_gate_{name}',
                  oldoutput/'checkpoints'/f'b3_lambda_gate_{name}', oldoutput/'figures'/f'b3_lambda_gate_{name}']:
            paths += [p for p in d.rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'lambda_gate/b2_baseline').rglob('*') if p.is_file()]
    paths += [ROOT/'lambda_gate/progress.json',ROOT/'lambda_gate/stdout.log']
    write_json(target,{str(p):dict(sha256=digest(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))})
    (output/'original_report.md').write_bytes(report.read_bytes())
    (output/'original_gate_summary.json').write_bytes((ROOT/'lambda_gate/summary.json').read_bytes())
    (output/'effective_protocol.yaml').write_text(yaml.safe_dump(RULES,sort_keys=False))


def metric_values(result):
    v=result['voxel']['mean_per_case']
    return dict(voxel_l1=v['relative_l1'],voxel_l2=v['relative_l2'],mae=v['mae'],rmse=v['rmse'],
                observed_l1=result['observed']['relative_l1'],observed_l2=result['observed']['relative_l2'],
                heldout_l1=result['heldout']['relative_l1'],heldout_l2=result['heldout']['relative_l2'])


def pair_rows(candidate_rows, control_rows):
    by_case={r['case_id']:r for r in control_rows}
    if len(by_case) != len(control_rows) or len(candidate_rows) != len(control_rows):
        raise AssertionError('Repeated/missing validation cases')
    if {r['case_id'] for r in candidate_rows} != set(by_case):
        raise AssertionError('Validation case IDs differ')
    paired=[]
    for r in candidate_rows:
        c=by_case[r['case_id']]
        for key in ('input_station_ids','hold_station_ids','pool','draw'):
            if r[key] != c[key]:
                raise AssertionError('Case/station/geometry protocol mismatch: '+key)
        for domain in ('observed','heldout'):
            if r[domain]['valid_rays'] != c[domain]['valid_rays']:
                raise AssertionError('Valid ray masks differ')
        paired.append(dict(case_id=r['case_id'],input_station_ids=r['input_station_ids'],hold_station_ids=r['hold_station_ids'],
            voxel_l1_candidate=r['relative_l1'],voxel_l1_control=c['relative_l1'],
            voxel_l1_delta=r['relative_l1']-c['relative_l1'],
            heldout_l1_candidate=r['heldout']['relative_l1'],heldout_l1_control=c['heldout']['relative_l1'],
            heldout_l1_delta=r['heldout']['relative_l1']-c['heldout']['relative_l1']))
    return paired


def paired_statistics(rows,key):
    delta=np.asarray([r[key+'_delta'] for r in rows]); eps=RULES['case_equality_tolerance']
    return dict(case_count=len(rows),improved=int((delta < -eps).sum()),worsened=int((delta > eps).sum()),
                unchanged=int((np.abs(delta) <= eps).sum()),mean_paired_change=float(delta.mean()),
                median_paired_change=float(np.median(delta)))


def choose_candidate(comparisons):
    eligible=[name for name in CANDIDATES if comparisons[name]['eligible']]
    if not eligible:
        return None
    largest=max(comparisons[n]['heldout_improvement_absolute'] for n in eligible)
    return next(n for n in eligible if largest-comparisons[n]['heldout_improvement_absolute'] <= RULES['close_heldout_absolute'])


def verify_integrity(output,repo):
    before=json.loads((output/'inventory_before.json').read_text())
    changed=[n for n,v in before.items() if not Path(n).is_file() or digest(n)!=v['sha256']]
    dataset=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
    expected={n for n in before if Path(n).is_relative_to(dataset)}
    actual={str(p) for p in dataset.rglob('*') if p.is_file()}
    original=json.loads((output/'original_gate_summary.json').read_text())
    updated=json.loads((ROOT/'lambda_gate/summary.json').read_text())
    preserved=all(updated.get(k)==v for k,v in original.items())
    old_report=(output/'original_report.md').read_text()
    report=(Path(repo)/'docs/reports/supervised_b3_measurement_consistency_v1.md').read_text()
    result=dict(passed=not changed and actual==expected and preserved and report.startswith(old_report),
        changed_files=changed,dataset_files=len(actual),dataset_unchanged=actual==expected and not any(Path(n).is_relative_to(dataset) for n in changed),
        b2_unchanged=not changed,projector_unchanged=not changed,original_candidates_unchanged=not changed,
        original_summary_fields_preserved=preserved,original_report_body_preserved=report.startswith(old_report),
        candidates_retrained=False,formal_training_started=False,pool_b_used=False,test_used=False)
    write_json(output/'integrity_summary.json',result)
    if not result['passed']:
        raise AssertionError('Original artifacts/reference changed: '+str(changed))
    return result


def evaluate_control(repo):
    repo=Path(repo); output=ROOT/'lambda_gate/control'
    config=load_config(repo/'configs/train_b3_gate_control.yaml')
    checkpoint=output/'checkpoint/last.pt'
    model,ck=load_model(checkpoint,config['training']['device'])
    if ck['epoch'] != 19 or ck['config'] != config or ck['initial_checkpoint_sha256'] != digest(B2_CHECKPOINT):
        raise AssertionError('Evaluate only the matched final20 checkpoint')
    result,rows=evaluate_measurements(model,config)
    write_json(output/'metrics.json',result)
    (output/'per_case_metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    original=json.loads((output/'original_gate_summary.json').read_text())
    control=metric_values(result); comparisons={}; means=[]; paired_all=[]
    means.append(dict(model='Frozen B2',**metric_values(original['baseline'])))
    means.append(dict(model='FT-control lambda0',**control))
    for name in CANDIDATES:
        r=original['candidates'][name]; candidate=metric_values(r['metrics'])
        delta={k:candidate[k]-control[k] for k in control}
        improvement={k:(control[k]-candidate[k])/control[k] for k in control}
        candidate_rows=[json.loads(l) for l in (ROOT/'lambda_gate'/name/'per_case_metrics.jsonl').read_text().splitlines()]
        paired=pair_rows(candidate_rows,rows)
        stats={key:paired_statistics(paired,key) for key in ('voxel_l1','heldout_l1')}
        for key in stats:
            if abs(stats[key]['mean_paired_change']-delta[key]) > 1e-10:
                raise AssertionError('Mean and case-paired deltas disagree')
        reasons=[]
        if delta['voxel_l1'] > RULES['voxel_worsening_absolute_max']: reasons.append('voxel worsening exceeds .001')
        if delta['observed_l1'] > RULES['observed_worsening_absolute_max']: reasons.append('observed worsening exceeds .001')
        if improvement['heldout_l1'] < RULES['meaningful_heldout_improvement_relative_min']: reasons.append('held-out improvement below predeclared 1%')
        comparisons[name]=dict(delta_candidate_minus_control=delta,relative_improvement=improvement,
            casewise=stats,eligible=not reasons,rejection_reasons=reasons,
            heldout_improvement_absolute=control['heldout_l1']-candidate['heldout_l1'],
            lambda_obs=r['lambda_obs'],lambda_hold=r['lambda_hold'])
        means.append(dict(model=name,**candidate)); paired_all.extend([dict(candidate=name,**p) for p in paired])
    selected=choose_candidate(comparisons)
    no_hold_gain=all(r['heldout_improvement_absolute'] <= 0 for r in comparisons.values())
    reason=('current soft measurement-consistency regularization does not improve the matched fine-tuning control'
            if no_hold_gain else 'no candidate meets the predeclared measurement gain and voxel preservation criteria')
    decision='GO' if selected else 'definitive NO-GO'
    # Retain every original field/value, including selected=None and gate_passed=False.
    updated=copy.deepcopy(original)
    updated.update(original_decision='NO-GO',original_selected=original['selected'],
        control=dict(metrics=result,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                     evaluated_epoch=20,parameter_count=319042),candidate_vs_control=comparisons,
        revised_decision=decision,revised_selected=selected,revised_reason='positive evidence versus matched control' if selected else reason,
        control_decision_protocol=RULES,formal_training_started=False,ready_for_formal_training=selected is not None)
    write_json(ROOT/'lambda_gate/summary.json',updated)
    for filename,data in [('five_model_comparison.csv',means),('per_case_paired.csv',paired_all)]:
        with (output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    flat=[dict(candidate=name,**r['delta_candidate_minus_control'],
               **{'relative_improvement_'+k:v for k,v in r['relative_improvement'].items()},eligible=r['eligible']) for name,r in comparisons.items()]
    with (output/'candidate_vs_control.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write_json(output/'comparison_summary.json',dict(models=means,comparisons=comparisons,
        revised_decision=decision,revised_selected=selected,reason=updated['revised_reason'],rules=RULES,
        control_vs_frozen_b2={k:control[k]-metric_values(original['baseline'])[k] for k in control},
        lowest_heldout_candidate=min(CANDIDATES,key=lambda n:original['candidates'][n]['metrics']['heldout']['relative_l1']),
        formal_training_started=False))
    print(json.dumps(dict(revised_decision=decision,revised_selected=selected,control=control,comparisons=comparisons)),flush=True)


def append_report(repo):
    repo=Path(repo); output=ROOT/'lambda_gate/control'
    s=json.loads((ROOT/'lambda_gate/summary.json').read_text())
    if 'revised_decision' not in s:
        raise RuntimeError('Evaluate the final control checkpoint first')
    tests=(output/'tests.log').read_text()
    if not tests.rstrip().endswith('OK'):
        raise AssertionError('All tests must pass')
    test_line=next(l for l in tests.splitlines() if l.startswith('Ran '))
    comparison=json.loads((output/'comparison_summary.json').read_text())
    text=['## 15. Matched supervised-only fine-tuning control',
        'The original gate compared fine-tuned physics candidates against the unfine-tuned frozen B2 checkpoint. A matched λ=0 continuation control was therefore added to separate continued-training drift from the effect of measurement-consistency regularization.',
        '前14章节保留原gate的历史结果与NO-GO记录。本节给出补充control后的最新判断。没有重训weak/medium/strong，没有新λ/beta/loss/architecture实验。',
        'Control从同一个冻结B2 best启动，严格匹配166train、Pool A、V3–6 input+hold1、seed20261011、20 epochs、AdamW lr5e−5/wd1e−4、constant scheduler、FP32/batch2/clip1。仅λobs=λhold=0；训练不调用observed/heldout projection loss或其backprop。保持相同模型319042参数。评价唯一的epoch20 final checkpoint，未从20 epochs选best。',
        '自动重放全部166×20个case/index/epoch的冻结sampler，与三组candidate核对input与hold station IDs；三组共9960次比较，全部一致、重叠0。另核对36个Val-A fixedV4+hold1逐case station IDs和valid ray counts，全部一致。此处重放证明采样协议一致；旧candidate没有保存全部实际batch轨迹。',
        '仅评价Val-A/Pool A；raw ppm·m valid rays，固定K96与同一projection metrics实现。Pool B/test均未使用。']
    table=['| Model | Voxel L1 | Voxel L2 | MAE | RMSE | Obs L1 | Obs L2 | Hold L1 | Hold L2 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison['models']:
        table.append('| '+r['model']+' | '+' | '.join(f'{r[k]:.10g}' for k in ('voxel_l1','voxel_l2','mae','rmse','observed_l1','observed_l2','heldout_l1','heldout_l2'))+' |')
    text.append('\n'.join(table))
    text.append('Candidate vs frozen B2表示20 epoch continuation后的整体变化，包含continued-training与regularization共同作用；candidate vs λ=0 control表示这个匹配实验中regularization的增量影响。两者均保留，旧candidate-vs-B2结果见第6/14节。单seed、CUDA grid_sample backward非逐bit确定性；不做复杂显著性检验或普适因果结论。')
    drift=comparison['control_vs_frozen_b2']
    text.append('Control − frozen B2：'+', '.join(f'{k}={drift[k]:+.9f}' for k in ('voxel_l1','voxel_l2','observed_l1','heldout_l1'))+'。')
    table=['| Candidate | Δvoxel L1 | Δvoxel L2 | Δobs L1 | Δhold L1 | Voxel L1 improvement | Obs L1 improvement | Hold L1 improvement | Eligible |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,r in s['candidate_vs_control'].items():
        d=r['delta_candidate_minus_control']; imp=r['relative_improvement']
        table.append('| '+name+' | '+' | '.join(f'{d[k]:+.9f}' for k in ('voxel_l1','voxel_l2','observed_l1','heldout_l1'))+' | '+' | '.join(f'{100*imp[k]:+.4f}%' for k in ('voxel_l1','observed_l1','heldout_l1'))+f" | {r['eligible']} |")
    text.append('Δ=candidate−control，负值为candidate更好；relative improvement=(control−candidate)/control，正值为candidate更好。完整8指标delta与relative improvement见control/candidate_vs_control.csv。\n\n'+'\n'.join(table))
    table=['| Candidate | Domain | Improved cases | Worsened cases | Equal cases | Mean paired change | Median paired change |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,r in s['candidate_vs_control'].items():
        for domain,p in r['casewise'].items():
            table.append(f"| {name} | {domain} | {p['improved']} | {p['worsened']} | {p['unchanged']} | {p['mean_paired_change']:+.9f} | {p['median_paired_change']:+.9f} |")
    text.append('36个case逐对比较，change=candidate−control；abs(change)≤1e−12归为相等，无显著性检验。\n\n'+'\n'.join(table))
    text.append('评估前冻结规则：voxel L1 worsening≤.001，observed L1 worsening≤.001，held-out明确改善定义为相对下降≥1%；多个eligible优先held-out改善最大，差≤.001则weak>medium>strong较小λ优先。')
    for name,r in s['candidate_vs_control'].items():
        text.append(name+': '+('eligible' if r['eligible'] else '; '.join(r['rejection_reasons'])))
    decision=s['revised_decision']; selected=s['revised_selected']
    if selected:
        r=s['candidate_vs_control'][selected]
        conclusion=f"**Revised B3 gate: GO**。Selected={selected}，λobs={r['lambda_obs']}、λhold={r['lambda_hold']}。Ready for formal random-init300 epoch training；本轮未启动，等待负责人决定。Formal protocol沿用第7节，不修改既有formal config或自动开启训练。"
    else:
        conclusion='**Revised B3 gate: definitive NO-GO**。'+s['revised_reason']+'。B2 remains the current main model。不追加λ搜索，不正式训练B3，不修改projector或增加projection loss。下一阶段建议network architecture innovation；本轮未实现。'
    text.append(conclusion)
    text.append(f"{test_line}，全部通过。数据/B2/projector/旧weak-medium-strong source-config-checkpoint-metrics-artifacts冻结hash均未变；原summary全部字段值与原报告正文保留。控制实验产物：`{output}`，包括checkpoint、training log、effective config、metrics、per-case paired CSV、sampling与integrity验证。Control源码commit见environment/checkpoint；最终GitHub同步commit由终端给出。")
    report=repo/'docs/reports/supervised_b3_measurement_consistency_v1.md'
    report.write_text((output/'original_report.md').read_text()+'\n\n'+'\n\n'.join(text)+'\n')
    verification=verify_integrity(output,repo)
    write_json(output/'terminal_summary.json',dict(revised_decision=decision,revised_selected=selected,
        original_decision=s['original_decision'], original_selected=s['original_selected'],
        control_epoch=20,parameter_count=319042,sampling_identical=True,integrity=verification,tests_passed=True,
        candidates_retrained=False,formal_training_started=False,pool_b_used=False,test_used=False))
