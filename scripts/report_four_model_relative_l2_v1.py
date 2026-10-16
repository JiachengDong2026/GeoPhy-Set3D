"""Aggregate frozen per-case 3D L2 metrics without inference or label reads."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1'
OUT = ROOT / 'outputs/research/four_model_relative_l2_v1'


def main():
    evidence = {}

    def read(path, lines=False):
        path = Path(path)
        raw = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(x) for x in raw.splitlines() if x.strip()] if lines else json.loads(raw)

    registration = read(EVAL / 'registration.json')
    cases = registration['cases']
    values = {}
    per_case = []

    def put(model, condition, draw, case, metric):
        key = model, condition, draw, case
        assert case in cases and key not in values
        l2 = metric['relative_l2']
        assert math.isfinite(l2) and l2 >= 0
        if 'nmse' in metric:
            assert math.isclose(l2*l2, metric['nmse'], rel_tol=1e-10, abs_tol=1e-12)
        values[key] = l2

    for row in read(EVAL / 'per_case_metrics.jsonl', True):
        if row['stage'] == 'C1':
            put('Multiscale', row['condition'], row['draw'], row['case_id'], row['volume'])
    for setting in registration['settings']:
        for model, path in setting['references'].items():
            if model not in ('History', 'A', 'B'):
                continue
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == registration['evidence_sha256'][path]
            data = read(path, True)
            assert len(data) == 36 and {x['case_id'] for x in data} == set(cases)
            for row in data:
                assert row['selected_station_ids'] == setting['input_ids'][row['case_id']]
                put(model, setting['condition'], setting['draw'], row['case_id'], row)
    extra = ROOT / 'outputs/research/ab_remaining_draws_v1/per_case_metrics.jsonl'
    assert hashlib.sha256(extra.read_bytes()).hexdigest() == registration['evidence_sha256'][str(extra)]
    for row in read(extra, True):
        setting = next(s for s in registration['settings'] if s['condition'] == row['condition'] and s['draw'] == row['draw'])
        assert row['input_ids'] == setting['input_ids'][row['case_id']]
        put(row['model'], row['condition'], row['draw'], row['case_id'], row['volume'])
    assert len(values) == 4*17*36
    panels = []
    for old in read(EVAL / 'summary.json')['panels']:
        condition, draws = old['condition'], old['draws']
        panel = dict(condition=condition, draws=draws, models={})
        for model in ('History', 'A', 'B', 'Multiscale'):
            a = [sum(values[model, condition, d, case] for d in draws)/len(draws) for case in cases]
            for case, value in zip(cases, a):
                per_case.append(dict(condition=condition, draws=draws, case_id=case, model=model, relative_l2=value))
            panel['models'][model] = sum(a)/len(a)
        h = panel['models']['History']
        panel['relative_error_reduction_vs_history'] = {m: 1-v/h for m, v in panel['models'].items()}
        panels.append(panel)
    OUT.mkdir(exist_ok=False, parents=True)
    result = dict(panels=panels, cases=36, training_seeds=1,
        metric='mean_case(mean_draw(sqrt(sum_xyz((prediction-GT)^2)/sum_xyz(GT^2))))',
        not_pooled=True, not_sqrt_of_mean_nmse=True, model_stage='C1 for History and Multiscale; final output for A/B',
        evidence_sha256=evidence, scalar_rows=2448, new_inference=False, GT_arrays_loaded=False,
        final_test_used=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'per_case_panel_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in per_case))
    section = ['### 5.3 四模型三维相对 L2（后续补充）', '',
        '由已封存的逐案例指标重新汇总，无需重新推理或读取GT数组。历史与多尺度使用最终C1，A/B使用最终输出；相同36个验证案例、站位和64³节点。', '',
        '$$', r'E_{L2}=\frac{\|C_{pred}-C_{GT}\|_2}{\|C_{GT}\|_2}=\sqrt{\frac{\sum_i(C_{pred,i}-C_{GT,i})^2}{\sum_iC_{GT,i}^2}}.', '$$', '',
        '先逐例、逐布局计算相对L2，五布局时先在案例内平均，再对36例等权平均。不是将所有案例拼接后计算的pooled L2，也不是平均NMSE再开根号。', '',
        '| 条件 | 布局数 | 历史MRGR | 方案A | 方案B | 多尺度MRGR |', '|---|---:|---:|---:|---:|---:|']
    for p in panels:
        section.append('| '+p['condition']+' | '+str(len(p['draws']))+' | '+' | '.join(f"{p['models'][m]:.4%}" for m in ('History','A','B','Multiscale'))+' |')
    section += ['', 'V3/V6/聚簇的布局数1均为draw0。相对L2对大幅度局部误差更敏感，应与L1、峰位和形态指标联合解释。以上仍是单种子验证集开发结果。', '',
        '[可复核汇总](../../outputs/research/four_model_relative_l2_v1/summary.json)、[逐案例面板值](../../outputs/research/four_model_relative_l2_v1/per_case_panel_metrics.jsonl)、[汇总脚本](../../scripts/report_four_model_relative_l2_v1.py)。', '']
    text = '\n'.join(section)
    report = ROOT / 'docs/reports/three_model_comprehensive_physics_analysis_zh.md'
    content = report.read_text()
    assert '### 5.3 四模型三维相对 L2' not in content
    marker = '## 6. 从重建图像、峰和质心看问题'
    assert content.count(marker) == 1
    report.write_text(content.replace(marker, text+'\n'+marker))
    (OUT / 'report_section.md').write_text(text)
    print(json.dumps(panels, indent=2))


if __name__ == '__main__': main()
