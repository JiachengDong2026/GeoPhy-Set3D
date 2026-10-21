"""Reporting-only supplement; frozen protocol and numerical pipeline unchanged.

Registered separately when the geometry audit established zero actual historical
mask contamination. The original generated report/evidence remain immutable.
"""
from runtime import *

FINAL_REPORT = PROJECT / 'docs/reports/mrgr_oari_mechanism_v2_final_zh.md'


def register():
    path = BASE / 'report_supplement_registration.json'
    identity = dict(at=now(), parent_protocol_sha256=sha(BASE / 'protocol.json'),
                    source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                    scope='report wording and supplementary case-level descriptive tables only; original numerical pipeline/source identities unchanged',
                    reason='CPU geometry audit shows zero actual contamination in all registered P1 permutations of selected five draw0 layouts; avoid generic wording implying observed contamination',
                    no_model_forward=True, no_new_intervention=True, no_GT=True)
    if not path.exists():
        freeze(path, identity)
    else:
        old = verified(path)
        assert old['source_sha256'] == sha(__file__) and old['parent_protocol_sha256'] == sha(BASE / 'protocol.json')


def run():
    validate()
    register()
    original = verified(BASE / 'acceptance.json')
    verify_hashes(original['exports'])
    summary = verified(BASE / 'summary.json')['summaries']
    def get(condition, group, metric, statistic='median'):
        return next(r for r in summary if (r['condition'], r['group'], r['metric']) == (condition, group, metric))[statistic]
    def num(x):
        return f'{x:.6g}'
    def loadcsv(name):
        with (BASE / name).open() as f:
            return list(csv.DictReader(f))
    legacy = loadcsv('historical_P1_mask_audit.csv')
    manifests = verified(BASE / 'permutation_manifest.json')
    zero_contamination = all(int(r['contaminated_valid_pairs']) == 0 and int(r['diagonal_changed']) == 0 for r in legacy)
    all_visible = all(r['valid_view_count_histogram'][r['views']] == 32 ** 3 for r in manifests)
    stage1 = ['| 布局 | 历史P1 JS | P1* JS | P2 JS | P1* context相对L1 | P1* C0相对L1 | P1* C1相对L1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    paths = ['| 布局 | 干预 | JS 中位数 | JS P90 | context相对L1中位数 | C1相对L1中位数 | C1相对L1 P90 | C1最大值 | 变化案例比例 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        stage1.append('| ' + condition + ' | ' + ' | '.join(num(get(condition, group, metric)) for group, metric in [
            ('A2_P1', 'attention_js_mean'), ('A3_P1star_mean8', 'attention_js_mean'), ('A1_P2', 'attention_js_mean'),
            ('A3_P1star_mean8', 'context_relative_l1'), ('A3_P1star_mean8', 'C0_relative_l1'), ('A3_P1star_mean8', 'C1_relative_l1')]) + ' |')
        for group in ('drop_cos', 'drop_sin', 'drop_angle', 'drop_azimuth', 'obs_bias', 'obs_token', 'obs_both'):
            vals = [get(condition, group, 'attention_js_mean'), get(condition, group, 'attention_js_mean', 'p90'),
                    get(condition, group, 'context_relative_l1'), get(condition, group, 'C1_relative_l1'),
                    get(condition, group, 'C1_relative_l1', 'p90'), get(condition, group, 'C1_relative_l1', 'maximum'),
                    get(condition, group, 'C1_affected_case_fraction', 'mean')]
            paths.append('| ' + condition + ' | ' + group + ' | ' + ' | '.join(num(x) for x in vals) + ' |')
    case0 = {r['condition']: r for r in manifests if r['case_index'] == 0}
    scales = ['| 布局 | 有效k | bias RMS均值 | key中心化bias RMS均值 | content RMS均值 | 中心化bias/content比 |', '|---|---:|---:|---:|---:|---:|']
    bias = loadcsv('bias_statistics.csv')
    center_delta = {}
    for condition in CONDITIONS:
        rows_ = [r for r in bias if r['condition'] == condition and r['group'] == 'A0' and r['valid_views'] == 'all']
        means = {key: np.mean([float(r[key]) for r in rows_]) for key in ('bias_rms', 'centered_bias_rms', 'content_rms')}
        scales.append('| ' + condition + ' | ' + str(case0[condition]['views']) + ' | ' + ' | '.join(num(means[key]) for key in ('bias_rms', 'centered_bias_rms', 'content_rms')) + ' | ' + num(means['centered_bias_rms'] / means['content_rms']) + ' |')
    # Independently compute the key-centered CHANGE caused by removing obs bias,
    # on predetermined probes only, averaged within input before cases/layouts.
    with Guard(projections=False) as guard:
        for condition in CONDITIONS:
            vals = []
            for plan in verified(BASE / 'plans.json'):
                if plan['condition'] != condition:
                    continue
                folder = BASE / 'units' / f'p{plan["plan_id"]:03d}'
                a, b = [verified(folder / f'{group}.json') for group in ('A0', 'obs_bias')]
                with np.load(a['probe_path'], allow_pickle=False) as before, np.load(b['probe_path'], allow_pickle=False) as after:
                    mask = before['mask']
                    pair = mask[:, None, :, None] & mask[:, None, None, :]
                    delta = after['bias'].astype(np.float64) - before['bias'].astype(np.float64)
                    count = mask.sum(1)[:, None, None, None].clip(1)
                    centered = delta - np.where(pair, delta, 0).sum(-1, keepdims=True) / count
                    selected = centered[np.broadcast_to(pair, centered.shape)]
                    vals.append(float(np.sqrt(np.mean(selected ** 2))))
            center_delta[condition] = dict(probe_centered_obs_bias_delta_rms_case_median=float(np.median(vals)), n_cases=35,
                                            scope='retained geometry-selected probes only, not whole-volume estimate')
    supplemental = dict(at=now(), zero_actual_historical_contamination=zero_contamination,
                        all_32768_voxels_all_input_views_visible=all_visible, obs_bias_probe_checks=center_delta,
                        sources=verified(BASE / 'report_supplement_registration.json'), new_forward_count=0,
                        GT64_read=False, GT128_read=False)
    if not (BASE / 'interpretation_evidence.json').exists():
        freeze(BASE / 'interpretation_evidence.json', supplemental)
    original_text = REPORT.read_text()
    if zero_contamination:
        original_text = original_text.replace('历史全部排列的离线审计说明P1可能同时改变关系分配及有效pair的来源，甚至改变原有效自关系。历史结果原样保留。',
            '在本轮覆盖的五布局draw0的全部历史P1排列中，实际污染pair数和对角改变数均为0。源码在局部可见性不齐时具有污染可能，但当前输入没有触发；不能把这批历史attention变化归因于无效来源迁移。历史结果原样保留。')
        original_text = original_text.replace('历史P1存在无效来源污染，上表量化其pair比例；source_clean/dirty分层提供伴随证据，但不能识别attention变化的mask可归因百分比。',
            '本轮五布局draw0历史P1的实际无效来源污染为0，有效自关系也未变化，没有source_contaminated query。因此当前历史attention变化没有该类污染证据；不能把代码在其他可见性条件下的风险当作本数据的实测解释。')
    if all_visible:
        original_text = original_text.replace('历史P1还分别给出source_clean/source_contaminated query的注意力分层。',
            '历史P1在本数据只有source_clean query，没有非空source_contaminated分层。')
    detail = f'''\n\n## 最终解释补充：以实际审计为准\n\n本节与前文共同构成最终报告。原自动生成版本保留为审计证据；最终报告修正了其中将“源码存在污染可能”误写为“当前数据存在污染”的措辞。报告补充登记与源码SHA见 report_supplement_registration.json；未改变原协议、模型、干预、数值统计或推理代码，未增加前向。\n\n实际历史P1污染是否全为0：**{zero_contamination}**。全部175输入的32³体素是否均有所有输入视角可见：**{all_visible}**。因此当前没有k不足或图不变导致的不可诊断体素。普通V4与聚簇V4具有相同k=4，不能用有效视角数量差异解释二者响应差；可比较的证据是关系几何和有效bias尺度。k分层虽然完整登记，但没有覆盖同一布局内不同k的实际情况，不得声称已验证遮挡/局部缺视角鲁棒性。工程padding/全无效检查只证明实现边界正确。\n\n### 第一阶段的逐布局直接比较\n\n{chr(10).join(stage1)}\n\nP1*八个场先在每案例内平均，历史P1使用预定排列0；两种扰动分布不同。即使两者响应不同，也不能将差值称为mask污染贡献。当前mask来源迁移计数为0，这一具体竞争解释在本组输入中没有依据。表中context/C0/C1是实际响应，不能称GT误差变化。\n\n### 普通与聚簇的尺度解释\n\n{chr(10).join(scales)}\n\n上表尺度为35案例×4head的描述性均值；完整head分布与案例分位数保留在CSV。共同key偏移会抵消，中心化尺度比原始bias RMS更贴近改变attention的有效量级，但它与输出的相关不等于独立因果解释。普通与聚簇输入的k相同，内容尺度和中心化bias差异应联合看，不能仅凭原始bias绝对大小判断模块是否工作。\n\n### 通道与路径的逐布局完整响应\n\n{chr(10).join(paths)}\n\n以上提供各布局中位数、P90、最大案例和变化案例比例，数值单位均为比例。角通道可同时携带确定性冗余而具有不同参数化敏感性。方位角干预在其余输入不变时的响应支持当前函数使用该输入；不能区分统计信息冗余、训练适应和泛化独特性。obs_token改变内容QKV以及token残差，不能将其输出响应全部归因于attention变化。\n\n共享obs的非线性key差异：在预定probe上，obs_bias引起的**key中心化bias变化RMS**案例中位数为 `{json.dumps(center_delta, ensure_ascii=False)}`。这些是少量几何预定probe的直接数值检查，不声称全体素独立估计；若非零，说明该输入路径的变化并非完全共同常数。全体素attention响应与bias统计是主要对应证据。\n\n八个问题的最终判断以这些直接数据为依据：历史mask搬零解释未被当前数据触发；受控关系影响的大小必须按布局看，context与最终输出是否响应直接由表验证；前两通道确定性冗余；方位角与obs两路径的冻结敏感性已分离，但信息独立性及重训练贡献尚不可识别；普通/聚簇差异不能归因于k差异。优先完成现有数据的离线条件冗余分析，再决定是否值得预登记最小配对训练；本轮不追加训练，不重复已完成消融。\n'''
    final_text = original_text + detail
    if not FINAL_REPORT.exists():
        FINAL_REPORT.write_text(final_text)
    else:
        assert FINAL_REPORT.read_text() == final_text
    for name in ('report_supplement_registration.json', 'report_supplement_registration.json.sha256', 'interpretation_evidence.json', 'interpretation_evidence.json.sha256'):
        target = ARTIFACT / name
        if not target.exists():
            import shutil
            shutil.copy2(BASE / name, target)
    import shutil
    if not (ARTIFACT / FINAL_REPORT.name).exists():
        shutil.copy2(FINAL_REPORT, ARTIFACT / FINAL_REPORT.name)
    final_acceptance = dict(at=now(), completed=True, supersedes_report_only=str(REPORT), final_report=str(FINAL_REPORT),
                            final_report_sha256=sha(FINAL_REPORT), original_numerical_acceptance_sha256=sha(BASE / 'acceptance.json'),
                            original_protocol_sha256=sha(BASE / 'protocol.json'), correction='actual zero mask contamination; original numerical pipeline unchanged',
                            source_sha256=sha(__file__), evidence_sha256=sha(BASE / 'interpretation_evidence.json'),
                            new_forwards=0, GT64_read=False, GT128_read=False, training=False,
                            original_acceptance_exports_still_valid=True)
    if not (BASE / 'acceptance_final.json').exists():
        freeze(BASE / 'acceptance_final.json', final_acceptance)
    for name in ('acceptance_final.json', 'acceptance_final.json.sha256'):
        if not (ARTIFACT / name).exists():
            shutil.copy2(BASE / name, ARTIFACT / name)
    verify_hashes(original['exports'])
    print('FINAL REPORT ACCEPTED; ORIGINAL NUMERICAL EVIDENCE UNCHANGED', flush=True)


if __name__ == '__main__':
    if '--register' in sys.argv:
        register()
    else:
        run()
