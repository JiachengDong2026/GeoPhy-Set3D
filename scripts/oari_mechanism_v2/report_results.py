"""Evidence export and Chinese report; no model calls and no GT."""
from runtime import *
import gzip
import shutil
import zipfile


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def run():
    p = validate()
    verification = verified(BASE / 'independent_verification.json')
    assert verification['passed']
    summaries = verified(BASE / 'summary.json')['summaries']
    def value(condition, group, metric, key='median'):
        return next(r for r in summaries if (r['condition'], r['group'], r['metric']) == (condition, group, metric))[key]
    def fmt(x):
        return f'{x:.6g}'
    def table(groups, metrics, condition):
        lines = ['| 干预 | ' + ' | '.join(title for title, metric in metrics) + ' |',
                 '|---|' + '---:|' * len(metrics)]
        for group in groups:
            lines.append('| ' + group + ' | ' + ' | '.join(fmt(value(condition, group, metric)) for title, metric in metrics) + ' |')
        return '\n'.join(lines)
    legacy = read_csv(BASE / 'historical_P1_mask_audit.csv')
    perms = verified(BASE / 'permutation_manifest.json')
    per_case = read_csv(BASE / 'case_results_collapsed.csv')
    interaction = []
    for condition in CONDITIONS:
        selected = {}
        for r in per_case:
            if r['condition'] == condition:
                selected[(r['case_id'], r['group'])] = r
        for pair, joint, singles in [('angle', 'drop_angle', ('drop_cos', 'drop_sin')), ('observability', 'obs_both', ('obs_bias', 'obs_token'))]:
            for metric in ('attention_js_mean', 'context_relative_l1', 'C1_relative_l1'):
                vals = [float(selected[(case, joint)][metric]) - sum(float(selected[(case, s)][metric]) for s in singles) for case in verified(BASE / 'cases.json')]
                interaction.append(dict(condition=condition, channels=pair, metric=metric, n_cases=35,
                                        median_joint_minus_sum=float(np.median(vals)), mean_joint_minus_sum=float(np.mean(vals)),
                                        p90_joint_minus_sum=float(np.quantile(vals, .9))))
    if not (BASE / 'nonadditivity.csv').exists():
        csv_write(BASE / 'nonadditivity.csv', interaction)
    history_lines = ['| 布局 | 历史全部 P1 的污染 pair 比例均值 | 不可诊断活跃体素比例 |', '|---|---:|---:|']
    for condition in CONDITIONS:
        old = [r for r in legacy if r['condition'] == condition]
        per = [r for r in perms if r['condition'] == condition]
        contamination = np.mean([float(r['contaminated_fraction']) for r in old])
        ineligible = np.mean([1 - r['eligible_voxels'] / r['active_voxels'] for r in per])
        history_lines.append(f'| {condition} | {contamination:.4%} | {ineligible:.4%} |')
    layout_lines = ['| 布局 | P1* attention JS 中位数 | P1* context 相对L1中位数 | P1* C1相对L1中位数 | C1变化案例比例 |', '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        group = 'A3_P1star_mean8'
        layout_lines.append('| ' + condition + ' | ' + ' | '.join(fmt(value(condition, group, m)) for m in ('attention_js_mean', 'context_relative_l1', 'C1_relative_l1')) + ' | ' + f"{value(condition, group, 'C1_affected_case_fraction', 'mean'):.1%}" + ' |')
    clustered_js = value('clustered-V4', 'A3_P1star_mean8', 'attention_js_mean')
    clustered_c1 = value('clustered-V4', 'A3_P1star_mean8', 'C1_relative_l1')
    ordinary_azi = value('Val-A-V4', 'drop_azimuth', 'C1_relative_l1')
    cluster_azi = value('clustered-V4', 'drop_azimuth', 'C1_relative_l1')
    text_report = f'''# OARI 掩码受控关系干预与几何通道分解：正式冻结诊断 v2

## 范围、身份与证据边界

本轮完成第一、第二阶段，只做冻结推理及离线统计，未训练、未反向传播、未更新参数，未读取 GT64 或 GT128。全部结果属于已经使用过的测试集上的探索性机制诊断，不能称为独立测试确认；输出变化不等于重建误差增加。

正式模型为 **Multi-scale MRGR + Uniform Fusion**，seed 20261011，epoch 285，342,725 参数。通过登记的官方构造器 `scripts/uniform_final_v1/uniform_models.py:model_for` 严格加载，checkpoint SHA256 `{HIST.EXPECTED}`。源码审计发现实际架构仅有一次 OARI，位于 C0 前；反馈分支有第二次 GeometryLifting，但没有第二次 OARI。因此本轮 OARI 干预可能同时改变 C0/C1。正式模型、其他几何通路和历史实验均未改动。

登记前源码提交为 `{verified(BASE / 'checkpoint_audit.json')['source_commit']}`。协议 SHA256 `{sha(BASE / 'protocol.json')}`。可执行脚本及全部模型源码 SHA、配置 SHA、数据身份、视角 ID、排列种子均记录在本实验目录。协议在 GPU 运行前冻结，后续失败及恢复证据保留。

35 个固定案例、5种布局、draw 0，共175份输入。A0、历史P1的排列0、P2的525份封存 C0/C1 预测直接复用，缺失的 OARI 中间量只重放必要前缀/OARI；8个P1*空间排列场及7个通道干预共新增2,625份 C0/C1 预测。每组完整身份及哈希见 prediction_manifest。工程全模型重复仅用于预登记的两个输入，成功结果直接复用于正式记录。

## 实现与工程检查

A1/P2 精确为第一仿射保留 `W_obs o + b`；既非最终 bias 置零，也非从 logits 减去 bias。所有通道屏蔽均仅移除第一仿射中指定列的输入贡献并保留 learned bias。obs_token 仅去掉 observability_embedding 第一线性层的五维输入贡献，保留偏置后再 SiLU；obs_bias 只改 pairwise_bias，obs_both 同时改两条路径。

P1*对每个体素原有效视角集合共同置换 query/key，固定无效索引，保留自关系、对称性和有效pair数量，不搬入无效pair零值。每份输入固定8个空间排列场，种子 `20261024 + 1009×plan_id`，完全依据几何生成并拒绝恒等和图自同构。k≤2或关系图恒定的体素不可诊断。V3每体素最多5种非恒等排列，8个场可重复局部选择，不能作为8个独立样本。

工程检查见 engineering_checks.json：普通V4/clustered-V4各固定首案例的无hook重复、只读hook no-op和历史A0/P1/P2输出在登记容差内一致；padding不变性、全无效安全性、CPU登记mask与实际GPU mask逐元素一致；P1*确实改变所有eligible体素，mask、对角、自关系和对称性不变。所有组实际输入/参数hash不变，始终eval且参数无梯度；输出有限且在[0,1]。全过程只使用一张通过UUID锁和15秒双快照确认的空闲GPU。

独立复核对全部3,150份预测记录重算输出差异，并以独立 NumPy float64 重建预定少量probe的 token projection、LayerNorm、Q/K/V、MLP bias、masked softmax、完整context，使用SciPy复核JS/熵公式；另由原始组记录独立重组案例统计。该独立重建不涵盖所有体素，完整体素的mask和softmax校验由worker完成。独立复核最大误差：`{json.dumps(verification['max_errors'], ensure_ascii=False)}`。

## 统计规则

先在每案例每布局内平均8个P1*场，再以35案例统计中位数、P90、最大值及均值；跨布局先在同案例内平均5布局。体素、head及排列都不充当独立案例。逐head和有效视角数分层完整保留。下表均为案例中位数；均值、P90、最大值、全部案例详见 paired_statistics.csv / per_case_results.csv。相对变化为比例，乘100为百分比；这是预测敏感性，不是相对GT误差。

预登记 epsilon=1e-6；输出变化案例定义为最大绝对输出差>1e-5，P1*及跨布局先平均同案例响应后判定，阈值仅标记数值敏感性。context同时报告near-zero基准体素比例和相对变化分布，防止小分母造成误解。JS使用自然对数散度（非平方根距离），只包含相同有效query/key。

## 第一阶段：历史污染与受控关系效应

{chr(10).join(history_lines)}

历史全部排列的离线审计说明P1可能同时改变关系分配及有效pair的来源，甚至改变原有效自关系。历史结果原样保留。P1*污染pair数恒为0。历史P1还分别给出source_clean/source_contaminated query的注意力分层。**这些比较不能因果量化“attention变化多少来自mask”**：历史全局排列与新局部受控排列改变了不同关系分布；污染比例不是注意力变化的可归因比例。还需同扰动强度的额外设计才能分离这一比例，当前不追加结果导向的实验。

{chr(10).join(layout_lines)}

在clustered-V4，P1* attention JS案例中位数为 {fmt(clustered_js)}，C1相对L1变化中位数为 {fmt(clustered_c1)}。这些实际数值用于判断其量级，不能仅用“聚簇无效/有效”二分结论。受控关系效应与历史P1/P2及通道干预的跨布局结果如下。

{table(['A1_P2', 'A2_P1', 'A3_P1star_mean8'], [('attention JS', 'attention_js_mean'), ('context相对L1', 'context_relative_l1'), ('C0相对L1', 'C0_relative_l1'), ('C1相对L1', 'C1_relative_l1')], 'across5_within_case_mean')}

## 第二阶段：通道审计与路径分解

离线审计确认前两通道满足cos²+sin²≈1，最大偏差为 {verified(BASE / 'relation_channel_audit.json')['max_cos_squared_plus_sin_squared_error']:.6g}，是确定性冗余，不能称两份独立几何信息。方位角差通道范围为 {verified(BASE / 'relation_channel_audit.json')['azimuth_range']}，完整分布、零值、有效pair覆盖、与cos相关性见 relation_channel_distributions.csv。权重范数及基于实际输入512个预定有效pair的解析局部Jacobian见关系审计与 analytic_bias_jacobian.csv；输入方差、权重范数和局部导数均不能单独证明功能贡献。

{table(['drop_cos', 'drop_sin', 'drop_angle', 'drop_azimuth', 'obs_bias', 'obs_token', 'obs_both'], [('bias变化RMS', 'bias_delta_rms'), ('content变化RMS', 'content_delta_rms'), ('attention JS', 'attention_js_mean'), ('context相对L1', 'context_relative_l1'), ('C1相对L1', 'C1_relative_l1')], 'across5_within_case_mean')}

方位角单通道屏蔽在普通V4、clustered-V4的C1相对L1中位数分别为 {fmt(ordinary_azi)}、{fmt(cluster_azi)}。若产生响应，直接支持冻结模型在给定输入下对该输入路径敏感；不单独证明它在夹角之外含有可泛化增量信息，也不等同重训练后的独立贡献。角通道单独屏蔽可能离开cos/sin确定性流形，属于可能分布外敏感性检查。

obs_bias保持token和QKV不变；obs_token保持原始pairwise bias逐元素不变；obs_both同时干预。共享obs输入经过非线性bias MLP可产生随关系不同的key差异，不能预先断言全部会在softmax抵消。bias_statistics.csv含每head原始/沿有效key中心化RMS、content RMS、二者比例、对角/非对角分布、bias变化以及head间中心化差异审计。attention_change_statistics.csv保留所有head与有效k、JS/绝对差/熵及key均值变化；context_change_statistics.csv保留有效k/空间变化分布和近零分母比例。

nonadditivity.csv给出角双通道与obs双路径“联合响应−单独响应之和”，只描述非可加性。JS、范数本身亦非线性，不能据其差值分配独立因果贡献。

## 八个研究问题的证据回答

1. 历史P1存在无效来源污染，上表量化其pair比例；source_clean/dirty分层提供伴随证据，但不能识别attention变化的mask可归因百分比。
2. 聚簇有效对内错配的实际attention JS与C1响应见第一阶段，所有P1*均满足零污染；应按实测量级解释，不能把历史P1直接当成纯关系依赖。
3. context、C0和C1响应均逐例记录，上表直接检验信号是否传递到输出；数值变化不说明更接近或远离GT。
4. 前两个夹角通道存在确定性冗余；单独移除的不同响应反映已训练参数化与非线性，不能称独立信息。
5. 方位角路径的冻结敏感性可直接观察；“夹角之外增量信息”和训练后性能收益需要额外可辨识证据。
6. obs_bias与obs_token分别隔离显式bias和内容/残差token路径；后者还直接改变token残差，并非仅Q/K。obs_both揭示联合敏感性，不能线性相加为贡献。
7. 有效k、中心化bias及content相对尺度的分层证据能解释与布局差异一致的现象，但布局还共同改变其他几何条件，不能仅据相关分层作因果解释。逐布局逐head完整统计允许核查近零/常数/相同head。
8. 当前两条obs路径及角/方位角敏感性已在全案例冻结检查，机制响应已定位；不能仅凭本轮决定泛化最优结构。需要重新训练才能回答参数适应后是否仍有性能增益。

## 事实、解释、假设与最小下一步

直接事实是上述有限输出响应、历史污染、确定性冗余、隔离路径对照和工程不变性。与证据一致的解释是关系/可观测性可通过bias及token的不同路径传递，且布局条件会改变其响应尺度。未验证的假设包括某通道对泛化独有、聚簇小响应说明训练时不需要该通道、以及历史P1的attention主要来自污染。冻结敏感性不能回答重新训练的性能贡献，也不能证明模型学到了可泛化几何规律。

资源最省的下一步：先利用本轮逐head/k及已完成训练消融，核对竞争假设；若研究目标仍是独立信息而非参数化敏感性，优先离线分析现有几何中方位角在给定cos/sin后的剩余变化（不需要GPU/GT），不足时才预登记针对该剩余量的冻结干预。若最终需要验证重训练的增量价值，仅考虑保留Uniform Fusion的“保留全部其他模块、删除方位角bias输入”最小配对训练对照，使用已有正式种子和既有完整对照，不重复已完成的整OARI/GeometryLifting消融。本任务不启动任何训练或新增第三阶段。

## 产物、异常与可复现

完整数组、分批group记录、GPU调度日志及原始统计位于 `{BASE}`。仓库同步协议、完整身份/哈希清单、CSV统计（大表gzip）、每布局预定probe压缩包、权重代数证据、独立核验和验收。大体积预测保留scratch并由manifest SHA256链接，未修改或覆盖历史产物。故障记录与恢复日志若存在则一并导出，最终acceptance只在完整独立核验通过后生成。
'''
    if not REPORT.exists():
        REPORT.write_text(text_report)
    else:
        assert REPORT.read_text() == text_report
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    # Export small audit files and all tables. Very large CSVs remain directly
    # inspectable after gzip decompression; no prediction is checked into git.
    for path in BASE.glob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.npz', '.sha256') and path.name not in ('status.json',):
            destination = ARTIFACT / path.name
            if path.stat().st_size > 8 * 1024 ** 2 and path.suffix in ('.csv', '.json', '.jsonl'):
                destination = ARTIFACT / (path.name + '.gz')
                if not destination.exists():
                    with path.open('rb') as src, destination.open('xb') as dst, gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
                        shutil.copyfileobj(src, compressed)
            elif not destination.exists():
                shutil.copy2(path, destination)
            else:
                assert sha(path) == sha(destination)
    for li, condition in enumerate(CONDITIONS):
        archive = ARTIFACT / f'probes_layout{li}_{condition}.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as dest:
                for pid in range(li * 35, (li + 1) * 35):
                    unit = BASE / 'units' / f'p{pid:03d}'
                    for group in GROUPS:
                        path = unit / f'{group}_probes.npz'
                        dest.write(path, f'p{pid:03d}/{path.name}')
    raw_path = ARTIFACT / 'raw_group_records.jsonl.gz'
    if not raw_path.exists():
        with raw_path.open('xb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as out:
            for plan in verified(BASE / 'plans.json'):
                for group in GROUPS:
                    r = verified(BASE / 'units' / f'p{plan["plan_id"]:03d}' / f'{group}.json')
                    out.write((json.dumps(r, allow_nan=False) + '\n').encode())
    for sub in ('scheduling', 'interrupted_artifacts'):
        if (BASE / sub).exists() and not (ARTIFACT / sub).exists():
            shutil.copytree(BASE / sub, ARTIFACT / sub)
    for name in ('controller.lock',):
        for path in ARTIFACT.rglob(name):
            path.unlink()
    shutil.copy2(REPORT, ARTIFACT / REPORT.name)
    exports = {str(path.relative_to(PROJECT)): sha(path) for path in ARTIFACT.rglob('*') if path.is_file() and path.name not in ('acceptance.json', 'acceptance.json.sha256')}
    acceptance = dict(at=now(), completed=True, model='Multi-scale MRGR + Uniform Fusion', checkpoint_sha256=HIST.EXPECTED,
                      inputs=175, case_n=35, groups=18, group_records=3150, new_predictions=2625, reused_predictions=525,
                      engineering_passed=True, independent_verification_passed=True, sources_unchanged=True,
                      GT64_read=False, GT128_read=False, training=False, maximum_GPUs=1,
                      report=str(REPORT), report_sha256=sha(REPORT), exports=exports,
                      failures_preserved=[str(x) for x in BASE.glob('*failure*.json')],
                      protocol_sha256=sha(BASE / 'protocol.json'), scratch=str(BASE), independent_test_confirmation=False)
    if not (BASE / 'acceptance.json').exists():
        freeze(BASE / 'acceptance.json', acceptance)
    for name in ('acceptance.json', 'acceptance.json.sha256'):
        if not (ARTIFACT / name).exists():
            shutil.copy2(BASE / name, ARTIFACT / name)
    verify_hashes(verified(BASE / 'acceptance.json')['exports'])
    print('REPORT AND EVIDENCE ACCEPTED', flush=True)


if __name__ == '__main__':
    run()
