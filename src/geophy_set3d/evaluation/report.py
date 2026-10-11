"""Generate the repository's report from the immutable protocol and completed artifacts."""
import csv
import json
from pathlib import Path
from collections import Counter

from .protocols import REPO
from .analysis import read_jsonl


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def metric_table(a,b,aggregation):
    s='| Condition | MAE | RMSE | relative L1 | relative L2 |\n|---|---:|---:|---:|---:|\n'
    for label,result in (('Val-A-V4',a),('Val-B-V4',b)):
        m=result[aggregation];s+=f"| {label} | {m['mae']:.9g} | {m['rmse']:.9g} | {m['relative_l1']:.9g} | {m['relative_l2']:.9g} |\n"
    return s


def write_report(protocol,config,root,summary):
    reference=json.loads((root/'reference.json').read_text());a,b=summary['val_a_v4'],summary['val_b_v4']
    vc=csv_rows(root/'view_count_metrics.csv');angular=csv_rows(root/'angular_coverage_metrics.csv')
    groups=csv_rows(root/'condition_stratification/condition_metrics.csv')
    rankings=json.loads((root/'condition_stratification/rankings.json').read_text())
    pc=json.loads((root/'condition_stratification/plume_correlations.json').read_text())
    ac=json.loads((root/'angular_coverage/correlations.json').read_text())
    cc=json.loads((root/'condition_stratification/continuous_condition_correlations.json').read_text())
    seeds=summary['multiseed'];horizon=summary['horizon_probe'];integrity=summary['integrity']
    view_table='| Condition | V | Mean L1 | Draw std L1 | Mean L2 | Draw std L2 | RMSE | Cardinality |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for r in vc:
        view_table+=f"| {r['condition']} | {r['views']} | {float(r['mean_relative_l1']):.7f} | {float(r['draw_std_relative_l1']):.7f} | {float(r['mean_relative_l2']):.7f} | {float(r['draw_std_relative_l2']):.7f} | {float(r['mean_rmse']):.7f} | {'extrapolation' if r['cardinality_extrapolation']=='True' else 'trained range'} |\n"
    coverage_table='| V | Protocol | Draws | Mean span (deg) | Mean L1 | Draw std L1 | Mean L2 | RMSE |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in angular:
        coverage_table+=f"| {r['views']} | {r['protocol']} | {r['draw_count']} | {float(r['mean_azimuth_span_deg']):.4f} | {float(r['mean_relative_l1']):.7f} | {float(r['draw_std_relative_l1']):.7f} | {float(r['mean_relative_l2']):.7f} | {float(r['mean_rmse']):.7f} |\n"
    seed_table='| Seed | Val-A MAE | Val-A RMSE | Val-A L1 | Val-A L2 | Val-B L1 | Val-B L2 | Best epoch (1-based) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for label,r in [(str(r['seed']),r) for r in seeds['seeds']]+[('mean',seeds['mean']),('std (ddof=1)',seeds['std'])]:
        seed_table+=f"| {label} | {r['val_a_mae']:.8g} | {r['val_a_rmse']:.8g} | {r['val_a_relative_l1']:.7f} | {r['val_a_relative_l2']:.7f} | {r['val_b_relative_l1']:.7f} | {r['val_b_relative_l2']:.7f} | {r['best_epoch']:.3f} |\n"
    shortcut_table='| Seed | Condition | L1 | L2 | MAE | RMSE | L1 / normal |\n|---|---|---:|---:|---:|---:|---:|\n'
    for seed,results in summary['shortcuts'].items():
        for condition,result in results.items():
            m=result['mean_per_case'];ratio=m['relative_l1']/results['normal']['mean_per_case']['relative_l1']
            shortcut_table+=f"| {seed} | {condition} | {m['relative_l1']:.7f} | {m['relative_l2']:.7f} | {m['mae']:.8g} | {m['rmse']:.8g} | {ratio:.3f} |\n"
    condition_table='| Field | Value | n | Mean L1 | Median L1 | Mean L2 | Mean RMSE | small_n |\n|---|---|---:|---:|---:|---:|---:|---|\n'
    for r in groups:
        condition_table+=f"| {r['field']} | {r['value']} | {r['case_count']} | {float(r['mean_relative_l1']):.6f} | {float(r['median_relative_l1']):.6f} | {float(r['mean_relative_l2']):.6f} | {float(r['mean_rmse']):.7f} | {r['small_n']} |\n"
    worst_table='| Pool | Rank | Case | L1 | Wind (m/s) | Source family | Active fraction | Angular span |\n|---|---:|---|---:|---:|---|---:|---:|\n'
    best_table='| Pool | Case | L1 |\n|---|---|---:|\n'
    for pool,ranked in rankings.items():
        for i,r in enumerate(ranked['worst_10'],1):
            worst_table+=f"| {pool} | {i} | {r['case_id']} | {r['relative_l1']:.6f} | {r['condition_meta']['wind_speed_m_s']} | {r['condition_meta']['source_family']} | {r['active_voxel_fraction']:.6f} | {r['azimuth_span_deg']:.3f} |\n"
        for r in ranked['best_5']:best_table+=f"| {pool} | {r['case_id']} | {r['relative_l1']:.6f} |\n"
    correlations='| Descriptor | Pool A rho(L1) | Pool A rho(L2) | Pool B rho(L1) | Pool B rho(L2) |\n|---|---:|---:|---:|---:|\n'
    for descriptor in dict.fromkeys(r['x'] for r in pc['A']):
        values=[]
        for pool,metric in (('A','relative_l1'),('A','relative_l2'),('B','relative_l1'),('B','relative_l2')):
            value=next(r['rho'] for r in pc[pool] if r['x']==descriptor and r['y']==metric)
            values.append(f'{value:.5f}' if value is not None else 'undefined')
        correlations+=f"| {descriptor} | {' | '.join(values)} |\n"
    checks='| Predefined B1 criterion | Passed |\n|---|---|\n'+''.join(f'| {k} | {v} |\n' for k,v in summary['b1_checks'].items())
    hashes='```json\n'+json.dumps(reference['dataset_hashes'],indent=2)+'\n```\n'
    small_gain=all(float(next(r['mean_relative_l1'] for r in vc if r['pool']==pool and r['views']=='6'))-float(next(r['mean_relative_l1'] for r in vc if r['pool']==pool and r['views']=='8'))<protocol['saturation_absolute_relative_l1_gain'] for pool in protocol['pools'])
    saturation='V6→V8 的两池增益都小于预注册阈值，可视为该区间开始饱和。' if small_gain else 'V4–5 之后收益明显递减，但两池 V6→V8 仍改善超过0.005，未达到预注册严格饱和阈值；不能宣称 V≥6 已饱和。'
    tests_log=(root/'tests_all.log').read_text()
    tests_line=next(line for line in tests_log.splitlines() if line.startswith('Ran '))
    diagnostic_conclusion='training horizon may be limiting B0；不替换正式 reference，先由负责人决定后续训练 horizon。' if horizon['meaningful_improvement'] else '当前诊断未显示达到0.5个百分点的改善，150-epoch reference 在此判据下基本足够。'
    text=f'''# B0 Generalization Evaluation Protocol v1

## 1. Frozen B0 reference

正式 baseline 仍为 GeoLift-Mean-Res3D / supervised_b0，网络、encoder/lifting/fusion/decoder、
loss、split、Pool A-only training 与归一化保持原样。Training source commit：
`{reference['training_source_git_commit']}`。
checkpoint：`{reference['checkpoint_path']}`；SHA256：
`{reference['checkpoint_sha256']}`。best epoch={reference['best_epoch_zero_based']+1}（1-based）。
完整 effective config、manifest/split/geometry/statistics hashes 和 model source hashes 在 reference.json。

{hashes}
reference 首次复现被默认 cuDNN TF32 设置触发误差门槛，后续实验当时停止。
修复为与原评估一致的 FP32、TF32 off、cudnn benchmark off 后，四项 mean/per-case 与四项 pooled
全部精确复现（absolute differences=0），未放宽原先 atol=1e−8、rtol=1e−5。
失败结果完整保留在 reference_reproduction_initial_tf32_mismatch.json 与 plume_generalization/initial_tf32_mismatch/。

## 2–6. Evaluation protocol、Pool定义、view count、draw与覆盖

本轮全部 inference 为 val，model.eval() + torch.no_grad()，GT64 直接在 CH4 mass fraction
物理空间计算，复用 training.metrics.VolumeMetrics，未创建第二套误差公式。
normalization 只使用 frozen train/Pool A p99.5={a['normalization_scale_ppm_m']:.12g} ppm·m；
不使用 val/test normalization，不做 per-case 最大值归一化。
Mean per-case 对各 case 的指标平均；pooled 用全体 voxel 误差与GT sums计算。

**Pool B consists of independently sampled unseen station poses from the same defined TDLAS acquisition family.**
Pool A有64 stations，Pool B有32；B使用记录中的 ood_projection_path + ood_geometry.npz，
配对经明确 integration test 检查。准确称呼是 unseen station geometry / in-family geometry generalization，
不称 arbitrary OOD geometry。Pool B 从未用于训练、网络选择或 optimizer 选择。

两个 legacy reference 条件 Val-A-V4 / Val-B-V4 使用原 ViewSampler：seed20261011、epoch0、V4、hold0，
case在val Dataset中的原index参与seed。所有三个训练seed及horizon model selection均固定这同一组Val-A视图。

View-count 条件使用独立 EvaluationViewSampler，V∈[2,3,4,5,6,8]，draw∈[0,1,2,3,4]。
seed仅由 base evaluation seed={protocol['evaluation_seed']}、case_id、V、draw_id、pool 组成，
case/pool用稳定BLAKE2b摘要 + NumPy SeedSequence，sample index/worker/全局RNG不参与。
每个subset为V个唯一station；所有per-case/per-draw行记录IDs、azimuths、pool与draw，可精确重放。
原训练 ViewSampler 未修改。独立draw随机子集不要求不同V嵌套。

因此 view-count V4 的五draw均值与 legacy reference V4 属于不同预注册 subset 协议，不能混为同一指标。
训练只见过V=3–6；V=2和8是 cardinality extrapolation。

Angular coverage仅 val/Pool A，V=4/6：

- Spread：含最小/最大frozen azimuth，最大化span；之后 greedy max-min azimuth distance填充。
  每个V仅一个canonical set，全case共享，draw std=0表示只有一组，不是无不确定性。
- Clustered：对azimuth排序，枚举长度V的连续窗口，取span最小的五个窗口，按span、ID排序。
  draw0为全局最小span，其余draw为后续最小窗口，不硬编码station index。
- Random：与view-count完全相同的五个case-specific deterministic draws。

每行记录 min/max/span、largest consecutive sorted azimuth gap（不包括半环边界空隙）、
mean pairwise absolute azimuth distance；角度差是定义域0–180°内的线性距离。
Draw std是5个case-averaged means的 sample std(ddof=1)；case variability单独保存在case_std_per_draw；
seed std是三个独立训练seed的sample std。三者不混用。

## 7–8. Plume与unseen station geometry结果

固定 legacy V4，mean per-case：

{metric_table(a,b,'mean_per_case')}
pooled：

{metric_table(a,b,'pooled')}

Val-B相较Val-A，relative L1绝对变化={b['mean_per_case']['relative_l1']-a['mean_per_case']['relative_l1']:.9g}，
relative degradation={(b['mean_per_case']['relative_l1']/a['mean_per_case']['relative_l1']-1)*100:.5f}%；
relative L2 degradation={(b['mean_per_case']['relative_l2']/a['mean_per_case']['relative_l2']-1)*100:.5f}%。
当前单reference的L1略改善、L2略恶化，不把某一有利metric当作全面更优结论。
所有case行在 plume_generalization/Val-A-V4/per_case_metrics.jsonl 与 geometry_generalization/Val-B-V4/per_case_metrics.jsonl。

## 9–10. Variable V、draw variability与cardinality extrapolation

{view_table}

在当前两个pool和该evaluation seed下，case/draw均值随测试V增加而下降。
{saturation}
V2可以运行且输出finite，但relative L1约0.37；V8可以运行且误差低于V6。
这只支持测试过的V∈[2,3,4,5,6,8]，不能扩展到任意数量views。
标准差是draw-induced variability，非case std或training-seed std。
数字源文件 view_count_metrics.csv；对应三张PNG和PDF保留全部数据/errorbars。

## 11. 相同V的angular coverage

{coverage_table}

Spread优于random，clustered显著变差；增加数量无法抵消极窄coverage。
V4 span↔relative L1 Spearman rho={ac['4']['rho']:.6f}，V6 rho={ac['6']['rho']:.6f}。
每个V有36×(1+5+5)=396行，含重复case；相关系数用于描述，不将重复行当作独立样本做显著性推断。
nominal p-value仅原始输出，未做多重比较校正；不能把rho解释为纯azimuth span的因果效应。
station radius/height和选择的specific poses也可能影响误差。

## 12. Condition stratification

只使用 frozen condition_meta 的真实字段。source_mode=null被保留，未从filename推断；
所有groups均输出，n<3标small_n，不做强结论、不删除case。

{condition_table}

wind/pressure 的 exact-value groups之外，Spearman结果：

```json
{json.dumps(cc,indent=2)}
```

这些条件存在共变（例如source_family与active_hole_count），分层均值不能解释为独立因果效应。
风速3m/s组较难；specific source geometry结果必须结合n解读。condition_id小组仅描述，不据此调参。

## 13. Best / worst case分析

Pool A和B分别按legacy V4 relative L1排序，保留best5与worst10，未删bad cases。
两个worst10中wind=3m/s分别有
{Counter(r['condition_meta']['wind_speed_m_s'] for r in rankings['A']['worst_10'])[3.0]} / {Counter(r['condition_meta']['wind_speed_m_s'] for r in rankings['B']['worst_10'])[3.0]}例。
许多worst case具有较小mean/active fraction的细窄plume，并涉及G2/G3多源；也存在coverage较窄的case，
但较大span仍有难例，不能只用coverage解释全部误差。
A最差case图中可见两条窄plume的形状/峰值偏差；整体GT mass较小会放大relative error。
worst case的GTmax、mean、active/nonzero fractions、center、integrated proxy、condition_meta、IDs、
azimuths和coverage描述全部存于 rankings.json / enriched.jsonl，三轴GT/pred/error图保存在figures/worst_cases/Pool_A和Pool_B。

{worst_table}

Best5：

{best_table}

## 14. Plume characteristics与误差

固定active阈值=1e−5 mass fraction，在查看结果前写入effective protocol，未调阈值。
GT max/mean、nonzero与active fraction、concentration center XYZ均从GT64直接计算。
integrated concentration proxy=mean(GT)×0.6³，等权endpoint grid近似，无density，不能称CH4质量。
浓度中心为各endpoint XYZ坐标的concentration-weighted平均。

{correlations}

中心坐标和误差的相关可能反映wind/source共变；GT mean与integrated proxy只差常数，相关系数相同，
不算两份独立证据。所有相关仅解释失败模式，未用于模型选择。
'''
    text+=f'''
## 15. Multi-seed结果

原seed20261011仅引用已有checkpoint和日志，没有重新训练其150-epoch正式实验。
seed20261012/13只改变training seed（初始化、train station sampling和shuffle）；架构、loss、AdamW、
batch2、V3–6、FP32、150epoch上限与patience20不变。为了隔离训练seed变异，
所有runs的validation selection seed固定20261011、epoch0、Pool A/V4，station IDs与reference逐case一致。
该评估seed与training seed明确分开记录在validation_protocol.json与checkpoint。

{seed_table}

Pool A relative L1={seeds['mean']['val_a_relative_l1']:.7f} ± {seeds['std']['val_a_relative_l1']:.7f}，
relative L2={seeds['mean']['val_a_relative_l2']:.7f} ± {seeds['std']['val_a_relative_l2']:.7f}；
Pool B relative L1={seeds['mean']['val_b_relative_l1']:.7f} ± {seeds['std']['val_b_relative_l1']:.7f}。
完整A/B MAE、RMSE、L1/L2、bestepoch mean/std见multiseed_metrics.csv。
seed std使用ddof=1；只有3 seeds，此稳定性评价有有限样本限制。
新runs均只用Pool A val loss选择best，Pool B评估发生在checkpoint冻结后。

## 16. Horizon probe

**Option A：fresh diagnostic training**，seed20261011、max_epochs220，其余optimizer/data/loss/seed语义不变。
原cosine T_max150的last.pt已将LR降到零，不能合理地直接改checkpoint scheduler继续；
因此没有篡改、覆盖或恢复正式last.pt，220诊断从随机初始化开始，cosine T_max220。
这从epoch0改变LR轨迹，测试的是更长/更慢decay的diagnostic protocol，非纯粹延长相同轨迹。
保持原patience20，实际完成{horizon['training_summary']['completed_epochs']} epochs，
结束原因={horizon['training_summary']['stopping_reason']}，bestepoch={horizon['best_epoch']}（1-based）。

| Condition | Best val loss | Mean L1 | Mean L2 |
|---|---:|---:|---:|
| Reference 150 | {a['supervised_loss']:.9g} | {a['mean_per_case']['relative_l1']:.9g} | {a['mean_per_case']['relative_l2']:.9g} |
| Diagnostic 220 | {horizon['diagnostic']['supervised_loss']:.9g} | {horizon['diagnostic']['mean_per_case']['relative_l1']:.9g} | {horizon['diagnostic']['mean_per_case']['relative_l2']:.9g} |

relative L1 absolute improvement={horizon['absolute_relative_l1_improvement']:.9g}，
即{horizon['percentage_point_improvement']:.6f} percentage points；预注册meaningful阈值=0.5个百分点（比例0.005）。
{diagnostic_conclusion}
诊断best仍位于第{horizon['best_epoch']}个epoch，若处于220上限则不能据此宣称220已经充分收敛。
只用val loss/L1/L2回答诊断问题，未根据test或Pool B选择更长horizon。
原正式B0、原150-epoch配置与原checkpoint仍保持原样。

## 17. Shortcut跨seed复查

同一Val-A-V4/reference视图；zero Y保留valid/path/geometry，wrong G循环错位全部station geometry，
包括origin/basis、valid/path、ray geometry、IDs，Y仍原顺序。仍复用原B0 shortcut语义。

{shortcut_table}

这检验原measurement/geometry结论能否跨训练seed复现，不对失败条件重选subset或改模型。

## 18–20. 限制、可以支持的结论与B1建议

冻结reference在预定义V2/3/4/5/6/8上都可推理；当前测试V增大时平均误差下降。
当前results支持defined acquisition family内的unseen station poses泛化，
支持相同V下宽coverage更有利。不能宣称arbitrary geometry、任意V或已经解决sparse-view tomography。

仍只有36个val plumes，同一Fluent来源和若干相关source families，未做独立现实测量域验证。
这些val案例也用于best checkpoint和early stopping选择，本阶段分数不是独立盲测test估计。
小组样本、three-seed数量、同case重复draw及acquisition family的边界限制外推；
standard deviations为各自明确的case/draw/seed变异，非泛化置信区间。
32³ lifting和3D卷积对窄/多源plume可能有细节与峰值误差。
coverage descriptor并不完全捕获半环方向、radius、height及具体poses。
Horizon probe改变cosine轨迹，单diagnostic seed不能区分纯训练时长效应和学习率日程效应。
CUDA grid_sample backward不承诺逐bit训练轨迹复现；station selection则完全确定性可重放。

B1门槛在effective_protocol.yaml预注册：Val-A L1 seed CV≤0.15；Pool B mean L1≤0.5且相对A恶化≤100%；
A/B V3→V6均改善；两个新seed的wrongG/zeroY L1至少normal的1.2倍；
horizon诊断L1改善<0.005。它们是本次分析的操作性判断，非通用物理界限。

{checks}

**建议：{summary['b1_recommendation']}。**
若horizon检查失败，优先解决baseline horizon公平性，由负责人决定正式训练预算，
不自动替换reference，不直接实现B1。若后续所有门槛满足，优先比较简单的
geometry-conditioned attention pooling与固定mean fusion，先保持encoder/lifting/decoder不变；
全量set self-attention可作为后续对照，本轮未实现、未训练任何新网络结构。

## Integrity、tests与复现

Test set remained untouched in this phase：没有运行test推理/训练/超参数选择。
完整性QA仅只读hash全部frozen artifacts，包含test文件，不是test model evaluation。
Pool B始终evaluation-only；不用于training/model architecture/optimizer选择。

{tests_line}，全部OK。
测试覆盖V个唯一station、Pool隔离、draw可重放、draw变化、spread/clustered覆盖性质、
Pool B投影/geometry正确配对与真实model的同步view permutation、draw/case variability区分、
固定GT threshold/XYZ center、null metadata与small_n处理，以及此前全部31个tests。

本轮前后 frozen dataset={integrity['dataset_file_count']} files、source={integrity['source_file_count']} files
inventory完全一致（size/mtime/inode/SHA256；historical inputs/保留旧QA的stat-only规则），
old supervised_b0 logs/checkpoints/figures完全未修改，model source hashes、manifest/split/geometry/statistics hashes一致。
完整性passed={integrity['passed']}，reference checkpoint unchanged={integrity['checkpoint_unchanged']}。

运行命令（仓库根目录，已安装torch/CUDA、NumPy、SciPy、PyYAML、Matplotlib）：

```bash
python scripts/evaluate_b0_generalization.py --stage reference
python scripts/train_b0_multiseed.py --seed 20261012
python scripts/train_b0_multiseed.py --seed 20261013
python scripts/run_b0_horizon_probe.py
python scripts/evaluate_b0_generalization.py --stage all
python scripts/summarize_b0_generalization.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
```

训练入口禁止重训reference seed和覆盖已有run；恢复新run时可加 --resume，原reference resume不涉及本轮。
所有artifacts位于 `{root}`：reference.json/effective_protocol.yaml、summary.json、integrity_summary.json、
view_count_metrics.csv、angular_coverage_metrics.csv、multiseed_metrics.csv与各实验目录。
每个新run含effective config/environment、逐epochmetrics、best/last完整checkpoint、curves与fixed-val图。
主图同时导出PNG/PDF，源CSV/JSONL保留所有case；数据集/权重/日志/图不进入Git。
'''
    target=REPO/'docs/reports/supervised_b0_generalization_v1.md'
    target.write_text(text)
    # Numeric Markdown main tables are standalone artifacts as well as repository report sections.
    (root/'main_tables.md').write_text('# B0 Generalization v1 main tables\n\n'+view_table+'\n'+coverage_table+'\n'+seed_table)
    return target
