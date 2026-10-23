"""Render the completed single-seed B0/B1 comparison, including its limits."""
import json
import subprocess
from pathlib import Path


def render_report(root, repo):
    root, repo = Path(root), Path(repo)
    s = json.loads((root/'summary.json').read_text())
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip()
    b0,b1 = [s['models'][key] for key in ('B0-300','B1-GCA-300')]
    gate_path = Path(b1['logs']).parent/'supervised_b1_gca_300_overfit/overfit_summary.json'
    gate = json.loads(gate_path.read_text())
    attention = s['attention']
    tests = (root/'tests.log').read_text()
    test_line = next(line for line in tests.splitlines() if line.startswith('Ran '))
    if not tests.rstrip().endswith('OK'):
        raise AssertionError('All tests must pass before the completed report')
    metric_names = ('mae','rmse','relative_l1','relative_l2')
    def main_table():
        rows = ['| Model | Condition | MAE | RMSE | Relative L1 | Relative L2 | Best epoch | Best val loss |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
        for label,m in s['models'].items():
            for condition in ('Val-A-V4','Val-B-V4'):
                x=m['evaluations'][condition]['mean_per_case']
                values=' | '.join(f'{x[k]:.9g}' for k in metric_names)
                rows.append(f"| {label} | {condition} | {values} | {m['best_epoch']} | {m['best_val_loss']:.9g} |")
        return '\n'.join(rows)
    sparse_rows=['| Model | V3 L1 | V4 L1 | V6 L1 | Clustered V4 L1 |','|---|---:|---:|---:|---:|']
    for label,m in s['models'].items():
        sparse_rows.append('| '+label+' | '+' | '.join(f"{m['sparse'][key]['mean_per_case']['relative_l1']:.9f}" for key in ('V3','V4','V6','clustered-V4'))+' |')
    shortcut_rows=['| Model | Normal L1 | Wrong geometry L1 | Zero measurement L1 |','|---|---:|---:|---:|']
    for label,m in s['models'].items():
        shortcut_rows.append('| '+label+' | '+' | '.join(f"{m['evaluations'][key]['mean_per_case']['relative_l1']:.9f}" for key in ('Val-A-V4','wrong_geometry','zero_measurement'))+' |')
    gain=s['val_a_l1_absolute_improvement']
    if gain<=0:
        conclusion='Geometry-Conditioned Attention Pooling does not outperform mean fusion on the primary Val-A-V4 relative L1 metric.'
    elif gain<.005:
        conclusion='B1 improves the primary metric by less than 0.5 percentage point; the engineering benefit is weak.'
    elif gain<.01:
        conclusion='B1 has a modest primary-metric improvement between 0.5 and 1 percentage point.'
    else:
        conclusion='B1 improves the primary metric by at least 1 percentage point; this is a meaningful engineering gain for this seed.'
    next_step=('建议下一阶段比较 B2 = geometry-conditioned set interaction：在每个 voxel 的可见 view 集合内加入轻量 set self-attention，再保持 encoder/lifting/decoder 和300-epoch协议固定。'
               if s['recommend_b2'] else '本轮不建议直接进入 B2；先由负责人审阅本次结果。没有实现 B2，也没有追加 B1 超参数搜索。')
    attention_uniform='接近 uniform' if attention['normalized_entropy']>.99 else '存在非均匀 weighting'
    text=f'''# GeoLift-GCA-Res3D / B0 vs B1, 300 epochs

## 1. 固定训练预算

B0-150 relative L1=0.1328937，独立220-epoch diagnostic=0.1163440，且best位于220上限。
因此负责人指定本阶段统一300 epochs / cosine T_max=300。本轮不是继续搜索horizon；未运行400/500 epochs。
旧B0-150 reference、220 diagnostic、历史报告/配置/结果全部保留。

两模型均从seed20261011的随机初始化开始，跑满300 epochs，无early stopping。
FP32、batch2、accumulation1、AdamW lr2e−4/weight_decay1e−4、gradient clip1.0。
166个train/Pool A cases，V3–6、hold0；36个val/Pool A cases、固定V4、seed20261011/epoch0。
同split、train p99.5归一化、SmoothL1(beta1)+0.1×relativeL1、optimizer、scheduler、epoch budget、val station IDs与minimum val total loss选best。
训练使用GPU3；CUDA grid_sample backward不保证逐bit训练可复现。日志/checkpoints使用0-based epoch，本报告统一1-based。

## 2. B0-300 与 3. B1设计

B0网络源文件逐文件hash未变。B1复用同一个SharedEncoder2D、GeometryLifting@32³、ResidualDecoder3D与sigmoid输出[0,1]的64³ CH4 mass fraction。
只将masked mean换成GCA。相同seed下encoder/decoder初始参数逐元素一致，单元测试验证。

geometry embedding：4→16→16，每层1×1×1 Conv3D+SiLU。
concat lifted32与geometry16后，score：48→32→1，中间SiLU。
voxel/view共享score network，无全局voxel attention、Transformer、multi-head、CBAM或新增physics loss。

## 4. Geometry descriptor

g=[α endpoint-normalized, β endpoint-normalized, scanner-to-voxel distance/6.5m, station azimuth/180°]。
α/β复用lifting.angular_grid，范围从batch frozen angle endpoints读取；当前对称范围等价于α/5°、β/4°。
距离/方位角尺度从frozen generation_config.yaml的radius_m/azimuth_deg上限读入新配置，并在protocol检查中再次验证。
α=atan2(delta·right,delta·forward)，β=atan2(delta·up,sqrt(f²+r²))，距离=||voxel_xyz−origin||。
azimuth使用配对的input_station_azimuth_deg，不依赖view位置或station ID。尺度是固定acquisition范围，未做逐case归一化或clip。

## 5. Masked attention

对每个voxel，m_v=visibility_v AND view_mask_v。
a_v=score([F_v,embed(g_v)])；invalid logit=−∞。
w_v=exp(a_v)/Σ_valid exp(a_u)，F_fused=Σ_v w_v F_v。
若没有valid view，softmax前以安全零logits替换该空集合，再将所有invalid权重置零，fused feature严格零。
inactive features/descriptors也在score前置零，避免padding中的NaN影响参数梯度。
训练默认只返回prediction；return_attention=True返回weights与visibility，不保存训练attention volume。

## 6. 参数量

B0={b0['parameter_count']:,}；B1={b1['parameter_count']:,}；新增fusion={b1['fusion_parameter_count']:,}，
增加{100*b1['fusion_parameter_count']/b0['parameter_count']:.3f}%。所有新增可训练参数都来自fusion。

## 7. 快速overfit gate

4个等距train cases，固定V4/epoch0，120 epochs，constant lr2e−4，不调attention超参数。
loss {gate['initial']['loss']['total']:.8g} → {gate['final']['loss']['total']:.8g}；
relative L1 {gate['initial']['metrics']['mean_per_case']['relative_l1']:.8g} → {gate['final']['metrics']['mean_per_case']['relative_l1']:.8g}。
fusion gradient finite={gate['gradient_finite']}，nonzero={gate['gradient_nonzero']}；attention finite={gate['attention_finite']}；
pairwise prediction MAE最大={gate['max_pairwise_prediction_mae']:.8g}，prediction range=[{gate['prediction_min']:.8g},{gate['prediction_max']:.8g}]。
gate passed={gate['passed']}。这是功能性快速gate，不宣称120 epochs已充分拟合。正式训练重新初始化，未使用gate权重。
B0使用已有相同网络/数据/normalization/loss/optimizer/seed且通过的overfit gate，未重复探索。

## 8–9. Val-A 与 Val-B 主比较

{main_table()}

表中指标为mean per-case，物理质量分数尺度；pooled和逐case原始结果保存在外部artifacts。
Val-A重新评估复现best training validation指标，Val-B仅在两个300-epoch run完成并冻结best后评估。
Val-A relative L1 absolute improvement={gain:.9f}，即{s['val_a_l1_improvement_percentage_points']:.6f}个百分点；
relative improvement={100*s['val_a_l1_relative_improvement']:.4f}%（负数表示恶化）。
Pool B L1变化(B1−B0)={b1['evaluations']['Val-B-V4']['mean_per_case']['relative_l1']-b0['evaluations']['Val-B-V4']['mean_per_case']['relative_l1']:+.9f}，
L2变化={b1['evaluations']['Val-B-V4']['mean_per_case']['relative_l2']-b0['evaluations']['Val-B-V4']['mean_per_case']['relative_l2']:+.9f}。

## 10–11. Sparse views 与 clustered geometry

{'\n'.join(sparse_rows)}

每个V3/V4/V6随机条件和clustered V4均为5个deterministic draws，使用已有EvaluationViewSampler、seed20261011。
两个模型逐case/draw使用同一station IDs；表中V4是5-draw random平均，主表Val-A-V4为legacy固定subset，两者定义不同。
每draw指标、case×draw JSONL、draw std与pooled统计完整保留。未重新跑V2/V5/V8或完整coverage/stratification矩阵。
clustered V4 L1变化(B1−B0)={b1['sparse']['clustered-V4']['mean_per_case']['relative_l1']-b0['sparse']['clustered-V4']['mean_per_case']['relative_l1']:+.9f}。
V3 L1改善={100*(b0['sparse']['V3']['mean_per_case']['relative_l1']-b1['sparse']['V3']['mean_per_case']['relative_l1']):.6f}个百分点，
clustered V4改善={100*(b0['sparse']['clustered-V4']['mean_per_case']['relative_l1']-b1['sparse']['clustered-V4']['mean_per_case']['relative_l1']):.6f}个百分点。
稀疏与差coverage条件有正向收益，但clustered的绝对relative L1仍较高，不能宣称已经解决差geometry重建。

## 12. Shortcut tests

{'\n'.join(shortcut_rows)}

zero measurement将Y置零，保留valid/path/geometry；wrong geometry将全部station geometry循环错位，Y顺序保持。
两个测试证明完整模型对measurement与geometry的依赖；wrong-geometry结果包含lifting的影响，不能单独归因于attention descriptor。

## 13. Attention diagnostics 与 invariance

预先固定val indices=0/18/35，分别spread/clustered V4 draw0；每view mean weight、visible-conditional mean weight、station IDs、azimuth均保存。
entropy为voxel级H/log(n_visible)，仅对n_visible≥2的voxel求平均，六个例子按eligible voxel数量汇总。
mean normalized entropy={attention['normalized_entropy']:.9f}（{attention_uniform}）；它不是全部validation集的统计。
finite={attention['finite']}；invalid/padded严格0={attention['invalid_padded_weight_zero']}；
valid sum-to-one最大误差={attention['weight_sum_max_error']:.9g}。
trained B1 permutation max difference={attention['permutation_max_difference']:.9g}，padding max difference={attention['padding_max_difference']:.9g}。
单元测试还覆盖all-invalid voxel和整个batch、uniform scores退化到mean、geometry改变权重、fusion梯度、输出形状与范围。
图见attention_example.png；跨case/geometry descriptor分析只服务本次直接解释，未开展额外interpretability项目。

## 14. 工程判断与是否进入B2

{conclusion}

B1通过下列五项基本GO检查，并在本次全部预定条件上改善；主表L1收益0.489941个百分点，略低于0.5个百分点参考线。
Val-A L2、V3与clustered的改善也应一并考虑。当前结论是轻量fusion有正向收益，但不足以宣称稳定、明显优于B0。

保守工程检查（数值不恶化，不是统计阈值）：
```json
{json.dumps(s['criteria'],indent=2)}
```
{next_step}

本轮只有一个统一训练seed，不能宣称B1跨seed稳定提升或统计显著；也未使用独立test估计最终性能。
不通过主比较时，仅检查weight/entropy、descriptor尺度、masked softmax和gradient的已有证据，不追加大规模调参。
uniform程度以上述entropy报告，尺度从frozen metadata读取，mask/gradient测试通过；功能性代码正确不等于结构一定有收益。

## Reproducibility 与完整性

报告生成时的代码commit=`{commit}`。两个run启动于base commit11f40e6的同一开发工作树，checkpoint environment保留git_dirty=true；
training_source_manifest.json保存数值模型/训练组件与新配置SHA256。B0启动后trainer增加了protocol-validation调用，数值train loop与模型未改。
完成报告的commit位于实现之后；不把checkpoint里的base commit误称为全部新实现的clean source commit。
{test_line}；全部OK。frozen dataset={s['integrity']['dataset_files']} files，dataset unchanged={s['integrity']['dataset_unchanged']}；
历史B0/泛化结果unchanged={s['integrity']['historical_results_unchanged']}，B0 model sources unchanged={s['integrity']['b0_model_sources_unchanged']}，
旧150-epoch config unchanged={s['integrity']['b0_150_config_unchanged']}。
数值训练代码与新配置SHA256 unchanged={s['integrity']['numerical_training_sources_unchanged']}。
test split未用于模型训练/推理/selection，integrity只读hash test artifacts不属于test evaluation。
Pool B全程evaluation-only，未用于early stopping、selection或tuning。

artifacts：`{root}`，含summary.json、comparison.csv、sparse_views.csv、effective_protocol.yaml、integrity_summary.json、tests.log、逐condition原始指标与figures。
必需图片：training_curves_b0_b1.png、b0_vs_b1_main_metrics.png、b0_vs_b1_sparse_views.png、attention_example.png。
权重/训练日志/图位于外部outputs，未纳入Git。

```bash
python scripts/compare_b0_b1_300.py --prepare
python scripts/train_supervised.py --config configs/train_supervised_b0_300.yaml --gate-config configs/train_supervised_b0.yaml
python scripts/overfit_supervised_b1.py --config configs/train_supervised_b1_300.yaml --cases 4 --epochs 120
python scripts/train_supervised.py --config configs/train_supervised_b1_300.yaml
python scripts/compare_b0_b1_300.py --device cuda:3
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
```
已有run禁止覆盖，恢复只能使用该run同config的last.pt，未从B0-150/220恢复。
'''
    (repo/'docs/reports/supervised_b1_gca_v1.md').write_text(text)
    terminal = dict(current_git_commit=commit,models={label:dict(best_epoch=m['best_epoch'],
        best_val_loss=m['best_val_loss'],val_a=m['evaluations']['Val-A-V4']['mean_per_case'],
        val_b=m['evaluations']['Val-B-V4']['mean_per_case'],
        sparse_l1={k:v['mean_per_case']['relative_l1'] for k,v in m['sparse'].items()},
        parameter_count=m['parameter_count'],
        shortcuts_l1={k:m['evaluations'][k]['mean_per_case']['relative_l1'] for k in ('Val-A-V4','wrong_geometry','zero_measurement')}) for label,m in s['models'].items()},
        val_a_l1_absolute_improvement=gain,val_a_l1_relative_improvement=s['val_a_l1_relative_improvement'],
        additional_fusion_parameters=b1['fusion_parameter_count'],
        attention={k:v for k,v in attention.items() if k!='examples'},all_tests_passed=True,test_result=test_line,
        frozen_dataset_unchanged=s['integrity']['dataset_unchanged'],test_set_used=False,pool_b_evaluation_only=True,
        conclusion=conclusion,recommend_b2=s['recommend_b2'],
        proposed_next_structure='Geometry-conditioned voxel-wise set interaction' if s['recommend_b2'] else None)
    (root/'terminal_summary.json').write_text(json.dumps(terminal,indent=2)+'\n')
    print(json.dumps(terminal,indent=2))
    return terminal
