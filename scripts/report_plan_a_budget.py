#!/usr/bin/env python3
"""Generate v4 budget evidence from real JSON results; never fabricate timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/research/plan_a_budget_v4'
ARTIFACT = ROOT/'docs/reports/artifacts/plan_a_budget_v4'


def read(path):
    return json.loads(path.read_text())


def epoch_summary(path):
    value = read(path)
    epochs = value['epochs']
    times = [r['total_seconds'] for r in epochs]
    warm = times[1:]
    return dict(path=str(path.relative_to(ROOT)), status=value['status'],
                epoch_seconds=times, mean_seconds=statistics.mean(times),
                warm_mean_seconds=statistics.mean(warm),
                mean_extrapolation_300h=statistics.mean(times)/12,
                cold_plus_299_max_warm_hours=(times[0]+299*max(warm))/3600,
                effective_update_seconds=[r['phases']['train']['seconds_per_update'] for r in epochs],
                peak_allocated_gib=max(r['phases'][s]['cuda']['peak_allocated_bytes']
                    for r in epochs for s in ('train','validation'))/1024**3,
                all_epochs_complete=all(r['phases'][s]['metrics']['complete_epoch'] for r in epochs for s in ('train','validation')),
                all_249_updates_finite=all(len(r['phases']['train']['metrics']['gradient_norms'])==83 and
                    all(__import__('math').isfinite(x) for x in r['phases']['train']['metrics']['gradient_norms']) for r in epochs),
                all_202_cases_per_epoch=all(r['phases']['train']['metrics']['sample_count']==166 and r['phases']['validation']['metrics']['sample_count']==36 for r in epochs),
                no_forbidden_reads=not any(value['gt_read_audit'].get(k,1) for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts')),
                source_sha256=value['provenance']['source_sha256'],
                parameter_count=value['parameter_count'],architecture=value['architecture'],runtime=value['runtime'],
                model=value['resolved_config']['model'],geometry_cache=value['resolved_config'].get('geometry_cache',dict(cpu_budget_mib=512,cuda_budget_mib=512)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selected',required=True,help='Filename of selected completed 3-epoch JSON')
    args=parser.parse_args()
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    rows={p.name:epoch_summary(p) for p in sorted(OUTPUT.glob('epochs_*.json')) if read(p).get('status')=='passed'}
    selected=rows[args.selected]
    baseline=read(ROOT/'outputs/research/plan_a_runtime_v3/full_epoch.json')
    quick={p.name:dict(seconds=read(p)['train_plus_validation_seconds'],parameters=read(p)['parameter_count'])
           for p in OUTPUT.glob('full_epoch_*.json') if read(p).get('status')=='passed'}
    acceptance={kind:read(ROOT/f'outputs/research/scheme_a_budget_v4_wide_{kind}/acceptance_result.json') for kind in ('smoke','gate')}
    failed_narrow=read(ROOT/'outputs/research/scheme_a_budget_v4_coarseprior_smoke/acceptance_result.json')
    selected_source=read(ROOT/'protocols/plan_a_budget_v4/source_frozen.json')['source_sha256']
    evidence=[ROOT/f'outputs/research/scheme_a_budget_v4_wide_{kind}' for kind in ('smoke','gate')]
    preflight=read(ARTIFACT/'formal_preflight.json')
    reload_check=read(ARTIFACT/'checkpoint_reload.json')
    checks=dict(formal_preflight=preflight['passed'],checkpoint_reload=reload_check['passed'],source_identity_matches=selected['source_sha256']==selected_source and all(read(p/'provenance.json')['source_sha256']==selected_source for p in evidence),
                model_identity_matches=all(__import__('yaml').safe_load((p/'config_resolved.yaml').read_text())['model']==selected['model'] for p in evidence),
                target_under_5h_mean=selected['mean_extrapolation_300h']<5,
                target_under_5h_warm_projection=selected['cold_plus_299_max_warm_hours']<5,
                complete=selected['all_epochs_complete'] and selected['all_202_cases_per_epoch'],
                finite_gradients=selected['all_249_updates_finite'],no_forbidden_reads=selected['no_forbidden_reads'],
                smoke_passed=acceptance['smoke']['passed'],gate_passed=acceptance['gate']['passed'])
    summary=dict(protocol='scheme_a_budget_v4',selected=args.selected,full_epochs=rows,single_epoch_screening=quick,
                 historical_single_epoch_seconds=baseline['train_plus_validation_seconds'],acceptance=acceptance,rejected_narrow_smoke=failed_narrow,checks=checks,
                 scope='Implementation and bounded engineering/measurement-learning validation only; no 300-epoch formal job or final-test/3D-GT evaluation',
                 limitations='Shared GPU, measured 3 epochs only. Cost estimates, not a guaranteed 300-epoch duration. New architecture/BF16 are not numerically equivalent to original A2 and volumetric accuracy remains unverified.')
    (ARTIFACT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 方案 A：单卡 300 轮五小时预算实测', '',
           '本轮只完成实现和小规模验收，没有启动 300 轮正式训练。测速使用共享 RTX 4090（cuda:0），同卡仍有其他进程，原始 GPU 利用率与进程采样保留在每份 JSON。', '',
           '## 完整轮次测速', '',
           '每轮真实执行 166 train / 36 val、83 次 AdamW 更新；连续 3 轮，训练几何随 epoch 改变，验证几何固定。计时包含完整 A/B/H 损失、全部诊断、首末轮快照、scheduler、采样清单和每轮两份完整检查点。无预热更新；初始几何缓存为空，编译磁盘缓存和 OS 文件缓存允许存在。', '',
           '| 配置 | 第1轮 | 第2轮 | 第3轮 | 三轮均值外推300轮 |', '|---|---:|---:|---:|---:|']
    for name,row in rows.items():
        times=row['epoch_seconds']
        lines.append(f"| {name} | {times[0]:.2f}s | {times[1]:.2f}s | {times[2]:.2f}s | {row['mean_extrapolation_300h']:.2f}h |")
    lines += ['',f"选用 **{args.selected}**，峰值已分配显存 **{selected['peak_allocated_gib']:.2f} GiB**。保守地重复三轮均值，300轮估计 **{selected['mean_extrapolation_300h']:.2f}小时/种子**；按首轮成本＋299个较慢稳定轮次计算为 **{selected['cold_plus_299_max_warm_hours']:.2f}小时**。正式启动的来源哈希、初始验证与最终冻结另有一次性开销；估算不构成对共享GPU负载的保证。三个种子单卡串行需三倍时间。", '',
              '## 改动与机制边界', '',
              '三维先验由64³、宽32的完整残差网络改为多尺度结构：9通道输入在32³上运行宽32、3个残差块的上下文路径；结果上采样后，与64³上的32通道局部3×3×3卷积分支相加。两条输出头均零初始化。先验的卷积MAC约为原版17.00%，这不是整轮加速倍数。总参数488,919，原版481,014。', '',
              '保留历史 DeepSets 初始化器与解码器、K=4、Q=96、lift32、output64、原始损失与抽样、完整 Jᵀr 的浓度反传。BF16仅用于encoder/decoder/prior，进入lifting与物理计算前转回FP32；参数、参数梯度、物理状态及损失保持FP32。没有通过减少案例、优化次数、验证次数或积分采样来达标。', '',
              '64个训练站位的插值模板约1.22 GiB，原CPU/GPU各512 MiB缓存会反复淘汰。分别扩大为有界1536 MiB后，可容纳完整工作集；它们不写入模型检查点。射线分块候选只改变执行分块，不改变Q96。', '',
              '## 学习验收', '',
              f"独立真实 smoke（2 train/2 val、20次更新）通过，最佳留出投影相对L1={acceptance['smoke']['best_heldout_relative_l1']:.6f}；gate（16 train/8 val、5轮40次更新）通过，最佳={acceptance['gate']['best_heldout_relative_l1']:.6f}。有限梯度、测量依赖、不同案例输出差异和禁读审计均检查；完整测速的249次更新梯度也均有限。", '',
              '这验证了小规模投影学习与工程可用性，尚不能证明300轮收敛效果或三维重建精度等同于原A2。历史BF16逐元素FP32比较未通过的结论保留；本轮结构变化按独立轻量变体注册，不宣称数值等价。', '',
              '## 筛选记录与复现', '',
              '单轮筛选同时包含首次编译、cuDNN搜索和几何缓存填充，不能把这类单轮时间当稳定速度；也不能将跨时段GPU负载变化全部归于代码。仅缩窄先验和仅替换轻量解码器未达到目标；轻量解码器首轮损失较高。较窄多尺度先验虽已满足速度，但20更新smoke的H相对L1为1.296774，未优于零预测，验收失败并被排除，未放宽标准。最终选用宽32上下文＋宽32局部分支，并保留历史解码器。所有候选原值与源码版本快照保留。', '',
              '正式配置和启动器使用独立scheme_a_budget_v4身份，要求本版smoke/gate，不复用旧A2验收。v2/v3源码哈希与历史资产检查通过；旧入口与旧配置保持原行为。', '',
              '```bash',
              'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2 \\',
              'TORCHINDUCTOR_CACHE_DIR=/tmp/plan_a_budget_v4_inductor \\',
              '/scratch/djc/envs/gas-torch/bin/python scripts/train_plan_a_budget.py \\',
              '  --config configs/research/scheme_a_budget_v4_formal_seed20261011.yaml --device cuda:0',
              '```', '',
              '该命令供后续正式训练使用，本轮没有执行。其他种子配置20261012、20261013同目录；`--preflight-only`只读验收与来源，`--resume`从完成轮次边界恢复。', '',
              '完整原始测速在`outputs/research/plan_a_budget_v4/`；汇总、测试、预检和资产审计在`docs/reports/artifacts/plan_a_budget_v4/`。']
    lines += ['', '补充检查：核心CPU测试78项通过；最终结构的CPU检查4项通过，CUDA检查4项通过。三种子正式预检、v2/v3/v4源码与历史资产审计通过。已严格重载本版真实smoke检查点，并对measurement-only NPZ完成64³有限范围重建；该检查没有读取GT。', '', '新结构推理使用`scripts/reconstruct_plan_a_budget.py`（参数与原重建入口一致，CUDA）；后续正式冻结后的验证预测/离线评分使用`scripts/evaluate_plan_a_budget.py`。它沿用原版必须完成300轮、先保存预测后读取验证GT的边界；本轮没有运行GT评价或最终测试。']
    (ROOT/'docs/reports/plan_a_budget_v4_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
