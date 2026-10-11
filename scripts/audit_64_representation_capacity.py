#!/usr/bin/env python3
"""Train-only 64³ measurement representation Gate; no network or B3 training."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml

from geophy_set3d.data.build_utils import digest, source_snapshot, write_json, write_jsonl, save_npy
from geophy_set3d.data.projection_audit import ErrorTotals
from geophy_set3d.data.projection_pool_dataset import load_source_index
from geophy_set3d.evaluation.view_selection import coverage_descriptor
from geophy_set3d.geometry.tdlas import subset_geometry
from geophy_set3d.physics.projectors import differentiable_project
from geophy_set3d.physics.representation_capacity import (
    Y_SCALE, assert_projector_unchanged, select_train_cases, select_stations,
    load_audit_case, measurement_metrics, optimize_volume, volume_diagnostics)

REPO=Path(__file__).resolve().parents[1]
DATASET=Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
OUTPUT=Path('/data/djc/GeoPhy-Set3D/outputs/audits/representation_capacity_64_v1')
STAGES=('operator','naive64','optimized64')
SUBSETS=('fit','heldout')


def capture_inventory():
    base=OUTPUT.parents[1]
    protected_dirs=[base/'checkpoints'/name for name in
                    ('supervised_b0_300','supervised_b1_gca_300','supervised_b2_gcsi_300')]
    protected_dirs += [base/'comparisons'/name for name in ('b0_vs_b1_300','b0_b1_b2_300')]
    protected_dirs += [base/'audits/c64_projection_mismatch_v1']
    protected_code=[*(REPO/'src/geophy_set3d/models').glob('*.py'),
                    REPO/'src/geophy_set3d/physics/projectors.py',
                    REPO/'src/geophy_set3d/physics/reference_projector.cpp',
                    REPO/'src/geophy_set3d/geometry/tdlas.py',
                    *(REPO/'configs').glob('train_supervised*.yaml'),
                    *(REPO/'docs/reports').glob('supervised_b*.md')]
    return dict(dataset=source_snapshot(DATASET),
                artifacts={str(p):digest(p) for d in protected_dirs for p in d.rglob('*') if p.is_file()},
                protected_code={str(p):digest(p) for p in protected_code})


def check_integrity(before, selected):
    assert_projector_unchanged()
    result=dict(dataset_unchanged=source_snapshot(DATASET)==before['dataset'],
        dataset_files=len(before['dataset']),
        b0_b1_b2_checkpoints_and_old_artifacts_unchanged=all(digest(p)==sha for p,sha in before['artifacts'].items()),
        protected_model_config_projector_report_sources_unchanged=all(digest(p)==sha for p,sha in before['protected_code'].items()),
        selected_source_gt128_unchanged=all(digest(r['source_target_path'])==r['source_gt_sha256'] for r in selected),
        projector_source_unchanged=True,test_model_evaluation=False,pool_b_used=False)
    manifest=json.loads((OUTPUT/'implementation_manifest.json').read_text())
    result['audit_implementation_unchanged']=all(digest(REPO/p)==sha for p,sha in manifest['sha256'].items())
    result['passed']=all(value for key,value in result.items() if key.endswith('unchanged'))
    write_json(OUTPUT/'integrity_summary.json',result)
    if not result['passed']:
        raise AssertionError('Protected audit inputs changed')
    return result


def distribution(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
                max=float(np.max(values)),min=float(np.min(values)))


def summarize(cases, totals, protocol, integrity, zero=None):
    distributions={stage:{subset:{metric:distribution([r[stage][subset][metric] for r in cases])
                   for metric in ('relative_l1','relative_l2')} for subset in SUBSETS} for stage in STAGES}
    f=distributions['optimized64']['fit']['relative_l1']
    h=distributions['optimized64']['heldout']['relative_l1']
    nf=distributions['naive64']['fit']['relative_l1'];nh=distributions['naive64']['heldout']['relative_l1']
    fit_pass=max(f['mean'],f['median'])<.005
    improved=all(distributions['optimized64'][subset]['relative_l1'][stat]
                 <distributions['naive64'][subset]['relative_l1'][stat]
                 for subset in SUBSETS for stat in ('mean','median'))
    if fit_pass and max(h['mean'],h['median'])<.01 and improved:
        classification='PASS'
    elif fit_pass and max(h['mean'],h['median'])<=.015 and improved:
        classification='CONDITIONAL PASS'
    else:
        classification='FAIL'
    ratios=[r['recovered_fraction']['fit'] for r in cases if r['recovered_fraction']['fit'] is not None]
    volume_summary={key:distribution([r['volume']['optimized'][key] for r in cases])
                    for key in ('min','max','mean','active_fraction','total_concentration_proxy')}
    comparison={key:distribution([r['volume'][key] for r in cases])
                for key in ('relative_l1_vs_gt64','relative_l2_vs_gt64')}
    worst=max(cases,key=lambda r:r['optimized64']['fit']['relative_l1'])
    best=min(cases,key=lambda r:r['optimized64']['fit']['relative_l1'])
    return dict(completed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip(),
        case_count=len(cases),fit_station_count=16,heldout_station_count=16,
        case_selection=protocol['case_selection'],protocol=protocol,
        distributions=distributions,pooled_metrics={stage:{subset:totals[stage,subset].metrics() for subset in SUBSETS} for stage in STAGES},
        recovered_fraction=distribution(ratios) if ratios else None,
        best_fit_case=dict(case_id=best['case_id'],relative_l1=best['optimized64']['fit']['relative_l1']),
        worst_fit_case=dict(case_id=worst['case_id'],relative_l1=worst['optimized64']['fit']['relative_l1']),
        cases_fit_l1_above_one_percent=[r['case_id'] for r in cases if r['optimized64']['fit']['relative_l1']>.01],
        volume_summary=volume_summary,volume_vs_gt64_diagnostics=comparison,
        implausible_case_ids=[r['case_id'] for r in cases if r['volume']['measurement_compatible_but_implausible']],
        zero_init_sanity=zero,classification=classification,recommend_b3=classification!='FAIL',
        gate_criteria=dict(fit_mean_and_median_below_half_percent=fit_pass,
                          heldout_mean_and_median_below_one_percent=max(h['mean'],h['median'])<.01,
                          fit_and_heldout_mean_and_median_improved=improved),
        integrity=integrity,test_used=False,pool_b_used=False,network_trained=False,
        exact_theoretical_lower_bound=False,
        next_stage='Add supervised observed-view and held-out-view measurement-domain consistency to B2.'
                   if classification!='FAIL' else 'NO-GO for strong 64³ projection-domain constraint')


def make_figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUTPUT/'figures';directory.mkdir(exist_ok=True)
    x=np.arange(len(cases));labels=[str(r['train_index']) for r in cases]
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for i,stage in enumerate(STAGES):
        ax.bar(x+(i-1)*.25,[100*r[stage]['fit']['relative_l1'] for r in cases],.25,label=stage)
    ax.set_xticks(x,labels);ax.set(xlabel='Train manifest case index',ylabel='Fit valid-ray relative L1 (%)')
    ax.legend();fig.savefig(directory/'projection_error_before_after.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for i,(stage,subset) in enumerate((('naive64','fit'),('naive64','heldout'),('optimized64','fit'),('optimized64','heldout'))):
        ax.bar(x+(i-1.5)*.2,[100*r[stage][subset]['relative_l1'] for r in cases],.2,label=f'{stage} {subset}')
    ax.set_xticks(x,labels);ax.set(xlabel='Train manifest case index',ylabel='Valid-ray relative L1 (%)')
    ax.axhline(.5,color='gray',ls='--',lw=.7);ax.axhline(1.,color='gray',ls=':',lw=.7)
    ax.legend();fig.savefig(directory/'fit_vs_heldout.png',dpi=150);plt.close(fig)


def render_report(summary):
    cases=[json.loads(line) for line in (OUTPUT/'per_case_metrics.jsonl').read_text().splitlines()]
    table=['| Stage | Subset | L1 mean / median / max (%) | L2 mean / median / max (%) |',
           '|---|---|---:|---:|']
    for stage in STAGES:
        for subset in SUBSETS:
            values=summary['distributions'][stage][subset]
            table.append(f'| {stage} | {subset} | '+' | '.join(' / '.join(f"{100*values[m][s]:.6f}" for s in ('mean','median','max')) for m in ('relative_l1','relative_l2'))+' |')
    per_case=['| Train index | Fit naive → optimized L1 (%) | Held-out naive → optimized L1 (%) | Steps / best step | Recovery (fit) |',
              '|---:|---:|---:|---:|---:|']
    for r in cases:
        per_case.append(f"| {r['train_index']} | {100*r['naive64']['fit']['relative_l1']:.6f} → {100*r['optimized64']['fit']['relative_l1']:.6f} | "
                        f"{100*r['naive64']['heldout']['relative_l1']:.6f} → {100*r['optimized64']['heldout']['relative_l1']:.6f} | "
                        f"{r['optimization']['completed_steps']} / {r['optimization']['best_step']} | {r['recovered_fraction']['fit']:.6f} |")
    stations=json.loads((OUTPUT/'selected_stations.json').read_text())
    tests=(OUTPUT/'logs/tests.log').read_text()
    test_line=next(line for line in tests.splitlines() if line.startswith('Ran '))
    if not tests.rstrip().endswith('OK'):
        raise AssertionError('Audit report requires passing old and new tests')
    summary['all_tests_passed']=True;summary['test_result']=test_line
    conclusion={'PASS':'64³ representation is sufficiently expressive for measurement-domain consistency within the tested setup. GO B3.',
                'CONDITIONAL PASS':'GO B3 with model-error-aware projection loss. Projection consistency has non-negligible representation/model error; use a robust, moderate-weight loss and avoid exact equality.',
                'FAIL':'NO-GO for strong 64³ projection-domain constraint. This finite optimizer run does not establish a theoretical representation floor; the owner should decide tolerance or another physics representation.'}[summary['classification']]
    zero=summary['zero_init_sanity']
    zero_text=('未执行zero-init。' if zero is None else
               f"仅首个选定train case执行同一配置zero-init sanity：fit L1从{100*zero['optimization']['initial_fit_relative_l1']:.6f}%降至{100*zero['metrics']['fit']['relative_l1']:.6f}%，held-out={100*zero['metrics']['heldout']['relative_l1']:.6f}%，steps={zero['optimization']['completed_steps']}。下降={zero['fit_error_decreased']}；它不参与主Gate分类。")
    text=f'''# 64³ Representation Capacity Audit v1

## 1–2. Why this audit / current C64 mismatch

既有全train/Pool A audit显示：K96 P_diff(C_GT128)对saved K256 reference的pooled relative L1≈0.0958%、L2≈0.3157%；naive endpoint-aligned GT64的L1≈1.3993%、L2≈3.0778%。
operator mismatch、naive downsample mismatch、64³ representation capacity floor是三个不同概念。
本轮直接优化64³体积，只检验measurement representability，既不证明唯一重建，也不将有限优化结果称为exact theoretical lower bound。

## 3–5. Protocol, cases/stations, optimization

只用12 train cases，按frozen source_index的train顺序evenly-spaced indices={summary['protocol']['train_indices']}；未按测量误差挑选案例、未做复杂难度分层、无val/test或Pool B实验。
复用EvaluationViewSampler的deterministic farthest-point spread逻辑：先从64个Pool A stations选16 fit，再从剩余48个选16 held-out；所有case共用同一geometry IDs。
fit IDs={stations['fit']['station_ids']}，azimuth span={stations['fit']['coverage']['azimuth_span_deg']:.3f}°；held-out IDs={stations['heldout']['station_ids']}，span={stations['heldout']['coverage']['azimuth_span_deg']:.3f}°。
目标严格是saved projections_train.npy = P_ref(C_GT128,K256)，ppm·m；没有生成新reference或使用P_ref(C_GT64)。
GT64直接读取现有gt_volume_64.npy，GT128仅用于operator reference阶段；所有数组/source hashes核对。

C_opt=sigmoid(Z)，Z0=logit(clamp(GT64,1e−6,1−1e−6))；FP32/cuda:3，Adam lr0.01、eps1e−12，最多2000 steps。
Adam的小epsilon用于已归一化、按valid rays平均、低浓度sigmoid变量的微小梯度，不改变measurement objective；没有optimizer/lr搜索。
冻结P_diff：grid_sample、align_corners=True、midpoint K96、mass-to-mole转换、path length与valid mask，源码未修改。
loss=mean_valid SmoothL1(Y_pred/32284.894580078173,Y_ref/32284.894580078173)，beta1，仅fit rays；没有voxel supervision、TV、sparsity、mass penalty或任何网络。
200 steps内best fit relative L1改善不足1e−5 absolute即early stop；保留最低fit loss iterate，held-out从未用于优化、early stop或iterate selection。
Naive阶段使用未clamp的原GT64；sigmoid初始化的epsilon背景影响另存initial_fit_relative_l1，未混入naive结果。

## 6–8. Operator / naive64 / optimized64; fit and held-out

{'\n'.join(table)}

以上为mean/median/max per-case ratios；summary.json另外保存按原始valid-ray sums累计的pooled指标。
逐case与summary.csv完整保留raw ppm·m MAE、RMSE、bias(prediction−saved reference)、max absolute error和relative L1/L2，所有分母只含valid rays。

{'\n'.join(per_case)}

recovered_fraction=(E_naive64−E_opt64)/(E_naive64−E_operator)，以relative L1计算；fit mean={summary['recovered_fraction']['mean']:.6f}，median={summary['recovered_fraction']['median']:.6f}。
它可能超过1：有限ray拟合可以低于GT128的operator mismatch，operator结果是比较参考而非数学下界；未将recovery截断到[0,1]。
best fit case={summary['best_fit_case']}；worst fit case={summary['worst_fit_case']}；fit L1>1%的cases={summary['cases_fit_l1_above_one_percent']}。

## 9. Voxel plausibility diagnostics

active fraction固定定义为C>1e−4 mass fraction；total concentration proxy=sum(C)，只是网格proxy，不称真实气体质量。
voxel similarity只作diagnostic，不用于PASS条件或优化。以下均按12个最终体积汇总：

```json
{json.dumps(dict(volume=summary['volume_summary'],difference_vs_gt64=summary['volume_vs_gt64_diagnostics'],implausible_case_ids=summary['implausible_case_ids']),indent=2,ensure_ascii=False)}
```

补充screens：max>0.9且>5×GTmax、GT≤1e−4背景中超过25%升至>1e−4、平均邻居差分>3×GT且>1%×GTmax；这些是粗略诊断，不是可证明的物理合理性标准。
没有加入TV variant；主结果保持measurement-only，B3仍应保留supervised voxel loss。{zero_text}

## 10–11. Capacity conclusion and B3 recommendation

**{summary['classification']}**。{conclusion}

工程Gate要求fit mean和median均<0.5%，held-out均<1.0%，且两个subset的mean/median均优于其naive结果；held-out在1%–1.5%以内则为CONDITIONAL PASS。不是理论阈值或统计显著性检验。
本轮只测试12个GT64-initialized train volumes、16 fit+16 held-out stations，FP32误差远小于本Gate尺度；结果不证明所有64³体积都能精确表示GT128，也不证明网络能自动找到这些解。
GT64初始化提供很强的空间先验，少量zero-init诊断不能取代这一限定；voxel非唯一性仍存在。

下一阶段：{summary['next_stage']}
本轮没有实现B3，没有改成128³网络，也未重新生成dataset、训练B0/B1/B2或研究其它inverse methods。

## Evidence and integrity

Audit implementation commit见implementation_manifest.json；报告生成时commit=`{summary['git_commit']}`。
{test_line}，全部OK。完整性：

```json
{json.dumps(summary['integrity'],indent=2)}
```

Test未用于模型/volume优化或评估，dataset inventory只读取hash用于immutability；Pool B没有使用。
输出：`{OUTPUT}`，含protocol、selected cases/stations、per_case_metrics.jsonl、summary.json/CSV、optimized_volumes、logs与两张图。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:tests python -m unittest discover -s tests -v
python scripts/audit_64_representation_capacity.py --device cuda:3
# 已完成结果只重新生成报告：
python scripts/audit_64_representation_capacity.py --report-only
```
'''
    (REPO/'docs/reports/representation_capacity_64_v1.md').write_text(text)
    write_json(OUTPUT/'summary.json',summary)
    write_json(OUTPUT/'terminal_summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        render_report(json.loads((OUTPUT/'summary.json').read_text()));return
    assert_projector_unchanged()
    if (OUTPUT/'per_case_metrics.jsonl').exists():
        raise ValueError('Existing audit results must not be overwritten; use --report-only')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for folder in ('optimized_volumes','logs','figures'):
        (OUTPUT/folder).mkdir(exist_ok=True)
    before=capture_inventory();write_json(OUTPUT/'inventory_before.json',before)
    frozen=json.loads((OUTPUT.parents[1]/'comparisons/b0_b1_b2_300/inventory_before.json').read_text())['dataset']
    if before['dataset']!=frozen:
        raise AssertionError('Frozen dataset differs from B2 inventory')
    build=json.loads((DATASET/'dataset_build_manifest.json').read_text())
    if build['status']!='frozen' or build['reference_samples']!=256:
        raise ValueError('Expected saved frozen K256 reference')
    rows=load_source_index(DATASET);indices,selected=select_train_cases(rows)
    with np.load(DATASET/'geometry/train_geometry.npz',allow_pickle=False) as arrays:
        geometry=dict(arrays)
    if digest(DATASET/'geometry/train_geometry.npz')!=build['geometry_hashes']['train_geometry.npz']:
        raise AssertionError('Frozen geometry hash mismatch')
    statistics=json.loads((DATASET/'train_statistics.json').read_text())
    if statistics['projection_valid_rays']['p99.5']!=Y_SCALE or statistics['split']!='train':
        raise ValueError('Frozen train scale changed')
    fit,heldout=select_stations(geometry['station_azimuth_deg'])
    write_json(OUTPUT/'selected_cases.json',dict(protocol='deterministic evenly-spaced train manifest indices',indices=indices,cases=selected))
    write_json(OUTPUT/'selected_stations.json',{subset:dict(station_ids=ids.tolist(),
        station_azimuth_deg=geometry['station_azimuth_deg'][ids].tolist(),
        coverage=coverage_descriptor(geometry['station_azimuth_deg'][ids])) for subset,ids in [('fit',fit),('heldout',heldout)]})
    protocol=dict(case_count=12,case_selection='12 deterministic evenly-spaced cases in frozen train manifest order',
        train_indices=indices,fit_stations=fit.tolist(),heldout_stations=heldout.tolist(),
        geometry_pool='A only',test_used=False,pool_b_used=False,reference='saved P_ref(C_GT128,K256), ppm_m',
        projector_samples=96,volume_shape=[64]*3,parameterization='sigmoid(logit(clamp(GT64,1e-6,1-1e-6)))',
        scale_ppm_m=Y_SCALE,loss='mean_valid SmoothL1(pred/scale, saved_ref/scale), beta1',
        optimizer='Adam',lr=.01,adam_epsilon=1e-12,max_steps=2000,
        early_stop=dict(patience_steps=200,relative_l1_absolute_improvement=1e-5),
        iterate_selection='minimum fit loss; held-out never used',precision='float32',device=args.device,
        zero_init_case_indices=[indices[0]],weak_regularized_variant=False,active_voxel_threshold=1e-4)
    (OUTPUT/'effective_protocol.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
    sources=[Path(__file__),REPO/'src/geophy_set3d/physics/representation_capacity.py']
    write_json(OUTPUT/'implementation_manifest.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip(),
        sha256={str(p.relative_to(REPO)):digest(p) for p in sources}))
    torch.set_num_threads(2);torch.manual_seed(20261011)
    totals={(stage,subset):ErrorTotals() for stage in STAGES for subset in SUBSETS}
    station_ids=np.concatenate([fit,heldout]);all_geometry=subset_geometry(geometry,station_ids)
    fit_geometry=subset_geometry(geometry,fit)
    def project(volume):
        with torch.no_grad():
            tensor=torch.tensor(np.array(volume,copy=True),dtype=torch.float32,device=args.device)
            return differentiable_project(tensor,all_geometry,samples=96).cpu().numpy()
    valid=all_geometry['valid_ray'];cases=[];started=time.monotonic()
    for index,row in zip(indices,selected):
        reference,gt64,gt128=load_audit_case(DATASET,row)
        if reference.shape!=geometry['valid_ray'].shape or np.any(reference[~geometry['valid_ray']]!=0):
            raise ValueError('Frozen measurement geometry/mask mismatch')
        ref=np.array(reference[station_ids],copy=True)
        predictions={'operator':project(gt128),'naive64':project(gt64)}
        optimized,optimization=optimize_volume(gt64,fit_geometry,ref[:16],device=args.device,
            trace_path=OUTPUT/'logs'/f'{index:03d}_gt64_init.jsonl',label=f'train[{index}]')
        save_npy(OUTPUT/'optimized_volumes'/f'{index:03d}.npy',optimized)
        predictions['optimized64']=project(optimized)
        result=dict(case_id=row['case_id'],train_index=index,split='train',condition_meta=row['condition_meta'],
                    optimization=optimization,volume=volume_diagnostics(optimized,gt64),recovered_fraction={})
        for stage,prediction in predictions.items():
            result[stage]={}
            for subset,sl in [('fit',slice(0,16)),('heldout',slice(16,32))]:
                result[stage][subset]=measurement_metrics(prediction[sl],ref[sl],valid[sl])
                totals[stage,subset].update(prediction[sl],ref[sl],valid[sl])
        for subset in SUBSETS:
            op,naive,opt=[result[stage][subset]['relative_l1'] for stage in STAGES]
            result['recovered_fraction'][subset]=(naive-opt)/(naive-op) if naive>op else None
        cases.append(result);write_jsonl(OUTPUT/'per_case_metrics.jsonl',cases)
        print(f"Completed {len(cases)}/12 train[{index}] fit={result['optimized64']['fit']['relative_l1']:.6%} "
              f"heldout={result['optimized64']['heldout']['relative_l1']:.6%} steps={optimization['completed_steps']} elapsed={time.monotonic()-started:.1f}s",flush=True)
    reference,gt64,_=load_audit_case(DATASET,selected[0]);ref=np.array(reference[station_ids],copy=True)
    volume,optimization=optimize_volume(np.zeros_like(gt64),fit_geometry,ref[:16],device=args.device,
        trace_path=OUTPUT/'logs/zero_init_first_case.jsonl',label='zero-init')
    prediction=project(volume)
    zero=dict(case_id=selected[0]['case_id'],optimization=optimization,
              metrics={subset:measurement_metrics(prediction[sl],ref[sl],valid[sl])
                       for subset,sl in [('fit',slice(0,16)),('heldout',slice(16,32))]},
              volume=volume_diagnostics(volume,gt64))
    zero['fit_error_decreased']=zero['metrics']['fit']['relative_l1']<optimization['initial_fit_relative_l1']
    save_npy(OUTPUT/'optimized_volumes/zero_init_first_case.npy',volume)
    write_json(OUTPUT/'zero_init_sanity.json',zero)
    integrity=check_integrity(before,selected)
    summary=summarize(cases,totals,protocol,integrity,zero)
    write_json(OUTPUT/'summary.json',summary)
    csv_rows=[dict(case_id=r['case_id'],train_index=r['train_index'],stage=stage,subset=subset,**r[stage][subset])
              for r in cases for stage in STAGES for subset in SUBSETS]
    with (OUTPUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    make_figures(cases);render_report(summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
