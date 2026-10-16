#!/usr/bin/env python3
"""Reproject sealed MRGR C0/C1 fields; audit difficult-layout data consistency."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.projection_losses import project_measurements
from geophy_set3d.training.utils import move_batch

SOURCE = ROOT/'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1'
DEFAULT_OUT = ROOT/'outputs/research/mrgr_difficult_projection_audit_v1'


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def jsonl(path, values): Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n' for v in values))


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
        p90=float(np.quantile(a, .9)), maximum=float(a.max()), minimum=float(a.min()))


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = b-a
    rng = np.random.default_rng(20261017)
    ci = np.quantile(delta[rng.integers(0, len(a), (10000, len(a)))].mean(1), [.025, .975])
    return dict(mean_C0=float(a.mean()), mean_C1=float(b.mean()), mean_delta_C1_minus_C0=float(delta.mean()),
        relative_reduction_of_means=float(1-b.mean()/a.mean()), delta_ci95=ci.tolist(),
        improved=int((delta < -1e-8).sum()), tied=int((abs(delta)<=1e-8).sum()), worsened=int((delta>1e-8).sum()))


def summarize(records, cases, condition, draws):
    # Always aggregate repeated layouts within case before distribution/interval calculations.
    selected = [r for r in records if r['condition']==condition and r['draw'] in draws]
    values = {}
    for stage in ('C0','C1'):
        values[stage] = {}
        for case in cases:
            rr = [r for r in selected if r['case_id']==case and r['stage']==stage]
            assert len(rr)==len(draws)
            values[stage][case] = {key: float(np.mean([r[key] for r in rr])) for key in (
                'observed_l1','observed_l2','heldout_l1','heldout_l2','volume_l1','mae_ppm_m','reference_mean_ppm_m')}
    result = dict(condition=condition, draws=draws, cases=len(cases), case_stage_values=values)
    for stage in values:
        result[stage] = {key: distribution([values[stage][c][key] for c in cases])
                        for key in values[stage][cases[0]]}
        result[stage]['observed_l1_exceedance'] = {str(t):sum(values[stage][c]['observed_l1']>t for c in cases)
                                                  for t in (.05,.1,.2)}
    result['feedback'] = {key: paired([values['C0'][c][key] for c in cases], [values['C1'][c][key] for c in cases])
                          for key in ('observed_l1','heldout_l1','volume_l1')}
    result['joint_input_and_volume_changes'] = {
        f'input_{input_direction}_volume_{volume_direction}':sum(
            (values['C1'][c]['observed_l1']-values['C0'][c]['observed_l1'] < -1e-8 if input_direction=='improves'
             else values['C1'][c]['observed_l1']-values['C0'][c]['observed_l1'] > 1e-8) and
            (values['C1'][c]['volume_l1']-values['C0'][c]['volume_l1'] < -1e-8 if volume_direction=='improves'
             else values['C1'][c]['volume_l1']-values['C0'][c]['volume_l1'] > 1e-8)
            for c in cases)
        for input_direction in ('improves','worsens') for volume_direction in ('improves','worsens')}
    return result


def report(out, summary, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panel_lookup = {(p['condition'],tuple(p['draws'])):p for p in summary['panels']}
    ordinary = panel_lookup['Val-A-V4',(0,)]
    hard0 = panel_lookup['clustered-V4',(0,)]
    hard5 = panel_lookup['clustered-V4',tuple(range(5))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for stage,color in [('C0','tab:blue'),('C1','tab:orange')]:
        x = np.arange(6)
        pp = [ordinary]+[panel_lookup['clustered-V4',(d,)] for d in range(5)]
        y = [100*p[stage]['observed_l1']['mean'] for p in pp]
        axes[0].plot(x,y,'o-',label=stage,color=color)
    axes[0].set_xticks(np.arange(6),['ordinary','cluster0','cluster1','cluster2','cluster3','cluster4'],rotation=25)
    axes[0].set(ylabel='Input projection relative L1 (%)',title='36-case means before/after feedback')
    axes[0].legend()
    for condition,draw,color,label in [('Val-A-V4',0,'tab:green','Ordinary V4'),('clustered-V4',0,'tab:red','Cluster draw0')]:
        pp = panel_lookup[condition,(draw,)]['case_stage_values']
        axes[1].scatter([100*v['observed_l1'] for v in pp['C0'].values()],
                        [100*v['observed_l1'] for v in pp['C1'].values()],s=20,c=color,label=label)
    limit=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,limit],[0,limit],'k--',lw=.8)
    axes[1].set(xlabel='C0 input L1 (%)',ylabel='C1 input L1 (%)',title='Below diagonal: input error improves')
    axes[1].legend(fontsize=8)
    for condition,draw,color,label in [('Val-A-V4',0,'tab:green','Ordinary V4'),('clustered-V4',0,'tab:red','Cluster draw0')]:
        pp=panel_lookup[condition,(draw,)]['case_stage_values']['C1']
        axes[2].scatter([100*v['observed_l1'] for v in pp.values()], [100*v['volume_l1'] for v in pp.values()],
                        s=20,c=color,label=label)
    axes[2].set(xlabel='C1 input projection L1 (%)',ylabel='C1 3D relative L1 (%)',title='Projection fit and 3D accuracy differ')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'projection_audit.png',dpi=160);plt.close(fig)
    rel='../../'+str(out.relative_to(ROOT))
    f=hard0['feedback']['observed_l1']
    lines=['# 多尺度 MRGR：困难布局输入投影误差核查','',
        f'完成时间：{summary["completed_at"]}。36个已用于开发的羽流验证案例，冻结第278轮多尺度MRGR；复用全部17组布局、612次重建的已封存C0/C1三维场。没有重新运行网络或优化模型。','',
        f'**困难布局确实仍有明显输入投影残差。** 最窄聚簇draw0的C1输入投影L1为{100*f["mean_C1"]:.2f}%，普通V4为{100*ordinary["C1"]["observed_l1"]["mean"]:.2f}%；聚簇五布局先在每例内平均后为{100*hard5["C1"]["observed_l1"]["mean"]:.2f}%。','',
        f'**反馈有效，但不足以解决困难布局的数据一致性。** 最窄布局C0→C1输入L1从{100*f["mean_C0"]:.2f}%降至{100*f["mean_C1"]:.2f}%，均值相对降低{100*f["relative_reduction_of_means"]:.2f}%，逐例改善/持平/变差为{f["improved"]}/{f["tied"]}/{f["worsened"]}。','',
        '## 1. 重投影口径','',
        '从封存NPZ分别读取C0/C1，使用原FP32可微投影器、每条射线96点、原站位顺序、有效射线掩码和原始ppm·m测量重算。无须打开GT64/GT128数组或加载网络权重。三维L1仅引用已有冻结评分JSON，用于检查投影和三维变化是否一致；没有新算三维评分。','',
        '每例相对L1为Σ_valid|P(C)−Y|/Σ_valid|Y|，先按例计算再对36例等权平均。P90/最大值对应案例分布；五布局汇总先在每例内平均，180次重建不视为180个独立案例。报告中的5%/10%/20%仅为描述性分层，不是预注册验收阈值。额外H4不输入网络，也不等同训练日志固定验证H面板。','',
        f'全部612例C1重算与原始封存指标的最大绝对差为{summary["max_C1_archived_difference"]:.3g}；读取审计GT尝试次数为{summary["gt_audit"]["gt_open_attempts"]}。','',
        '## 2. 反馈前后的输入投影L1','',
        '| 布局 | draw | 方位跨度范围 | C0均值 | C1均值 | C1中位数 | C1 P90 | C1最大 | 输入误差改善/平/变差 | C1额外H4 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for condition,draws in [('Val-A-V4',[0]),('spread-V4',[0]),('V3',list(range(5))),('V6',list(range(5)))]+[('clustered-V4',[d]) for d in range(5)]+[('clustered-V4',list(range(5)))]:
        p=panel_lookup[condition,tuple(draws)];a=p['C0']['observed_l1'];b=p['C1']['observed_l1'];fb=p['feedback']['observed_l1']
        rr=[r for r in records if r['condition']==condition and r['draw'] in draws and r['stage']=='C1']
        spans=[r['azimuth_span_deg'] for r in rr]
        lines.append(f'| {condition} | {draws} | {min(spans):.2f}–{max(spans):.2f}° | {100*a["mean"]:.2f}% | {100*b["mean"]:.2f}% | {100*b["median"]:.2f}% | {100*b["p90"]:.2f}% | {100*b["maximum"]:.2f}% | {fb["improved"]}/{fb["tied"]}/{fb["worsened"]} | {100*p["C1"]["heldout_l1"]["mean"]:.2f}% |')
    lines += ['',f'![反馈、案例分布及三维关系]({rel}/projection_audit.png)','',
        '## 3. 尾部、反馈稳定性与三维取舍','',
        '| 面板 | C1>5%例数 | C1>10%例数 | C1>20%例数 | Δ输入L1的95%区间（百分点） | 输入改善但三维变差 |',
        '|---|---:|---:|---:|---|---:|']
    for p in [ordinary,hard0,hard5]:
        e=p['C1']['observed_l1_exceedance'];ci=p['feedback']['observed_l1']['delta_ci95']
        lines.append(f'| {p["condition"]} {p["draws"]} | {e["0.05"]}/36 | {e["0.1"]}/36 | {e["0.2"]}/36 | [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {p["joint_input_and_volume_changes"]["input_improves_volume_worsens"]}/36 |')
    lines += ['', 'Δ为C1−C0；案例bootstrap为36例、10,000次、seed20261017。区间只反映案例变化，不能代表训练种子或独立新数据的不确定性，未做多重检验校正。C0/C1为同一联合训练模型轨迹，不是重训消融。','',
        '下表取最窄聚簇draw0中最终输入误差最高的5个案例，不删除失败样本。','',
        '| case_id | C0输入L1 | C1输入L1 | C1输入MAE ppm·m | 有效射线平均测量 ppm·m | C1三维L1 |',
        '|---|---:|---:|---:|---:|---:|']
    lookup={(r['condition'],r['draw'],r['case_id'],r['stage']):r for r in records}
    worst=sorted([r for r in records if r['condition']=='clustered-V4' and r['draw']==0 and r['stage']=='C1'],key=lambda r:r['observed_l1'],reverse=True)[:5]
    for r in worst:
        c0=lookup[r['condition'],r['draw'],r['case_id'],'C0']
        lines.append(f'| {r["case_id"]} | {100*c0["observed_l1"]:.2f}% | {100*r["observed_l1"]:.2f}% | {r["mae_ppm_m"]:.1f} | {r["reference_mean_ppm_m"]:.1f} | {100*r["volume_l1"]:.2f}% |')
    lines += ['', '## 4. 历史模型的相同输入面板','',
        '仅比较原历史C1已保存的observed指标；逐例核对输入站位、顺序和有效射线数相同。历史额外H口径不用于比较。','',
        '| 面板 | 历史C1输入L1 | 多尺度C1输入L1 |', '|---|---:|---:|']
    for p in [ordinary,hard0,hard5]:
        rr=[r for r in records if r['condition']==p['condition'] and r['draw'] in p['draws'] and r['stage']=='C1']
        lines.append(f'| {p["condition"]} {p["draws"]} | {100*np.mean([r["historical_observed_l1"] for r in rr]):.2f}% | {100*p["C1"]["observed_l1"]["mean"]:.2f}% |')
    lines += ['', '## 5. 对方法与论文表述的含义','',
        '这些结果支持：多尺度MRGR在普通布局能达到较低投影误差，在聚簇困难布局仍存在测量拟合不足；残差反馈有纠错作用，但不能称为充分满足输入投影约束。有限角度会使三维解不唯一，但不唯一性本身不能解释所有已观测射线上的高残差，不能把困难三维误差全部归为未观测方向。','',
        '同时，输入投影残差降低不等于三维结构必然改善；应同时验收输入残差、额外视角和三维指标。后续若研究更严格的数据一致性更新，需要检验投影改善是否以峰、质心或三维误差为代价，不能只以输入残差选方案。','',
        '本轮未读取GT体场，未计算P96(GT64)相对原128³测量的离散基线，也未求解每个布局的数据拟合下限。因此，不能把全部剩余误差都定为网络可消除误差，或据此确定需要多少物理更新。结论适用于当前单种子开发验证集，最终test未读取。','',
        f'产物：[汇总]({rel}/summary.json)、[逐例C0/C1投影指标]({rel}/per_case_projection_metrics.jsonl)、[独立核验]({rel}/independent_verification.json)、[核查脚本](../../scripts/audit_mrgr_difficult_projection_v1.py)。']
    path=ROOT/'docs/reports/mrgr_difficult_projection_audit_zh.md'
    path.write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registration=read(SOURCE/'registration.json');boundary=read(SOURCE/'predictions_complete.json')
    assert sha(SOURCE/'registration.json')==boundary['registration_sha256']
    assert sha(SOURCE/'prediction_manifest.jsonl')==boundary['manifest_sha256']
    manifest=rows(SOURCE/'prediction_manifest.jsonl');assert len(manifest)==612
    old_scores={(r['condition'],r['draw'],r['case_id'],r['stage']):r['volume']['relative_l1']
                for r in rows(SOURCE/'per_case_metrics.jsonl')}
    settings={(s['condition'],s['draw']):s for s in registration['settings']}
    history={}
    for key,s in settings.items():
        path=Path(s['references']['History'])
        assert sha(path)==registration['evidence_sha256'][str(path)]
        history[key]={r['case_id']:r for r in rows(path)}
    config=registration['config'];data=Path(config['dataset']['root'])
    for name in ('geometry/train_geometry.npz','train_statistics.json'):
        assert sha(data/name)==registration['evidence_sha256'][str(data/name)]
    write(out/'registration.json',dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source=str(SOURCE),script_sha256=sha(Path(__file__)),checkpoint_sha256=registration['checkpoint_sha256'],
        evidence_sha256={str(SOURCE/name):sha(SOURCE/name) for name in ('registration.json','predictions_complete.json','prediction_manifest.jsonl','per_case_metrics.jsonl','summary.json')},
        scope='all 612 saved reconstructions, C0/C1; original Pool A train geometry and validation measurements',
        network_inference=False, optimization=False, new_volumetric_gt_access=False,
        projector='original FP32, 96 midpoint samples; TF32 disabled; original valid rays',
        thresholds_descriptive=[.05,.1,.2],aggregation='per case, then mean draws within each case for multi-layout panels'))
    records=[];max_difference=0.;station_records=[];cached={}
    with GTReadGuard(data,SOURCE.parent.parent/'checkpoints') as guard:
        ds=MeasurementOnlyPoolDataset(data,split='val',geometry_pool='train',min_input_views=3,max_input_views=6,
            hold_views=0,normalization=config['normalization']['method'],statistics_file=config['normalization']['statistics_file'])
        index={r['case_id']:i for i,r in enumerate(ds.records)}
        for i,r in enumerate(manifest):
            condition,draw,case=r['condition'],r['draw'],r['case_id'];setting=settings[condition,draw]
            assert r['input_ids']==setting['input_ids'][case] and r['heldout_ids']==setting['heldout_ids'][case]
            assert not set(r['input_ids'])&set(r['heldout_ids'])
            historical=history[condition,draw][case]
            assert historical['selected_station_ids']==r['input_ids']
            path=SOURCE/r['prediction'];assert sha(path)==r['sha256']
            if case not in cached:
                row,measurement=ds._projections(index[case])
                measure_path=data/row['train_projection_path']
                assert sha(measure_path)==registration['evidence_sha256'][str(measure_path)]
                cached[case]=(row,measurement)
            row,measurement=cached[case]
            sample=ds._measurement_sample(row,measurement,np.array(r['input_ids'],dtype=np.int64),np.array(r['heldout_ids'],dtype=np.int64))
            batch=move_batch(projection_pool_collate([sample]),args.device)
            az=ds.geometry['station_azimuth_deg'][r['input_ids']]
            with np.load(path,allow_pickle=False) as fields:
                for stage in ('C0','C1'):
                    volume=torch.from_numpy(fields[stage].copy())[None,None].to(args.device)
                    result=dict(condition=condition,draw=draw,case_id=case,stage=stage,input_ids=r['input_ids'],
                        heldout_ids=r['heldout_ids'],azimuth_span_deg=float(az.max()-az.min()),
                        volume_l1=old_scores[condition,draw,case,stage],
                        historical_observed_l1=historical['observed']['relative_l1'])
                    with torch.no_grad():
                        for prefix,label in [('input','observed'),('hold','heldout')]:
                            p,y,valid=project_measurements(volume,batch,prefix)[0]
                            metrics=relative_projection_metrics(p,y,valid)
                            result[label+'_l1']=metrics['relative_l1'];result[label+'_l2']=metrics['relative_l2']
                            if stage=='C1':
                                archived=r['measurement'][label]
                                assert metrics['valid_rays']==archived['valid_rays']
                                for key in ('relative_l1','relative_l2'):
                                    difference=abs(metrics[key]-archived[key]);max_difference=max(max_difference,difference)
                                    assert difference<1e-6,(condition,draw,case,label,key,difference)
                            if prefix=='input':
                                assert metrics['valid_rays']==historical['observed']['valid_rays']
                                err=(p-y)[valid].double();reference=y[valid].double()
                                result.update(mae_ppm_m=float(err.abs().mean()),reference_mean_ppm_m=float(reference.mean()),
                                    signed_sum_bias=float(err.sum()/reference.sum().clamp_min(1e-6)),
                                    absolute_residual_sum_ppm_m=float(err.abs().sum()),reference_sum_ppm_m=float(reference.sum()),
                                    observed_valid_rays=metrics['valid_rays'])
                                for j,sid in enumerate(r['input_ids']):
                                    sm=relative_projection_metrics(p[j],y[j],valid[j])
                                    station_records.append(dict(condition=condition,draw=draw,case_id=case,stage=stage,
                                        station_id=sid,**sm))
                    records.append(result)
            if (i+1)%36==0: print(f'{i+1}/612 fields reprojected; C1 max archive difference {max_difference:.3g}',flush=True)
        audit=guard.summary()
        assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    assert len(records)==1224
    jsonl(out/'per_case_projection_metrics.jsonl',records);jsonl(out/'per_station_projection_metrics.jsonl',station_records)
    panels=[summarize(records,registration['cases'],s['condition'],[s['draw']]) for s in registration['settings']]
    panels += [summarize(records,registration['cases'],c,list(range(5))) for c in ('V3','V6','clustered-V4')]
    summary=dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),completed=True,
        reconstructions=612,stages=1224,panels=panels,gt_audit=audit,
        max_C1_archived_difference=max_difference,checkpoint_sha256=registration['checkpoint_sha256'],
        historical_observed_matched=True,final_test_used=False,
        metric_note='relative L1 on valid input rays, per-case ratio then macro mean; repeated draws averaged within case before inference on cases')
    write(out/'summary.json',summary)
    path=report(out,summary,records)
    write(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(path,flush=True)


if __name__=='__main__':main()
