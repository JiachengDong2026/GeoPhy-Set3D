#!/usr/bin/env python3
"""One preregistered posthoc checkpoint average, measurement-only validation."""
import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import probe_mrgr_multiscale_v1 as base
import train_mrgr_multiscale_formal_v1 as formal
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.utils import move_batch

ROOT = base.ROOT
OUT = ROOT / 'outputs/research/mrgr_checkpoint_average_v1'
ORIGINAL = formal.DEFAULT_OUT
PANEL = ORIGINAL / 'evaluation/frozen_model_test_v1/registration.json'
STATIONS = base.OUT / 'station_epoch0.json'
NAMES = ('epoch_250.pt', 'epoch_275.pt', 'epoch_300.pt', 'best.pt')
ARMS = ('best278', 'average250_275_300')


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    p = read(OUT / 'protocol.json')
    assert base.digest(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    assert base.digest(Path(__file__)) == p['script_sha256']
    formal.verify(ORIGINAL)
    formal.verify_data(p)
    for path, expected in p['source_sha256'].items():
        assert base.digest(path) == expected, path
    for name in NAMES:
        assert base.digest(OUT / 'checkpoints' / name) == p['source_sha256'][str(ORIGINAL / 'checkpoints' / name)]
    return p


def register():
    original = formal.verify(ORIGINAL)
    formal.verify_data(original)
    assert not OUT.exists(), f'Refuse existing directory {OUT}'
    config = copy.deepcopy(original['config'])
    config['training']['device'] = 'cuda:1'
    config['experiment'] = dict(name=OUT.name, output_root=str(OUT))
    paths = [ORIGINAL / 'protocol.json', ORIGINAL / 'summary.json', PANEL, STATIONS,
             *[ORIGINAL / 'checkpoints' / name for name in NAMES]]
    settings = [s for s in read(PANEL)['settings'] if s['condition'] == 'clustered-V4']
    assert sorted(s['draw'] for s in settings) == list(range(5))
    case_ids = original['validation_case_ids']
    assert len(case_ids) == len(set(case_ids)) == 36
    for s in settings:
        assert set(s['input_ids']) == set(s['heldout_ids']) == set(case_ids)
        for case in case_ids:
            a, h = s['input_ids'][case], s['heldout_ids'][case]
            assert len(a) == len(h) == len(set(a)) == len(set(h)) == 4
            assert not set(a) & set(h)
    p = dict(registered_at=base.now(), experiment='posthoc checkpoint averaging; not SWA training or EMA',
        script_sha256=base.digest(Path(__file__)), config=config, data_files=original['data_files'],
        frozen_formal_sources=original['sources'], source_sha256={str(x):base.digest(x) for x in paths},
        validation_case_ids=case_ids, station_map=read(STATIONS)['val'], cluster_settings=settings,
        candidate_rule='Exactly epoch250,275,300, equal arithmetic weights. No window search or reselection.',
        averaging_rule='Named floating parameters accumulated in float64 and cast to original dtype; all buffers and nonfloating tensors must be identical and are copied unchanged; no geometry averaging.',
        reference_rule='Frozen best.pt selected at epoch278 during original training',
        primary='per-case average of A->H and B->H relative L1, then equal case mean and P90',
        cluster='fixed same five layouts, heldout H4 relative L1; average within case then across36 cases',
        gate=dict(ordinary_mean_ratio_max=.99, ordinary_p90_ratio_max=1.02, cluster5_mean_ratio_max=1.02),
        statistics='Paired36-case bootstrap, 20000 draws, seed20261017; no claim of statistical significance from point estimates',
        gt_allowed=False, final_test_allowed=False, training=False, no_automatic_gt_scoring=True,
        classification='single-seed validation-development measurement-only experiment',
        gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'], text=True))
    OUT.mkdir(parents=True)
    (OUT / 'checkpoints').mkdir()
    formal.write(OUT / 'protocol.json', p)
    (OUT / 'protocol.sha256').write_text(base.digest(OUT / 'protocol.json') + '\n')
    # Only these already approved measurement-only source files are copied. All
    # torch.load calls happen later inside GTReadGuard and inside this new root.
    for name in NAMES:
        shutil.copyfile(ORIGINAL / 'checkpoints' / name, OUT / 'checkpoints' / name)
    verify()
    print(json.dumps(dict(stage='registered', out=str(OUT))), flush=True)


def average(states, parameter_names):
    assert states and all(s.keys() == states[0].keys() for s in states)
    result = {}
    for key, value in states[0].items():
        assert all(s[key].shape == value.shape and s[key].dtype == value.dtype for s in states)
        assert all(bool(torch.isfinite(s[key]).all()) for s in states)
        if key in parameter_names and value.is_floating_point():
            accumulator = torch.zeros_like(value, dtype=torch.float64)
            for s in states:
                accumulator.add_(s[key].to(torch.float64))
            result[key] = (accumulator / len(states)).to(value.dtype)
        else:
            assert all(torch.equal(value, s[key]) for s in states), f'Nonaveragable tensor differs: {key}'
            result[key] = value.clone()
    return result


class Replay:
    def __init__(self, setting):
        self.setting = setting

    def sample(self, station_count, *, epoch, sample_index, case_id):
        return (np.asarray(self.setting['input_ids'][case_id], dtype=np.int64),
                np.asarray(self.setting['heldout_ids'][case_id], dtype=np.int64))


def heldout(batch):
    result = {k:v for k,v in batch.items() if not k.startswith(('input_','hold_')) and k != 'view_mask'}
    result.update({'input_'+k[5:]:v for k,v in batch.items() if k.startswith('hold_') and k != 'hold_view_mask'})
    result['view_mask'] = batch['hold_view_mask']
    return result


def score(field, target):
    prediction = base.project_input(field, target)
    valid = target['input_valid_ray'].bool() & target['view_mask'][:,:,None,None,None]
    assert torch.isfinite(field).all() and torch.isfinite(prediction).all()
    return [base.relative_projection_metrics(prediction[i], target['input_projection_raw'][i], valid[i]) for i in range(len(field))]


def paired(reference, candidate):
    delta = candidate - reference
    rng = np.random.default_rng(20261017)
    boot = delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
    return dict(reference_mean=float(reference.mean()), candidate_mean=float(candidate.mean()),
        reference_p90=float(np.quantile(reference,.9)), candidate_p90=float(np.quantile(candidate,.9)),
        mean_ratio=float(candidate.mean()/reference.mean()), p90_ratio=float(np.quantile(candidate,.9)/np.quantile(reference,.9)),
        mean_delta=float(delta.mean()), paired_delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
        wins=int((delta < -1e-8).sum()), losses=int((delta > 1e-8).sum()), ties=int((abs(delta) <= 1e-8).sum()))


def run():
    p = verify()
    assert not (OUT / 'per_case_metrics.jsonl').exists(), 'Refuse to overwrite completed/partial evaluation'
    began = time.perf_counter()
    config = p['config']; device = config['training']['device']
    records = []; timing = {}; identity = {}
    with base.GTReadGuard(config['dataset']['root'], OUT / 'checkpoints') as guard:
        template = base.model_for(config,'multiscale','cpu')
        parameter_names = set(dict(template.named_parameters()))
        checkpoints = {name:torch.load(OUT / 'checkpoints' / name, map_location='cpu', weights_only=False) for name in NAMES}
        assert [checkpoints[name]['epoch'] for name in NAMES] == [250,275,300,278]
        assert all(c['volumetric_gt_used'] is False for c in checkpoints.values())
        for c in checkpoints.values():
            template.load_state_dict(c['model'], strict=True)
        state = average([checkpoints[name]['model'] for name in NAMES[:3]], parameter_names)
        for name in NAMES:
            single = average([checkpoints[name]['model']], parameter_names)
            assert all(torch.equal(single[k],v) for k,v in checkpoints[name]['model'].items())
        identity['single_checkpoint_average_bitwise_equal'] = True
        identity['buffer_names_not_averaged'] = sorted(set(state)-parameter_names)
        identity['parameter_tensors_averaged'] = len(parameter_names)
        identity['all_shapes_dtypes_finite'] = True
        base.atomic_checkpoint(OUT / 'checkpoints/average250_275_300.pt', dict(model=state,
            config=config, epoch='average250_275_300', protocol_sha256=base.digest(OUT/'protocol.json'),
            source_epochs=[250,275,300], volumetric_gt_used=False))
        states = dict(best278=checkpoints['best.pt']['model'], average250_275_300=state)
        ds = base.selfsup.make_dataset(config, validation=True)
        assert [r['case_id'] for r in ds.records] == p['validation_case_ids']
        for arm in ARMS:
            model = base.model_for(config,'multiscale',device)
            model.load_state_dict(states[arm],strict=True); model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); start = time.perf_counter()
            with torch.no_grad():
                for first in range(0,len(ds),2):
                    samples = [ds[j] for j in range(first,min(first+2,len(ds)))]
                    for sample in samples:
                        case = sample['A']['case_id']
                        for k in ('A','B','H'):
                            assert sample[k]['input_station_indices'].tolist() == p['station_map'][case][k]
                    b = base.selfsup.move_dual_batch(base.dual_measurement_collate(samples),device)
                    metrics = {k:score(model(b[k]),b['H']) for k in ('A','B')}
                    for j,case in enumerate(b['A']['case_id']):
                        row = dict(arm=arm,panel='ordinary',case_id=case,
                            input_ids={k:p['station_map'][case][k] for k in ('A','B')}, heldout_ids=p['station_map'][case]['H'],
                            A=metrics['A'][j],B=metrics['B'][j],
                            relative_l1=.5*(metrics['A'][j]['relative_l1']+metrics['B'][j]['relative_l1']))
                        records.append(row); base.append(OUT/'per_case_metrics.jsonl',row)
                for setting in p['cluster_settings']:
                    hard = MeasurementOnlyPoolDataset(config['dataset']['root'],split='val',geometry_pool='train',
                        evaluation_only=True,min_input_views=4,max_input_views=4,hold_views=4,base_seed=20261011,
                        normalization='linear_p99_5',statistics_file='train_statistics.json')
                    hard.sampler = Replay(setting)
                    for first in range(0,len(hard),2):
                        samples = [hard[j] for j in range(first,min(first+2,len(hard)))]
                        b = move_batch(projection_pool_collate(samples),device)
                        inputs = {k:v for k,v in b.items() if not k.startswith('hold_')}
                        metrics = score(model(inputs),heldout(b))
                        for j,case in enumerate(inputs['case_id']):
                            assert samples[j]['input_station_indices'].tolist() == setting['input_ids'][case]
                            assert samples[j]['hold_station_indices'].tolist() == setting['heldout_ids'][case]
                            row = dict(arm=arm,panel='clustered',draw=setting['draw'],case_id=case,
                                input_ids=setting['input_ids'][case],heldout_ids=setting['heldout_ids'][case],**metrics[j])
                            records.append(row); base.append(OUT/'per_case_metrics.jsonl',row)
                    print(f'{arm} cluster draw{setting["draw"]} complete',flush=True)
            torch.cuda.synchronize(device)
            timing[arm] = dict(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                ordinary_cases=36,ordinary_reconstructions=72,clustered_reconstructions=180)
            del model; gc.collect(); torch.cuda.empty_cache()
        audit = guard.summary()
    assert len(records) == 432
    assert all(audit[k] == 0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
    comparisons = {}
    for panel in ('ordinary','clustered'):
        arrays = {arm:np.array([np.mean([r['relative_l1'] for r in records if r['arm']==arm and r['panel']==panel and r['case_id']==case])
                               for case in p['validation_case_ids']]) for arm in ARMS}
        comparisons[panel] = paired(arrays[ARMS[0]],arrays[ARMS[1]])
    ordinary,clustered = comparisons['ordinary'],comparisons['clustered']
    checks = dict(ordinary_mean=ordinary['mean_ratio'] <= p['gate']['ordinary_mean_ratio_max'],
                  ordinary_p90=ordinary['p90_ratio'] <= p['gate']['ordinary_p90_ratio_max'],
                  cluster5_mean=clustered['mean_ratio'] <= p['gate']['cluster5_mean_ratio_max'])
    verify()
    result = dict(completed=True,finished_at=base.now(),promoted=all(checks.values()),checks=checks,
        comparisons=comparisons,identity=identity,gt_audit=audit,timing=timing,
        total_elapsed_seconds=time.perf_counter()-began,protocol_sha256=base.digest(OUT/'protocol.json'),
        raw_metrics_sha256=base.digest(OUT/'per_case_metrics.jsonl'),average_checkpoint_sha256=base.digest(OUT/'checkpoints/average250_275_300.pt'),
        source_sha256=p['source_sha256'],gt_evaluation_performed=False,final35_test_evaluated=False,
        limitations='Single seed and development validation measurements only; no 3D improvement or convergence claim. Fixed one-window posthoc average is not SWA/EMA training.')
    formal.write(OUT/'summary.json',result)
    report(p,result)
    print(json.dumps(result),flush=True)


def report(p,r):
    lines = ['# 多尺度 MRGR 后期 checkpoint 固定平均：测量验证','',
        '本实验是 **posthoc checkpoint averaging**，没有实施 SWA 优化过程或 EMA。冻结正式多尺度模型的 epoch 250、275、300 等权平均；窗口提前固定，未搜索或根据结果挑选。对照为原训练选出的 best epoch 278。','',
        '仅使用训练池的 36 个验证案例投影。普通指标逐案例平均 A→H、B→H；聚簇沿用冻结的 5 组布局，先案例内平均再对 36 案例等权平均。以下不是三维浓度场误差，最终 35 条测试集没有使用。','',
        '| 留出投影相对 L1 | best278 | 固定平均 | 相对变化 |','|---|---:|---:|---:|']
    for panel,label in [('ordinary','普通均值'),('clustered','聚簇五布局均值')]:
        x=r['comparisons'][panel]
        lines.append(f'| {label} | {100*x["reference_mean"]:.6f}% | {100*x["candidate_mean"]:.6f}% | {100*(x["mean_ratio"]-1):+.4f}% |')
    x=r['comparisons']['ordinary']
    lines.append(f'| 普通 P90 | {100*x["reference_p90"]:.6f}% | {100*x["candidate_p90"]:.6f}% | {100*(x["p90_ratio"]-1):+.4f}% |')
    lines += ['',f'晋级结果：**{"通过" if r["promoted"] else "未通过"}**。预注册条件：普通均值至少改善 1%，普通 P90 和聚簇五布局均值均不得恶化超过 2%。检查：`{json.dumps(r["checks"])}`。没有自动启动训练或三维 GT 评价。','',
        '| 配对案例统计（候选－对照） | 均值差（百分点） | 95% bootstrap 区间（百分点） | 改善/退化/持平 |', '|---|---:|---:|---:|']
    for panel,label in [('ordinary','普通'),('clustered','聚簇五布局')]:
        x=r['comparisons'][panel];lo,hi=x['paired_delta_ci95']
        lines.append(f'| {label} | {100*x["mean_delta"]:+.6f} | [{100*lo:+.6f}, {100*hi:+.6f}] | {x["wins"]}/{x["losses"]}/{x["ties"]} |')
    lines += ['', 'bootstrap 按 36 个独立案例成对重采样，20,000 次、固定种子 20261017。聚簇的 180 次重建不作为 180 个独立样本。区间仅反映此开发验证面板，不代表多种子稳定性。','',
        '所有命名浮点参数以 float64 累加再转换回原 dtype；非浮点与 buffer 要求完全一致后保留，不平均几何 buffer。每个源 checkpoint 的单模型平均均通过逐张量 bitwise identity 检查；形状、类型、有限值检查通过。', '',
        f'实测总耗时 **{r["total_elapsed_seconds"]:.2f} 秒**；评估 best278 为 {r["timing"]["best278"]["seconds"]:.2f} 秒，平均候选为 {r["timing"]["average250_275_300"]["seconds"]:.2f} 秒。GPU1，FP32/TF32 off；没有训练更新，部署仍为单模型，参数量 344,678。', '',
        f'GTReadGuard 审计：`{json.dumps(r["gt_audit"])}`。原源文件及权重 hash 事后复核一致。', '',
        '证据：', '',
        '- [注册协议](../../outputs/research/mrgr_checkpoint_average_v1/protocol.json)',
        '- [完整汇总](../../outputs/research/mrgr_checkpoint_average_v1/summary.json)',
        '- [逐案例原始指标](../../outputs/research/mrgr_checkpoint_average_v1/per_case_metrics.jsonl)', '',
        '原始列：`arm, panel, case_id, input_ids, heldout_ids, relative_l1`；普通还记录 `A/B.relative_l1/relative_l2`，聚簇记录 `draw, relative_l2`。', '',
        '| 源文件 | SHA256 |','|---|---|']
    for path,h in p['source_sha256'].items():
        lines.append(f'| `{path}` | `{h}` |')
    lines += ['',f'平均权重 SHA256：`{r["average_checkpoint_sha256"]}`。',
        f'注册 SHA256：`{r["protocol_sha256"]}`；原始指标 SHA256：`{r["raw_metrics_sha256"]}`。','']
    (ROOT/'docs/reports/mrgr_checkpoint_average_v1_zh.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run','verify'))
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.action=='register': register()
    elif args.action=='run': run()
    else: verify(); print('Verified',flush=True)


if __name__=='__main__':
    main()
