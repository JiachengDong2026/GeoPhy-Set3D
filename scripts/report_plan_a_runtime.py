#!/usr/bin/env python3
"""Report runtime experiments without hiding failed candidates or reading GT."""
import argparse
import json
from pathlib import Path
import torch
from report_plan_a_speed import comparison, sha, timed_summary, run_evidence


def load(path):
    raw = json.loads(path.read_text())
    if raw['protocol'] != 'plan_a_network_runtime_speed_v3':
        raise ValueError(f'Wrong protocol: {path}')
    result = dict(path=str(path.resolve()), raw=raw)
    if raw['status'] == 'passed':
        artifact = path.with_suffix('.numerics.pt')
        if sha(artifact) != raw['numerical_artifact']['sha256']:
            raise ValueError(f'Artifact hash mismatch: {path}')
        result['artifact'] = torch.load(artifact, map_location='cpu', weights_only=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--smoke-run',type=Path)
    parser.add_argument('--gate-run',type=Path)
    parser.add_argument('--probe',type=Path,action='append',default=[])
    parser.add_argument('--epoch-result',type=Path)
    parser.add_argument('--reference-pair',action='append',default=[],help='candidate=reference filenames without .json; compare within a similar load window')
    args = parser.parse_args()
    torch.set_num_threads(2)
    baseline = load(args.directory/'baseline.json')
    pairs=dict(value.split('=',1) for value in args.reference_pair)
    results = dict(protocol='plan_a_network_runtime_comparison_v3',
                   limitations='Shared single GPU; fixed cases and warm caches; strict numerical checks are not learning/3D accuracy equivalence. No formal jobs. Failed candidates retained.',
                   baseline=baseline['raw'], candidates={})
    lines = ['# 方案 A 第二轮执行优化实验', '',
             '保持 A2、K4/Q96、宽度32、64³ 输出和完整梯度。网络执行候选独立于 v2 正式配置；物理计算与损失保持 FP32。所有时间为共享单张 RTX 4090，固定案例预热后的完整 batch2 更新。', '',
             '| 候选 | 比较基准 | 更新中位数 秒 | 两案例验证 秒 | 更新加速比 | 峰值显存 GiB | 严格数值核验 |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for path in sorted(args.directory.glob('*.json')):
        raw = json.loads(path.read_text())
        if raw.get('protocol') != 'plan_a_network_runtime_speed_v3':
            continue
        candidate = load(path)
        row = dict(path=str(path), result_sha256=sha(path), status=raw['status'], runtime=raw['runtime'])
        if raw['status'] == 'passed':
            reference_name=pairs.get(path.stem,'baseline')
            reference=baseline if reference_name=='baseline' else load(args.directory/(reference_name+'.json'))
            comp = comparison(reference, candidate)
            row.update(comparison=comp, train=timed_summary(raw['train']), validation=timed_summary(raw['validation']),
                       reference_name=reference_name,
                       peak_allocated_gib=max(r['memory']['peak_allocated_bytes'] for r in raw['train'])/2**30,
                       train_warmup=raw['train_warmup'], validation_warmup=raw['validation_warmup'],
                       gt_read_audit=raw['gt_read_audit'],
                       same_source=raw['source_sha256']==baseline['raw']['source_sha256'],
                       elapsed_seconds=raw['elapsed_seconds'])
            lines.append(f"| {path.stem} | {reference_name} | {row['train']['median']:.3f} | {row['validation']['median']:.3f} | {comp['phases']['train']['median_speedup']:.2f}× | {row['peak_allocated_gib']:.2f} | {'通过' if comp['numerical_passed'] else '未通过'} |")
        else:
            row.update(error=raw.get('error'), error_type=raw.get('error_type'))
            lines.append(f"| {path.stem} | — | — | — | — | — | 执行{raw['status']}，原始异常保留 |")
        results['candidates'][path.stem] = row
    lines += ['', '逐元素容差沿用 v2：体素/H预测 rtol=1e−4、atol=1e−6；完整clip后梯度 rtol=1e−3、atol=1e−6；更新后参数 rtol=1e−4、atol=1e−6。初始化与输入必须相同。不因 TF32/BF16 结果放宽阈值。未通过不等于不能学习，也不允许自动替代当前已验收 FP32 配置。', '',
              '编译和算法搜索的首次开销记录在 train_warmup，表中只列预热后时间。不同候选的最小/最大值、完整数值比较和禁读审计见配套 JSON。短测速不能直接外推 300 轮。', '',
              '本轮中途 GPU 0 的大任务结束，复测时仍有另一个 Python 进程。baseline 与 baseline_repeat 的耗时差包含明显负载变化，不能视为代码加速。repeat 候选应与 baseline_repeat 比较；其他早期候选仅与早期基准比较，不将跨阶段比值解释为算法收益。', '']
    for name, row in results['candidates'].items():
        if 'comparison' not in row:
            continue
        lines += [f'## {name} 数值检查', '', '| 张量 | 相对 L2 | 超差元素数 |', '| --- | ---: | ---: |']
        for field, metric in row['comparison']['phases']['train']['numerical'].items():
            lines.append(f"| {field} | {metric['relative_l2']:.3e} | {metric['outside_tolerance']} |")
        lines += ['']
    results['acceptance_runs']={}
    for label,path in (('smoke',args.smoke_run),('gate',args.gate_run)):
        if path:
            evidence=run_evidence(path)
            results['acceptance_runs'][label]=evidence
    if results['acceptance_runs']:
        lines += ['## 独立 FP32 编译/自动算法配置验收', '',
                  '此配置通过独立启动器运行，来源 hash 包含启动器及执行配置实现；与 v2 验收身份不同。', '',
                  '| 验收 | 完成/通过 | 最佳留出投影相对 L1 | 当前源码/执行代码匹配 |',
                  '| --- | --- | ---: | --- |']
        for name,evidence in results['acceptance_runs'].items():
            acc=evidence.get('acceptance') or {}
            lines.append(f"| {name} | {evidence['completed']} / {evidence['acceptance_passed']} | {acc.get('best_heldout_relative_l1',float('nan')):.6f} | {evidence['source_matches_current']} |")
        lines += ['']
    results['learning_probes']={}
    if args.probe:
        lines += ['## 降精度学习探针', '',
                  '使用实际 epoch runner、相同初始化种子与 A/B/H 采样，固定 16 个训练/8 个验证案例，5 轮40次更新。无正式 checkpoint、无正式验收身份，不能据此认定三维精度等价。严格逐元素比较失败仍保留。', '',
                  '| 探针 | 学习检查 | 初始 / 最佳 H 相对 L1 | 更新数 | 禁读尝试 |',
                  '| --- | --- | ---: | ---: | ---: |']
        for path in args.probe:
            probe=json.loads(path.read_text())
            if probe['protocol']!='plan_a_runtime_learning_probe_v3':
                raise ValueError('Wrong learning probe protocol')
            results['learning_probes'][path.stem]=dict(path=str(path),sha256=sha(path),result=probe)
            attempts=sum(probe['gt_read_audit'].get(k,0) for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
            lines.append(f"| {path.stem} | {probe['learning_checks_passed']} | {probe.get('initial_heldout_relative_l1',float('nan')):.6f} / {probe.get('best_heldout_relative_l1',float('nan')):.6f} | {probe.get('optimizer_updates',0)} | {attempts} |")
        lines += ['']
    if args.epoch_result:
        raw=json.loads(args.epoch_result.read_text())
        if raw['protocol']!='plan_a_runtime_full_epoch_speed_v3':
            raise ValueError('Wrong full epoch protocol')
        train=raw.get('phases',{}).get('train',{})
        val=raw.get('phases',{}).get('validation',{})
        full=(raw['status']=='passed' and train.get('metrics',{}).get('sample_count')==166
              and val.get('metrics',{}).get('sample_count')==36
              and train.get('metrics',{}).get('optimizer_updates')==83
              and train.get('metrics',{}).get('complete_epoch')
              and val.get('metrics',{}).get('complete_epoch'))
        results['full_epoch']=dict(path=str(args.epoch_result),sha256=sha(args.epoch_result),raw=raw,full_166_36_complete=bool(full))
        lines += ['## FP32 组合配置完整一轮', '']
        if full:
            seconds=train['seconds']+val['seconds']
            results['full_epoch'].update(seconds=seconds,extrapolated_300_epochs_hours=seconds/12,extrapolated_three_seeds_hours=seconds/4)
            lines += [f"真实166案例、83次更新：训练 {train['seconds']:.2f} 秒，36案例验证 {val['seconds']:.2f} 秒，合计 {seconds/60:.2f} 分钟/轮。按单轮线性估算，300轮约 {seconds/12:.2f} 小时，三种子单卡串行约 {seconds/4:.2f} 小时。", '',
                      '这是共享负载下的一轮测量，包含首次编译/算法搜索（磁盘编译缓存可能已存在），OS文件缓存未清空；外推不是独占卡结果、收敛承诺或完整正式训练，未计正式checkpoint/freeze序列化。', '']
        else:
            lines += [f"未获得完整166/36一轮证据，状态 {raw['status']}；不外推300轮。", '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    args.report.write_text('\n'.join(lines))
    print(json.dumps({k:dict(status=v['status'],seconds=v.get('train',{}).get('median'),numerics=v.get('comparison',{}).get('numerical_passed')) for k,v in results['candidates'].items()}))


if __name__ == '__main__':
    main()
