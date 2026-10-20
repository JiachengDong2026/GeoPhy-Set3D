"""Read-only status plus a replaceable progress report; never opens arrays/weights."""
from runtime import *
from scope import effective_runs, deferred_runs, evaluation_stage

def main(quiet=False):
    s=read(BASE/'status.json')
    lines=['# MRGR-UniformFusion v1 执行进展（非最终结果）','',f'更新时间：{now()}（北京时间）；调度状态：`{s["state"]}`。','',
        '任务尚未全部完成时，不将本文当作正式评价报告。原有35例测试报告和已验收结果保留。本轮数据审计未发现额外独立案例；后续35例评价属于补充结果。','',
        '固定主模型：MRGR-UniformFusion-v1，342725参数，seed20261011第285轮，权重SHA256 `64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc`。两历史均匀融合种子核验通过；新主线种子20261020/20261021，与独立支线反馈OARI匹配。','',
        '## 工程与统计检查','']
    for name in ('cpu','gpu','statistics_synthetic'):
        p=BASE/'checks'/f'{name}.json'
        lines.append(f'- {name}：'+('通过' if p.exists() and verified(p)['passed'] else '未完成'))
    lines+=['','## 训练队列','', '| 任务 | 轮次 | 实际更新 | 最佳H4 L1 | GPU UUID | 状态 |','|---|---:|---:|---:|---|---|']
    for name in effective_runs():
        target=Path(name);path=target/'status.json'
        if path.exists():
            r=read(path);epoch=r.get('epoch',r.get('epochs',0));best=r.get('best_val_H')
            lines.append(f'| {target.name} | {epoch}/300 | {r.get("updates",0)} | {100*best:.4f}% | {r.get("gpu_uuid","")} | {r.get("state","")} |')
        else:lines.append(f'| {target.name} | 0/300 | 0 | — | — | 排队 |')
    if evaluation_stage():
        lines+=['','## 后续消融暂缓','',
            '按最新用户要求，先完成当前四个实验的300轮训练及35例三维配对评价，再讨论均匀融合＋反馈OARI是否作为最终模型。以下7项保持登记但暂不启动，评价完成后也不会自动恢复；此前取消的3个第二种子仍取消。','']
        lines.extend('- `'+Path(n).name+'`' for n in deferred_runs())
        lines+=['','本阶段只有四项纳入训练完成门槛；各自H4选轮和预测封存后再评分。原模型、训练预算和统计校正分母不变。反馈OARI不自动晋级，阶段完成不代表所有消融或最终模型决策完成。']
    elif (BASE/'seed_scope_amendment_v1.json').exists():
        lines+=['','按用户缩短排期要求，在本轮测试前追加种子范围变更：无反馈和零残差仍各双种子；无射线关系、无一致性、单尺度仅保留seed20261011，各取消尚未启动的seed20261019。三项标为单种子探索性消融，不声称跨训练种子稳定性。原始design及每项300轮预算不变；完整记录见scratch的seed_scope_amendment_v1.json。']
    lines+=['','每正式训练300轮/24900更新，FP32、原始预算不变；OARI先20轮验证H4工程筛查，门槛在运行前登记。训练期禁止体积GT与测试投影；全部纳入任务训练完成后才生成和封存新增预测。','',
        '[执行协议](../mrgr_uniform_final_v1_protocol_zh.md)；代码入口：`scripts/uniform_final_v1/controller.py`；运行目录：`/scratch/djc/GeoPhy-Set3D/mrgr_uniform_final_v1`。',
        '', '完整GPU进程、UUID锁、资源快照、异常与恢复记录保存在运行目录`scheduling/`。完成评分和独立验收后自动生成正式报告并尝试按限定文件清单同步GitHub；任何失败都会阻断后续阶段并保留日志。']
    path=PROJECT/'docs/reports/mrgr_uniform_final_progress_v1_zh.md'
    if (BASE/'acceptance.json').exists():
        lines+=['','正式评价及独立验收已完成，结果以[最终总报告](mrgr_uniform_final_v1_zh.md)为准；GitHub同步状态另见运行目录`github_sync.json`。']
    path.write_text('\n'.join(lines)+'\n')
    if not quiet:print(path)

if __name__=='__main__':main()
