"""Assemble Chinese report, compact evidence, integrity and experiment status."""
import csv,gzip,shutil,datetime
from context import *

def main():
    verify();historic=read(ART/'history_audit.json');status={};checkpoints={}
    protected=read(ROOT/'protected_tracked.json')
    changed=[p for p,h in protected.items() if not (PROJECT/p).is_file() or sha(PROJECT/p)!=h]
    assert not changed
    assert all(sha(p)==h for p,h in historic['evidence_sha256'].items())
    for arm in ['deeponet','fno']:
        root=ROOT/arm;pilot=root/'pilot';complete=(pilot/'result.json').exists()
        source=read(pilot/'result.json') if complete else read(root/'terminal_audit/result.json')
        review=read(ART/(arm+'_independent_review.json'));assert review['all_passed']
        status[arm]=dict(status=review['status'],engineering=True,formal_epochs=0,formal_updates=0,GT64_evaluated=False,
            pilot_completed=complete,pilot_epochs=source.get('epochs',source.get('checkpoint',{}).get('epoch')),
            actual_updates=source.get('optimizer_steps',source.get('actual_training_updates')),independently_reviewed=True)
        if complete:assert source['status']=='NO_GO','GO requires completing formal phase, not final delivery now'
        for p in sorted(pilot.glob('*.pt')):
            checkpoints[arm+'_'+p.stem]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        for filename in ['epoch0.json','result.json','failure_access.json','access.json']:
            p=pilot/filename
            if p.exists():shutil.copyfile(p,ART/(arm+'_'+filename))
        for filename in ['metrics.jsonl','actual_updates.jsonl','actual_training_inputs.jsonl','updates.json']:
            p=pilot/filename
            if p.exists():
                with p.open('rb') as src,gzip.open(ART/(arm+'_'+filename+'.gz'),'wb') as dest:shutil.copyfileobj(src,dest)
        for p in (ROOT/'scheduling').glob('*'+arm+'*'):
            if p.is_file() and p.suffix in ['.json','.log']:shutil.copyfile(p,ART/p.name)
        history=[json.loads(s) for s in (pilot/'metrics.jsonl').read_text().splitlines()]
        fields=['epoch','updates','train_loss','train_H_L1','val_loss','val_H_L1','val_mean','val_std','seconds','peak_gpu_bytes']
        with (ART/(arm+'_curve.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for r in history:writer.writerow(dict(epoch=r['epoch'],updates=r['optimizer_steps'],train_loss=r['train']['loss']['total'],train_H_L1=r['train']['measurement']['heldout']['relative_l1'],
                val_loss=r['val']['loss']['total'],val_H_L1=r['val']['measurement']['heldout']['relative_l1'],val_mean=r['val']['output']['mean'],val_std=r['val']['output']['std'],seconds=r['seconds'],peak_gpu_bytes=r['peak_gpu_bytes']))
    status['geolift_deepsets_sum']=dict(status='GO_FROZEN_EXISTING_FORMAL',epochs=300,optimizer_steps=24900,retrained=False,checkpoint=historic['deepsets'])
    write(ART/'experiment_status.json',status);write(ART/'checkpoint_manifest.json',checkpoints)
    write(ART/'integrity.json',dict(all_original_tracked_files_unchanged=True,checked_files=len(protected),original_checkpoint_history_and_queue_sha256_unchanged=True,
        primary_bitwise_regression=read(ART/'primary_regression.json')['bitwise_equal'],new_formal_updates=0,new_GT64_reads=0,
        github_main_verified=False,resource_conflict_preempted=False,scientific_thresholds_not_relaxed=True))
    log=ROOT/'regression.log'
    if log.exists():shutil.copyfile(log,ART/'regression.log')
    lines=['# GeoPhy-Set3D：DeepONet / FNO 经典基线实施与决策','',
        '本轮已完成已有基线冻结、两种经典模型实现、工程验证和各自科学判定。**GeoLift-DeepSets-Sum复用已有正式结果；DeepONet与FNO的当前适配均NO-GO，不进入正式精度排名。** 没有新300轮训练，没有新GT64/GT128/Pool B/test读取，没有恢复Senseiver或失败Ray系列。','',
        f"交付日期：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M %Z')}。独立分支`experiment/classical_operators_v1`；工作区`{PROJECT}`。",'',
        '## 1. 已有结果与主模型冻结身份','',
        f"GeoLift-DeepSets-Sum：306,625参数，300epoch/24,900更新，best第292轮。实际checkpoint `{historic['deepsets']['checkpoint']}`；SHA256 `{historic['deepsets']['sha256']}`。本轮对900条prediction-row JSON和900个原预测数组逐一复算SHA并匹配冻结清单；已有逐案例指标、50项组/指标统计与独立bootstrap复核均存在并通过。本轮是身份重核，不声称重新执行全部旧GT评价。",'',
        '原配置train_deepsets_selfsup_v1.yaml、训练源码清单f56b55d0f2ff20c1d667ee4aa58378b6e0017325f16ebb539edcaea761852128及评价源码清单完整保存。原始SharedEncoder2D/GeometryLifting/masked sum÷4/ResidualDecoder3D保持原样；这是共享GeoLift受控集合聚合对照，不是DeepSets论文官方三维重建实现，与失败Ray-DeepSets-V1不同。','',
        '既有五组3D Relative L1：G1 .135463、G2 .131762、G3 .137796、G4 .156366、G5 .618805。完整Relative L2、observed/H误差、峰值/积分及95%CI仍引用原deepsets_selfsup_v1_zh.md和其paired_statistics.csv；不挑选新模型版本。','',
        f"当前主模型：`{historic['current_primary']['key']}`，{historic['current_primary']['parameters']:,}参数；checkpoint `{historic['current_primary']['checkpoint']}`；SHA `{historic['current_primary']['checkpoint_sha256']}`。冻结配置见history_audit.json；本轮CPU固定输入C0/C1逐位复现通过。",'',
        '**比较身份限制：** 历史DeepSets报告对照的是旧SelfSup-MRGR v1 SHA1573cd84…，不是当前Plain主模型。本轮没有伪造当前主模型与新NO-GO模型的配对三维表。900固定清单为36个开发Val案例×5组×5布局，不是最终独立test；最终test禁止访问。','',
        '## 2. 固定架构与公平性','',
        '| 算法 | 适配 | 参数 | 共同目标/数据 |','|---|---|---:|---|',
        '| GeoLift-DeepSets-Sum | 原共享几何提升及masked sum/4 | 306,625 | 已有正式自监督协议 |',
        '| DeepONet | 全站完整测量/几何CNN branch；station-ID排序5槽；坐标MLP trunk；64维内积 | 222,033 | 原selfsup目标，H不输入 |',
        '| FNO | 固定K96线性射线伴随3通道+XYZ；4层3D谱卷积，width16/modes8 | 4,196,657 | 原selfsup目标，H不输入 |','',
        'DeepONet依据Lu et al., NMI2021, arXiv:1910.03193；FNO依据Li et al., ICLR2021, arXiv:2010.08895。文献标识清楚；网络不可达，未声称本轮取得并逐段核对全文。没有MRGR/OARI/GCA/新attention，也没有使用GT构造输入。模型只读取合法measurement/geometry字段。','',
        'DeepONet保留全部有效射线（含0测量），常规CNN下采样不是筛射线；padding在排序后剔除，最多5真实station，完整XYZ/z最快64³查询分块。FNO的T为归一化**摩尔分数线积分**，非线性质量投影P没有全局线性伴随：用已知chi导数将J_P(0)^T转换为T^T，固定计算加权回投影、log1p支持度和可见性，不把测量积分称作局部浓度真值。CPU固定顺序累积与512条有界缓存保持相同输入重复一致。薄弱支持处归一化对舍入敏感；输入在模型参数外确定，不经GT选择。','',
        '所有原训练字段逐项严格一致：Pool A、166/36拆分、原A/B/H sampler与mask、FP32、AdamW2e-4/wd1e-4、clip1、seed20261011、cosine T_max300、batch2、loader4workers。pilot固定前32个train/完整36Val，每轮16更新，上限30轮480更新，独立于正式预算。直接调用training.selfsup.objective/run_epoch/selection_metric；不复制loss、不改epsilon/权重。','',
        '预登记交集门槛：最后两次Val H<=.99，相对真正epoch0至少改善5%且绝对改善.01；非零/非恒定/非饱和；train0/1/2至少2例置零响应>=1e-3且绝对MA>=1e-8，错配H增加>=1e-6；Val前4例至少2例正常H优于置零.001；工程/访问/独立复核通过。不同loss及解析GT可辨识性实验的专用门槛不混用。所有阈值在训练前登记，失败后未降低。','',
        '## 3. 工程运行与pilot结果','',
        '| 模型 | smoke | 完整pilot轮 | Val H epoch0→最后完整轮 | Val输出mean/std | 科学结论 |','|---|---|---:|---|---|---|']
    for arm in ['deeponet','fno']:
        p=ROOT/arm/'pilot';h=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()];first=read(p/'epoch0.json')['val'];last=h[-1]['val']
        lines.append(f"| {arm} | batch2/all-ray/64³/原loss/反传/1更新通过 | {len(h)} | {selection_metric(first):.9g} → {selection_metric(last):.9g} | {last['output']['mean']:.6g} / {last['output']['std']:.6g} | {status[arm]['status']} |")
    lines+=['','两项smoke均检查了finite/range、真实参数变化、H精确排除、station置换、NaN invalid/padding、全invalid、normal/zero/wrong-case、seed重复与序列化。DeepONet峰值653,123,584 bytes/工程段.821秒；FNO峰值1,566,914,048 bytes/3.526秒；这些是多次前向/反传/更新的工程段，**不是独立推理benchmark**。未OOM，未减少有效射线或分辨率。','']
    for arm in ['deeponet','fno']:
        p=ROOT/arm/'pilot';h=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()];r=read(ART/(arm+'_independent_review.json'))
        lines += [f'### {arm}','']
        if not (p/'result.json').exists():
            t=read(ROOT/arm/'terminal_audit/result.json');u=t['actual_training_updates']
            number=str(u.get('exact_process_updates')) if u.get('exact_interrupted_epoch_count_available') else f"{u['process_lower_bound']}–{u['process_upper_bound']}（精确中断计数缺失）"
            lines += [f"pilot在第{len(h)+1}轮因下一次更新前FP32梯度norm非正而停止；权威last.pt为第{len(h)}轮/{t['checkpoint']['updates']}更新，进程实际更新{number}。完整轮训练+Val实测{sum(x['seconds'] for x in h):.3f}秒，峰值{max(x['peak_gpu_bytes'] for x in h):,} bytes，不含epoch0/干预/中断轮；全部进程时间见调度日志。",'',
                f"只读末态重放完整36Val与日志精确一致；另封存{r['full_fields_recomputed']}个完整干预场/原始logit/H投影，第二套NumPy FP64重算sigmoid、场统计、测量L1及科学门槛通过。",'']
        else:lines += [f"完成30轮/{status[arm]['actual_updates']}更新，但科学门槛失败；不推广pilot权重为正式结果。",'']
        checks=r.get('necessary_conditions',r.get('diagnostic_checks',{}))
        lines += ['```json',json.dumps(checks,ensure_ascii=False,indent=2),'```','']
        if 'cases' in r:
            lines += ['| train案例索引 | normal-zero相对MA | 绝对MA | 错配H增加 |','|---|---:|---:|---:|']
            for i,v in enumerate(r['cases']['train']):lines.append(f"| {i} | {v['normal_zero_relative_MA']:.6g} | {v['normal_zero_absolute_MA']:.6g} | {v['wrong_H_increase']:.6g} |")
            lines+=['']
    lines+=['**解释边界：** loss从巨大初始值下降不等于学会利用测量；接近零预测与响应消失均违反必要条件。工程断言在30轮上限之前停止，因此不写成30轮完成。必要塌缩失败已证实，NO-GO不因缺失最终两次成功而降格为“可能继续训练”。末态梯度FP32/FP64范数见terminal_audit；单次末态梯度诊断不证明中断瞬间唯一机制。未实施任何学习率/损失/初始化搜索或修复候选，未将失败外推为经典算法能力上限。','',
        '## 4. 正式三维对比状态','',
        '两个新模型均未获得正式训练准入：formal=0epoch/0update、新900预测=0、新GT64评分=0。故无新三维Relative L1/L2、峰值/积分、95%CI或精度排名；不拿pilot单例代替正式全量。新增正式/selector/infer/GT入口采用完整预算、checkpoint哈希、measurement-only replay和900封存前置校验；由于本轮未获准入，这些完整路径未做端到端科学运行，不声称已经验证。','',
        '## 5. 独立复核、保护、偏离与资源','',
        '独立复核是第二套NumPy数值实现，不是外部人员复核。CPU测试初次4/5通过（合成边界fixture差异），保留失败日志；改解析射线避开舍入网格边界后5/5通过，不改算子/容差；新模型、原DeepSets、原selfsup与配对统计30项回归通过。当前主模型C0/C1固定CPU输出逐位复现。原始4835个tracked文件及原checkpoint/历史结果/旧queue哈希前后全部不变。模型独立registry没有改旧build_model白名单。','',
        '偏离/限制逐项保留：','',
        '- 远端GitHub main最新性未确认：7897失效，17890 CONNECT和直连超时；本地main/cache origin/main均3f977a3。未推送、未合并main；正式训练即使pilot GO也需要补齐远端身份核验。',
        '- home worktree因配额失败，使用scratch独立worktree；失败新目录仅属本轮，历史工作区保持不变。',
        '- DeepONet中断轮精确计数未持久化，只能给96–111范围；last完整epoch权威96。之后FNO在首个pilot更新前修复逐step fsync与异常快照，独立source manifest保留前后身份。',
        '- GPU0被其他用户占用使FNO首次预约60秒退出、零更新；登记资源修订后GPU1执行。全程既有UUID flock、15秒双空闲检查、外来冲突只停止自身；没有抢占或终止任何他人任务。GPU3既有任务保持。',
        '- DeepONet/FNO均未完成正式预算，不生成伪造的主模型配对精度；单seed/小pilot/开发Val限制保留。','',
        '## 6. 交付与复算','',
        '- [训练前协议](../classical_operators_v1_protocol_zh.md)、[中断审计/资源修订](../classical_operators_v1_terminal_audit_zh.md)。',
        '- [历史冻结审计](artifacts/classical_operators_v1/history_audit.json)、[状态](artifacts/classical_operators_v1/experiment_status.json)、[checkpoint路径与SHA](artifacts/classical_operators_v1/checkpoint_manifest.json)、[保护核验](artifacts/classical_operators_v1/integrity.json)。',
        '- [DeepONet独立复核](artifacts/classical_operators_v1/deeponet_independent_review.json)、[FNO独立复核](artifacts/classical_operators_v1/fno_independent_review.json)；逐例干预/曲线/压缩日志均同目录。',
        f'- 大数组、checkpoint、完整资源快照仅在`{ROOT}`；配置`configs/classical_operators_v1/{{deeponet,fno}}.yaml`，代码`scripts/classical_operators_v1`与独立models模块。',
        '- CPU复算已保存失败末态：`CUDA_VISIBLE_DEVICES="" PYTHONPATH=src OMP_NUM_THREADS=2 /scratch/djc/envs/gas-torch/bin/python scripts/classical_operators_v1/terminal_review.py deeponet`（FNO已完成30轮用review.py，否则terminal_review.py）。复算只读已有数组，不恢复训练。','',
        '本轮明确结论：GeoLift-DeepSets-Sum已有正式基线保持有效；DeepONet与FNO当前适配不得进入论文正式精度排名。下一步若研究明确修复假设，须独立预登记，不能自行调整本轮门槛或把新版本覆盖本轮失败。']
    (PROJECT/'docs/reports/classical_operators_v1_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
