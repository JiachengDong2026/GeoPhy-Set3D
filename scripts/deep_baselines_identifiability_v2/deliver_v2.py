"""Publish small auditable evidence only, preserving all prior results."""
from ctx_v2 import *
import csv
import shutil

def table_rows():
    with (ROOT/'case_metrics.csv').open() as f:cases=list(csv.DictReader(f))
    with (ROOT/'pair_metrics.csv').open() as f:pairs=list(csv.DictReader(f))
    with (ROOT/'swap_metrics.csv').open() as f:swaps=list(csv.DictReader(f))
    return cases,pairs,swaps
def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases,_,_=table_rows();fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,arm in enumerate(design()['arms']):
        for role,style in [('seen','-'),('unseen','--')]:
            data=[r for r in cases if r['arm']==arm and r['role']==role]
            for ax,key in zip(axes[row],('AB_L1','H_L1','volume_L1')):
                values=[np.mean([float(r[key]) for r in data if int(r['update'])==u]) for u in design()['evaluation_updates']]
                ax.plot(design()['evaluation_updates'],values,style,marker='.',label=role);ax.axhline(1,color='gray',ls=':');ax.set_title(arm+' '+key);ax.set_xlabel('updates');ax.legend();ax.grid(alpha=.2)
    fig.savefig(ART/'learning_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(9,11),constrained_layout=True)
    for row,index in enumerate([0,1,6,7]):
        truth=np.load(ROOT/f'inputs/truth_{index}.npy',allow_pickle=False);values=[truth]
        for arm in design()['arms']:
            with np.load(ROOT/f'runs/{arm}/raw/update200_case{index}.npz',allow_pickle=False) as a:values.append(a['field_A'].reshape(64,64,64))
        maximum=max(v.max() for v in values)
        for col,(value,label) in enumerate(zip(values,['analytic','GNOT','DeepSets'])):
            image=axes[row,col].imshow(value[:,:,32].T,origin='lower',extent=(0,.6,0,.6),vmin=0,vmax=maximum,cmap='magma');axes[row,col].set_title(f'{label} plume{index} '+('seen' if index<6 else 'unseen'))
        fig.colorbar(image,ax=list(axes[row]),shrink=.7,label='mass fraction; same scale per row')
    fig.savefig(ART/'fixed_geometry_slices.png',dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    assert not ART.exists()
    audit=verified(ROOT/'independent_review.json');assert audit['passed']
    reg=verified(ROOT/'registration.json');verify(reg['sources']);verify(reg['inputs'])
    pre=verified(ROOT/'preregistration.json');verify(pre['historical_tracked']);verify(pre['frozen_main_checkpoints'])
    ART.mkdir(parents=True)
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.suffix in ('.pt','.npy','.npz','.lock','.tmp') or '__pycache__' in path.parts or path.name=='resources.jsonl':continue
        dest=ART/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    resource=ROOT/'scheduling/resources.jsonl'
    if resource.exists():
        for row in rows(resource):append(ART/'scheduling/gpu_resources.jsonl',{k:row[k] for k in ('at','gpus','gpu_processes','memory','load')})
    plots();cases,pairs,swaps=table_rows();summary=[];pair_summary=[]
    for arm in design()['arms']:
        result=verified(ROOT/'runs'/arm/'evaluation_result.json')
        for role in ('seen','unseen'):
            r=[x for x in cases if x['arm']==arm and x['role']==role and int(x['update'])==200]
            summary.append(dict(arm=arm,role=role,AB=float(np.mean([float(x['AB_L1']) for x in r])),H=float(np.mean([float(x['H_L1']) for x in r])),
                volume=float(np.mean([float(x['volume_L1']) for x in r])),volume_better=sum(float(x['volume_L1'])<1 for x in r),branches=len(r),
                zero_MAE=float(np.mean([float(x['zero_MAE']) for x in r])),status=result['status']))
        for group in ('seen_seen','seen_unseen','unseen_unseen'):
            r=[x for x in pairs if x['arm']==arm and int(x['update'])==200 and x['group']==group]
            pair_summary.append(dict(arm=arm,group=group,MAE=float(np.mean([float(x['mae']) for x in r])),
                fixed_response_min=min(float(x['fixed_scale_response']) for x in r),cosine_median=float(np.median([float(x['direction_cosine'] or 0) for x in r])),
                positive_fraction=float(np.mean([float(x['direction_cosine'] or 0)>0 for x in r]))))
    lines=['# 固定几何、变化测量的条件重建诊断 v2','',f"完成时间：{now()}。起始提交 `{pre['commit']}`；远端main已核对一致。",'',
        '本轮只回答同一几何下的测量条件依赖。两个fresh模型各200次更新，v1累计2000加本轮400为2400，剩余100未使用。固定update200，不通过H/真场选模，正式队列仍暂停。诊断结论不能授权正式300epoch或第二种子。','',
        '## 协议与数据隔离','',
        '直接geometry-only读取唯一train_geometry.npz，没有实例化历史dataset或调用真实投影getter，也不读取真实测量、GT/GT128、val/test/Pool B或真实GT统计。A=[0,3,6,9]、B=[1,4,7,10]、H=[2,5,8,11]两两不交，8例全部几何、掩码、站位及case_id/sample_id逐位相同。6个seen与2个unseen定义和全部28个配对在design.json预登记。配对按seen-seen15、seen-unseen12、unseen-unseen1分别报告；8个测量与解析真场的SHA各不相同。',
        '复用同一seed20261011的v2初始化，与v1同模型初态SHA一致；一致性固定为原v3齐次公式。AdamW2e-4/wd1e-4/batch2，固定[0,1]/[2,3]/[4,5]循环、FP32/TF32关闭、clip1、无调度，loss=.5cross+.1observed+.05consistency。前6例AB是训练worker唯一可打开的数组。CPython文件open审计对整个训练worker启用，拒绝H、真场、unseenAB和所有真实数组；拒绝测试有记录。这是文件访问层隔离，不冒充OS沙箱。',
        '两个训练进程全部退出后，独立评价进程按预定0/10/25/50/100/150/200权重节点评价；因此训练不接收H/真场结果。H仅在评价中用于投影误差，真场仅用于独立三维评价；训练无H/GT损失、无warm-start。checkpoint200为唯一终态。每节点8例双分支记录全部测量/三维指标、零场、置零/置换/交换、全体积观测损失的measurement-token与末层梯度，梯度审计0次更新。',
        '交换时除raw和相应归一化测量外的全部输入不变；构造后的输入逐位等于donor原始输入。因此复用donor已重复逐位核验的完整预测，明确标记复用来源，没有把同一输出伪称为新独立样本。全部56个配对分支、112个有向交换每节点保留。输出绝对变化、固定C_init尺度、变化方向余弦、匹配/错配真场L1优势均独立重算。','',
        '## 参考积分与预登记门槛','',
        f"独立NumPy FP64解析生成，128/256阶GL最大参考绝对差={max(r['reference_max_abs_ppm_m'] for r in verified(ROOT/'input_registration.json')['precision']):.4g} ppm·m；64³/K96对实际FP32目标相对L1偏差范围={min(r['grid64_K96_relative_l1'] for r in verified(ROOT/'input_registration.json')['precision']):.4%}–{max(r['grid64_K96_relative_l1'] for r in verified(ROOT/'input_registration.json')['precision']):.4%}。解析真场保存FP64；模型、测量和逆投影仍FP32，投影生成与反演路径独立。", 
        'GO要求seen各例AB<=.99并较初始改善>=.01，所有例H末两次<=.99，输出非近零且CV>=.05；全部配对绝对MAE/C_init>=.001且重复一致；每分支每配对组方向cos>0至少80%、中位数>=.1；seen至少4/6、unseen2/2三维L1<=.99；seen/unseen recipient分别至少75%交换具有>=.01匹配真场L1优势；输入测量梯度非零且有限。响应或近零失败为NO-GO，其余证据不足为INCONCLUSIVE。阈值运行前封存，结果后未调整。','',
        '## 终态结果：seen与unseen分别报告','',
        '| 模型 | 羽流 | AB L1 | H L1 | 三维L1 | 优于零场分支/总数 | 置零MAE | 结论 |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"| {r['arm']} | {r['role']} | {r['AB']:.6f} | {r['H']:.6f} | {r['volume']:.6f} | {r['volume_better']}/{r['branches']} | {r['zero_MAE']:.3g} | {r['status']} |")
    lines+=['','优于零场计数展示A/B分支；验收按各案例及各分支检查。零场测量L1/L2和三维L1均按同目标重算为1。unseen的AB在评价时是可见输入测量，不属于训练样本。逐例/逐分支及所有节点完整数据见case_metrics.csv，不用seen平均掩盖unseen。','',
        '| 模型 | 配对组 | 平均预测MAE | 最小固定尺度响应 | 方向cos中位数 | 正方向比例 |','|---|---|---:|---:|---:|---:|']
    for r in pair_summary:lines.append(f"| {r['arm']} | {r['group']} | {r['MAE']:.3g} | {r['fixed_response_min']:.3g} | {r['cosine_median']:.4f} | {r['positive_fraction']:.1%} |")
    lines+=['','![测量与体积误差曲线](artifacts/deep_baselines_identifiability_v2/learning_curves.png)','',
        '![固定案例中心切面，行内共用色标](artifacts/deep_baselines_identifiability_v2/fixed_geometry_slices.png)','',
        '## 证据、替代解释与逐项回答','']
    for arm in design()['arms']:
        result=verified(ROOT/'runs'/arm/'evaluation_result.json');failed=[k for k,v in result['checks'].items() if not v]
        r=[x for x in pair_summary if x['arm']==arm]
        s=[x for x in swaps if x['arm']==arm and int(x['update'])==200]
        lines.append(f"- **{arm}：{result['status']}**。固定几何终态配对MAE平均{np.mean([float(x['mae']) for x in pairs if x['arm']==arm and int(x['update'])==200]):.6g}，有向交换匹配优势>=.01比例{np.mean([float(x['matching_advantage_L1'])>=.01 for x in s]):.1%}；未通过条件：{', '.join(failed) or '无'}。数值变化存在与有效条件重建必须区别。")
    response={arm:np.mean([float(x['mae']) for x in pairs if x['arm']==arm and int(x['update'])==200]) for arm in design()['arms']}
    higher=max(response,key=response.get)
    lines += ['',f"**固定几何能否区分羽流、谁更依赖测量：** 当前配对绝对响应较大的是{higher}（{response[higher]:.4g}），这是测量依赖幅度比较，不自动代表质量更好；能否有效区分必须同时检查上表变化方向及每个模型的pair_response/direction/matching门槛。seen与unseen误差和配对组已分别报告。", 
        '**已验证事实：** 本轮排除了“不同羽流输入携带不同几何/标识”的捷径；模型所有几何输入一致，任何逐位可复现的输出差异只能来自测量及其确定性计算。差异是否有正确重建意义由真场差分、匹配优势和H共同判断。没有达到全部门槛时，不把微小非零变化写成学会重建。',
        '**较强假设与未解决问题：** 共享几何下仍可学习共同空间模板，因而弱条件响应与共同先验/条件读出受抑制相符。它不能反向证明v1一定依赖几何识别，也不能把200次预算不足、全局聚合/decoder优化和输出层初始化的影响唯一分离。本轮没有调整初始化或输出非线性，故不声称已锁定输出层为唯一根因。8例/单几何/单seed/有限更新不能代表算法普遍能力。',
        '**正式训练授权与最小下一步：** 本任务无正式训练授权，即便诊断GO也不启动。建议先只读固定同几何的测量编码/latent，对已登记羽流的解析场差分做预登记线性可读性对照（与原decoder输出比较），区分内部保留的条件信息和读出利用失败；对照必须固定划分与正则，不作宽泛搜索。本轮未执行，也不使用剩余100次预算自动训练。','',
        '## 独立审计、异常与资源','',
        f"独立review_v2.py未导入运行期metrics/decide，直接从保存数组重算{audit['measurement_fields_recomputed']}个L1/L2字段、{audit['cases']}条case/branch记录、{audit['pairs']}条配对和{audit['directed_swaps']}条有向交换，独立重建科学判定。全部400条更新、7节点AdamW状态、初态SHA、输入和checkpoint身份核验通过。全部历史跟踪文件及冻结主模型双种子checkpoint哈希不变；旧队列保持paused_by_user。", 
        '5项CPU工程测试通过，覆盖geometry-only/互斥/文件拒绝/GL参考/完整64³物理反传/序列化/干预；两个完整配置GPU batch2前向、反向和checkpoint保存smoke通过，均无optimizer更新。训练worker访问记录只包含train_AB.pt，评价worker单独记录H/真场读取。拒绝测试属于预期拒绝；实际运行异常/资源冲突以scheduling事件与日志为准，没有自动重试或参数更改。',
        '启动前静态复核发现独立审计的几何读取白名单断言过严：生成器为计算新真场SHA会重新读自己刚写的合成文件，不能把这些自有文件误判为额外真实数据读取。第一次GPU/optimizer运行前仅修正该审计断言及对应封存/报告逻辑，原源码保存在engineering/preupdate_original_sources并有前后SHA修订记录。没有改模型、loss、阈值、输入或合成定义，未丢弃任何训练结果。',
        '首个GNOT GPU smoke在任何optimizer更新之前因grid_sampler_3d_backward_cuda不支持严格确定性模式而失败。保留失败日志/目录，登记前后源码及协议SHA后，将框架严格确定性开关恢复为冻结流程原设置；未替换CUDA物理算子、精度、模型或训练设置。固定种子/采样与逐位重复推理检查保留，CUDA反传的原子累加不宣称逐位可复现。按明确recover-reason重新完成smoke后才封存并开始训练，无训练更新重放。',
        '只用GPU0既有UUID全局锁、15秒双空闲快照、最多60秒获取及外来冲突停止。未抢占或停止GPU1–3其他任务。原始数组、权重和完整资源快照在scratch，Git只含小型记录。','',
        '## 交付索引','',
        '- [执行状态](artifacts/deep_baselines_identifiability_v2/final_status.json) · [独立重算](artifacts/deep_baselines_identifiability_v2/independent_review.json) · [逐例指标](artifacts/deep_baselines_identifiability_v2/case_metrics.csv)',
        '- [配对差分](artifacts/deep_baselines_identifiability_v2/pair_metrics.csv) · [有向测量交换](artifacts/deep_baselines_identifiability_v2/swap_metrics.csv) · [初始化/协议封存](artifacts/deep_baselines_identifiability_v2/registration.json) · [几何/输入哈希](artifacts/deep_baselines_identifiability_v2/input_registration.json)',
        '- [完整运行前协议](../deep_baselines_identifiability_v2_protocol_zh.md)',
        '',f'本地完整原始产物：`{ROOT}`。所有报告结果属于有界诊断，不进入论文正式基线排名。']
    report=PROJECT/'docs/reports/deep_baselines_identifiability_v2_zh.md';assert not report.exists();report.write_text('\n'.join(lines)+'\n')
    status=dict(at=now(),state='COMPLETE',new_updates=400,prior_updates=2000,cumulative_updates=2400,remaining_budget=100,
        independent_review_passed=True,old_queue_paused=True,formal_training_authorized=False,second_seed_authorized=False,
        candidates=audit['summaries'],real_measurement_or_GT_access=False,historical_hashes_unchanged=True,main_checkpoint_hashes_unchanged=True)
    freeze(ROOT/'final_status.json',status);freeze(ART/'final_status.json',status);write(ROOT/'status.json',status)
    files={str(p.relative_to(PROJECT)):sha(p) for root in (ART,HERE,CONFIG.parent) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for p in [report,PROJECT/'docs/deep_baselines_identifiability_v2_protocol_zh.md',PROJECT/'tests/test_baseline_identifiability_v2.py']:files[str(p.relative_to(PROJECT))]=sha(p)
    freeze(ART/'publication_manifest.json',dict(at=now(),files_sha256=files,manifest_self_excluded=True))
    print('DELIVERED',report,flush=True)

if __name__=='__main__':main()
