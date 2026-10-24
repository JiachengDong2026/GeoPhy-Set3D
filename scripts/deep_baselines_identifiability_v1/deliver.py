"""Publish bounded diagnostics honestly; numerical runs cannot be rerun here."""
from diag_context import *
import csv
import shutil
import subprocess

REPORT=PROJECT/'docs/reports/deep_baselines_identifiability_v1_zh.md'

def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def collect():
    runs=[];crows=[];gradrows=[];cases=[]
    completed_sensitivity={}
    if (ROOT/'final_case_sensitivity.json').exists():
        completed_sensitivity={r['run']:r for r in verified('final_case_sensitivity.json')['runs']}
    for result in sorted((ROOT/'runs').glob('*/result.json')):
        r=c.verified(result);p=c.verified(result.parent/'protocol.json');evaluations=c.rows(result.parent/'evaluations.jsonl');last=evaluations[-1]
        volumes=[b['synthetic_volume']['relative_l1'] for case in last['cases'] for b in case['branches'].values() if 'synthetic_volume' in b]
        volume_cases=[float(np.mean([b['synthetic_volume']['relative_l1'] for b in case['branches'].values()])) for case in last['cases'] if all('synthetic_volume' in b for b in case['branches'].values())]
        row=dict(experiment=p['experiment'],arm=p['arm'],version=p['version'],n=p['n'],updates=r['updates'],
            initial_AB=r['initial_train'],final_AB=r['final_train'],initial_H=r['initial_H'],final_H=r['final_H'],zero=1.,
            mean=float(np.mean([b['field']['mean'] for case in last['cases'] for b in case['branches'].values()])),
            spatial_cv=float(np.mean([b['field']['spatial_cv'] for case in last['cases'] for b in case['branches'].values()])),
            zero_intervention_MAE=float(np.mean([b['interventions']['zero']['mae'] for case in last['cases'] for b in case['branches'].values()])),
            synthetic_L1=None if not volumes else float(np.mean(volumes)),status=r['status'],interpretation=r['interpretation'],
            synthetic_cases_better_than_zero=None if not volume_cases else sum(v<1. for v in volume_cases),
            train_gate=r['checks']['train_improvement'],H_gate=r['checks']['independent_H'],response_gate=r['checks']['meaningful_repeated_response'],
            structure_gate=r['checks']['nontrivial_structure'],input_gradient_gate=r['checks']['input_gradient'],FD_gate=r['checks']['finite_difference'],
            H_failed_cases_last_two=sum(any(ev['cases'][i]['H_relative_l1']>.99 for ev in evaluations[-2:]) for i in range(p['n'])),
            checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],protocol_sha256=c.sha(result.parent/'protocol.json'),
            initial_state_sha256=r['initial_state_sha256'],peak_GiB=r['peak_allocated_bytes']/1024**3,elapsed_seconds=r['elapsed_seconds'])
        runs.append(row)
        row['original_run_status']=r['status']
        if result.parent.name in completed_sensitivity:
            proof=completed_sensitivity[result.parent.name]
            row['input_gradient_gate']=proof['all_case_input_gradient'];row['FD_gate']=proof['all_case_finite_difference']
            if r['status']=='GO_DIAGNOSTIC_ONLY' and not (row['input_gradient_gate'] and row['FD_gate']):row['status']='INCONCLUSIVE_SENSITIVITY_AUDIT'
        for case in last['cases']:
            for branch,b in case['branches'].items():
                cases.append(dict(run=result.parent.name,case=case['case'],case_index=case['case_index'],branch=branch,
                    AB_L1=case['train_relative_l1'],AB_L2=case['train_relative_l2'],H_L1=case['H_relative_l1'],H_L2=case['H_relative_l2'],
                    **b['field'],zero_input_MAE=b['interventions']['zero']['mae'],perm_input_MAE=b['interventions']['permuted']['mae'],
                    **{'volume_'+k:v for k,v in b.get('synthetic_volume',{}).items()}))
    for arm in ARMS:
        states=c.verified(ROOT/f'C/{arm}/result.json')['states']
        for state in states:
            normals=[r for case in state['cases'] for r in case['responses'] if r['intervention']=='normal']
            zeros=[r for case in state['cases'] for r in case['responses'] if r['intervention']=='zero']
            perms=[r for case in state['cases'] for r in case['responses'] if r['intervention']=='permuted']
            enc='phi' if arm=='deepsets' else 'measurement_encoder'
            latent='masked_mean' if arm=='deepsets' else 'query_1_first_chunk'
            crows.append(dict(arm=arm,state=state['state']['name'],state_sha256=state['state_sha256'],checkpoint=state['state']['path'],
                checkpoint_sha256=state['checkpoint_sha256'],identical_state_alias=state.get('reused_identical_state'),
                H=float(np.mean([r['H']['relative_l1'] for r in normals])),mean=float(np.mean([r['output']['mean'] for r in normals])),
                zero_MAE=float(np.mean([r['difference']['mae'] for r in zeros])),
                zero_fixed_scale_relative=float(np.mean([r['difference']['stabilized_relative_mae'] for r in zeros])),
                perm_MAE=float(np.mean([r['difference']['mae'] for r in perms])),
                perm_latent_relative_delta=float(np.mean([r['representation'][latent]['relative_delta_l2'] for r in perms])),
                encoder_relative_delta=float(np.mean([r['representation'][enc]['relative_delta_l2'] for r in zeros])),
                latent_relative_delta=float(np.mean([r['representation'][latent]['relative_delta_l2'] for r in zeros])),
                logit_relative_delta=float(np.mean([r['representation']['logits_first_chunk']['relative_delta_l2'] for r in zeros])),
                saturated_logits_fraction=float(np.mean([case['gradients']['logits']['fraction_below_minus12'] for case in state['cases']])),
                sigmoid_derivative_mean=float(np.mean([case['gradients']['sigmoid_derivative_mean'] for case in state['cases']])),
                optimizer_updates=0,status='C_NO_GO_EFFECTIVE_RESPONSE' if all(r['difference']['stabilized_relative_mae']<.001 for r in zeros+perms) else 'C_RESPONSE_ONLY_INCONCLUSIVE'))
            for case in state['cases']:
                for loss in ('observed','cross','consistency'):
                    for layer,value in case['gradients'][loss]['layers'].items():
                        gradrows.append(dict(arm=arm,state=state['state']['name'],case=case['case'],loss=loss,layer=layer,**value))
    return runs,crows,gradrows,cases

def plots(runs,crows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for r in runs:
        target=ROOT/'runs'/f"{r['experiment']}_{r['arm']}_{r['version']}_n{r['n']}"
        ev=[c.verified(target/'initial.json'),*c.rows(target/'evaluations.jsonl')]
        for column,key in enumerate(('train_relative_l1','H_relative_l1')):
            ax=axes[0 if r['experiment']=='A' else 1,column]
            ax.plot([v['update'] for v in ev],[v[key] for v in ev],marker='.',label=f"{r['arm']} {r['version']} n{r['n']}")
    for i in range(2):
        for j in range(2):
            ax=axes[i,j];ax.axhline(1,color='black',ls='--',label='zero=1');ax.axhline(.99,color='gray',ls=':')
            ax.set_title(('A real train' if i==0 else 'B synthetic')+(' A/B' if j==0 else ' independent H'))
            ax.set_xlabel('optimizer updates');ax.set_ylabel('Relative L1');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(ART/'measurement_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm,ax in zip(ARMS,axes):
        rows=[r for r in crows if r['arm']==arm];xx=np.arange(len(rows))
        ax.bar(xx,[max(r['zero_MAE'],1e-30) for r in rows]);ax.set_yscale('log')
        ax.axhline(design()['initial_mass_fraction']*.001,color='red',ls='--',label='meaningful response threshold')
        ax.set_xticks(xx,[r['state'] for r in rows],rotation=45,ha='right');ax.set_title(arm+' C measurement-zero intervention')
        ax.set_ylabel('absolute field MAE');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.savefig(ART/'C_absolute_response.png',dpi=160);plt.close(fig)
    # Synthetic fields used solely for the requested independent diagnostic view.
    fig,axes=plt.subplots(3,5,figsize=(13,8),constrained_layout=True)
    for i in range(3):
        truth=np.load(ROOT/f'inputs/synthetic_truth_{i}.npy',allow_pickle=False)
        fields=[truth]
        for arm in ARMS:
            for version in ('v2','v3'):
                with np.load(ROOT/f'runs/B_{arm}_{version}_n4/raw/update200_case{i}.npz',allow_pickle=False) as a:fields.append(a['field_A'][0,0])
        maximum=max(float(x.max()) for x in fields)
        for j,(label,value) in enumerate(zip(['analytic','DeepSets v2','DeepSets v3','GNOT v2','GNOT v3'],fields)):
            im=axes[i,j].imshow(value[:,:,32].T,origin='lower',extent=(0,.6,0,.6),vmin=0,vmax=maximum,cmap='magma')
            axes[i,j].set_title(label if i==0 else f'case {i}');axes[i,j].set_xlabel('x (m)');axes[i,j].set_ylabel('y (m)')
        fig.colorbar(im,ax=list(axes[i]),label='CH4 mass fraction, shared scale within row',shrink=.8)
    fig.savefig(ART/'synthetic_slices.png',dpi=160);plt.close(fig)

def main():
    assert not ART.exists() and not REPORT.exists(),'Never overwrite previous publication'
    independent=verified('independent_review.json');assert independent['passed']
    assert verified('integrity_audit.json')['passed'] and verified('spatial_audit.json')['passed']
    assert verified('final_case_sensitivity.json')['completed']
    registration=verified('registration.json');pre=verified('preregistration.json');amendment=verified('engineering_memory_v2.json')
    c.verify_hashes(registration['sources']);c.verify_hashes(registration['data_identity']);c.verify_hashes(registration['inputs'])
    c.verify_hashes(pre['historical_hashes']);c.verify_hashes(amendment['sources'])
    for name in ('engineering_dispatch_v3.json','engineering_bookkeeping_v4.json'):c.verify_hashes(verified(name)['sources'])
    c.verify_hashes(verified('final_case_sensitivity_plan.json')['sources'])
    for arm in ARMS:
        for state in c.verified(ROOT/f'C/{arm}/result.json')['states']:
            if state['state']['path'] is not None:assert c.sha(state['state']['path'])==state['checkpoint_sha256']
    main_train_anchor('main_after.json')
    assert c.read(legacy.ROOT/'queue_status.json')['state']=='paused_by_user'
    runs,crows,gradrows,cases=collect();assert sum(r['updates'] for r in runs)==2000
    assert len(runs)==10 and len(crows)==14
    old=c.verified(ROOT/'historical_git_tree.json')
    changed=subprocess.check_output(['git','diff','--name-only',old['commit']],text=True).splitlines()
    assert not changed,'Historical tracked file modified before publication: '+str(changed)
    ART.mkdir(parents=True)
    # Full resource snapshots and all large arrays stay local; publish GPU facts.
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.suffix in ('.pt','.npy','.npz','.lock','.tmp') or 'cache' in path.parts:continue
        if path.name=='resources.jsonl':continue
        dest=ART/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    resources=c.rows(ROOT/'scheduling/resources.jsonl')
    c.jsonl(ART/'scheduling/gpu_resources.jsonl',[{k:r[k] for k in ('at','gpus','gpu_processes','memory','load')} for r in resources])
    write_csv(ART/'experiment_summary.csv',runs);write_csv(ART/'C_state_summary.csv',crows)
    write_csv(ART/'C_layer_gradients.csv',gradrows);write_csv(ART/'final_case_branch_metrics.csv',cases)
    interventions=[]
    for arm in ARMS:
        for state in c.verified(ROOT/f'C/{arm}/result.json')['states']:
            for case in state['cases']:
                for value in case['responses']:
                    interventions.append(dict(arm=arm,state=state['state']['name'],case=case['case'],branch=value['branch'],
                        intervention=value['intervention'],H_L1=value['H']['relative_l1'],H_L2=value['H']['relative_l2'],
                        H_L1_minus_zero=value['H']['relative_l1']-1.,input_MAE_ppm_m=value['input_mae_ppm_m'],
                        input_RMSE_ppm_m=value['input_rmse_ppm_m'],repeat_bitwise=value['repeat_bitwise'],
                        **value['output'],**value['difference'],
                        **{layer+'_relative_change':v['relative_delta_l2'] for layer,v in value['representation'].items()}))
    write_csv(ART/'C_interventions.csv',interventions)
    plots(runs,crows)
    findings=[]
    for arm in ARMS:
        fresh=next(r for r in crows if r['arm']==arm and r['state']=='fresh_v2')
        original_fresh=next(r for r in crows if r['arm']==arm and r['state']=='fresh_v1')
        old=next(r for r in crows if r['arm']==arm and r['state']=='original_failed_best')
        findings.append(f"- {arm} fresh v2：置零测量引起encoder相对L2变化{fresh['encoder_relative_delta']:.6g}、latent/query变化{fresh['latent_relative_delta']:.6g}，输出绝对MAE仅{fresh['zero_MAE']:.6g}。原失败权重饱和logit比例{old['saturated_logits_fraction']:.6g}、sigmoid平均导数{old['sigmoid_derivative_mean']:.6g}，输出干预MAE为{old['zero_MAE']:.6g}。")
        findings.append(f"- {arm} fresh v1→v2输出置零响应缩小约{original_fresh['zero_MAE']/max(fresh['zero_MAE'],1e-30):.0f}倍；fresh v2的logits<-12比例为{fresh['saturated_logits_fraction']:.3g}。因此新初态的弱绝对响应不能直接等同于末态深饱和；readout×.01和低浓度sigmoid斜率的作用需与训练后饱和分开。置换后的latent/query相对变化仅{fresh['perm_latent_relative_delta']:.6g}，逐层完整变化见CSV；不同层的相对范数比不是信息容量或唯一因果证据。")
    empirical=[]
    real=[r for r in runs if r['experiment']=='A']
    if all(r['final_AB']>=.99 for r in real):
        empirical.append('- 六项真实少样本实验的训练A/B本身均未实质优于零场，故当前失败不能仅解释为“训练已拟合但独立H泛化差”；但200次预算也不能证明架构永远不能拟合。移除H后仍失败，说明H项不是出现失败的必要条件，不代表H对原训练完全无影响。')
    if any(r['experiment']=='B' and r['final_AB']<.99 for r in runs):
        empirical.append('- 合成条件中确有A/B测量改善；结合已知64³真场可将测量误差压到约0.2%以内的正对照，不支持“整个物理/梯度链完全不能拟合”这一绝对解释。测量改善、条件响应和三维恢复仍须分开判断。')
    if any(r['experiment']=='B' and r['synthetic_cases_better_than_zero']>0 for r in runs):
        empirical.append('- 部分合成案例的三维场误差确实低于零场，并出现非均匀空间结构，削弱“空间解码器完全不能表达非均匀场”的解释；这仍不能证明输出空间结构正确依赖输入测量。共同模板、几何捷径与有效条件映射需要区分。')
    for arm in ARMS:
        aa=[r for r in runs if r['experiment']=='A' and r['arm']==arm]
        bb=[r for r in runs if r['experiment']=='B' and r['arm']==arm]
        empirical.append(f"- {arm}真实1/2/4例：最终AB L1范围{min(r['final_AB'] for r in aa):.6g}–{max(r['final_AB'] for r in aa):.6g}，H范围{min(r['final_H'] for r in aa):.6g}–{max(r['final_H'] for r in aa):.6g}；通过条件门槛{sum(r['status']=='GO_DIAGNOSTIC_ONLY' for r in aa)}/3。")
        for r in bb:empirical.append(f"- {arm}合成{r['version']}：AB={r['final_AB']:.6g}、H={r['final_H']:.6g}，三维L1={r['synthetic_L1']:.6g}（{r['synthetic_cases_better_than_zero']}/4案例优于零场），置零MAE={r['zero_intervention_MAE']:.3g}；独立H连续两次/所有case门槛={r['H_gate']}、条件响应门槛={r['response_gate']}。")
    precision=registration['synthetic_precision'];go=[r for r in runs if r['status']=='GO_DIAGNOSTIC_ONLY']
    lines=['# DeepSets / GNOT 条件学习能力与塌缩诊断 v1','',f"完成时间：{c.now()}。起始main commit：`{pre['commit']}`，已fetch并核验远端main一致。",'',
        f"A/B共10项、每项200次有效optimizer更新，合计2000；C覆盖14个模型状态、每状态2个固定train案例，更新数0。初步条件学习GO为{len(go)}/10。**所有结果均为诊断，不授权正式300epoch、第二种子或Senseiver；原正式队列保持暂停。**",'',
        '## 1. 目的、预登记与信息边界','',
        'A检验真实少样本A/B拟合与同案例独立H泛化，B检验受控解析非均匀场，C定位测量信号传播及饱和。三者分开执行和解释。C先运行。A固定train索引0/1/2/3的嵌套1/2/4例；C为前2例；A4/B4/H4按原sampler、seed20261011、epoch0固定且两两不交。完整案例/站位及输入SHA见registration.json。',
        '全新v2测量尺度初始化，四个B条件同一模型初态逐位相同；没有从失败或监督权重warm-start。64³/FP32/全部有效射线/K96不变。独立诊断loss=0.5cross+0.1observed+0.05consistency，完全移除hold。A只用v2；B仅v2/v3 consistency差别。AdamW2e-4/wd1e-4/clip1、固定LR、batch=min(2,N)、固定顺序，每项恰好200次。固定LR是本轮诊断协议，不冒充原cosine300正式训练。',
        '真实实验A的H只在评价函数抽取，无梯度评价，不作为输入、不进入训练目标；训练函数只有A/B能力。B的H在合成生成期封存，训练步骤不读取，只有评价函数读取。生成B几何时复用getter曾临时载入所选真实train测量（包括H），随后全部覆盖为合成积分，原值未用于合成定义、初始化或目标。合成真场仅在生成及独立评价访问。新的train-only适配器复用原getter，但不调用读取混合统计JSON的构造函数；Y_SCALE和初始化只复用既有合法train测量统计常量。历史train_statistics.json只由冻结身份核验逐字节读取以计算SHA，未解析其GT统计字段，也未用其设计初始化；数据适配器不打开该文件。无真实GT数组、GT统计数值、val/test数组、GT128、Pool B或监督权重使用。冻结主模型回归改用同一个合法train输入，避免为回归额外读取val。',
        '首次读取测量前封存preregistration.json及SHA。阈值：全部case训练L1<=.99且较初态改善>=.01，H连续两次评价<=.99，全部分支mean>=1e-8、空间CV>=.05，置零或固定有效射线置换的绝对MAE/固定C_init>=.001（绝对约4.20e-6），重复前向逐位相同；输入measurement-channel gradient L2>1e-12且AD/FD有可辨认证据。低于分辨率的FD不冒充有效响应。阈值未改变。每个候选全部条件满足才GO；预算耗尽而证据不足为INCONCLUSIVE。',
        'C的operator/query/readout梯度tap只覆盖首个8192-query块；encoder、token、DeepSets聚合为完整表示，整体输出及H投影仍为完整64³。没有将局部query梯度冒充整个解码器。C无训练改善条件，因此最多提供响应证据，不能单独判训练科学GO。A/B原运行详细梯度/FD采样首例；独立验收再以零optimizer更新补齐其余20个case终态审计，合计30个case，完全复用原阈值。最终GO按全部case检查；原run初筛status保留，experiment_summary.csv另列original_run_status以区分。',
        '', '## 2. A：真实train少样本测量过拟合','',
        '| 模型 | 案例数 | updates | AB初始→最终L1 | H初始→最终L1 | mean / 空间CV | 置零MAE | 结果 |',
        '|---|---:|---:|---|---|---|---:|---|']
    for r in runs:
        if r['experiment']=='A':lines.append(f"| {r['arm']} | {r['n']} | 200 | {r['initial_AB']:.6f} → {r['final_AB']:.6f} | {r['initial_H']:.6f} → {r['final_H']:.6f} | {r['mean']:.3g} / {r['spatial_cv']:.3g} | {r['zero_intervention_MAE']:.3g} | {r['status']} |")
    lines+=['','AB为两分支对A/B全部四条预测-目标组合先在case内平均，再case宏平均；H为两分支先case内平均。零场L1/L2均按相同目标/掩码重算，精确1.0。表中mean/CV/干预是分支与case均值，仅展示；GO按全部case逐一检查，不以平均掩盖失败。L2、discrepancy、每条投影、每步loss/LR/模块梯度、峰值显存均有原始记录。spatial_and_logit_statistics.csv给出每个评价点的跨案例绝对MAE/voxel std和logits分位数；单案例的跨案例差异记NA。中间logits由保存FP32 sigmoid输出反算并注明端点clamp，初始/最终另有真实hook logits，不混淆两者。训练L1从高尺度下降到1附近不自动等于学会空间反演。','',
        '## 3. B：解析羽流合成反演','',
        '固定四例：左侧各向异性单高斯、右侧旋转高斯、上侧旋转高斯和双羽流；center/sigma/rotation/peak在预登记design.json中固定，峰值由既有train-only初始化尺度的固定倍数定义，不来自真实GT。连续解析质量分数通过独立NumPy FP64 Gauss-Legendre256阶积分和显式物理换算生成目标，反演使用原64³/K96可微投影器。',
        f"128/256阶参考积分最大绝对差为{max(r['max_abs_reference_difference_ppm_m'] for r in precision):.3g} ppm·m，最大相对L1差{max(r['reference_relative_l1_difference'] for r in precision):.3g}；64³/K96对独立参考的相对L1离散差范围{min(r['grid64_K96_vs_reference_relative_l1'] for r in precision):.4%}–{max(r['grid64_K96_vs_reference_relative_l1'] for r in precision):.4%}。128/256阶比较在FP64转存前计算，训练目标另保存为FP32并有舍入；上述64³/K96比较使用实际FP32目标。避免完全同路径生成/反演，但仍是相同物理假设、理想无噪声的有限合成例，不能证明真实反演。",'',
        '| 模型/条件 | updates | AB初始→最终 | H初始→最终 | H末两次失败case数 | 合成三维L1 | mean / CV | 置零MAE | 结果 |',
        '|---|---:|---|---|---:|---:|---|---:|---|']
    for r in runs:
        if r['experiment']=='B':lines.append(f"| {r['arm']} {r['version']} | 200 | {r['initial_AB']:.6f} → {r['final_AB']:.6f} | {r['initial_H']:.6f} → {r['final_H']:.6f} | {r['H_failed_cases_last_two']} / 4 | {r['synthetic_L1']:.6f} | {r['mean']:.3g} / {r['spatial_cv']:.3g} | {r['zero_intervention_MAE']:.3g} | {r['status']} |")
    lines+=['','合成体积L1为case/分支宏平均，零场为1；逐case的MAE/RMSE、峰值幅度与位置、空间std在final_case_branch_metrics.csv。H测量改善和合成三维恢复必须分别判断；平滑或共同空间模板也可能降低测量误差。四例使用不同的固定站位子集，不能完全排除几何识别案例的捷径；置零/置换响应也不等于对有效测量分布上的正确条件映射。',
        '', '![A/B完整预算测量曲线](artifacts/deep_baselines_identifiability_v1/measurement_curves.png)','',
        '![合成真场及最终预测中心切片；每行共用色标](artifacts/deep_baselines_identifiability_v1/synthetic_slices.png)','',
        '## 4. C：信息传播、逐层梯度与输入敏感性','',
        '| 模型/状态 | H L1 | 置零 / 置换MAE | encoder相对变化 | latent/query相对变化 | logits<-12比例 | C判定 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in crows:lines.append(f"| {r['arm']} {r['state']} | {r['H']:.6g} | {r['zero_MAE']:.3g} / {r['perm_MAE']:.3g} | {r['encoder_relative_delta']:.3g} | {r['latent_relative_delta']:.3g} | {r['saturated_logits_fraction']:.3g} | {r['status']} |")
    lines+=['','C表为前2train案例、A/B输入的平均；梯度/AD-FD为A输入路径，B参与相同A/B诊断目标。正常、0、.5、1.5、有效射线固定置换全部保存。输入measurement通道与202个geometry通道分别记录gradient L2/max/finite_fraction，另含phi/mean/latent或measurement/geometry encoder、每层operator/query、末层weight/bias和logits。中间表示的相对变化使用自身稳定范数；最终输出一律另报绝对变化及固定初始化尺度，不能用近零输出作分母夸大响应。相同权重state哈希的best/last明确复用诊断，checkpoint身份仍各自记录。',
        *findings,'',
        'AD/FD对输入共同幅度方向分别用完整输出均值/固定尺度及平均logits，epsilon=.01/.001；允许5%相对误差加1e-8绝对容差。所有原始AD/FD和误差可复核；低幅度、FP32相消或非线性处不满足数值核验的项目标precision-limited/不通过，不强行声称所有状态敏感性都被精确验证。',
        '最终30个case的补充审计见[全部case敏感性记录](artifacts/deep_baselines_identifiability_v1/final_case_sensitivity.json)。下表按每项全部case要求判定；AD/FD未通过表示当前数值证据不足，不等于梯度为零，也不覆盖已有独立失败门槛。','',
        '| 实验 | 输入测量梯度全部通过 | 输出AD/FD全部通过 |',
        '|---|---|---|',
        *[f"| {r['experiment']}_{r['arm']}_{r['version']}_n{r['n']} | {r['input_gradient_gate']} | {r['FD_gate']} |" for r in runs],
        '', '![C输出绝对干预变化](artifacts/deep_baselines_identifiability_v1/C_absolute_response.png)','',
        '## 5. 证据分级与当前不能证明的结论','',
        *empirical,'',
        '**已验证事实：** 测量信息并非在token入口被全部丢弃；C的encoder和聚合/operator表示存在可重复变化。旧失败权重深负logits及其sigmoid导数衰减保留了饱和机制证据；饱和是末态现象，不能由此证明唯一训练起因。v2 readout权重×.01及负bias虽修正幅度，也会在初态进一步减弱输出对同一latent变化的绝对响应；fresh v1/v2共享其余参数，C可直接核对此影响。A/B的条件门槛、逐case预测/测量指标及失败条件均来自保存数值，见表和CSV。',
        '**较强假设：** 当前表示、条件解码及稀疏测量loss之间的优化匹配仍有困难，不能只归因于consistency固定epsilon。若B的H降低而三维结构或测量响应失败，则共同空间先验/几何捷径和非唯一解是合理替代解释；不能仅以非零encoder梯度说明它学会了正确反演。',
        '**未解决问题：** 200次更新是否足以获得最优条件映射、全局聚合是否构成决定性瓶颈、同几何下测量变化的可读性、坐标解码器与相对L1的作用各占多少，本轮不能唯一分离。A前4例为来源清单顺序中的相近条件，B仅4个理想例；本轮不支持完整数据分布或充分调优算法的普遍能力结论。没有真实三维GT评价，不能声称真实浓度场恢复。NO-GO仅表示本轮固定条件和预算下未过门槛；即便如此，架构在更充分且合理优化下的能力仍是未解决问题，不能自动增加预算来回答。',
        '', '## 6. 独立复核、工程异常与资源','',
        f"独立review.py直接从保存的逐射线预测/目标/掩码以NumPy重算{independent['recomputed_measurement_fields']}个L1/L2字段、零场、case宏平均、空间std/干预MAE和B体积指标，并独立重建GO条件；不调用训练scores/decide。复核2000次AdamW step、所有C状态0更新、B初态及输入身份一致。额外独立合成能力测试将H放大1000倍，A/B诊断loss及模型参数梯度逐位相同，H输入/几何梯度均不存在，网络只见A/B；该审计不执行optimizer更新。新CPU工程13 passed+3 subtests，显存adapter等价1 passed、精确路径dispatch1 passed；前后冻结主模型同合法train输入C0/C1逐位一致。双种子权重、冻结源码/配置、原loss及历史已跟踪结果内容均未改变。",'',
        '保留三项工程异常与恢复记录。其一：GNOT C在完整输入梯度FD阶段OOM，发生在任何训练更新之前。原控制器停止该任务并释放GPU0，失败日志与部分数组移到engineering_failures，未覆盖。仅新增engineering_memory_v2.py：tap用weakref避免checkpoint closure保留Trace/计算图，独立梯度调用间gc.collect；CPU验证前向和输入梯度逐位一致。原登记数值文件和阈值不改，amendment及新源码SHA先封存，再按--recover-reason重试；没有减小64³、查询范围、射线数、loss或科学预算。C DeepSets保留原成功版本，GNOT及十项训练调用等价的内存adapter，命令和映射可追溯。其二：首项训练启动按通用名称导入了历史train模块，在建立run目录或任何更新之前退出；新增显式文件路径dispatch，CPU回归通过，原登记文件不改。其三：首项DeepSets1例已封存200次更新、checkpoint及NO-GO结果，末尾status写入重复at参数报错。核验全部200条日志、AdamW step及checkpoint哈希后接收该科学结果，仅修复状态，不重跑更新；其余九项使用逐行一致、仅修正该状态写入的一份新train_fixed_v2.py，差异与源码SHA预先登记。',
        '全程只用GPU0全局UUID flock，15秒双空闲快照/FD继承/外来冲突监控；等待最多60秒。GPU1–3其他用户任务未停止或抢占。全部训练更新恰好2000，无额外smoke optimizer更新；C/CPU测试均0。每项耗时/峰值显存、调度、访问和恢复记录已保存。主模型回归只读权重并前向；未重训/微调。',
        '', '## 7. 唯一建议的下一步与停止边界','',
        '建议另行授权一个**固定完全相同A/B/H几何、多个不同解析羽流的配对条件读出实验**，同时比较正常测量与跨羽流测量交换。理由：本轮已能定位“内部响应存在但输出是否有效”的差距，现有不同案例的站位子集仍允许几何捷径，而置零/射线置换可能偏离正常测量分布。相同几何下交换物理合法测量可以更直接检验条件解码，而不需要新架构、GT或宽泛搜索。此建议未执行；即使本轮有诊断GO，也需用户审阅后决定下一步。',
        '', '## 8. 产物入口','',
        '- [独立诊断状态](artifacts/deep_baselines_identifiability_v1/final_status.json) · [十项实验汇总](artifacts/deep_baselines_identifiability_v1/experiment_summary.csv) · [逐case指标](artifacts/deep_baselines_identifiability_v1/case_metrics.csv)',
        '- [C状态及checkpoint身份](artifacts/deep_baselines_identifiability_v1/C_state_summary.csv) · [全部五种干预](artifacts/deep_baselines_identifiability_v1/C_interventions.csv) · [逐层梯度](artifacts/deep_baselines_identifiability_v1/C_layer_gradients.csv) · [最终case/branch及合成指标](artifacts/deep_baselines_identifiability_v1/final_case_branch_metrics.csv)',
        '- [独立复核](artifacts/deep_baselines_identifiability_v1/independent_review.json) · [H隔离和历史完整性](artifacts/deep_baselines_identifiability_v1/integrity_audit.json) · [跨案例/中间logits](artifacts/deep_baselines_identifiability_v1/spatial_and_logit_statistics.csv) · [输入/源码登记](artifacts/deep_baselines_identifiability_v1/registration.json) · [首次运行前协议](artifacts/deep_baselines_identifiability_v1/preregistration.json) · [内存修复登记](artifacts/deep_baselines_identifiability_v1/engineering_memory_v2.json)',
        '- [预登记说明](../deep_baselines_identifiability_v1_protocol_zh.md) · [最终主模型回归](artifacts/deep_baselines_identifiability_v1/main_after.json)',
        '',f'全部原始数组及checkpoint位于 `{ROOT}`，每个run独立，哈希在JSON/CSV。Git仅保存小体积日志、配置、SHA、CSV、图片及报告，不上传真实GT或完整预测大数组。']
    REPORT.write_text('\n'.join(lines)+'\n')
    status=dict(at=c.now(),state='DIAGNOSTICS_COMPLETE',A=dict(runs=6,updates=1200),B=dict(runs=4,updates=800),C=dict(states=14,updates=0),
        total_updates=2000,limit=2500,GO_diagnostic_candidates=[r for r in runs if r['status']=='GO_DIAGNOSTIC_ONLY'],
        formal_training_authorized=False,second_seed_authorized=False,old_queue_paused=True,
        independent_review_passed=True,main_regression_passed=True,historical_evidence_unchanged=True,
        real_GT_access=False,val_test_array_access=False,engineering_OOMs=1,engineering_import_failures=1,post_result_bookkeeping_failures=1,updates_replayed=0)
    status['all_case_sensitivity_audit_completed']=True;status['supplementary_sensitivity_optimizer_updates']=0
    c.freeze(ART/'final_status.json',status);c.freeze(ROOT/'final_status.json',status)
    c.write(ROOT/'diagnostic_status.json',status)
    files={str(p.relative_to(PROJECT)):c.sha(p) for root in (ART,HERE,PROJECT/'configs/baselines/identifiability_v1') for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files[str(REPORT.relative_to(PROJECT))]=c.sha(REPORT);files[str(PROTOCOL.relative_to(PROJECT))]=c.sha(PROTOCOL)
    for p in PROJECT.glob('tests/test_baseline_identifiability*.py'):files[str(p.relative_to(PROJECT))]=c.sha(p)
    c.freeze(ART/'publication_manifest.json',dict(at=c.now(),files_sha256=files,manifest_self_excluded=True))
    print('DELIVERED',REPORT,'GO',len(go),flush=True)

if __name__=='__main__':main()
