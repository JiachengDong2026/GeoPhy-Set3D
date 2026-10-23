"""Deterministic Chinese reports from audited machine evidence; no hand-filled results."""
from context import *
from analysis import KEYS

REPORTS=('mrgr_plain_final_model_freeze_v1_zh.md','mrgr_plain_final_classic_comparison_v1_zh.md',
         'mrgr_plain_final_tables_v1_zh.md','mrgr_plain_final_results_v1_zh.md','mrgr_independent_test_readiness_v1_zh.md')
LABELS={MAIN:'Plain 主模型',SECOND:'Plain 第二种子','NNLS-NL':'NNLS-NL','Tikhonov':'Tikhonov','TV':'TV','SART':'SART（补充）'}
LAYOUT={'Val-A-V4':'普通 V4','spread-V4':'spread V4','V3':'V3','V6':'V6','clustered-V4':'聚簇 V4'}
ARMS={core.GEOMETRY:'无 GeometryLifting',core.SINGLE:'单尺度 MRGR',core.NO_PRF:'无 PRF',PLAIN:'完整模型'}

def number(x,scale=1):return '缺失' if x is None else f'{x*scale:.4f}'
def interval(v,scale=1):return '缺失' if v is None else '['+', '.join(number(x,scale) for x in v)+']'
def table(head,rr):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rr])
def save(name,paragraphs):
    p=PROJECT/'docs/reports'/name;assert not p.exists();p.write_text('\n\n'.join(paragraphs)+'\n')

def main():
    verify_hashes(verified(BASE/'protocol.json')['sources']);manifest=verified(BASE/'model_manifest.json');comp=verified(BASE/'comparison_manifest.json')
    summary=verified(BASE/'summary.json');access=verified(BASE/'GT64_access.json');independent=verified(BASE/'independent_data_audit.json')
    verification=verified(BASE/'independent_statistics_verification.json');assert verification['passed']
    models=manifest['models'];primary=models[0];stat=verified(CORE/'statistics.json');core_rows=[r for r in stat['summaries'] if r['stage']=='C1' and r['metric']=='relative_l1' and (r['condition']=='Val-A-V4' or len(r['draws'])==5)]
    costs=list(csv.DictReader((BASE/'paper_complexity.csv').open()));panels=summary['panels'];paired=summary['paired']
    def panel(method,condition,stage='C1'):return next(r for r in panels if (r['method'],r['condition'],r['stage'])==(method,condition,stage))['metrics']
    def pair(method,condition,metric='volume.relative_l1'):return next(r for r in paired if (r['reference'],r['condition'],r['metric'])==(method,condition,metric))
    stamp=f'生成时间：{now()}。所有数字由本轮机器证据自动生成。'
    limitation='本轮沿用已参与研发/比较的35个历史案例，全部比较与区间均属探索性结果，不能替代独立盲测。bootstrap单位为案例；多draw先在案例内平均；两个训练种子分别报告，案例区间不是跨种子不确定性。'
    save(REPORTS[0],[ '# 普通集合交互最终主模型冻结报告',stamp,
        f"已冻结 `{MAIN}`，最佳epoch {primary['best_epoch']}，参数量 {primary['parameters']:,}。主种子由用户指定，不按历史测试误差重新排名。",
        table(['身份','种子','最佳epoch','参数量','checkpoint SHA256'],[[m['role'],m['seed'],m['best_epoch'],m['parameters'],f"`{m['checkpoint_sha256']}`"] for m in models]),
        '两份checkpoint均实际加载并严格校验state dict、内部config、protocol哈希、epoch及GT标记，文件大小与实际路径见[model_manifest](artifacts/mrgr_plain_final_v1/model_manifest.json)。完整300epoch/24,900次提交更新及日志最低验证H4选轮已逐行复核；last.pt保留优化器/调度器/RNG恢复状态。',
        '架构：Encoder → GeometryLifting → Plain Set Interaction → 多尺度MRGR → UniformFusion → PRF。当前主模型不是历史MRGR、MRGR-UniformFusion或LR4E4。继承config里的`geolift_oari_mrgr_res3d`名称不能独立决定工厂；必须通过普通交互arm分派。',
        '训练166案例、开发验证36案例。线性p99.5归一化标尺32284.894580078173，仅训练统计；数据划分/数据文件与统计哈希已登记。batch2，300轮，AdamW lr2e-4/wd1e-4，cosine300，FP32/TF32关闭，clip1；损失hold+0.5cross+0.1observed+0.05consistency，全部沿用原始配置。',
        'C0为初始解码；C1为带符号测量残差反馈后的解码。输出[B,1,64,64,64]、XYZ端点[0,0.6]m、CH4质量分数[0,1]。正式P_diff在质量分数体上三线性采样后转换摩尔分数，K96中点积分，观测单位ppm·m。',
        '访问日志、训练摘要及源码显示这两次正式训练和验证选轮未读取测试GT；本轮无optimizer、无反向训练、无种子/超参搜索。历史35案例已参与模型研发，故不能据此声称整个架构开发完全未利用测试信息。',
        'CPU与独占GPU smoke均通过：严格加载、前向形状/有限值/范围、eval/no_grad及权重buffer规范张量哈希前后一致。主模型全部595组预测已封存，原checkpoint哈希在最终验收再次核验。',
        '复现（在仓库根目录）：\n```python\nimport sys, torch\nsys.path.insert(0, "scripts/plain_final_v1")\nfrom context import verified, BASE, model_for, PLAIN\nm = verified(BASE / "model_manifest.json")["models"][0]\nmodel = model_for(m["config"], PLAIN, "cpu").eval()\nck = torch.load(m["checkpoint"], map_location="cpu", weights_only=False)\nmodel.load_state_dict(ck["model"], strict=True)\n# 测量输入及布局由context.dataset和batch_for构造，无GT输入\n```',
        f"环境、完整配置、源码快照与协议位于`{BASE}`；Python可执行文件`{sys.executable}`。轻量身份及哈希已导出到[证据目录](artifacts/mrgr_plain_final_v1/model_manifest.json)。",limitation])
    rows_main=[]
    for condition in CONDITIONS:
        for method in (MAIN,'NNLS-NL','Tikhonov','TV'):
            p=panel(method,condition);l=p['volume.relative_l1']
            rows_main.append([LAYOUT[condition],LABELS[method],number(l['mean'],100),number(l['median'],100),interval(l['ci95_mean'],100),number(p['volume.relative_l2']['mean'],100),number(p['morphology.dice_0.5']['mean'])])
    pairedrows=[]
    for condition in CONDITIONS:
        for method in CLASSICS:
            p=pair(method,condition);pairedrows.append([LAYOUT[condition],LABELS[method],number(p['mean'],100),number(p['median'],100),interval(p['ci95'],100),interval(p['bonferroni4_ci'],100),interval(p['historical_bonferroni20_ci'],100),f"{p['decreased']}/{p['increased']}/{p['unchanged']}"])
    secondary=[]
    for condition in CONDITIONS:
        for method in (MAIN,*CLASSICS):
            p=panel(method,condition)
            secondary.append([LAYOUT[condition],LABELS[method],number(p['morphology.absolute_peak_ratio_error']['mean'],100),number(p['volume.peak_location_error_mm']['mean']),number(p['volume.centroid_error_mm']['mean']),number(p['morphology.dice_0.5']['mean']),number(p['measurement.observed.relative_l1']['mean'],100),number(p['measurement.heldout.relative_l1']['mean'],100)])
    runtime=[]
    raw=rows(BASE/'metrics.jsonl')
    for method in (MAIN,*CLASSICS):
        r=[r for r in raw if r['method']==method and r['stage']=='C1'];seconds=[x['reconstruct_seconds'] for x in r];setup=[x['setup_seconds'] for x in r if x['setup_seconds'] is not None]
        runtime.append([LABELS[method],number(float(np.median(seconds))*1000),number(float(np.median(setup))*1000) if setup else '未纳入前向计时',number(max(x['peak_allocated_bytes'] for x in r)/1024**2),'CPU CSR+checkpoint I/O' if method=='SART' else 'PG-BB+checkpoint I/O' if method in CLASSICS else 'GPU forward，同批量混合视图数、含首次冷启动'])
    worst=[]
    for r in summary['worst_main_cases']:
        if r['condition'] in ('Val-A-V4','clustered-V4'):worst.append([LAYOUT[r['condition']],r['case_id'],number(r['metrics']['volume.relative_l1'],100)])
    save(REPORTS[1],['# 最终普通交互模型与经典算法的探索性比较',stamp,limitation,
        '复用审计通过：四经典各595组、共2380组，主模型既有210组保持原路径与文件哈希，只新增385组。未重新求解经典算法、未重新训练任何模型；第二种子只保留既有210组及核心消融稳健性证据。共2975组预测/3570场先封存后评分，实际仅访问35份已授权GT64，GT128/Pool B未读取。',
        '源文件、历史全局封存、数据身份、每条配置/停止曲线、案例顺序、输入/H4站位、几何/有效mask、输出shape/范围/哈希均通过。SART最初登记之后有正式v2缓存修订；审计沿该修订与final_lock核验，未忽略源码差异。新input_manifest逐tensor登记原始/归一化投影、几何与mask哈希。',
        f"3570场从封存数组用同一正式volume/morphology实现重评分；其中{access['historical_fields_recomputed_and_matched']}场与旧同口径结果一致，最大差{access['maximum_old_metric_difference']:.3g}。经典投影诊断复用已冻结指标，并以固定首案例×五布局×四算法共20组复投影核验；主模型595组C0/C1投影诊断全部本轮补算。",
        'NNLS-NL/Tikhonov/TV共用非线性P_diff；上界/下界[0,1]，PG-BB最多200、最少40次，tol1e-5、patience8。Tikhonov λ0.001；TV λ1e-5、ε1e-4；参数来自训练校准冻结选择，不按历史测试调参。SART为节点摩尔分数线性CSR算子，K96、relaxation0.5，仅补充；其评价测量也使用非线性P_diff。',
        '主表为C1。L1/L2按百分比展示，区间为案例bootstrap均值95%。普通/spread draw0，其余五draw先案例内均值。唯一主要端点为普通V4 C1 L1，其余探索性。',
        table(['布局','方法','L1均值%','L1中位%','L1均值95%CI','L2均值%','Dice50'],rows_main),
        'SART补充表：'+ '\n\n'+table(['布局','L1均值%','L1中位%','L2均值%','Dice50'],[[LAYOUT[c],number(panel('SART',c)['volume.relative_l1']['mean'],100),number(panel('SART',c)['volume.relative_l1']['median'],100),number(panel('SART',c)['volume.relative_l2']['mean'],100),number(panel('SART',c)['morphology.dice_0.5']['mean'])] for c in CONDITIONS]),
        '配对Δ=主模型−经典，L1负值有利，单位百分点。增降相等阈值1e-8（未乘100前）。同时保存四比较Bonferroni区间与历史20对照保守区间；不会因SART放补表缩小family。',
        table(['布局','对照','Δ均值pp','Δ中位pp','95%CI','Bonf4 CI','历史Bonf20 CI','主模型更低/更高/相等'],pairedrows),
        '辅助指标均值（峰幅误差为|peak_ratio−1|%，位置/质心为mm；质心采用正式volume定义，另保存形态学质心，二者不可混用）：',
        table(['布局','方法','峰幅误差%','峰定位mm','质心mm','Dice50','输入投影L1%','H4投影L1%'],secondary),
        '失败/停止状态全部保留：'+ '\n\n'+table(['方法','应有/实际','缺失','数值失败','预算耗尽','目标稳定'],[[LABELS[r['method']],f"{r['expected']}/{r['count']}",r['missing'],r['numerical_failures'],r['budget_exhausted'],r['objective_stable']] for r in summary['failure_rates']]),
        '预算耗尽不是已证明收敛；本轮不增加经典预算。历史算法参数选择和非线性积分离散误差仍限制推论。',
        table(['布局','主模型高误差案例','L1%'],worst),
        '实际时间/内存记录（混合17设置，非统一性能benchmark）：'+ '\n\n'+table(['方法','重建/前向中位ms','setup中位ms','最大allocated MiB','边界'],runtime),
        '经典时间含求解及周期checkpoint I/O，setup另列；网络时间为batch1前向、排除加载/写盘/评分。SART运行CPU，GPU allocated=0并不表示无内存成本。CPU峰RSS见历史原始指标。不同边界不计算加速倍数。正式网络同GPU成本表见[论文表](mrgr_plain_final_tables_v1_zh.md)。',
        '[逐案例CSV](artifacts/mrgr_plain_final_v1/per_case_metrics.csv) · [汇总JSON，含全部指标均值/中位/P90/max/区间/配对向量](artifacts/mrgr_plain_final_v1/summary.json) · [来源与配置](artifacts/mrgr_plain_final_v1/comparison_manifest.json)。C0为诊断场，CSV/JSON完整保存，没有选择性删除。'])
    abltable=[]
    for r in core_rows:
        arm=r['model'].rsplit('_seed',1)[0];abltable.append([ARMS[arm],r['seed'],LAYOUT[r['condition']],number(r['baseline_mean'],100),number(r['treatment_mean'],100),number(r['mean'],100),number(r['median'],100),interval(r['ci95'],100),f"{r['increased']}/{r['decreased']}/{r['unchanged']}"])
    costtable=[]
    for r in costs:
        costtable.append([ARMS[r['key'].rsplit('_seed',1)[0]],r['seed'],r['parameters'],r['delta_parameters'],number(float(r['mean_forward_ms'])),number(float(r['delta_forward_ms'])),number(float(r['peak_allocated_MiB'])),number(float(r['train_seconds'])/3600)])
    save(REPORTS[2],['# 普通交互最终模型论文消融与成本表',stamp,limitation,
        '本表从已验收的[核心消融报告](mrgr_plain_core_ablations_v1_zh.md)和[实验追踪表](mrgr_plain_final_experiment_tracking_v1_zh.md)及其哈希封存机器产物导出。本轮未重复训练、未重做benchmark。六个消融均300epoch/24900更新；对应种子配对。',
        'C1 relative L1以%显示；Δ=消融−完整（百分点），正值为移除模块后变差。案例级20000次bootstrap，seed20261026，95%未校正探索性均值区间。普通draw0、聚簇五draw案例内平均，35案例。',
        table(['消融','种子','布局','完整L1%','消融L1%','Δ均值pp','Δ中位pp','Δ95%CI','变差/变好/相等'],abltable),
        '12组主要差值方向均为正，95%区间下界均大于0。两个种子均支持三模块在两布局下的独立L1收益；该一致性仅覆盖两个训练种子。多尺度MRGR在普通布局收益小、聚簇收益更明显；不能将这一观察写成所有指标一致优势。',
        '严格单因素定义：GeometryLifting替换为既有PlanarExtrusionLifting；单尺度只运行既有context基础尺度并复用正式通道处理（未删除MRGR）；无PRF删除residual_encoder/adapter且C1=C0，其他训练损失不变。前两项保留相同参数登记，因此参数不变不等于计算量不变。初始化RNG调用顺序差异见原audit，未声称所有共享参数逐一相同。',
        table(['模型','种子','参数','Δ参数','前向均值ms','Δ前向ms','峰allocated MiB','训练实际h'],costtable),
        '计时为同一独占RTX4090、FP32/TF32off、batch1，固定验证索引0/17/34，每例1次warmup+3次实测，return_stages=True。原始9次时间、标准差、UUID和训练日志路径见[成本CSV](artifacts/mrgr_plain_final_v1/paper_complexity.csv)及原benchmark。峰allocated是进程的实际峰，可能包含仍驻留对象/缓存，不能解释为独立增量工作集。训练h是训练引擎记录的elapsed，可能受恢复/暂停边界影响，不是GPU能耗。FLOPs、MACs、能耗均未测。',
        '辅助不利结果保留：seed11单尺度MRGR在峰定位上可优于完整模型；普通布局PRF的峰幅误差并非两种子统一改善。移除PRF节约12512参数且前向明显更短，应同时报告精度收益和成本。',
        '[消融CSV](artifacts/mrgr_plain_final_v1/paper_core_ablation.csv) · [成本CSV](artifacts/mrgr_plain_final_v1/paper_complexity.csv) · [统计独立复核](artifacts/mrgr_plain_final_v1/independent_statistics_verification.json)。'])
    paragraphs=['# 普通集合交互最终模型论文结果说明',stamp,limitation,
        '以下文字针对已冻结普通集合交互架构，历史MRGR/OARI结果仅作前序开发记录，不与本轮主模型混称。未绘方法架构图、未新增深度基线。']
    for arm in (core.GEOMETRY,core.SINGLE,core.NO_PRF):
        r=[r for r in core_rows if r['model'].rsplit('_seed',1)[0]==arm]
        paragraphs.append(f"{ARMS[arm]}：两个种子在普通/聚簇布局的C1 L1增量分别为"+'；'.join(f"seed{x['seed']} {LAYOUT[x['condition']]} {number(x['mean'],100)}个百分点（95%CI {interval(x['ci95'],100)}）" for x in r)+'。方向一致，案例区间不能替代训练种子分布。')
    paragraphs.append('多尺度MRGR的贡献依赖观测布局：普通V4差值较小，聚簇V4更大，适合表述为在非均匀观测下增强重建稳健性。GeometryLifting对L1的贡献最大；PRF提供额外精度，代价是第二阶段计算。三者可作为有消融证据支持的技术组成，但不能仅凭既有35案例和两个种子确证普适核心创新或独立泛化。')
    for condition in CONDITIONS:
        main=panel(MAIN,condition)['volume.relative_l1']['mean'];p=[pair(m,condition) for m in CLASSICS]
        paragraphs.append(f"{LAYOUT[condition]}：主模型C1 L1均值为{number(main,100)}%；"+'；'.join(f"相对{x['reference']}的配对均值差{number(x['mean'],100)}个百分点，95%CI {interval(x['ci95'],100)}，案例更低/更高/相等{x['decreased']}/{x['increased']}/{x['unchanged']}" for x in p)+'。以上为同案例配对，非新的盲测。')
    adverse=[]
    for r in paired:
        if r['metric'] not in KEYS or r['metric']=='volume.relative_l1':continue
        bad=r['mean']<0 if r['metric']=='morphology.dice_0.5' else r['mean']>0
        if bad:adverse.append([LAYOUT[r['condition']],r['reference'],r['metric'],number(r['mean']),interval(r['ci95'])])
    paragraphs.extend(['并非每个辅助指标都支持主模型。下表完整列出预先选定主要辅助指标中，主模型均值方向不利的项目（量纲为原始值，L1/L2为比例，定位为mm）；不将均值不利等同显著劣势。',table(['布局','对照','指标','主−经典均值','95%CI'],adverse) if adverse else '选定辅助指标没有主模型均值方向不利项目；仍需查看逐案例胜负和区间。',
        '结果存在布局和种子效应：MRGR普通布局效应较小且辅助定位结果可反向；PRF峰幅指标有种子差异。最小后续验证优先采用真正新独立案例及预注册布局，必要时另行预注册少量训练种子量化方差；本轮不自动扩大训练搜索。经典解可能受200步预算及正则校准范围限制，SART线性离散算子不同；不能据本轮结果断言所有经典配置的性能上限。',
        '主模型高误差案例、算法预算耗尽/数值失败、H4投影与体场误差差异均见[经典比较](mrgr_plain_final_classic_comparison_v1_zh.md)。案例level bootstrap只反映本35案例分布，不覆盖数据域偏移、训练随机性及重复开发选择。未证明独立测试泛化，状态见[独立测试审计](mrgr_independent_test_readiness_v1_zh.md)。'])
    save(REPORTS[3],paragraphs)
    save(REPORTS[4],['# 独立最终测试数据就绪审计',stamp,
        '**状态：BLOCKED_NO_INDEPENDENT_DATA。没有运行或宣称通过独立测试。**',
        f"审计范围：`{independent['scope']}`下项目自有raw/processed目录及现有分割、metadata、frame_manifest与GT访问记录；不涵盖尚未提供的服务器外数据。发现1套原始/1套processed数据，237个不同case/source-run/来源GT哈希，划分166训练、36验证、35历史测试；237个raw target仅列名/stat，不打开或重算GT128内容哈希。未发现未登记target或新增数据集。",
        '35历史案例已多次参与模型比较和探索，不能通过封存、改名、改变draw/视角/布局成为新独立测试。debug是既有案例子集，Pool B是相同case的另一观测池，不具案例独立性且本轮未启用。没有读取GT128/Pool B数组；本轮GT64评分仅限预测封存后的历史35案例。',
        f"source_index SHA256 `{independent['source_index_sha256']}`；raw manifest SHA256 `{independent['raw_manifest_sha256']}`。完整元数据、source-run、派生frame、split审计和历史访问证据见[机器审计](artifacts/mrgr_plain_final_v1/independent_data_audit.json)。来源GT哈希为既有metadata身份记录，不代表本轮读取GT128。",
        '下一阶段数据规划：至少先准备35个真正新增且case/source-run独立的案例，目标50–100。此数量是采集规划，未经功效计算；正式样本量须针对预注册效应/区间精度和目标域另行论证。覆盖普通/聚簇测量、已有与新增几何/工况的适用范围应预先明确。',
        '预注册与盲评分顺序：\n\n1. 新case/run/GT来源哈希与既有237案例去重，核查训练/调参/选模/GT访问历史；同一源运行派生数据按案例组处理。\n2. 固定主模型及第二种子角色、配置/源码/归一化/投影器与checkpoint哈希；预注册主要指标、布局、draw、样本量依据、统计family、失败和停止规则。\n3. 数据保管方登记GT64身份、生成流程及访问权限；模型方仅获得无GT输入；不自动访问GT128或启用Pool B。\n4. 全部方法预测与清单封存后，独立评分方一次性读取授权GT64；保留失败案例及全部主要端点，不回选模型。\n5. 对照既有探索性结论与新盲测结果，明确跨域适用边界。',
        '尚需用户决策：提供/组织新增独立source-run案例与GT64，并确定目标域和盲评保管人；确定后另行预注册采集数量与正式评价。当前冻结、经典比较、论文表格均已可交付，独立确认仍受数据阻塞。'])
    ART.mkdir(parents=True,exist_ok=True)
    # Explicit curated export; binary fields and checkpoints remain in scratch.
    names=['model_manifest.json','comparison_manifest.json','protocol.json','cases.json','layouts.json','input_manifest.json','independent_data_audit.json',
        'GT64_access.json','all_predictions_sealed.json','inference/complete.json','inference/manifest.jsonl','inference/measurement.jsonl',
        'prediction_manifest.json','summary.json','per_case_metrics.csv','paired_summary.csv','case_panels.jsonl','paper_core_ablation.csv','paper_complexity.csv',
        'independent_statistics_verification.json','checks/cpu.json','checks/gpu.json','scheduling/events.jsonl']
    exports={}
    for name in names:
        p=BASE/name;dst=ART/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(p,dst);exports[str(dst)]=sha(dst)
        side=p.with_suffix(p.suffix+'.sha256')
        if side.exists():shutil.copyfile(side,dst.with_suffix(dst.suffix+'.sha256'))
    report_hashes={str(PROJECT/'docs/reports'/n):sha(PROJECT/'docs/reports'/n) for n in REPORTS}
    for m in models:assert sha(m['checkpoint'])==m['checkpoint_sha256']
    for r in verified(BASE/'prediction_manifest.json'):assert sha(r['prediction'])==r['prediction_sha256']
    freeze(BASE/'acceptance.json',dict(at=now(),freeze_completed=True,classical_comparison_completed=True,paper_tables_completed=True,
        independent_test_status=independent['status'],independent_test_performed=False,primary_checkpoint_sha256=primary['checkpoint_sha256'],
        new_trainings=0,new_classical_reconstructions=0,new_main_predictions=385,reused_main_predictions=210,reused_classical_predictions=2380,
        scored_fields=3570,statistics_verified=verification['passed'],GT128_read=False,Pool_B_read=False,foreign_processes_signalled=0,
        sources_verified=True,reports=report_hashes,exports=exports))
    shutil.copyfile(BASE/'acceptance.json',ART/'acceptance.json');shutil.copyfile(BASE/'acceptance.json.sha256',ART/'acceptance.json.sha256')
    print('REPORTS_EXPORTED',ART,flush=True)


if __name__=='__main__':main()
