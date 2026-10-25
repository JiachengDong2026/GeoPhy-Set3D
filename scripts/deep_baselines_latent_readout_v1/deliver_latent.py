"""Publish compact diagnostic evidence; large arrays remain in scratch."""
from latent_context import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main():
    reg=verify_seal();review=read(ROOT/'independent_review.json');assert review['passed']
    ev=read(ROOT/'evaluation_fp64.json');seal=read(ROOT/'probe_seal.json');case=rows(ROOT/'case_metrics.csv');reps=rows(ROOT/'representation_differences.csv');improvements=rows(ROOT/'probe_improvements.csv')
    art=PROJECT/'docs/reports/artifacts/deep_baselines_latent_readout_v1';assert not art.exists();art.mkdir(parents=True)
    # Presentation does not update any existing historical file.
    allsources={str(p):sha(p) for p in HERE.glob('*.py')}
    allsources[str(PROJECT/'docs/deep_baselines_latent_readout_v1_protocol_zh.md')]=sha(PROJECT/'docs/deep_baselines_latent_readout_v1_protocol_zh.md')
    freeze(ROOT/'delivery_sources.json',dict(at=now(),sources=allsources,postregistration_additions=['exact_input_latent.py','review_latent.py','review_latent_fp64.py','pair_precision_latent.py','recheck_latent.py','swap_statistics_latent.py','deliver_latent.py'],scientific_preregistered_sources_unchanged=True))
    sourcechecks={k:sha(Path(k))==v for k,v in reg['sources'].items()};assert all(sourcechecks.values())
    originals={a:read(ROOT/f'extraction_{a}.json') for a in EXPECTED}
    status=dict(at=now(),state='COMPLETE',task='frozen representation / independent analytical ridge diagnostic',decisions=ev['decisions'],independent_review_passed=True,
        neural_updates=0,remaining_diagnostic_updates=100,original_checkpoints_unchanged=True,main_checkpoints_unchanged=True,historical_tracked_files_unchanged=True,old_queue_paused=True,
        formal_training_started=False,real_GT_access=False,GT128_access=False,synthetically_supervised_probe=True,case_split=dict(seen=list(range(6)),unseen=[6,7]),scratch=str(ROOT),source_seal_verified=sourcechecks)
    freeze(ROOT/'final_status.json',status)
    aggregate=[]
    for arm in EXPECTED:
        for b in ('A','B'):
            for role in ('seen','unseen'):
                for method in ('original','normal_ridge','zero_ridge','zero_field'):
                    subset=[r for r in case if r['arm']==arm and r['branch']==b and r['role']==role and r['method']==method]
                    aggregate.append(dict(arm=arm,branch=b,role=role,method=method,cases=len(subset),**{f'mean_{k}':float(np.mean([float(r[k]) for r in subset if r[k]!=''])) if any(r[k]!='' for r in subset) else None for k in ('relative_l1','mae','centroid_error_m','mass_relative_error','spatial_correlation','negative_fraction')}))
    csvwrite(ROOT/'aggregate_metrics.csv',aggregate)
    repaggregate=[]
    for arm in EXPECTED:
        for b in ('A','B'):
            layers=sorted({r['layer'] for r in reps if r['arm']==arm and r['branch']==b})
            for layer in layers:
                selected=[r for r in reps if r['arm']==arm and r['branch']==b and r['layer']==layer]
                repaggregate.append(dict(arm=arm,branch=b,layer=layer,median_pair_normalized_l2=float(np.median([float(r['normalized_l2']) for r in selected if r['condition']=='normal_pair'])),median_zero_normalized_l2=float(np.median([float(r['normalized_l2']) for r in selected if r['condition']=='normal_zero'])),median_pair_rms=float(np.median([float(r['rms_difference']) for r in selected if r['condition']=='normal_pair']))))
    csvwrite(ROOT/'representation_summary.csv',repaggregate)
    # All eight unseen model/branch/case combinations shown with common signed scale.
    labels={6:'bridge',7:'triple'};methods=['original','normal_ridge','zero_ridge','zero_field']
    fig,axes=plt.subplots(8,5,figsize=(13,19),constrained_layout=True)
    row=0
    for arm in EXPECTED:
        for b in ('A','B'):
            for i in (6,7):
                truth=np.load(OLD/f'inputs/truth_{i}.npy');projs=[truth.sum(2)*(.6/63)]
                for method in methods:projs.append(np.load(ROOT/f'predictions/{arm}/{b}/{method}/{i}.npy').reshape(64,64,64).sum(2)*(.6/63))
                vmax=max(float(np.max(np.abs(a))) for a in projs)
                for col,proj in enumerate(projs):
                    ax=axes[row,col];im=ax.imshow(proj.T,origin='lower',extent=(0,.6,0,.6),vmin=-vmax,vmax=vmax,cmap='RdBu_r');ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(['Analytic truth','Original frozen','Normal ridge','Zero-feature ridge','Zero field'][col],fontsize=9)
                    if col==0:ax.set_ylabel(f'{arm} {b}\n{i}: {labels[i]}',fontsize=9)
                fig.colorbar(im,ax=list(axes[row,:]),shrink=.7,label='Signed Z-integral (fraction m)');row+=1
    fig.suptitle('Synthetic-supervised readout diagnostic only; full-volume XY projections',fontsize=12);fig.savefig(art/'unseen_full_volume_projections.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,arm in zip(axes,EXPECTED):
        data=[r for r in repaggregate if r['arm']==arm and r['branch']=='A'];layers=[r['layer'] for r in data if r['layer'] not in ('valid_mask','spatial_valid_mask')]
        for b in ('A','B'):
            val=[next(r['median_pair_normalized_l2'] for r in repaggregate if r['arm']==arm and r['branch']==b and r['layer']==l) for l in layers]
            ax.plot(range(len(layers)),np.maximum(val,1e-12),'o-',label=b)
        ax.axhline(1e-4,color='grey',linestyle='--');ax.set_yscale('log');ax.set_xticks(range(len(layers)),layers,rotation=65,ha='right',fontsize=7);ax.set_title(arm);ax.set_ylabel('Median full-feature pair normalized L2');ax.legend()
    fig.savefig(art/'layer_response.png',dpi=130);plt.close(fig)
    names=['registration.json','probe_seal.json','cross_validation.csv','case_metrics.csv','pair_metrics.csv','layer_statistics.csv','representation_differences.csv','truth_distance_alignment.csv','probe_improvements.csv','aggregate_metrics.csv','representation_summary.csv','evaluation.json','independent_review.json','final_status.json','delivery_sources.json','exact_raw_supplement.json','cpu_tests.log','ridge.log','evaluate.log','review.log','access_fit.json','access_evaluate.json','access_review.json','access_extract_gnot.json','access_extract_deepsets.json','access_extract_raw.json','extraction_gnot.json','extraction_deepsets.json','pair_metrics_fp64.csv','evaluation_fp64.json','pair_precision_amendment.json','pair_precision.log','review_fp64.log','access_evaluate_fp64.json','access_review_fp64.json','swap_layer_statistics.csv','swap_statistics_audit.json','access_extract_swapstats.json']
    for name in names:shutil.copy2(ROOT/name,art/name)
    shutil.copytree(ROOT/'engineering',art/'engineering')
    for p in (ROOT/'probes').glob('*.npz'):
        (art/'probes').mkdir(exist_ok=True);shutil.copy2(p,art/'probes'/p.name)
    (art/'scheduling').mkdir()
    for p in (ROOT/'scheduling').iterdir():
        if p.suffix in ('.json','.log') or p.name in ('events.jsonl','recovery.jsonl'):shutil.copy2(p,art/'scheduling'/p.name)
    lines=['# DeepSets / GNOT 冻结内部表示与线性可读性诊断 v1','',f'完成时间：{now()}；基于Git `{reg["git_head"]}`。所有结果为单seed/单固定几何、6 seen＋2 unseen解析合成羽流上的诊断，不能进入正式重建精度榜单。','']
    statuses={d['arm']:d['status'] for d in ev['decisions']}
    lines+=['**最终判定：'+ '；'.join(f'{a} = {s}' for a,s in statuses.items())+'。** 本轮未得到正常测量特征读出在两个unseen上同时优于原decoder和测量置零读出的充分证据。本轮GNOT的INCONCLUSIVE不撤销v2对原重建模型的NO_GO，也不赋予训练GO。继续训练或输出头修复均不获本轮证据支持；建议保持当前实现暂停，不消耗剩余100次诊断更新。','', '## 冻结、数据边界与复现','',f'两个 checkpoint_200 按注册SHA核验：GNOT `{EXPECTED["gnot"]}`；DeepSets `{EXPECTED["deepsets"]}`。严格加载200次更新状态，eval()、requires_grad_(False)、inference_mode()，无神经optimizer实例或step调用。每模型16个正常/16个置零全64³前向与v2原数组逐位相同；18次normal/zero独立特征捕获全部hook开启/关闭逐位相同；每分支另实跑一次交换，各层均逐位等于donor。参数state_hash在推理前后相同，checkpoint未重存。','', '直接复用登记train_AB/unseen_AB、原A/B固定站位及几何；所有8例非测量字段逐位一致。全部28对、每分支56个有向交换严格复制raw和归一化伴随字段，验证全输入等于donor后复用其已验证表示。每分支zero输入逐位相同，保存一份并为8例登记别名。H输入/站位、几何和解析场SHA保存在原注册映射，本轮无需重新读取H测量数组；无真实GT/GT128、真实val/test体积、Pool B、真实GT统计。启动时SHA检查会读取unseen标签文件的原始字节用于身份核验；拟合阶段没有解析或打开两个unseen标签，访问清单可独立检查。','', 'CPU三项工程测试覆盖已知仿射ridge恢复及整羽流排除、边界拒绝、两模型hook透明性。原token通道反归一化存在FP32往返舍入，因此保留raw_measurement捕获并另以CPU脚本直接保存exact_raw_measurement及完整spatial_valid_mask；未改源输入或神经前向，补充源码/数组SHA单独登记。原packed valid_mask是token排除掩码的反值，不能与完整站位有效射线mask混淆。','', '## 实际提取层与响应','', 'DeepSets：203维ray token → phi per-ray128 → masked mean pooled128 → concat坐标Fourier96 → decoder Linear128/GELU/Linear128/GELU → **prehead128** → scalar logit → sigmoid。首decoder Linear输出也保存。','', 'GNOT：203维measurement_encoder64和202维geometry分支 → 两块异质cross算子、expert、全局self算子、expert → head Linear64/GELU → **prehead64** → scalar logit → sigmoid。保存每块cross后全空间状态、after_self最终状态、测量源key/context统计和全查询global self统计；不存在DeepSets式单一pooled latent，不能虚构该层。','', '下表为28对完整特征的归一化L2中位数。所有逐例均值/标准差/shape/有限整数计数、置零/有向交换和余弦见附件CSV。有向交换层统计另表逐一展开：验证完整输入等于donor后，引用已独立复核的donor正常特征统计；未虚称重复执行相同前向。距离与解析真场差分RMS的相关另列，28对共享案例，不能当28个独立样本做显著性推断。','', '| 模型 | 分支 | 层 | normal配对L2中位 | normal−zero中位 |','|---|---|---|---:|---:|']
    for r in repaggregate:
        if r['layer'] in ('exact_raw_measurement','measurement_encoder','pooled_latent','block0_cross_state','block1_measurement_operator','block1_global_operator','block1_final_state','prehead','logit','concentration'):
            lines.append(f"| {r['arm']} | {r['branch']} | {r['layer']} | {r['median_pair_normalized_l2']:.6g} | {r['median_zero_normalized_l2']:.6g} |")
    lines+=['','![逐层条件响应](artifacts/deep_baselines_latent_readout_v1/layer_response.png)','','## 独立线性读出：解析监督、完整体积','', '每个模型/分支分别拟合normal与zero两种读出：DeepSets129个系数、GNOT65个系数（含截距）。原生特征维数不同，不能称等参数比赛；流程、标签、全部64³体素权重和ridge网格完全相同。FP64充分统计，按整羽流6折，每fold中心/尺度只读5例seen，std下限1e−6；lambda固定[1e−6,1e−4,1e−2,1]，按平均验证relative L1选取、平局取大值。最终6例重拟合并封存后才读unseen标签。读出是独立解析监督诊断，不能回填原checkpoint，也不是原无监督网络的正式成绩。','', '线性输出不裁剪；negative/above-one比例均保存。质心仅用预测正部计算，零质量时未定义；质量误差用原始有符号积分，体素体积(.6/63)^3；此为mass fraction的体积积分而非使用额外气体密度换算的kg质量。所有指标计算完整体积，无内部羽流mask或体素筛选。','', '| 模型 | 分支 | 条件 | lambda | 6折平均L1 | fold最佳与所选一致数 | cond |','|---|---|---|---:|---:|---:|---:|']
    for s in seal['selected']:lines.append(f"| {s['arm']} | {s['branch']} | {s['kind']} | {s['lambda_value']:g} | {s['mean_CV_relative_l1']:.6f} | {s['best_lambda_agreement']}/6 | {s['condition']:.4g} |")
    lines+=['','下表完整列出A/B各分支seen/unseen的案例宏平均relative L1。两unseen分开保留在case_metrics.csv，未把体素或分支当独立案例。0场L1恒1；ridge标签为解析监督，原输出从未使用本轮标签更新。','','| 模型 | 分支 | 集合 | 原decoder | normal ridge | zero-feature ridge | 零场 |','|---|---|---|---:|---:|---:|---:|']
    for a in EXPECTED:
        for b in ('A','B'):
            for role in ('seen','unseen'):
                m={r['method']:r['mean_relative_l1'] for r in aggregate if r['arm']==a and r['branch']==b and r['role']==role}
                lines.append(f"| {a} | {b} | {role} | {m['original']:.6f} | {m['normal_ridge']:.6f} | {m['zero_ridge']:.6f} | {m['zero_field']:.6f} |")
    lines+=['','全部MAE、输出mean/std/max/结构比例、正部质心误差、质量/积分误差、相关性见逐案例CSV和aggregate_metrics.csv；每例normal读出相对三对照的L1改善量见probe_improvements.csv。seen分数是在6例重拟合后的描述性拟合分数，不能代替留一验证成绩。','','![两个unseen、两模型及A/B全体积投影](artifacts/deep_baselines_latent_readout_v1/unseen_full_volume_projections.png)','','每行共享包含所有预测的有符号色标，避免逐图自动放大近零原输出；图为完整Z积分，不是只选羽流局部。数值判断使用三维全体积指标。','','## 瓶颈判断与后续建议','']
    pair=rows(ROOT/'pair_metrics_fp64.csv')
    for d in ev['decisions']:
        lines += [f"### {d['arm']}：{d['status']}",'']
        for e in d['evidence']:
            p=next(r for r in pair if r['arm']==d['arm'] and r['branch']==e['branch'] and r['method']=='normal_ridge' and r['i']=='6' and r['j']=='7')
            lines.append(f"- {e['branch']}分支：上游 `{e['upstream_layer']}` 配对L2中位={e['median_normalized_l2']:.6g}，预登记响应判据={'通过' if e['meaningful_response'] else '未通过'}；两个unseen相对全部三对照改善>=.01分别为{e['unseen_improvement']}；unseen差分cos={p['difference_cosine']}，差分relative L1={float(p['difference_relative_l1']):.6g}；稳定性={'通过' if e['probe_stable'] else '未通过'}；证据类别{e['category']}。")
        lines+=['']
        cats={e['category'] for e in d['evidence']}
        if 'D' in cats:
            lines+=['预登记的整羽流lambda稳定性门槛未全部通过，按D给出INCONCLUSIVE；不得把微小非零表示响应、求解器残差很小或共同模板拟合改写为可泛化条件信息。当前结果不足以支持readout瓶颈或追加训练，保持暂停；不是算法本身不可学的证明。','']
        elif cats=={'A'}:
            lines+=['上游聚合/算子条件响应已很弱，读出也未给出超越测量置零的空间恢复证据，符合A。重点怀疑条件信号在编码/聚合/交互中的保留或传递不足，不能直接归咎最终输出层。建议停止当前实现继续训练。','']
        else:
            lines+=['可测内部响应未转化为通过条件收益门槛的线性恢复。不能据此断言encoder完全丢失信息；非线性组织、所选层、6例覆盖和固定几何反演可辨识性均是替代解释。没有B证据，建议停止当前版本追加训练。','']
    lines+=['**是否继续：** 当前不进入decoder修复或正式训练。没有模型达到完整B/GO，因而本轮不提出自动执行的新架构方案、不扩大正则搜索，也不动剩余100次预算。若未来独立授权进一步验证，必须先针对已观察到的条件响应/跨羽流稳定性提出单一可证伪机制；本报告不将它当已获支持的修复方案。2个unseen、单几何/seed不足以外推算法普遍能力；线性失败也不能证明不存在非线性可读信息。','','## 独立复核、资源及完整性','',f"独立脚本未导入执行期ridge/metric/decision函数；从原始冻结特征重新累计不同分块的充分统计，重拟合所有192个lambda/fold分数及8个最终读出；逐一核验划分、选择、全部预测与独立系数映射，重算{review['counts']['metrics']}个逐例指标、{review['counts']['pairs']}条空间差分和{review['counts']['representations']}条表示差异，另核验{review['counts']['layer_stats']}条层统计，重建两个最终分类。全部通过。",'', '仅GPU0沿用全局UUID flock与15秒双空闲快照、60秒有界等待、冲突监控；两个模型顺序只读提取，GPU1–3未使用。ridge、评价和复核使用CPU双线程。调度/峰值显存/耗时与资源快照均保留，无神经更新、无checkpoint写入、无历史队列恢复。CPU测试中的拒绝为预期边界测试；正式提取/拟合/评价访问日志无违规读取。首次独立复核发现原输出配对余弦的执行期分母使用了FP32范数，极小差分下与独立FP64重算不符。保留原pair_metrics.csv/evaluation.json、review.log及失败访问清单，另行封存pair_precision_amendment.json；仅新增FP64配对重算脚本和review_latent_fp64.py，统一在相减/范数前转FP64，未放宽复核容差。权威表为pair_metrics_fp64.csv，分类逐项不变，系数、预测、指标阈值不动，0次新神经前向/更新。另有只读状态显示误用系统Python缺少NumPy，修正解释器后重跑显示，异常单独保留。精确raw补充为仪器读数精度说明而非科学协议调整。','', '| 模型 | 参数量 | GPU提取耗时秒 | 峰值allocated MiB |','|---|---:|---:|---:|']
    for a,e in originals.items():lines.append(f"| {a} | {e['parameter_count']} | {e['elapsed_seconds']:.2f} | {e['peak_gpu_bytes']/2**20:.2f} |")
    lines+=['', '原v2 checkpoint/全部登记输入、冻结主模型双种子checkpoint、所有既有Git跟踪文件及旧queue_status哈希前后不变。源代码运行前封存；后续仅新增独立复核/精确raw补充/配对FP64精度更正/报告脚本，身份见delivery_sources.json。没有修改任何旧报告或实验追踪表，新追踪单独保存。','', '## 产物索引','', '- [最终状态](artifacts/deep_baselines_latent_readout_v1/final_status.json) · [独立复核](artifacts/deep_baselines_latent_readout_v1/independent_review.json) · [运行前协议](../deep_baselines_latent_readout_v1_protocol_zh.md)','- [逐案例指标](artifacts/deep_baselines_latent_readout_v1/case_metrics.csv) · [汇总](artifacts/deep_baselines_latent_readout_v1/aggregate_metrics.csv) · [CV](artifacts/deep_baselines_latent_readout_v1/cross_validation.csv) · [读出封存](artifacts/deep_baselines_latent_readout_v1/probe_seal.json)','- [逐层统计](artifacts/deep_baselines_latent_readout_v1/layer_statistics.csv) · [逐有向交换层统计](artifacts/deep_baselines_latent_readout_v1/swap_layer_statistics.csv) · [逐层配对/zero/swap](artifacts/deep_baselines_latent_readout_v1/representation_differences.csv) · [空间差分](artifacts/deep_baselines_latent_readout_v1/pair_metrics_fp64.csv)','- [注册与全部输入/源码身份](artifacts/deep_baselines_latent_readout_v1/registration.json) · [预测清单及SHA](artifacts/deep_baselines_latent_readout_v1/evaluation_fp64.json) · [源码补充身份](artifacts/deep_baselines_latent_readout_v1/delivery_sources.json)', '', f'全体积特征、原输出、读出预测、系数及完整资源快照：`{ROOT}`。Git只导出小型表格、系数、哈希、日志与图；原权重留在v2路径。复算入口为同名scripts目录中的recheck_latent.py（内部调用独立review_latent_fp64.py，写入新的rechecks目录，不能覆盖本轮结果）；所有科学原始数组有SHA，可在相同环境独立复核。']
    report=PROJECT/'docs/reports/deep_baselines_latent_readout_v1_zh.md';assert not report.exists();report.write_text('\n'.join(lines)+'\n')
    tracking=PROJECT/'docs/reports/deep_baselines_latent_readout_v1_tracking_zh.md';assert not tracking.exists();tracking.write_text('# 冻结表示诊断追踪表 v1\n\n'+f'更新时间：{now()}。神经更新0，保留剩余100次预算，旧队列暂停。\n\n| 模型 | checkpoint | seed | 本轮更新 | 判定 |\n|---|---|---:|---:|---|\n'+'\n'.join(f'| {a} | `{EXPECTED[a]}` | 20261011 | 0 | {statuses[a]} |' for a in EXPECTED)+'\n\n[报告](deep_baselines_latent_readout_v1_zh.md)。此为解析监督线性诊断，不计入正式模型成绩；不修改历史追踪记录。\n')
    freeze(ROOT/'delivery_manifest.json',dict(at=now(),files={str(p.relative_to(PROJECT)):sha(p) for p in [report,tracking,*art.rglob('*')] if p.is_file()}))
    shutil.copy2(ROOT/'delivery_manifest.json',art/'delivery_manifest.json')
    print('DELIVERED',report,flush=True)
if __name__=='__main__':main()
