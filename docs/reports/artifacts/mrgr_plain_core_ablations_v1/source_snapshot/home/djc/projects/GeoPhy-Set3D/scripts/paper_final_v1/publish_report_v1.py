"""Readable accepted final35 results and fixed-case shared-scale classic figures."""
from common import *
import numpy as np


def main():
    assert read(OUT/'final_acceptance.json')['completed']
    base=OUT/'evaluation/test';summary=read(base/'summary.json');seal=verified(base/'all_predictions_sealed.json')
    assert read(base/'independent_verification.json')['passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from evaluation import target_for
    names=('MRGR','NNLS-NL','Tikhonov','TV','SART');cases=verified(OUT/'registration.json')['cases']['test']
    indices=(0,17,34);source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};lookup={}
    for name in names:
        m=next(m for m in seal['methods'] if m['key']==name);d=target_for(m,'test')
        lookup[name]=(d,{(r['condition'],r['draw'],r['case_id']):r for r in rows(d/'manifest.jsonl')})
    hashes={}
    with AccessGuard(gt_splits=['test']) as guard:
        for condition in ('Val-A-V4','clustered-V4'):
            for scale in ('linear','asinh'):
                fig,axes=plt.subplots(3,6,figsize=(19,10),layout='constrained',squeeze=False)
                for row,index in enumerate(indices):
                    case=cases[index];gt=np.load(DATA/source[case]['gt_volume_64_path'],allow_pickle=False).squeeze()
                    iz=np.unravel_index(gt.argmax(),gt.shape)[2];peak=float(gt.max());fields=[('GT',gt)]
                    for name in names:
                        d,manifest=lookup[name];r=manifest[condition,0,case];path=d/r['prediction'];assert sha(path)==r['sha256']
                        with np.load(path,allow_pickle=False) as z:fields.append((name,z['C1'].copy()))
                    for col,(name,field) in enumerate(fields):
                        norm=AsinhNorm(linear_width=.02*peak,vmin=0,vmax=peak) if scale=='asinh' else None
                        kwargs=dict(norm=norm) if norm else dict(vmin=0,vmax=peak)
                        im=axes[row,col].imshow(field[:,:,iz].T,origin='lower',cmap='magma',extent=(0,.6,0,.6),**kwargs)
                        axes[row,col].set(title=f'{name}\ncase {index}, peak/GT={field.max()/peak:.3f}',xlabel='x (m)',ylabel='y (m)')
                    fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.8,label='CH4 mass fraction')
                fig.suptitle(f'Final 35 test: {condition}, draw0; shared {scale} scale and GT-peak z slice')
                for ext in ('png','pdf'):
                    path=base/f'classical_test35_{condition}_{scale}.{ext}';fig.savefig(path,dpi=160);hashes[str(path)]=sha(path)
                plt.close(fig)
        audit=guard.summary();assert not audit['denied']
    freeze(base/'classical_test35_figure_audit.json',dict(at=now(),case_indices=list(indices),methods=['GT',*names],
        layouts=['Val-A-V4 draw0','clustered-V4 draw0'],slice='GT argmax z',scale='Per-case common [0, GT peak]; asinh linear_width=0.02*GT peak; no per-method amplitude normalization',
        files=hashes,audit=audit,all_prediction_manifests_unchanged=True,source_sha256=sha(__file__)))
    def panel(name,condition):
        draws=[0] if condition=='Val-A-V4' else list(range(5))
        return next(p for p in summary['panels'] if p['method']==name and p['stage']=='C1' and p['condition']==condition and p['draws']==draws)
    lines=['# MRGR：35 例最终测试与论文结果入口','',f'更新时间：{now()}（北京时间）。全部 21 方法、35 案例、17 布局已完成：12,495 次重建、22,610 个阶段场，独立评分和验收通过。','',
        '原始第278轮 MRGR 始终作为主模型；全部新增训练在测试评分前结束。测试预测全体封存后才读取本轮35份GT64，GT128未读取，评分后未重新训练或选模型。','',
        '## 主结果','', '| 方法 | 普通固定V4三维L1 | 聚簇V4五draw三维L1 |','|---|---:|---:|']
    for m in seal['methods']:
        lines.append(f'| {m["key"]} | {panel(m["key"],"Val-A-V4")["metrics"]["volume.relative_l1"]["mean"]:.4%} | {panel(m["key"],"clustered-V4")["metrics"]["volume.relative_l1"]["mean"]:.4%} |')
    lines+=['','唯一主要终点为普通固定V4 C1三维相对L1。聚簇先按每案例平均五draw，再跨35案例汇总。全部20个主要终点对照均提供Bonferroni同时区间，完整判断见[预登记规则结果判读](mrgr_paper_final_interpretation_zh.md)。','',
        '## 同种子独立消融','', '| 因素 | 普通V4两种子平均差值（百分点） | 案例95%区间（百分点） | 聚簇五draw差值（百分点） |','|---|---:|---|---:|']
    rr=read(base/'ablation_paired.json')['results']
    from module_registry_v2 import ablation_arms
    for arm in ablation_arms():
        a=next(r for r in rr if r['arm']==arm and r['seed']=='two_seed_average_descriptive' and r['condition']=='Val-A-V4' and r['metric']=='relative_l1')
        b=next(r for r in rr if r['arm']==arm and r['seed']=='two_seed_average_descriptive' and r['condition']=='clustered-V4' and r['metric']=='relative_l1')
        ci=np.array(a['ci95_case'])*100
        lines.append(f'| {arm} | {a["delta_ablation_minus_full"]*100:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {b["delta_ablation_minus_full"]*100:+.4f} |')
    lines+=['','差值为消融−完整，正值有利于完整模型。这里是探索性的同种子配对，两种子均值仅为描述性结果，案例区间不覆盖训练种子方差。各因素的容量与计算差异见[消融协议](mrgr_paper_final_module_ablations_v2_zh.md)。','',
        '## 原始MRGR与四种经典算法的固定切面','',
        '案例固定取索引0/17/34，均使用该案例GT峰的z切面；同一案例全部方法共享[0,GT峰值]色标。asinh只改变显示映射，浓度幅度未归一化；标题另列全体积峰值比。线性图和PDF也保留于同一结果目录。']
    prefix='../../outputs/research/mrgr_paper_final_v1/evaluation/test/'
    for condition in ('Val-A-V4','clustered-V4'):
        lines+=['',f'![{condition}原始MRGR与四种经典算法]({prefix}classical_test35_{condition}_asinh.png)', '',f'[同色标线性图]({prefix}classical_test35_{condition}_linear.png) · [asinh PDF]({prefix}classical_test35_{condition}_asinh.pdf)']
    lines+=['','## 结果、复现与同步范围','',
        '[总报告](mrgr_paper_final_experiments_zh.md) · [唯一主要终点与限制](mrgr_paper_final_interpretation_zh.md) · [封存恢复说明](mrgr_paper_final_seal_recovery_v2_zh.md)', '',
        f'[逐例指标CSV]({prefix}per_case_metrics.csv) · [统计JSON]({prefix}summary.json) · [独立复核]({prefix}independent_verification.json) · [最终验收](../../outputs/research/mrgr_paper_final_v1/final_acceptance.json)', '',
        'GitHub同步实验实现、预登记协议、逐例指标、图表、文件身份和恢复记录；数据集、权重、预测数组、恢复张量与原始进程资源快照保留本地。SHA256及绝对路径记录执行来源，跨机器复现需自行配置本地数据和权重。机器发布清单为`outputs/research/mrgr_paper_final_v1/github_publication_manifest.json`。', '',
        '测试时同时报告经典有限迭代预算、SART离散算子差异及CPU/GPU计时边界。两种子不足以估计种子总体不确定性；没有额外噪声、标定误差或全新几何域实验。']
    path=ROOT/'docs/reports/mrgr_paper_final_test35_zh.md';path.write_text('\n'.join(lines)+'\n');print(path)


if __name__=='__main__':main()
