"""Fixed development slices: original MRGR and four classical methods.

Preserve absolute linear views, add shared asinh and explicitly normalized
shape views. All visualization predictions are sealed before visualization GT.
"""
from common import *
import fcntl
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,AsinhNorm
from sart import Block,cached_blocks,solve,method,BASE
from geophy_set3d.reconstruction.classical import mole_to_mass

DEST=OUT/'evaluation/classical_slices_v2'
METHODS=['MRGR','NNLS-NL','Tikhonov','TV','SART']
CONDITIONS=['Val-A-V4','clustered-V4']
INDICES=[0,18,35]


def prepare():
    backend();registration=verified(OUT/'registration.json');settings=verified(OUT/'layouts.json')['val'];cases=registration['cases']['val']
    assert len(cases)==36
    model=next(m for m in verified(OUT/'model_manifest.json')['models'] if m['key']=='MRGR')
    assert model['best_epoch']==278 and sha(model['checkpoint'])==model['checkpoint_sha256']
    selected=method();amendment=verified(BASE/'implementation_amendment_v2.json');verify_hashes(amendment['sources'])
    manifests={'MRGR':OLD_EVAL/'prediction_manifest.jsonl'}
    manifests.update({m:OUT/'classical/val'/m/'frozen/manifest.jsonl' for m in METHODS[1:4]})
    lookup={m:{(r['condition'],r['draw'],r['case_id']):r for r in rows(p)} for m,p in manifests.items()}
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};records=[];cache={}
    # The legacy model arrays live outside OUT; expand only the own-output root.
    # GT128/test GT and foreign dataset arrays remain denied by AccessGuard.
    with AccessGuard(splits=['val'],output=ROOT/'outputs/research') as guard:
        with np.load(DATA/'geometry/train_geometry.npz') as z:geometry={k:z[k] for k in z.files}
        for condition in CONDITIONS:
            s=next(s for s in settings if s['condition']==condition and s['draw']==0)
            for index in INDICES:
                case=cases[index];ids=s['input_ids'][case];hids=s['heldout_ids'][case]
                for name in METHODS[:4]:
                    r=lookup[name][condition,0,case];assert r['input_ids']==ids and r['heldout_ids']==hids
                    path=manifests[name].parent/r['prediction'];assert sha(path)==r['sha256']
                    records.append(dict(method=name,condition=condition,case_id=case,index=index,input_ids=ids,heldout_ids=hids,prediction=str(path),sha256=r['sha256']))
                # Generate the six predefined visualization inputs separately;
                # the full registered 612-case-layout worker continues unchanged.
                path=DEST/'sart_fixed_inputs'/f'{condition}_case{index:02d}.npz'
                provenance=path.with_suffix('.json')
                if path.exists():
                    old=read(provenance);assert old['config']==selected['config'] and old['input_ids']==ids and old['case_id']==case and sha(path)==old['sha256']
                else:
                    blocks=cached_blocks(cache,ids,geometry);measurement=np.load(DATA/source[case]['train_projection_path'],mmap_mode='r',allow_pickle=False)
                    checkpoint=DEST/'sart_fixed_inputs/resume'/f'{condition}_case{index:02d}.pt'
                    chi,stats=solve(blocks,np.array(measurement[ids]),selected['config'],checkpoint=checkpoint,
                        identity=dict(case_id=case,input_ids=ids,registration_sha256=sha(BASE/'registration.json')))
                    c=mole_to_mass(chi.reshape((64,)*3));path.parent.mkdir(exist_ok=True,parents=True)
                    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,C1=c);temp.replace(path)
                    write(provenance,dict(case_id=case,input_ids=ids,heldout_ids=hids,config=selected['config'],sha256=sha(path),stats=stats,measurement_only=True))
                records.append(dict(method='SART',condition=condition,case_id=case,index=index,input_ids=ids,heldout_ids=hids,prediction=str(path),sha256=sha(path)))
        for r in records:
            assert sha(r['prediction'])==r['sha256']
            with np.load(r['prediction'],allow_pickle=False) as z:
                c=z['C1'];assert c.shape==(64,)*3 and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
        audit=guard.summary();assert not audit['denied'] and not any(k.startswith('GT64') for k in audit['counts'])
    seal=DEST/'visualization_predictions_sealed.json'
    if seal.exists():
        previous=verified(seal);assert previous['predictions']==records
    else:freeze(seal,dict(at=now(),split='val',fixed_indices=INDICES,conditions=CONDITIONS,predictions=records,fields=30,
        model_checkpoint_sha256=model['checkpoint_sha256'],SART_method=selected,SART_sources=amendment['sources'],audit=audit,
        purpose='six fixed development case-layout illustrations; separate from full 612 SART development scoring; no final test data'))
    return verified(seal)


def draw():
    seal=prepare();lookup={(r['method'],r['condition'],r['index']):r for r in seal['predictions']}
    source={r['case_id']:r for r in rows(DATA/'source_index.jsonl')};cases=verified(OUT/'registration.json')['cases']['val']
    metadata=[];figures=[];first_gt=None
    plt.rcParams.update({'font.size':10,'axes.titlesize':10,'figure.facecolor':'white','savefig.facecolor':'white'})
    with AccessGuard(gt_splits=['val'],output=ROOT/'outputs/research') as guard:
        fields={};ground_truth={}
        for index in INDICES:
            if first_gt is None:first_gt=now()
            ground_truth[index]=np.load(DATA/source[cases[index]]['gt_volume_64_path'],allow_pickle=False).squeeze()
        for condition in CONDITIONS:
            for index in INDICES:
                gt=ground_truth[index];iz=int(np.unravel_index(gt.argmax(),gt.shape)[2]);peak=float(gt.max())
                values=[('GT',gt)]
                for name in METHODS:
                    r=lookup[name,condition,index];assert sha(r['prediction'])==r['sha256']
                    with np.load(r['prediction'],allow_pickle=False) as z:values.append((name,z['C1'].copy()))
                fields[condition,index]=values
                fig,axes=plt.subplots(3,6,figsize=(19,10.4),layout='constrained',squeeze=False)
                for row in range(3):
                    for col,(name,c) in enumerate(values):
                        image=c[:,:,iz].T;slice_peak=float(image.max())
                        if row==0:norm=Normalize(0,peak);display=image
                        elif row==1:norm=AsinhNorm(linear_width=peak*.001,vmin=0,vmax=peak);display=image
                        else:norm=Normalize(0,1);display=image/slice_peak if slice_peak>0 else image
                        im=axes[row,col].imshow(display,origin='lower',extent=(0,.6,0,.6),cmap='magma',norm=norm,interpolation='nearest')
                        title=name if row<2 else f'{name}\nslice max={slice_peak:.4g}'
                        axes[row,col].set(title=title,xlabel='x (m)',ylabel='y (m)' if col==0 else '',xticks=[0,.2,.4,.6],yticks=[0,.2,.4,.6])
                        if col:axes[row,col].set_yticklabels([])
                        if row==0:
                            metadata.append(dict(method=name,condition=condition,index=index,case_id=cases[index],z_index=iz,z_m=iz*.6/63,
                                gt_peak=peak,predicted_3d_peak=float(c.max()),peak_ratio=float(c.max()/peak),slice_peak=slice_peak,
                                slice_peak_ratio=float(slice_peak/gt[:,:,iz].max()),saturated_slice_voxels=int((image>peak).sum()),zero_slice=slice_peak==0))
                    if row<2:
                        cb=fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.85,label='CH4 mass fraction (shared scale)')
                        if row==1:
                            ticks=peak*np.array([0,.001,.01,.1,1]);cb.set_ticks(ticks);cb.ax.set_yticklabels([f'{v:.3g}' for v in ticks])
                    else:fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.85,label='C / own displayed-slice max (shape only)')
                fig.suptitle(f'Development case {index} | {condition}, draw 0 | z={iz*.6/63:.4f} m\n'
                    'Row 1: shared linear amplitude | Row 2: shared asinh amplitude (width=0.001 x GT peak)\n'
                    'Row 3: each slice independently normalized; brightness cannot compare amplitudes',fontsize=13)
                name=f'{condition}_case{index:02d}_three_scales';fig.savefig(DEST/(name+'.png'),dpi=180);fig.savefig(DEST/(name+'.pdf'));plt.close(fig);figures.append(name+'.png')
            # A compact directly comparable overview: all six columns and three fixed cases.
            fig,axes=plt.subplots(3,6,figsize=(19,10.4),layout='constrained',squeeze=False)
            for row,index in enumerate(INDICES):
                gt=ground_truth[index];iz=int(np.unravel_index(gt.argmax(),gt.shape)[2]);peak=float(gt.max());norm=AsinhNorm(linear_width=.001*peak,vmin=0,vmax=peak)
                for col,(name,c) in enumerate(fields[condition,index]):
                    im=axes[row,col].imshow(c[:,:,iz].T,origin='lower',extent=(0,.6,0,.6),cmap='magma',norm=norm,interpolation='nearest')
                    ratio=float(c.max()/peak)
                    axes[row,col].set(title=f'{name}\n3D peak / GT={ratio:.3f}',xlabel='x (m)',ylabel=f'case {index}\ny (m)' if col==0 else '',xticks=[0,.2,.4,.6],yticks=[0,.2,.4,.6])
                    if col:axes[row,col].set_yticklabels([])
                cb=fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.85,label='CH4 mass fraction');ticks=peak*np.array([0,.001,.01,.1,1]);cb.set_ticks(ticks);cb.ax.set_yticklabels([f'{v:.3g}' for v in ticks])
            fig.suptitle(f'Development: {condition}, draw 0 | fixed cases 0 / 18 / 35 | original MRGR epoch 278\n'
                'Shared asinh scale within each case; identical GT-peak z slice for all methods; peak ratios retain amplitude context',fontsize=13)
            name=condition+'_shared_asinh_overview';fig.savefig(DEST/(name+'.png'),dpi=180);fig.savefig(DEST/(name+'.pdf'));plt.close(fig);figures.append(name+'.png')
        audit=guard.summary();assert not audit['denied']
    jsonl(DEST/'slice_amplitude_metadata.jsonl',metadata)
    write(DEST/'figure_audit.json',dict(at=now(),first_visualization_GT64_at=first_gt,prediction_seal_at=seal['at'],case_indices=INDICES,
        methods=['GT',*METHODS],conditions=CONDITIONS,split='val',test_GT_read=False,GT128_read=False,source_sha256=sha(__file__),
        slice='same GT-peak z index for every method',linear='shared [0,GT peak]; values above GT peak saturate and are counted',
        asinh='same asinh(C/(0.001*GT peak)) color mapping per case; colorbar remains actual mass fraction',
        shape='each displayed slice divided by its own slice maximum; linear [0,1]; cannot compare amplitudes',
        predictions_not_modified=True,figures=figures,audit=audit))
    report(metadata,figures)
    print('Created',len(figures),'combined comparison figures at',DEST,flush=True)


def report(metadata,figures):
    prefix='../../outputs/research/mrgr_paper_final_v1/evaluation/classical_slices_v2/'
    lines=['# 原始 MRGR 与四种经典算法：开发验证切面对比','',f'生成时间：{now()}（北京时间）。仅使用开发验证案例，未读取35例最终测试GT。','',
        '每图按 GT、原始 MRGR（epoch 278，C1）、NNLS-NL、Tikhonov、TV、SART 排列。普通固定V4与聚簇V4 draw0使用相同冻结输入站位和观测顺序。案例固定为清单索引0/18/35，切面为各案例GT三维峰所在z，所有方法共用同一物理切面，未按结果挑选案例或切面。','',
        '## 为什么旧图很暗','',
        '旧图采用[0,GT峰值]的统一线性色标。经典方法的预测峰值较低、羽流更分散，聚簇观测下尤其明显，因此低浓度羽流接近色图黑端。暗不代表数组为空；也不能仅靠增亮认定其恢复良好。下图保留峰值比和实际浓度色标，帮助区分幅值偏低与结构差异。','',
        '## 六方法同图：统一 asinh 色标','',
        '同一案例的六列共用同一个非线性色标，色条仍标实际质量分数；asinh线性宽度固定为GT峰值的0.001倍，对全部案例和方法使用同一规则。它让低浓度结构更可见，同时保留跨方法幅值可比性。每列标注三维峰值/GT峰值。','']
    for condition in CONDITIONS:lines += [f'### {condition}，draw0','',f'![{condition}六方法统一色标切面]({prefix}{condition}_shared_asinh_overview.png)','']
    lines+=['## 三种显示方式逐案例核对','',
        '每张详细图有三行：第一行保留统一线性绝对浓度；第二行采用统一asinh绝对浓度；第三行各显示切面除以自身最大值，仅比较切面形状。第三行每列会显示自己的切面峰值，不能将同样明亮理解为同样浓度。零切面保持零，不人为制造结构。超出GT峰值的区域在前两行饱和，实际峰比与饱和数量另存元数据。','']
    for condition in CONDITIONS:
        for index in INDICES:
            name=f'{condition}_case{index:02d}_three_scales'
            lines += [f'### {condition}，案例{index}','',f'![三种色标对照]({prefix}{name}.png)',f'[下载PDF]({prefix}{name}.pdf)','']
    lines+=['## 幅值诊断','', '| 布局 | 案例 | 方法 | 三维峰/GT峰 | 当前切面峰值 | 当前切面峰/GT切面峰 |','|---|---:|---|---:|---:|---:|']
    for r in metadata:
        if r['method']=='GT':continue
        lines.append(f'| {r["condition"]} | {r["index"]} | {r["method"]} | {r["peak_ratio"]:.4f} | {r["slice_peak"]:.6g} | {r["slice_peak_ratio"]:.4f} |')
    lines+=['','## 预测身份、SART进度与解释边界','',
        '原始MRGR与前三种经典方法复用已有预测，并核对manifest中的哈希与站位。SART采用已冻结松弛系数0.5、200轮预算；为了立即补齐固定图像，另生成预定的6个案例—布局预测，先保存并与其余24个方法预测一起单独封存，再读取开发GT绘图。该图像子集不替代612次完整SART开发验证，不根据切面调参。','',
        'SART完整开发曾在101次预测后因站位缓存淘汰触发KeyError。修复只保留本次重建已取到的矩阵引用；矩阵、求解器、选择规则与参数源码均核实未改变，原101份预测哈希保持一致，失败日志保留，完整开发已恢复。','',
        'SART采用节点摩尔分数离散，与MRGR前向存在已披露的插值/转换顺序差异。切面对比不能替代三维误差、峰值、定位和投影指标。增加显示动态范围不改变预测数组、评分或冻结模型。','',
        f'[图像审计]({prefix}figure_audit.json)；[图像输入封存]({prefix}visualization_predictions_sealed.json)；[幅值逐项记录]({prefix}slice_amplitude_metadata.jsonl)；[经典开发报告](mrgr_paper_final_classical_development_v1_zh.md)；[SART补充协议](mrgr_paper_final_sart_supplement_v1_zh.md)。']
    (ROOT/'docs/reports/mrgr_paper_final_classical_slices_v2_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    mutex=(DEST/'figures.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    draw()
