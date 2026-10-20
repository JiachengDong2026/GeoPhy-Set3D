#!/usr/bin/env python3
"""Frozen uniform MRGR on a filled gas bag; matched torus is a shape reference."""
import argparse
import fcntl
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/uniform_final_v1'), str(ROOT/'scripts'), str(ROOT/'src')]
from runtime import (AccessGuard, BASE, GPU_LOCKS, PROJECT, backend, now, read, write,
                     freeze, verified, sha, verify_hashes, training_identity)
from uniform_models import model_for
import numpy as np
import torch
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset, assert_no_gt
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
from geophy_set3d.evaluation.research import volume_metrics
from geophy_set3d.evaluation.measurement_consistency import relative_projection_metrics
from geophy_set3d.physics.projectors import mass_to_mole, differentiable_project
from geophy_set3d.geometry.tdlas import subset_geometry

DEFAULT_OUT = Path('/scratch/djc/GeoPhy-Set3D/mrgr_uniform_nonplume_gasbag_v1')
LAYOUT_SOURCE = ROOT/'outputs/research/mrgr_nonplume_views_v2/registration.json'
MODEL_CONFIG = ROOT/'configs/mrgr_uniform_fusion_v1.json'
NAMES = dict(ordinary_V4='Ordinary V4', spread_V4='Spread V4',
             ordinary_V6='Ordinary V6', spread_V6='Spread V6')
PARAMETERS = dict(center_xyz_m=[.3,.3,.3], peak_mass_fraction=.08,
    gasbag=dict(semiaxes_xyz_m=[.12,.09,.08], flat_core_radius=.85,
                edge='half-cosine from normalized radius0.85 to1; zero outside'),
    torus=dict(major_radius_m=.12,tube_sigma_m=.025,axis='y'))


def field(name, points):
    q=points-np.array(PARAMETERS['center_xyz_m'])
    if name=='gasbag':
        r=np.sqrt(np.square(q/np.array(PARAMETERS['gasbag']['semiaxes_xyz_m'])).sum(-1))
        t=np.clip((r-.85)/.15,0,1)
        return .08*.5*(1+np.cos(np.pi*t))
    if name=='torus':
        radial=np.sqrt(q[...,0]**2+q[...,2]**2)
        return .08*np.exp(-((radial-.12)**2+q[...,1]**2)/(2*.025**2))
    raise ValueError(name)


def target(name):
    xyz=np.stack(np.meshgrid(*[np.linspace(0,.6,64)]*3,indexing='ij'),-1)
    return field(name,xyz).astype(np.float32)


def analytic_projection(name, geometry, samples):
    """Independent float64 NumPy continuous-field trapezoidal integration.

    No volume interpolation, autograd projector or128-cubed target is involved.
    """
    result=np.zeros(geometry['valid_ray'].shape,np.float64)
    fraction=np.linspace(0,1,samples)
    for station,origin in enumerate(geometry['station_origin_m']):
        valid=geometry['valid_ray'][station].ravel()
        ids=np.flatnonzero(valid)
        directions=geometry['ray_direction'][station].reshape(-1,3)
        near=geometry['t_near'][station].ravel()
        length=geometry['path_length_m'][station].ravel()
        for start in range(0,len(ids),512):
            ix=ids[start:start+512]
            distance=near[ix,None]+length[ix,None]*fraction
            points=origin+directions[ix,None,:]*distance[...,None]
            chi=mass_to_mole(field(name,points))
            integrals=(chi[:,0]+chi[:,-1])*.5+chi[:,1:-1].sum(1)
            result[station].ravel()[ix]=integrals*length[ix]*1e6/(samples-1)
    assert np.isfinite(result).all() and result.min()>=0
    return result.astype(np.float32)


def project64(volume,geometry,device):
    with torch.inference_mode():
        tensor=torch.as_tensor(volume,device=device)
        return np.concatenate([differentiable_project(tensor,subset_geometry(geometry,slice(i,i+2)),samples=96).cpu().numpy()
            for i in range(0,len(geometry['station_origin_m']),2)])


def metrics(pred,gt):
    m=volume_metrics(pred,gt)
    m['peak_ratio']=float(pred.max()/gt.max())
    m['peak_ratio_absolute_error']=abs(m['peak_ratio']-1)
    for f in (.1,.5):
        a,b=pred>=f*gt.max(),gt>=f*gt.max()
        m[f'dice_gt_peak_{f}']=float(2*(a&b).sum()/(a.sum()+b.sum()))
    center=np.array([31,32]);core=pred[np.ix_(center,center,center)]
    m['center_mean_over_GT_peak']=float(core.mean()/gt.max())
    return m


def projection_metrics(p,y,g,ids):
    ix=np.array(ids,dtype=np.int64)
    return relative_projection_metrics(torch.from_numpy(p[ix]),torch.from_numpy(y[ix]),
        torch.from_numpy(g['valid_ray'][ix]))


def reserve_gpu(out):
    from controller import snapshot,devices,reserve
    # Use the established common UUID locks, idle thresholds and launch recheck.
    # This is a short inference job; it never submits training jobs.
    a=devices(snapshot());time.sleep(15);b=devices(snapshot())
    first={d['uuid'] for d in a if d['idle']}
    for d in b:
        if d['uuid'] in first and d['idle']:
            lock=reserve(d['uuid'])
            if lock is not None:
                write(out/'gpu_reservation.json',dict(at=now(),uuid=d['uuid'],index=d['index'],
                    repeated_idle_checks_seconds=15,common_UUID_flock=True,other_processes_signalled=False))
                return f'cuda:{d["index"]}',lock
    raise RuntimeError('No continuously idle GPU; use --device cpu or retry later')


def render(out,targets,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,AsinhNorm
    for name,gt in targets.items():
        vv=[gt]+[predictions[name+'/'+s]['C1'] for s in NAMES]
        labels=['GT']+list(NAMES.values())
        for scale in ('linear','asinh'):
            norm=Normalize(0,float(gt.max())) if scale=='linear' else AsinhNorm(linear_width=float(gt.max())*.03,vmin=0,vmax=float(gt.max()))
            fig,axes=plt.subplots(3,5,figsize=(16,9),constrained_layout=True)
            for col,(v,label) in enumerate(zip(vv,labels)):
                planes=[(v[:,31,:].T,'x','z','y'),(v[:,:,31].T,'x','y','z'),(v[31,:,:],'z','y','x')]
                for row,(plane,x,y,cut) in enumerate(planes):
                    im=axes[row,col].imshow(plane,origin='lower',extent=(0,.6,0,.6),cmap='magma',norm=norm,interpolation='nearest')
                    axes[row,col].set(title=f'{label}: {cut}=0.295m',xlabel=x+' (m)',ylabel=y+' (m)')
            fig.colorbar(im,ax=axes,shrink=.7,extend='max',label='CH4 mass fraction; same GT scale for every model')
            fig.suptitle(f'{name}: frozen MRGR-UniformFusion seed20261011 epoch285; C1')
            fig.savefig(out/f'{name}_slices_{scale}.png',dpi=150)
            if scale=='asinh':fig.savefig(out/f'{name}_slices_asinh.pdf')
            plt.close(fig)
        from skimage.measure import marching_cubes
        fig=plt.figure(figsize=(16,4),constrained_layout=True)
        for i,(v,label) in enumerate(zip(vv,labels)):
            ax=fig.add_subplot(1,5,i+1,projection='3d');level=.5*float(gt.max())
            if v.min()<level<v.max():
                vertices,faces,_,_=marching_cubes(v,level,spacing=(.6/63,)*3)
                ax.plot_trisurf(vertices[:,0],vertices[:,2],vertices[:,1],triangles=faces,color='steelblue',alpha=.8,lw=0)
            ax.set(title=label,xlabel='x (m)',ylabel='z (m)',zlabel='y (m)',xlim=(0,.6),ylim=(0,.6),zlim=(0,.6))
            ax.set_box_aspect((1,1,1));ax.view_init(25,40)
        fig.suptitle(name+': common isosurface at50% GT peak')
        fig.savefig(out/f'{name}_3d.png',dpi=150);plt.close(fig)


def report(out,summary):
    art=ROOT/'docs/reports/artifacts/mrgr_uniform_nonplume_gasbag_v1';art.mkdir(parents=True,exist_ok=False)
    for p in out.iterdir():
        if p.suffix in ('.png','.pdf','.json'):shutil.copy2(p,art/p.name)
    rel='artifacts/mrgr_uniform_nonplume_gasbag_v1/'
    lines=['# 均匀融合MRGR：实心气袋目标小测试','',f'完成时间：{now()}（北京时间）。','',
        '当前暂定主模型为均匀融合MRGR；停止反馈OARI后续研究，7个消融任务继续暂缓。本次只作冻结权重推理，不训练、不优化、不按本次目标修改权重或参数。','',
        '主模型：MRGR-UniformFusion-v1，seed20261011，第285轮，342725参数。权重SHA256：`'+summary['registration']['checkpoint_sha256']+'`。','',
        '## 目标和比较设计','',
        '气袋为实心椭球，中心(0.3,0.3,0.3)m，XYZ半轴(0.12,0.09,0.08)m，内部质量分数0.08。归一化椭球半径r≤0.85为均匀平台，0.85<r<1用半余弦降至零，r≥1为零；无空洞。它是静态合成标量场，未模拟袋膜光学效应、真实气袋或流体过程。','',
        '同时用本次同一均匀融合权重重算原参数圆环作为形态参照：中心相同、大半径0.12m、管高斯标准差0.025m、峰值0.08。两个目标同幅值，总浓度积分接近但并非严格匹配；形状、边界陡峭度与空间占据率仍同时不同。历史圆环使用不同权重，其指标不直接作为本次配对基线。','',
        '沿用历史圆环实验登记的普通V4、分散V4、普通V6(draw0)、分散V6四组站位，未按目标或模型结果选布局。每种目标只算一个案例，四组布局不是四个独立案例。普通V4为预先指定的主要展示，其他布局全部保留。','',
        '测量由独立NumPy float64连续解析场生成：先按原分子量做非线性质量分数→摩尔分数转换，再沿原Pool A射线做512点梯形积分，单位ppm·m；256/512点收敛核验。没有读取或生成任何128³标签，不读取已有案例GT/投影。模型使用原训练固定归一化，64³场仅用于本次合成评分；全部8组C0/C1预测保存并哈希封存后再计算指标。','',
        '| 目标 | 64³峰值 | 浓度积分(m³) | 平均质量分数 |','|---|---:|---:|---:|']
    for name,info in summary['targets'].items():lines.append(f'| {name} | {info["peak"]:.6f} | {info["integral_m3"]:.8f} | {info["mean"]:.6f} |')
    lines+=['','## 全部三维结果','',
        '| 目标/布局/阶段 | L1 | L2 | Dice50 | 峰值比 | 峰比绝对误差 | 质心mm | 输入投影L1 | 共同未输入投影L1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in summary['metrics'].items():
        v,p=r['volume'],r['projection']
        lines.append(f'| {key} | {100*v["relative_l1"]:.2f}% | {100*v["relative_l2"]:.2f}% | {v["dice_gt_peak_0.5"]:.3f} | {v["peak_ratio"]:.3f} | {v["peak_ratio_absolute_error"]:.3f} | {v["centroid_error_mm"]:.2f} | {100*p["input"]["relative_l1"]:.2f}% | {100*p["common_unobserved"]["relative_l1"]:.2f}% |')
    lines+=['','L1/L2以GT范数归一化；Dice50以同一目标GT峰值50%作共同阈值；峰比=max(pred)/max(GT)，质心按梯形积分。平台峰值没有唯一位置，不解释单个argmax坐标误差。C0/C1为同一训练模型的两阶段，不能作为独立训练消融。','',
        '## C1气袋与同权重圆环参照','',
        '| 布局 | 气袋L1 | 圆环L1 | 气袋Dice50 | 圆环Dice50 |','|---|---:|---:|---:|---:|']
    for layout in NAMES:
        a=summary['metrics']['gasbag/'+layout+'/C1']['volume'];b=summary['metrics']['torus/'+layout+'/C1']['volume']
        lines.append(f'| {NAMES[layout]} | {100*a["relative_l1"]:.2f}% | {100*b["relative_l1"]:.2f}% | {a["dice_gt_peak_0.5"]:.3f} | {b["dice_gt_peak_0.5"]:.3f} |')
    lines+=['','## 共同色标和固定切面','',
        '同一目标的GT与所有布局使用GT峰值固定色标，不逐模型归一化；线性和asinh图均保存，超过GT峰值的部分饱和。切面固定索引31，3D使用共同50%GT峰值等值面。','']
    for name in ('gasbag','torus'):
        lines += [f'![{name}三组正交切面]({rel}{name}_slices_asinh.png)','',f'![{name}三维形态]({rel}{name}_3d.png)','']
    lines+=['## 数值核验及边界','',
        'GT64表示基线的投影误差单列于summary.json，用来区分64³插值/96点中点积分误差与模型误差。全部预测范围/有限性、权重不变、输入无GT、独立NumPy L1/L2重算及参考积分收敛检查通过；详见验收记录。','',
        '本次只说明一个合成实心气袋和四组固定布局上的表现，不代表所有非羽流目标、实际气袋或独立泛化能力确认。不据此搜索形状、浓度、checkpoint或最佳布局；所有正负结果完整保存。','',
        f'[完整汇总]({rel}summary.json) · [登记]({rel}registration.json) · [验收]({rel}acceptance.json) · [可复现脚本](../../scripts/test_uniform_nonplume_gasbag_v1.py)','',
        '大型测量、目标、预测和重投影保存于服务器`'+str(out)+'`，不提交Git。','',
        '复现（新输出目录；自动检查并锁定空闲GPU，亦可用--device cpu）：','',
        '```bash','/scratch/djc/envs/gas-torch/bin/python scripts/test_uniform_nonplume_gasbag_v1.py --out /scratch/djc/GeoPhy-Set3D/mrgr_uniform_nonplume_gasbag_v1_repeat --no-publish','```']
    path=ROOT/'docs/reports/mrgr_uniform_nonplume_gasbag_v1_zh.md'
    assert not path.exists();path.write_text('\n'.join(lines).rstrip()+'\n')
    return path,art


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--no-publish',action='store_true',help='Reproduce into a new output directory without writing repository reports')
    o=parser.parse_args();out=o.out.resolve();out.mkdir(parents=True,exist_ok=False)
    backend();cfg=read(MODEL_CONFIG);checkpoint=Path(cfg['checkpoint']['path'])
    assert sha(checkpoint)==cfg['checkpoint']['sha256']
    verify_hashes(verified(BASE/'final_lock.json')['sources'])
    source=read(LAYOUT_SOURCE);settings=source['settings']
    assert [s['name'] for s in settings]==list(NAMES)
    lock=None
    if o.device=='auto':device,lock=reserve_gpu(out)
    else:
        assert o.device=='cpu','GPU inference must use automatic reservation'
        device=o.device
    registration=dict(at=now(),model_id=cfg['model_id'],checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),seed=20261011,selected_epoch=285,
        target_parameters=PARAMETERS,target_grid=[64]*3,axes='XYZ,y vertical,[0,.6]^3m',
        settings=settings,layout_source=str(LAYOUT_SOURCE),layout_source_sha256=sha(LAYOUT_SOURCE),
        config_sha256=sha(MODEL_CONFIG),script_sha256=sha(__file__),training_sources=training_identity(),
        primary='gasbag ordinary_V4/C1; other3 fixed layouts and same-model torus descriptive',
        projection='continuous analytic field, independent NumPy float64,512-point trapezoidal rays;256-point convergence check',
        convergence_max_relative_L1=.002,noise=False,training=False,optimizer_steps=0,
        feedback_OARI_research_stopped=True,ablations_held=True,
        model_selection='existing uniform deployment seed fixed; no synthetic-target selection',
        no_dataset_GT_or_test_projection_read=True,GT128_read=False,synthetic128_generated=False)
    freeze(out/'registration.json',registration)
    with AccessGuard(checkpoints=[checkpoint],output=out) as guard:
        ds=MeasurementOnlyPoolDataset(cfg['training_config']['dataset']['root'],split='val',geometry_pool='train',
            min_input_views=4,max_input_views=6,hold_views=0,
            normalization=cfg['training_config']['normalization']['method'],statistics_file=cfg['training_config']['normalization']['statistics_file'])
        np.savez_compressed(out/'geometry.npz',**ds.geometry)
        for s in settings:np.testing.assert_array_equal(ds.geometry['station_azimuth_deg'][s['input_ids']],s['station_azimuth_deg'])
        targets={name:target(name) for name in ('gasbag','torus')}
        assert field('gasbag',np.array([[.3,.3,.3]]))[0]==.08
        measurements={};convergence={}
        for name,gt in targets.items():
            np.save(out/f'{name}_GT64.npy',gt)
            a=analytic_projection(name,ds.geometry,256);b=analytic_projection(name,ds.geometry,512)
            error=float(np.abs(a.astype(float)-b).sum()/np.abs(b).sum())
            convergence[name]=dict(reference256_vs512_relative_l1=error,passed=error<=.002)
            assert error<=.002,(name,error)
            measurements[name]=b;np.save(out/f'{name}_measurements.npy',b)
            print('Continuous reference generated',name,'convergence L1',error,flush=True)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['epoch']==285 and not ck['volumetric_gt_used']
        assert ck['config']==cfg['training_config']
        model=model_for(ck['config'],'uniform',device).eval();model.load_state_dict(ck['model'],strict=True)
        assert sum(p.numel() for p in model.parameters())==342725
        import probe_mrgr_multiscale_v1 as probe
        before=probe.state_hash(model.state_dict());predictions={};manifest=[]
        for name in targets:
            for setting in settings:
                layout=setting['name'];ids=np.array(setting['input_ids'],np.int64)
                sample=ds._measurement_sample(dict(case_id='synthetic_'+name,sample_id='synthetic_'+name),measurements[name],ids,np.empty(0,np.int64))
                inputs=move_batch(projection_pool_collate([sample]),device)
                inputs={k:v for k,v in inputs.items() if not k.startswith('hold_')};assert_no_gt(inputs)
                with torch.inference_mode():stages=model(inputs,return_stages=True)
                fields={k:stages[k][0,0].cpu().numpy() for k in ('C0','C1')}
                assert all(v.shape==(64,)*3 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in fields.values())
                predictions[name+'/'+layout]=fields
                path=out/f'{name}_{layout}_predictions.npz';np.savez_compressed(path,**fields)
                manifest.append(dict(target=name,layout=layout,path=str(path),sha256=sha(path),input_keys=sorted(inputs)))
        assert probe.state_hash(model.state_dict())==before
        freeze(out/'predictions_sealed.json',dict(at=now(),manifest=manifest,weights_unchanged=True,optimizer_steps=0,all_saved_before_scoring=True))
        print('All8 C0/C1 predictions sealed before scoring',flush=True)
        common=source['common_unobserved_ids'];results={};baselines={};checks=0;details={}
        for name,gt in targets.items():
            y=measurements[name];reprojected=project64(gt,ds.geometry,device)
            baselines[name]=projection_metrics(reprojected,y,ds.geometry,list(range(64)))
            info=metrics(gt,gt);details[name]=dict(peak=float(gt.max()),mean=float(gt.mean()),integral_m3=info['reference_concentration_integral_m3'])
            for setting in settings:
                layout=setting['name']
                for stage,v in predictions[name+'/'+layout].items():
                    m=metrics(v,gt)
                    p,g=v.astype(np.float64),gt.astype(np.float64)
                    assert np.isclose(m['relative_l1'],np.linalg.norm((p-g).ravel(),1)/np.linalg.norm(g.ravel(),1),rtol=1e-12)
                    assert np.isclose(m['relative_l2'],np.linalg.norm((p-g).ravel())/np.linalg.norm(g.ravel()),rtol=1e-12)
                    q=project64(v,ds.geometry,device)
                    np.save(out/f'{name}_{layout}_{stage}_reprojection.npy',q)
                    results[name+'/'+layout+'/'+stage]=dict(volume=m,projection={label:projection_metrics(q,y,ds.geometry,ids)
                        for label,ids in [('input',setting['input_ids']),('common_unobserved',common),('all64',list(range(64)))]})
                    checks+=1
        render(out,targets,predictions)
        assert sha(checkpoint)==registration['checkpoint_sha256']
        assert not guard.denied
        freeze(out/'acceptance.json',dict(at=now(),passed=True,independent_L1_L2_fields=checks,
            predictions_sealed_before_scoring=True,weights_unchanged=True,optimizer_steps=0,
            convergence=convergence,checkpoint_sha256=sha(checkpoint),access=guard.summary(),GT128_read=False,
            existing_dataset_GT_read=False,main_model='MRGR-UniformFusion-v1',OARI_research_stopped=True,ablations_held=True))
        summary=dict(at=now(),registration=registration,targets=details,metrics=results,
            GT64_representation_projection_baseline=baselines,reference_convergence=convergence,
            input_max_over_training_scale={n:{s['name']:float(y[s['input_ids']].max()/ds.y_ref) for s in settings} for n,y in measurements.items()},
            normalization_scale_ppm_m=ds.y_ref,acceptance_sha256=sha(out/'acceptance.json'))
        freeze(out/'summary.json',summary)
    if lock is not None:lock.close()
    if o.no_publish:
        print('Completed reproduction',out,flush=True)
        return
    path,art=report(out,summary)
    freeze(out/'export.json',dict(at=now(),report=str(path),report_sha256=sha(path),
        files={str(p.relative_to(ROOT)):sha(p) for p in art.iterdir() if p.is_file()}))
    print('Completed',path,flush=True)


if __name__=='__main__':main()
