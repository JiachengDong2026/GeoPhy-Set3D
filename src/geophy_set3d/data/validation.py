"""Physics and geometry acceptance tests, independent analytical cross-checks."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.integrate import quad
from geophy_set3d.geometry.tdlas import make_geometry, subset_geometry, ray_box_intersection
from geophy_set3d.physics.projectors import mass_to_mole, differentiable_project
from geophy_set3d.data.build_utils import write_json


def errors(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    return dict(relative_l1=float(np.abs(d).sum()/max(np.abs(b).sum(), 1e-20)),
                relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b), 1e-20)),
                max_absolute_error=float(np.abs(d).max()))


def geometry_qa(train, ood, path):
    reports = {}
    for label, g in [('train', train), ('ood', ood)]:
        directions = g['ray_direction']
        norm_error = float(np.max(np.abs(np.linalg.norm(directions, axis=-1)-1)))
        aim = np.array([0.3, 0.3, 0.3]) - g['station_origin_m']
        aim /= np.linalg.norm(aim, axis=-1, keepdims=True)
        valid = g['valid_ray']
        entries = [g['station_origin_m'][:,None,None,:] + g[k][...,None]*directions
                   for k in ['t_near', 't_far']]
        assert norm_error < 1e-12
        assert np.max(np.abs(aim-g['center_direction'])) < 1e-12
        assert all(np.isfinite(v).all() for v in g.values())
        assert np.all(g['t_far'] >= g['t_near']) and np.all(g['path_length_m'] >= 0)
        assert np.all(g['path_length_m'] <= np.sqrt(3)*0.6 + 1e-12)
        assert all(np.all(e[valid] >= -1e-12) and np.all(e[valid] <= 0.6+1e-12) for e in entries)
        fraction = valid.mean(axis=(1,2))
        assert np.all((fraction > 0.2) & (fraction < 0.95))
        reports[label] = dict(count=len(aim), maximum_direction_norm_error=norm_error,
                              valid_ray_fraction_min=float(fraction.min()),
                              valid_ray_fraction_max=float(fraction.max()),
                              valid_ray_fraction_mean=float(fraction.mean()),
                              maximum_path_length_m=float(g['path_length_m'].max()))
    poses_a = set(map(tuple, train['station_origin_m']))
    poses_b = set(map(tuple, ood['station_origin_m']))
    assert not poses_a.intersection(poses_b)
    # Includes axis-parallel beams, boundary origins, inward/outward and diagonal.
    origins = np.array([[-1,.3,.3], [-1,.7,.3], [.3,.3,.3], [0,.3,.3], [0,.3,.3], [-1,-1,-1]])
    directions = np.array([[1,0,0], [1,0,0], [1,0,0], [-1,0,0], [1,0,0], [1,1,1]], float)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    _, _, length, valid = ray_box_intersection(origins, directions)
    np.testing.assert_array_equal(valid, [True, False, True, False, True, True])
    np.testing.assert_allclose(length, [.6,0,.3,0,.6,.6*np.sqrt(3)], atol=1e-12)
    reports.update(passed=True, pool_exact_pose_overlap_count=0, slab_edge_tests_passed=True,
                   convention='spherical yaw then elevation; row beta increases upward; column alpha increases rightward')
    write_json(path, reports)
    return reports


def projector_validation(projector, config, real_records, source, output):
    import torch
    torch.set_num_threads(8)
    from pathlib import Path
    g = make_geometry(config['geometry'], 3, 0, azimuth=[0,90,180])
    gg = subset_geometry(g, slice(0,1))
    report = dict(thresholds=config['validation'], tests={}, convergence=[], reference_diff=[])
    c = np.array([0,.001,.01,.1,.5,1.], dtype=np.float64)
    chi = mass_to_mole(c)
    assert chi[0] == 0 and chi[-1] == 1 and np.all(np.diff(chi)>0)
    report['tests']['mass_to_mole'] = dict(mass_fraction=c.tolist(), mole_fraction=chi.tolist(), passed=True)
    volume = np.zeros((128,128,128), np.float32)
    zero = projector(volume, g)
    assert np.count_nonzero(zero) == 0
    report['tests']['zero'] = dict(passed=True, max=float(zero.max()))
    for constant in [.001,.5,1.]:
        volume.fill(constant)
        observed = projector(volume, g)
        theoretical = 1e6*mass_to_mole(float(volume[0,0,0]))*g['path_length_m']
        e = errors(observed, theoretical)
        assert e['relative_l1'] < 2e-7
        report['tests'][f'constant_{constant}'] = dict(passed=True, **e, max_ppm_m=float(observed.max()))
    axis = np.linspace(0, .6, 128)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    affine = (.01 + .02*x + .03*y + .04*z).astype(np.float32)
    result = projector(affine, g)
    checks = []
    for view, i, j in [(0,32,32), (1,20,24), (2,42,40)]:
        o, d = g['station_origin_m'][view], g['ray_direction'][view,i,j]
        near, far = g['t_near'][view,i,j], g['t_far'][view,i,j]
        expected = 1e6*quad(lambda t: mass_to_mole(.01+np.dot([.02,.03,.04],o+t*d)), near, far,
                            epsabs=1e-12)[0]
        e = abs(float(result[view,i,j])-expected)/expected
        assert e < 2e-6
        checks.append(dict(view=view, row=i, column=j, relative_error=e))
    report['tests']['asymmetric_xyz_affine_integral'] = dict(passed=True, checks=checks)
    center = np.array([.22,.42,.38]); radius=.05
    sphere = (((x-center[0])**2+(y-center[1])**2+(z-center[2])**2)<=radius**2).astype(np.float32)*.01
    sphere_y = projector(sphere, g)
    oc = g['station_origin_m'][:,None,None,:]-center
    q = np.sum(oc*g['ray_direction'], axis=-1)
    disc = q*q - (np.sum(oc*oc, axis=-1)-radius**2)
    expected = 1e6*mass_to_mole(.01)*2*np.sqrt(np.maximum(disc,0))
    e = errors(sphere_y, expected)
    assert e['relative_l1'] < .1
    centers=[]
    for v in range(3):
        delta=center-g['station_origin_m'][v]
        f=np.dot(delta,g['center_direction'][v]); r=np.dot(delta,g['right_direction'][v]); u=np.dot(delta,g['up_direction'][v])
        predicted=[np.rad2deg(np.arctan2(r,f)),np.rad2deg(np.arctan2(u,np.hypot(r,f)))]
        weights=sphere_y[v]; observed=[float((weights*g['horizontal_angle_deg'][None,:]).sum()/weights.sum()),
                                    float((weights*g['vertical_angle_deg'][:,None]).sum()/weights.sum())]
        assert max(abs(np.array(predicted)-observed)) < .15
        centers.append(dict(station_azimuth_deg=float(g['station_azimuth_deg'][v]),
                            predicted_alpha_beta_deg=predicted, measured_alpha_beta_deg=observed))
    report['tests']['off_center_compact_sphere'] = dict(passed=True, **e, centers=centers)
    gaussian=(.01*np.exp(-((x-.27)**2+(y-.37)**2+(z-.23)**2)/(2*.045**2))).astype(np.float32)
    convergence_fields=[('gaussian',gaussian)]
    # Validation is confined to training cases; final test cases remain untouched.
    for record in real_records[:3]:
        assert record['split']=='train'
        convergence_fields.append((record['sample_id'],np.load(Path(source)/record['target_path'],allow_pickle=False)))
    for name, field in convergence_fields:
        projections={k:projector(field,g,samples=k) for k in [64,128,256,512]}
        item=dict(sample_id=name, comparisons={str(k)+'_vs_512':errors(projections[k],projections[512]) for k in [64,128,256]})
        e=item['comparisons']['256_vs_512']
        assert e['relative_l1'] < config['validation']['convergence_relative_l1_max'], item
        assert e['relative_l2'] < config['validation']['convergence_relative_l2_max'], item
        item['passed']=True
        report['convergence'].append(item)
        value=torch.tensor(field,dtype=torch.float64)
        diff=differentiable_project(value,g,96).detach().numpy()
        mismatch=errors(diff,projections[256])
        assert mismatch['relative_l1'] < config['validation']['reference_diff_relative_l1_max'], mismatch
        assert mismatch['relative_l2'] < config['validation']['reference_diff_relative_l2_max'], mismatch
        report['reference_diff'].append(dict(sample_id=name, passed=True, **mismatch,
                                            per_view=[errors(diff[v],projections[256][v]) for v in range(3)]))
    # Cross-check C++ interpolation against an independent scipy implementation.
    field=convergence_fields[-1][1]
    fraction=np.linspace(0,1,256)
    t=gg['t_near'][...,None]+gg['path_length_m'][...,None]*fraction
    points=gg['station_origin_m'][:,None,None,None,:]+gg['ray_direction'][...,None,:]*t[...,None]
    samples=map_coordinates(field, np.moveaxis(points/.6*127,-1,0),order=1,mode='nearest',prefilter=False)
    independent=np.trapezoid(mass_to_mole(samples.astype(np.float64)),dx=1/255,axis=-1)*gg['path_length_m']*1e6
    independent[~gg['valid_ray']]=0
    e=errors(projector(field,gg,samples=256),independent)
    assert e['relative_l1'] < 2e-6
    report['tests']['independent_scipy_reference'] = dict(passed=True, **e)
    value=torch.tensor(gaussian,dtype=torch.float64,requires_grad=True)
    differentiable_project(value,gg,96).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    report['tests']['differentiable_autograd'] = dict(passed=True, gradient_finite=True)
    report.update(passed=True, fluent_validation_split='train_only', reference_samples=256,
                  differentiable_samples=96, reference_source_sha256=projector.source_hash)
    write_json(output, report)
    return report


def make_preview(train, ood, volume, projection, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for g,label in [(train,'Pool A'),(ood,'Pool B')]:
        o=g['station_origin_m']
        axes[0,0].scatter(o[:,0],o[:,2],s=15,label=label)
        axes[0,1].scatter(np.linalg.norm(o[:,[0,2]]-.3,axis=1),o[:,1],s=15,label=label)
    axes[0,0].add_patch(plt.Rectangle((0,0),.6,.6,fill=False,color='black'))
    axes[0,0].set(xlabel='x (m)',ylabel='z (m)',title='Scanner stations: top view',aspect='equal');axes[0,0].legend()
    axes[0,1].set(xlabel='Horizontal radius (m)',ylabel='y (m)',title='Scanner heights: side view');axes[0,1].legend()
    ax=axes[0,2];g=train;v=0;o=g['station_origin_m'][v]
    for i in [24,32,40]:
        for j in [20,32,44]:
            d=g['ray_direction'][v,i,j]
            end=o+d*7
            ax.plot([o[0],end[0]],[o[2],end[2]],alpha=.5)
    ax.add_patch(plt.Rectangle((0,0),.6,.6,fill=False,color='black'))
    ax.set(xlabel='x (m)',ylabel='z (m)',title='Independent beam bundle',aspect='equal')
    extent=[g['horizontal_angle_deg'][0],g['horizontal_angle_deg'][-1],g['vertical_angle_deg'][0],g['vertical_angle_deg'][-1]]
    for ax,data,title in [(axes[1,0],g['valid_ray'][v],'Valid ray mask'),(axes[1,1],projection[v],'TDLAS PIC (ppm m)')]:
        im=ax.imshow(data,origin='lower',extent=extent,aspect='auto');fig.colorbar(im,ax=ax)
        ax.set(xlabel='Horizontal steering (deg)',ylabel='Vertical steering (deg)',title=title)
    im=axes[1,2].imshow(volume[:,:,64].T,origin='lower',extent=[0,.6,0,.6],aspect='equal');fig.colorbar(im,ax=axes[1,2])
    axes[1,2].set(xlabel='x (m)',ylabel='y (m)',title='GT mass fraction: z=0.302 m')
    fig.savefig(output,dpi=130);plt.close(fig)
