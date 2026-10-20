#!/usr/bin/env python3
"""Resume an audited gasbag run after missing optional3D-renderer dependency."""
import ast
import shutil
import test_uniform_nonplume_gasbag_v1 as task
from test_uniform_nonplume_gasbag_v1 import (ROOT, DEFAULT_OUT, MODEL_CONFIG, LAYOUT_SOURCE,
    AccessGuard, np, now, read, freeze, sha, verify_hashes, verified, BASE)


def main():
    out=DEFAULT_OUT;r=out/'render_recovery_v1'
    reg=verified(out/'registration.json');seal=verified(out/'predictions_sealed.json')
    assert not (out/'acceptance.json').exists()
    assert sha(r/'original_registered_source.py')==reg['script_sha256']
    original=ast.parse((r/'original_registered_source.py').read_text())
    current=ast.parse(task.Path(task.__file__).read_text())
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(original),functions(current)
    assert a.keys()==b.keys()
    assert all(a[k]==b[k] for k in a if k not in ('render','report'))
    verify_hashes(verified(BASE/'final_lock.json')['sources'])
    verify_hashes({m['path']:m['sha256'] for m in seal['manifest']})
    assert sha(reg['checkpoint'])==reg['checkpoint_sha256']
    freeze(out/'rendering_amendment_v1.json',dict(at=now(),reason='skimage unavailable after prediction/scoring; use established Matplotlib voxel surface instead',
        original_source_sha256=reg['script_sha256'],amended_source_sha256=sha(task.__file__),
        original_source=str(r/'original_registered_source.py'),recovery_script_sha256=sha(__file__),
        all_nonrendering_nonreport_functions_AST_identical=True,new_inference_or_training=False,
        new_target_or_model_selection=False,prediction_seal_sha256=sha(out/'predictions_sealed.json')))
    settings=read(LAYOUT_SOURCE)['settings'];source=read(LAYOUT_SOURCE)
    targets={};predictions={};results={};baselines={};details={};convergence={};checks=0
    with AccessGuard(checkpoints=[reg['checkpoint']],output=out) as guard:
        with np.load(out/'geometry.npz',allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
        for name in ('gasbag','torus'):
            gt=np.load(out/f'{name}_GT64.npy',allow_pickle=False);targets[name]=gt
            np.testing.assert_array_equal(gt,task.target(name))
            y=np.load(out/f'{name}_measurements.npy',allow_pickle=False)
            # Repeat only the independent measurement QA; require exact identity
            # to the measurements used by the original sealed model inference.
            coarse=task.analytic_projection(name,geometry,256)
            fine=task.analytic_projection(name,geometry,512)
            np.testing.assert_array_equal(fine,y)
            error=float(np.abs(coarse.astype(float)-fine).sum()/np.abs(fine).sum())
            convergence[name]=dict(reference256_vs512_relative_l1=error,passed=error<=reg['convergence_max_relative_L1'])
            assert convergence[name]['passed']
            info=task.metrics(gt,gt)
            details[name]=dict(peak=float(gt.max()),mean=float(gt.mean()),integral_m3=info['reference_concentration_integral_m3'])
            for setting in settings:
                layout=setting['name']
                with np.load(out/f'{name}_{layout}_predictions.npz',allow_pickle=False) as z:
                    fields={k:z[k] for k in ('C0','C1')}
                predictions[name+'/'+layout]=fields
                for stage,v in fields.items():
                    assert np.isfinite(v).all() and v.shape==(64,)*3 and 0<=v.min()<=v.max()<=1
                    m=task.metrics(v,gt)
                    p,g=v.astype(float),gt.astype(float)
                    assert np.isclose(m['relative_l1'],np.linalg.norm((p-g).ravel(),1)/np.linalg.norm(g.ravel(),1),rtol=1e-12)
                    assert np.isclose(m['relative_l2'],np.linalg.norm((p-g).ravel())/np.linalg.norm(g.ravel()),rtol=1e-12)
                    q=np.load(out/f'{name}_{layout}_{stage}_reprojection.npy',allow_pickle=False)
                    results[name+'/'+layout+'/'+stage]=dict(volume=m,projection={label:task.projection_metrics(q,y,geometry,ids)
                        for label,ids in [('input',setting['input_ids']),('common_unobserved',source['common_unobserved_ids']),('all64',list(range(64)))]})
                    checks+=1
        # CPU reprojection of synthetic GT64, no new model forward or dataset GT.
        for name,gt in targets.items():
            q=task.project64(gt,geometry,'cpu')
            baselines[name]=task.projection_metrics(q,np.load(out/f'{name}_measurements.npy'),geometry,list(range(64)))
        task.render(out,targets,predictions)
        assert not guard.denied and checks==16
        freeze(out/'acceptance.json',dict(at=now(),passed=True,independent_L1_L2_fields=checks,
            predictions_sealed_before_scoring=True,weights_unchanged=True,optimizer_steps=0,
            convergence=convergence,checkpoint_sha256=sha(reg['checkpoint']),access=guard.summary(),GT128_read=False,
            existing_dataset_GT_read=False,main_model='MRGR-UniformFusion-v1',OARI_research_stopped=True,ablations_held=True,
            rendering_amendment_sha256=sha(out/'rendering_amendment_v1.json')))
        scale=read(MODEL_CONFIG)['training_config']['loss']['measurement_scale']
        summary=dict(at=now(),registration=reg,targets=details,metrics=results,GT64_representation_projection_baseline=baselines,
            reference_convergence=convergence,input_max_over_training_scale={n:{s['name']:float(np.load(out/f'{n}_measurements.npy')[s['input_ids']].max()/scale) for s in settings} for n in targets},
            normalization_scale_ppm_m=scale,acceptance_sha256=sha(out/'acceptance.json'),rendering_amendment_sha256=sha(out/'rendering_amendment_v1.json'))
        freeze(out/'summary.json',summary)
    path,art=task.report(out,summary)
    shutil.copy2(r/'original_registered_source.py',art/'original_registered_source.py')
    shutil.copy2(r/'failure.json',art/'rendering_failure.json')
    freeze(out/'export.json',dict(at=now(),report=str(path),report_sha256=sha(path),files={str(p.relative_to(ROOT)):sha(p) for p in art.iterdir() if p.is_file()}))
    print('Recovered reports using original sealed predictions:',path,flush=True)


if __name__=='__main__':main()
