"""Synthetic whole-pipeline statistical checks; never reads real test arrays."""
from runtime import *
from scope import effective_runs, scope_evidence, evaluation_stage, deferred_runs
import inspect
import tempfile
import numpy as np
import finish as analysis

def statistics_digest():
    return hashlib.sha256((inspect.getsource(analysis.stats)+inspect.getsource(analysis.expressions)).encode()).hexdigest()

def report_digest():
    return hashlib.sha256(inspect.getsource(analysis.write_reports).encode()).hexdigest()

def main():
    real=BASE
    mm=[]
    for name in effective_runs(real):
        p=verified(Path(name)/'protocol.json')
        mm.append(dict(key=Path(name).name,kind='neural',arm=p['arm'],seed=p['seed'],historical=False,
            parameters=p['parameters'],best_epoch=300,checkpoint_sha256='synthetic-not-real'))
    mm+=verified(real/'reuse_methods.json')
    expected_methods=len(effective_runs(real))+len(verified(real/'reuse_methods.json'))
    assert len(mm)==expected_methods
    with tempfile.TemporaryDirectory(prefix='uniform_statistics_',dir='/scratch/djc') as temp:
        fake=Path(temp);(fake/'analysis').mkdir()
        freeze(fake/'design.json',verified(real/'design.json'))
        for path in scope_evidence(real):
            freeze(fake/Path(path).name,verified(path))
        cases=[f'synthetic_case{i}' for i in range(35)]
        settings=verified(HISTORY/'layouts.json')['test']
        freeze(fake/'registration.json',dict(cases={'test':cases}))
        freeze(fake/'layouts.json',dict(test=settings))
        freeze(fake/'analysis/independent_fields.json',dict(synthetic_fixture=True))
        lookup={};computed={};indices={m['key']:i for i,m in enumerate(mm)}
        for m in mm:
            i=indices[m['key']]
            for stage in (('C0','C1') if m['kind']=='neural' else ('C1',)):
                for setting in settings:
                    for j,case in enumerate(cases):
                        x=.05+.001*i+.002*j+(0 if stage=='C1' else .01)+setting['draw']*.0001
                        row=dict(volume=dict(relative_l1=x,relative_l2=x*2),morphology=dict(gt_peak_location_ratio=.7,
                            peak_ratio=.8,absolute_peak_ratio_error=.2,centroid_distance_mm=x*10,**{'dice_0.1':.8,'dice_0.5':.6}),
                            measurement=dict(observed=dict(relative_l1=x,relative_l2=x),heldout=dict(relative_l1=x,relative_l2=x)))
                        key=m['key'],stage,setting['condition'],setting['draw'],case
                        lookup[key]=row;computed[key]=x
        analysis.BASE=fake
        try:
            result=analysis.stats(mm,lookup,computed)
            expected_panels=sum(2 if m['kind']=='neural' else 1 for m in mm)*20
            assert len(result['panels'])==expected_panels
            expected_primary=9 if evaluation_stage(real) else 11 if scope_evidence(real) else 14
            assert sum(p['primary'] for p in result['paired'])==expected_primary
            assert result['missing_primary_comparisons']==14-expected_primary
            expr=analysis.expressions(mm)
            if evaluation_stage(real):
                for arm in ('no_feedback','zero_residual','no_ray_relations','no_consistency','single_scale'):
                    assert not any(e['comparison'].startswith(arm+'-minus-uniform') for e in expr)
                assert any(e['comparison']=='feedbackOARI2seed-minus-matchedUniform2seed' and e['primary_family'] for e in expr)
                assert len(deferred_runs(real))==7
            elif scope_evidence(real):
                for arm in ('no_ray_relations','no_consistency','single_scale'):
                    assert not any(e['comparison']==arm+'-minus-uniform2seed' for e in expr)
                    single=next(e for e in expr if e['comparison']==arm+'-minus-uniform_seed20261011')
                    assert not single['primary_family']
                    assert [t['weight'] for t in single['terms']]==[1,-1]
            expected={e['comparison']:sum(t['weight']*(.001*indices[t['method']]+(0 if t['stage']=='C1' else .01)) for t in e['terms']) for e in analysis.expressions(mm)}
            for p in result['paired']:
                if p['metric']!='volume.relative_l1':continue
                assert np.isclose(p['delta'],expected[p['comparison']],rtol=0,atol=1e-12)
                assert np.allclose(p['ci95'],expected[p['comparison']],rtol=0,atol=1e-12)
            # Render only inside the temporary project, ensuring report branches
            # and catalog generation work without publishing synthetic results.
            old_project,old_methods=analysis.PROJECT,analysis.all_methods
            project=fake/'synthetic_project';(project/'docs/reports').mkdir(parents=True)
            (fake/'scheduling').mkdir();(fake/'scheduling/events.jsonl').write_text('')
            analysis.PROJECT=project;analysis.all_methods=lambda:mm
            try:
                analysis.write_reports(result)
                assert len(list((project/'docs/reports').glob('*.md')))==5
                assert read(fake/'model_catalog.json')['deployment_seed']==20261011
                if evaluation_stage(real):
                    report=(project/'docs/reports/mrgr_uniform_final_ablations_v1_zh.md').read_text()
                    assert '暂缓的新消融（未训练、未评价）' in report
                    for path in deferred_runs(real):assert Path(path).name in report
                    overview=(project/'docs/reports/mrgr_uniform_final_v1_zh.md').read_text()
                    assert '最终模型待定' in overview and '五项新骨干消融全部暂缓' in overview
                    oari=(project/'docs/reports/mrgr_uniform_oari_v1_zh.md').read_text()
                    assert 'C0到C1的误差及峰值轨迹' in oari and '等待用户审阅' in oari
                    for report_path in (project/'docs/reports').glob('*.md'):
                        assert all(line==line.rstrip() for line in report_path.read_text().splitlines()),report_path
                elif scope_evidence(real):
                    report=(project/'docs/reports/mrgr_uniform_final_ablations_v1_zh.md').read_text()
                    assert '单种子探索性配对差值' in report
                    for arm in ('no_ray_relations','no_consistency','single_scale'):
                        assert arm+'-minus-uniform_seed20261011' in report
            finally:analysis.PROJECT,analysis.all_methods=old_project,old_methods
        finally:analysis.BASE=real
    out=real/'checks/statistics_synthetic.json'
    result=dict(at=now(),passed=True,synthetic=True,real_GT_read=False,methods=len(mm),panels=expected_panels,
        primary_comparisons=expected_primary,primary_family_denominator=14,scope_evidence=scope_evidence(real),
        independent_indexed_vs_weighted_bootstrap=True,analytic_signed_deltas=True,
        five_report_templates=True,statistics_source_sha256=statistics_digest(),report_source_sha256=report_digest(),checks_source_sha256=sha(__file__))
    if out.exists():
        previous=verified(out)
        assert previous['passed']  # Preserve original audit; amendment has its own current-source audit.
        existing=list((real/'checks').glob('statistics_and_reports_synthetic*.json'))
        matches=[p for p in existing if verified(p).get('checks_source_sha256')==sha(__file__)
            and verified(p).get('report_source_sha256')==report_digest()
            and verified(p).get('statistics_source_sha256')==statistics_digest()
            and verified(p).get('scope_evidence',{})==scope_evidence(real)]
        if not matches:
            version=len(existing)+1
            freeze(real/'checks'/f'statistics_and_reports_synthetic_v{version}.json',result)
    else:freeze(out,result)
    print(f'PASS synthetic35case/17layout/{len(mm)}method aggregation;{len(expr)}signed expressions;{expected_primary}simultaneous intervals; family denominator14',flush=True)

if __name__=='__main__':main()
