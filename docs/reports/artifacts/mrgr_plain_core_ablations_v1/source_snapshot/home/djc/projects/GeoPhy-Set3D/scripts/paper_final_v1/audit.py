"""Read-only historical identity audit and pre-test protocol registration."""
from common import *
import numpy as np
import torch
import train_mrgr_multiscale_formal_v1 as formal
from geophy_set3d.data.view_sampling import ViewSampler
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler
from geophy_set3d.evaluation.research import EvaluationWithHoldout


def layouts(cases,azimuth):
    settings=[]
    for condition,views in [('Val-A-V4',4),('spread-V4',4),('V3',3),('V6',6),('clustered-V4',4)]:
        for draw in ([0] if condition in ('Val-A-V4','spread-V4') else range(5)):
            s=dict(condition=condition,draw=draw,views=views,input_ids={},heldout_ids={})
            sampler=ViewSampler(4,4,0,20261011) if condition=='Val-A-V4' else EvaluationViewSampler(azimuth,views,draw,'train',20261011,'spread' if condition=='spread-V4' else 'clustered' if condition=='clustered-V4' else 'random')
            for i,c in enumerate(cases):s['input_ids'][c]=sampler.sample(64,epoch=0,sample_index=i,case_id=c)[0].tolist()
            replay=Replay(s);replay.views=views;replay.draw_id=draw;replay.pool='train';replay.protocol='matched_'+condition
            hold=EvaluationWithHoldout(replay)
            for i,c in enumerate(cases):s['heldout_ids'][c]=hold.sample(64,epoch=0,sample_index=i,case_id=c)[1].tolist()
            settings.append(s)
    return settings


def main():
    backend();assert not (OUT/'audit.json').exists()
    p=formal.verify(ORIGINAL);formal.verify_data(p)
    models=[]
    for key,run,epoch,expected in [('MRGR',ORIGINAL,278,'650b4f2c0f0ab345b5fd99a7d629e0db388cede0937acb7e3191462cd57898d7'),('LR4E4',LR,291,'9c4436d6d5c3a5c27f96f60fec9f99d373822bad796771438baee79eda79f5ad')]:
        protocol=read(run/'protocol.json');summary=read(run/'summary.json');ck=run/'checkpoints/best.pt'
        assert sha(ck)==summary['best_checkpoint_sha256']==expected
        assert sha(run/'protocol.json')==(run/'protocol.sha256').read_text().strip()
        verify_hashes({str(ROOT/k):v for k,v in protocol['sources'].items()})
        with AccessGuard(checkpoints=[ck]) as guard:
            state=torch.load(ck,map_location='cpu',weights_only=False)
        assert state['epoch']==epoch and state['protocol_sha256']==sha(run/'protocol.json') and not state['volumetric_gt_used']
        m=formal.probe.model_for(state['config'],'multiscale','cpu');m.load_state_dict(state['model'],strict=True)
        count=sum(x.numel() for x in m.parameters());assert count==344678
        log=rows(run/'metrics.jsonl');assert len(log)==300 and [r['epoch'] for r in log]==list(range(1,301))
        best=min(log,key=lambda r:r['val']['measurement']['heldout']['relative_l1']);assert best['epoch']==epoch
        models.append(dict(key=key,architecture='MRGR-16/8-context-v1',run=str(run),checkpoint=str(ck),checkpoint_sha256=expected,best_epoch=epoch,parameters=count,
            seed=20261011,epochs=300,updates=24900,config=state['config'],source_sha256=protocol['sources'],
            evidence_sha256={str(run/n):sha(run/n) for n in ('protocol.json','summary.json','metrics.jsonl','gt_access_audit.json')},
            checkpoint_audit=guard.summary(),validation=read((OLD_EVAL if key=='MRGR' else ROOT/'outputs/research/mrgr_autonomous_optimization_v2/frozen_formal_evaluation')/'summary.json')))
    source=rows(DATA/'source_index.jsonl');splits={s:[r['case_id'] for r in source if r['split']==s] for s in ('train','val','test')}
    assert [len(splits[s]) for s in splits]==[166,36,35]
    with np.load(DATA/'geometry/train_geometry.npz') as z:az=z['station_azimuth_deg']
    val=layouts(splits['val'],az);old=read(OLD_EVAL/'registration.json')
    assert len(val)==17 and splits['val']==old['cases']
    for a,b in zip(val,old['settings']):
        for k in a:assert a[k]==b[k],(a['condition'],a['draw'],k)
    test=layouts(splits['test'],az)
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    manifest=dict(frozen_at=now(),executor='paper_final_v1/audit.py',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),worktree=status,
        models=models,split_cases=splits,training_data_sha256=p['data_files'],
        grid='64^3 XYZ endpoints, [0,0.6]^3 m',field='methane mass fraction in [0,1]',measurement='ppm*m',
        forward='96 midpoint samples; trilinear mass-fraction interpolation FIRST then nonlinear mass_to_mole at sampled points; border padding; align_corners=True',
        normalization=read(DATA/'train_statistics.json'),forbidden='GT128 always; test arrays before final method lock; test GT64 before all-method prediction seal')
    freeze(OUT/'model_manifest.json',manifest)
    freeze(OUT/'layouts.json',dict(val=val,test=test))
    reg=dict(registered_at=now(),primary=dict(condition='Val-A-V4',stage='C1',metric='relative_l1',direction='lower'),
        cases=splits,settings_file=str(OUT/'layouts.json'),settings_sha256=sha(OUT/'layouts.json'),
        grid='64^3 XYZ endpoints',reconstructions_per_method=595,neural_fields_per_method=1190,classical_fields_per_method=595,
        secondary=['relative_l2','median','p90','max','centroid_distance_mm','peak_ratio','absolute_peak_ratio_error','gt_peak_location_ratio','dice_0.1','dice_0.5','background_predicted_fraction','observed_L1_L2','heldout_H4_L1_L2','seconds','peak_memory','failure_rate'],
        statistics=dict(resamples=20000,seed=20261019,unit='case; average five draws within each case first',ci='percentile 2.5/97.5%',primary_multiplicity='Also Bonferroni two-sided intervals for all predeclared comparator final outputs; improvement only if adjusted upper bound <0',secondary='exploratory, no corrected claims',training_seed_uncertainty=False),
        failure_policy='Never drop/substitute cases; incomplete predictions block scoring; preserve failure logs and repair only verifiable software errors with versioned audit; numerical failures reported and block completion until disposition is frozen before GT',
        final_method_lock_required=True,prediction_seal='all methods complete, keys/shapes/ranges/hashes verified BEFORE separate score process may open test GT64',
        no_test_tuning=True,no_gt128=True,model_manifest_sha256=sha(OUT/'model_manifest.json'))
    freeze(OUT/'registration.json',reg)
    write(OUT/'audit.json',dict(completed_at=now(),original_sources_and_train_data_verified=True,original_checkpoint_verified=True,lr_checkpoint_verified=True,
        historical_17_layout_input_and_H4_order_reproduced=True,test_GT_opened=False,GT128_opened=False,git_remote_status='unverified: configured proxies refused; direct connection timed out',resources=resources()))
    print('Audit and initial model/protocol freeze PASSED',flush=True)

if __name__=='__main__':main()
