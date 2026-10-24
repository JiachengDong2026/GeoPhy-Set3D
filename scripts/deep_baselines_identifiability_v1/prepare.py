"""Seal thresholds before experiments; generate synthetic sources with no real labels."""
from diag_context import *
from synthetic import generate

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'inputs').mkdir(exist_ok=False)
    c.backend()
    plan=[]
    for arm in ARMS:
        plan.extend(dict(experiment='A',arm=arm,version='v2',n=n) for n in (1,2,4))
        plan.extend(dict(experiment='B',arm=arm,version=v,n=4) for v in ('v2','v3'))
    historical=c.read(PROJECT/'docs/reports/artifacts/deep_baselines_repair_v2/publication_audit.json')['new_and_updated_files_sha256']
    # Protect all previous task evidence but permit current README/tracker updates later.
    historical={str(PROJECT/k):v for k,v in historical.items() if k not in ('README.md','docs/reports/mrgr_plain_final_experiment_tracking_v1_zh.md')}
    anchor=c.verified(legacy.ROOT/'main_anchor.json')
    historical.update(anchor['training_sources']);historical.update(anchor['protocol_hashes'])
    for role in ('model','reference'):historical[anchor[role]['checkpoint']]=anchor[role]['checkpoint_sha256']
    old=c.read(PROJECT/'docs/reports/artifacts/deep_baselines_repair_v2/audit.json');historical.update(old['old_evidence_sha256'])
    c.verify_hashes(historical)
    # This seal is written before reading measurements or generating synthetic targets.
    freeze('preregistration.json',dict(at=c.now(),commit=os.popen('git rev-parse HEAD').read().strip(),
        remote_main=os.popen('git rev-parse origin/main').read().strip(),config=design(),sources=sources(),
        historical_hashes=historical,jobs=plan,total_updates=2000,H_training_allowed=False,
        formal_training_allowed=False,second_seed_allowed=False))
    main_train_anchor('main_before.json')
    with TrainGuard(synthetic_truth=True) as access:
        ds=source_dataset();samples=[selected_sample(ds,i) for i in design()['train_case_indices']]
        atomic(ROOT/'inputs/real_AB.pt',samples)
        synth,precision=generate(ds);atomic(ROOT/'inputs/synthetic_AB.pt',synth)
        ids={str(c.DATA/'source_index.jsonl'):c.sha(c.DATA/'source_index.jsonl'),str(c.DATA/'geometry/train_geometry.npz'):c.sha(c.DATA/'geometry/train_geometry.npz')}
        cases=[]
        for i,sample in enumerate(samples):
            row=ds.records[i];ids[str(c.DATA/row['train_projection_path'])]=c.sha(c.DATA/row['train_projection_path'])
            groups=ds.sampler.sample(ds.station_count,epoch=0,sample_index=i,case_id=row['case_id'])
            cases.append(dict(index=i,case=row['case_id'],stations={k:v.tolist() for k,v in zip(('A','B','H'),groups)},
                AB_sha256={k:input_hash(v) for k,v in sample.items()}))
        inputs={str(p):c.sha(p) for p in sorted((ROOT/'inputs').glob('*'))}
        freeze('registration.json',dict(at=c.now(),sources=sources(),preregistration_sha256=c.sha(ROOT/'preregistration.json'),
            cases=cases,data_identity=ids,inputs=inputs,initial_states={a:state_hash(model(a).state_dict()) for a in ARMS},
            synthetic_precision=precision,access=access.summary(),GT_used=False,val_used=False,test_used=False,
            GT_statistics_read=False,H_in_training=False,total_updates=2000))
        assert not access.denied
    freeze('synthetic_projection_precision.json',precision)
    c.write(ROOT/'diagnostic_status.json',dict(at=c.now(),state='registered',A_updates=0,B_updates=0,C_updates=0,
        formal_training_authorized=False,second_seed_authorized=False))
    print('REGISTERED',ROOT,flush=True)

if __name__=='__main__':main()
