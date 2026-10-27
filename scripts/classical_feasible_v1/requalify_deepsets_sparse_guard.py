"""Read-only requalification of saved29/30 snapshots under the already fixed sparse guard."""
from sparse_guard_v2_context import *
import shutil,review_multicase


def verify_stage():
    verify();r=read(ART/'deepsets_guard_review_registration.json')
    for path,digest in r['sources'].items():assert sha(PROJECT/path)==digest,path
    for path,digest in r['boundary'].items():assert sha(path)==digest,path


def numpy_health(field):
    a=field.astype('float64');mask=field*(1-field)<4.54e-5
    return dict(mean=float(a.mean()),std=float(a.std()),nearzero=float((a<1e-10).mean()),
        saturated=float(mask.mean()),saturated_mass_fraction=float(a[mask].sum()/max(a.sum(),1e-30)))


def main():
    setup();verify_stage();source=BASE_ROOT/'pilot_deepsets';original=read(BASE_ART/'deepsets_pilot_summary.json')
    assert original['decision']=='STOP_MULTICASE_GATE_FAILED' and original['epochs']==30
    assert read(BASE_ART/'deepsets_multicase_review.json')['passed']
    rows=[json.loads(s) for s in (source/'metrics.jsonl').read_text().splitlines()]
    results=[]
    for epoch,filename in [(29,'best.pt'),(30,'last.pt')]:
        phase_root=ROOT/f'deepsets_review_epoch{epoch}';phase_art=ART/f'deepsets_review_epoch{epoch}'
        target=phase_root/'pilot_deepsets';target.mkdir(parents=True,exist_ok=False);phase_art.mkdir(exist_ok=False)
        checkpoint=source/filename;ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['epoch']==epoch and ck['optimizer_steps']==epoch*83;del ck
        os.link(checkpoint,target/'last.pt')
        for name in ['preheat_updates.jsonl','preheat_complete.json']:shutil.copy2(source/name,target/name)
        chosen=next(r for r in rows if r['stage']=='physical' and r['epoch']==epoch)
        selected_rows=[r for r in rows if r['stage']=='preheat' or r['epoch']<=epoch]
        (target/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected_rows))
        updates=(source/'physical_updates.jsonl').read_text().splitlines()[:epoch*83]
        (target/'physical_updates.jsonl').write_text('\n'.join(updates)+'\n')
        view=dict(original,epochs=epoch,physical_updates=epoch*83,last_checkpoint_sha256=sha(checkpoint),
            last_val_H=selection_metric(chosen['val']),evaluation_view_only=True,new_optimizer_updates=0)
        write(phase_art/'deepsets_pilot_summary.json',view)
        shutil.copy2(BASE_ART/'multicase_review_registration.json',phase_art/'multicase_review_registration.json')
        review_multicase.ART=phase_art;review_multicase.ROOT=phase_root;review_multicase.verify=verify_stage
        sys.argv=[sys.argv[0],'deepsets'];review_multicase.main()
        review=read(phase_art/'deepsets_multicase_review.json');assert review['passed'];stats=[]
        for row in review['records']:
            if row['index']>=(4 if row['split']=='val' else 3):continue
            with np.load(row['path'],allow_pickle=False) as arrays:
                for branch in ['A','B']:stats.append(dict(split=row['split'],index=row['index'],branch=branch,**numpy_health(arrays[branch+'_normal'])))
        val=[r for r in review['records'] if r['split']=='val' and r['index']<4]
        train=[r for r in review['records'] if r['split']=='train'];score=review['independent_val_H']
        checks=dict(val_H=score<=.99,improvement=score<=.95*original['initial_val_H'] and original['initial_val_H']-score>=.01,
            healthy=healthy(stats),val_benefit=sum(r['zero_benefit']>=.001 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in val)>=2,
            train_dependency=sum(r['wrong_increase']>=1e-6 and r['response_relative']>=.001 and r['response_absolute']>=1e-8 for r in train)>=2)
        from feasible_context import healthy as legacy_healthy
        results.append(dict(epoch=epoch,checks=checks,legacy_healthy=legacy_healthy(stats),normal_health=stats,
            independent_val_H=score,checkpoint_sha256=sha(checkpoint),projection_review=str(phase_art/'deepsets_multicase_review.json'),
            projection_review_sha256=sha(phase_art/'deepsets_multicase_review.json')))
        print('DEEPSETS_REQUALIFY',epoch,checks,flush=True)
    passed=all(all(r['checks'].values()) for r in results)
    write(ART/'deepsets_guard_review.json',dict(passed=True,pilot_GO=passed,
        decision='MULTICASE_REQUALIFIED_SPARSE_GUARD_V2' if passed else 'NO_GO_REVISED_GUARD',
        epochs=30,physical_updates=2490,new_physical_updates=0,preheat_updates=4980,new_preheat_updates=0,
        best_val_H=original['best_val_H'],last_val_H=original['last_val_H'],formal_complete=False,test_allowed=False,
        results=results,original_decision_preserved=original['decision'],new_optimizer_updates=0,no_test_or_GT=True))
    verify_stage();print('DEEPSETS_REQUALIFICATION_COMPLETE scientific_go=',passed,flush=True)


if __name__=='__main__':main()
