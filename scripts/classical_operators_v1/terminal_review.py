"""Independent NumPy replay of failed-pilot final fields, no model or GT."""
import argparse
from context import *

def main(arm):
    root=ROOT/arm/'pilot';terminal=ROOT/arm/'terminal_audit';result=read(terminal/'result.json');diag=read(terminal/'interventions/diagnostics.json')
    ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    assert sha(root/'last.pt')==result['checkpoint_sha256'] and ck['epoch']==result['checkpoint']['epoch']
    assert ck['optimizer_steps']==result['checkpoint']['updates'] and ck['pilot_only'] and not ck['volumetric_gt_used']
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={ck['optimizer_steps']}
    assert ck['scheduler']['last_epoch']==ck['epoch'] and ck['rng']
    hist=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    assert len(hist)==ck['epoch'] and all(r['optimizer_steps']==16*r['epoch'] for r in hist)
    assert all(r['train']['sample_count']==32 and r['val']['sample_count']==36 for r in hist)
    count=0;rows=[]
    for entry in diag['arrays']:
        assert sha(entry['path'])==entry['sha256'];split,index=Path(entry['path']).stem.rsplit('_',1);index=int(index)
        with np.load(entry['path'],allow_pickle=False) as z:
            for branch in ['A','B']:
                predictions={mode:z[f'{branch}_{mode}_volume'].astype(np.float64) for mode in ['normal','zero','wrong_case']}
                scores={}
                for mode,volume in predictions.items():
                    logits=z[f'{branch}_{mode}_logits'].astype(np.float64)
                    assert volume.shape==(1,1,64,64,64) and np.isfinite(volume).all() and np.isfinite(logits).all()
                    np.testing.assert_allclose(volume,1/(1+np.exp(-logits)),rtol=2e-6,atol=1e-7)
                    assert volume.min()>=0 and volume.max()<=1
                    y=z[branch+'_H_reference'].astype(np.float64);mask=z[branch+'_H_mask'].astype(bool);p=z[f'{branch}_{mode}_H_projection'].astype(np.float64)
                    score=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6));scores[mode]=score
                    old=next(r for r in diag['rows'] if r['split']==split and r['index']==index and r['branch']==branch and r['mode']==mode)
                    values=dict(H_relative_l1=score,mean=float(volume.mean()),std=float(volume.std()),nearzero=float((volume<1e-10).mean()),saturated=float((np.abs(logits)>10).mean()))
                    for k,v in values.items():assert np.isclose(v,old[k],rtol=2e-6,atol=1e-30),(k,v,old[k])
                    if mode=='normal':normal=dict(old,**values)
                    count+=1
                diff=np.abs(predictions['normal']-predictions['zero']);rel=float(diff.sum()/max(np.abs(predictions['normal']).sum(),1e-30))
                normal.update(normal_zero_relative_MA=rel,normal_zero_absolute_MA=float(diff.mean()),wrong_H_increase=scores['wrong_case']-scores['normal'],zero_H_increase=scores['zero']-scores['normal'])
                rows.append(normal)
    def cases(split):
        answer=[]
        for index in sorted({r['index'] for r in rows if r['split']==split}):
            pair=[r for r in rows if r['split']==split and r['index']==index]
            answer.append({k:float(np.mean([r[k] for r in pair])) for k in ['normal_zero_relative_MA','normal_zero_absolute_MA','wrong_H_increase','zero_H_increase']})
        return answer
    tr,va=cases('train'),cases('val');v=result['validation_replay'];o=v['output'];score=v['measurement']['heldout']['relative_l1'];initial=read(root/'epoch0.json')['val']['measurement']['heldout']['relative_l1']
    checks=dict(healthy_fields=all(r['mean']>=1e-8 and r['nearzero']<.95 and r['std']/max(r['mean'],1e-30)>=.05 and r['saturated']<.95 for r in rows),
        train_zero_2of3=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 for r in tr)>=2,
        train_wrong_2of3=sum(r['wrong_H_increase']>=1e-6 for r in tr)>=2,
        val_zero_benefit_2of4=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 and r['zero_H_increase']>=.001 for r in va)>=2,
        val_below_zero=score<=.99,val_initial_improvement=score<=.95*initial and initial-score>=.01,
        val_healthy=o['mean']>=1e-8 and o['std']/max(o['mean'],1e-30)>=.05 and o['case_to_case_mean_absolute_difference']/max(o['mean'],1e-30)>=1e-3)
    checks={k:bool(v) for k,v in checks.items()};assert checks==result['checks']
    assert not all(checks.values()) and result['status']=='NO_GO';assert result['model_hash_before']==result['model_hash_after']
    assert result['logged_validation_exact'] and result['new_updates']==0 and not result['optimizer_created'];no_forbidden(result['access'])
    if (root/'actual_updates.jsonl').exists():
        trace=[json.loads(s) for s in (root/'actual_updates.jsonl').read_text().splitlines()]
        fail=torch.load(root/'failure_state.pt',map_location='cpu',weights_only=False)
        assert len(trace)==fail['optimizer_steps'] and [r['update'] for r in trace]==list(range(1,len(trace)+1))
        assert all(r['actual_parameter_update_l2']>0 and r['postclip_gradient_norm']>0 for r in trace)
    out=dict(all_passed=True,scientific_GO=False,status='NO_GO',necessary_conditions=checks,full_fields_recomputed=count,
        cases=dict(train=tr,val=va),checkpoint=result['checkpoint'],actual_training_updates=result['actual_training_updates'],
        new_updates=0,GT_read=False,model_forwards=0,review_definition='Second NumPy FP64 numerical implementation, not external-person review',
        limitations='Incomplete 30-epoch pilot; necessary collapse/zero-baseline failure independently confirmed. No precision rank or formal training.')
    write(ART/(arm+'_independent_review.json'),out);write(ART/(arm+'_interventions.json'),rows)
    print(json.dumps(out),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['deeponet','fno']);main(ap.parse_args().arm)
