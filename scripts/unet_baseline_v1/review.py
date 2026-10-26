"""Second implementation: NumPy FP64 intervention/gate replay; no model forwards."""
import argparse,math
from unet_context import *
from numpy_projector import project

def close(a,b):assert np.isclose(a,b,rtol=2e-6,atol=2e-10),(a,b)

def review(arm):
    root=ROOT/arm/'pilot';summary=read(root/'result.json');initial=read(root/'epoch0.json')
    history=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    assert len(history)==30 and [r['epoch'] for r in history]==list(range(1,31))
    assert all(r['optimizer_steps']==16*r['epoch'] and r['train']['sample_count']==32 and r['val']['sample_count']==36 for r in history)
    assert all(not r[k]['gt_used'] for r in history for k in ['train','val'])
    updates=read(root/'updates.json');assert len(updates)==480 and all(r['actual_parameter_update_l2']>0 and r['postclip_gradient_norm']>0 for r in updates)
    assert sha(summary['checkpoint'])==summary['checkpoint_sha256']
    ck=torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['pilot_only'] and not ck['volumetric_gt_used'] and ck['epoch']==30 and ck['optimizer_steps']==480
    assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={480}
    assert ck['scheduler']['last_epoch']==30 and ck['rng'] and all(torch.isfinite(p).all() for p in ck['model'].values())
    count=0;all_gates={};rowsout=[]
    for epoch in [0,10,20,29,30]:
        d=read(root/f'diagnostic_epoch{epoch}/diagnostics.json');registered=d['rows'];rows=[];health=[]
        for entry in d['arrays']:
            assert sha(entry['path'])==entry['sha256']
            split,index=Path(entry['path']).stem.rsplit('_',1);index=int(index)
            with np.load(entry['path'],allow_pickle=False) as z:
                for branch in ['A','B']:
                    v={mode:z[f'{branch}_{mode}_volume'].astype('float64') for mode in ['normal','zero','wrong_case']}
                    scores={}
                    for mode in v:
                        volume=v[mode];logit=z[f'{branch}_{mode}_logits'].astype('float64')
                        assert volume.shape==(1,1,64,64,64) and np.isfinite(volume).all() and np.isfinite(logit).all()
                        assert volume.min()>=0 and volume.max()<=1
                        np.testing.assert_allclose(volume,1/(1+np.exp(-logit)),rtol=2e-6,atol=1e-7)
                        p=z[f'{branch}_{mode}_H_projection'].astype('float64');y=z[f'{branch}_H_reference'].astype('float64');mask=z[f'{branch}_H_mask'].astype(bool)
                        independent=project(volume,*(z[branch+'_H_'+k] for k in ['input_station_origin_m','input_ray_direction','input_t_near','input_path_length_m']),mask)
                        np.testing.assert_allclose(independent[mask],p[mask],rtol=3e-4,atol=.02)
                        score=float(np.abs(p[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6));scores[mode]=score
                        independent_score=float(np.abs(independent[mask]-y[mask]).sum()/max(np.abs(y[mask]).sum(),1e-6))
                        assert np.isclose(independent_score,score,rtol=1e-4,atol=1e-6)
                        r=next(r for r in registered if r['split']==split and r['index']==index and r['branch']==branch and r['mode']==mode)
                        for key,value in [('H_relative_l1',score),('mean',float(volume.mean())),('std',float(volume.std())),('nearzero',float((volume<1e-10).mean()))]:close(r[key],value)
                        if mode=='normal':normal=dict(r,mean=float(volume.mean()),std=float(volume.std()),nearzero=float((volume<1e-10).mean()),saturated=float((np.abs(logit)>10).mean()))
                        health.append(dict(mean=float(volume.mean()),std=float(volume.std()),nearzero=float((volume<1e-10).mean()),saturated=float((np.abs(logit)>10).mean())));count+=1
                    relative=float(np.abs(v['normal']-v['zero']).sum()/max(np.abs(v['normal']).sum(),1e-30));absolute=float(np.abs(v['normal']-v['zero']).mean())
                    wrong=scores['wrong_case']-scores['normal'];zero=scores['zero']-scores['normal']
                    for key,value in [('normal_zero_relative_MA',relative),('normal_zero_absolute_MA',absolute),('wrong_H_increase',wrong),('zero_H_increase',zero)]:close(normal[key],value)
                    normal.update(normal_zero_relative_MA=relative,normal_zero_absolute_MA=absolute,wrong_H_increase=wrong,zero_H_increase=zero);rows.append(normal)
        def aggregate(split):
            result=[]
            for i in sorted({r['index'] for r in rows if r['split']==split}):
                pair=[r for r in rows if r['split']==split and r['index']==i]
                result.append({k:np.mean([r[k] for r in pair]) for k in ['normal_zero_relative_MA','normal_zero_absolute_MA','wrong_H_increase','zero_H_increase']})
            return result
        tr,va=aggregate('train'),aggregate('val')
        current=initial['val'] if epoch==0 else history[epoch-1]['val'];score=current['measurement']['heldout']['relative_l1'];before=initial['val']['measurement']['heldout']['relative_l1'];o=current['output']
        checks=dict(healthy_fields=all(r['mean']>=1e-8 and r['nearzero']<.95 and r['std']/max(r['mean'],1e-30)>=.05 and r['saturated']<.95 for r in health),
            train_zero_2of3=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 for r in tr)>=2,
            train_wrong_2of3=sum(r['wrong_H_increase']>=1e-6 for r in tr)>=2,
            val_zero_response_2of4=sum(r['normal_zero_relative_MA']>=1e-3 and r['normal_zero_absolute_MA']>=1e-8 for r in va)>=2,
            val_zero_benefit_2of4=sum(r['zero_H_increase']>=.001 for r in va)>=2,
            val_below_zero=score<=.99,val_initial_improvement=score<=.95*before and before-score>=.01)
        checks={k:bool(v) for k,v in checks.items()};assert checks==read(root/f'diagnostic_epoch{epoch}/gates.json')
        all_gates[str(epoch)]=checks;rowsout.extend(rows)
    scientific=all(all(all_gates[str(e)].values()) for e in [29,30]) and summary['engineering'] and summary['training_improved'] and summary['parameter_net_change_l2']>0
    assert scientific==(summary['status']=='GO')
    no_forbidden(summary['access'])
    out=dict(all_passed=True,scientific_GO=scientific,status='GO' if scientific else 'NO_GO',
        recomputed_full_fields=count,diagnostic_checks=all_gates,optimizer_steps=480,epoch0_available=True,
        full_field_numpy_ray_reprojection=True,definition='Independent NumPy computation and independent gate replay; not external-person review',GT_read=False,model_forwards=0,new_updates=0)
    write(ART/(arm+'_independent_review.json'),out)
    write(ART/(arm+'_interventions.json'),rowsout)
    print(json.dumps(out),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['unet']);review(ap.parse_args().arm)
