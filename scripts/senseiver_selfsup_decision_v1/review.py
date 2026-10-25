"""Independent CPU recomputation from sealed arrays; never calls model/loss metrics."""
import csv
import math
import numpy as np
from audit import *

def main():
    a=read(ART/'history_audit.json');d=read(ART/'final_checkpoint_diagnostic.json')
    checks={}
    checks['all_historical_evidence_files_unchanged']=all(sha(p)==h for p,h in a['evidence_sha256'].items())
    checks['all_preexisting_tracked_files_unchanged']=all(sha(PROJECT/p)==h for p,h in read(ROOT/'protected_tracked_files.json').items())
    checks['selfsup_best_last_sha_unchanged']=all(sha(v['path'])==v['sha256'] for v in a['checkpoints'].values())
    checks['supervised_sha_unchanged']=sha(a['supervised']['path'])==a['supervised']['sha256']
    anchor=read('/scratch/djc/GeoPhy-Set3D/deep_baselines_v1/main_anchor.json')['model']
    checks['frozen_main_checkpoint_sha_correct']=sha(anchor['checkpoint'])==anchor['checkpoint_sha256']
    summary=[];recomputed=[]
    for i in [0,1,2]:
        path=ROOT/f'case{i}.npz'
        assert sha(path)==d['saved_arrays_sha256'][str(path)]
        with np.load(path,allow_pickle=False) as z:
            rels=[];errors={k:[] for k in ['normal','zero','wrong_case']}
            for branch in ['A','B']:
                normal=z[branch+'_normal_volume'].astype('float64')
                zero=z[branch+'_zero_volume'].astype('float64')
                rel=float(np.abs(normal-zero).sum()/max(np.abs(normal).sum(),1e-30));rels.append(rel)
                for mode in ['normal','zero','wrong_case']:
                    prefix=branch+'_'+mode
                    vol=z[prefix+'_volume'].astype('float64');logit=z[prefix+'_logit'].astype('float64')
                    assert vol.shape==(1,1,64,64,64) and logit.shape==(1,262144,1)
                    assert np.isfinite(vol).all() and np.isfinite(logit).all() and vol.min()>=0 and vol.max()<=1
                    np.testing.assert_allclose(vol.reshape(-1),1/(1+np.exp(-logit.reshape(-1))),rtol=3e-7,atol=0)
                    mask=z[prefix+'_H_mask'];p=z[prefix+'_H_projection'][mask].astype('float64');y=z[prefix+'_H_reference'][mask].astype('float64')
                    l1=float(np.abs(p-y).sum()/max(np.abs(y).sum(),1e-6))
                    l2=float(np.sqrt(np.square(p-y).sum()/max(np.square(y).sum(),1e-12)))
                    errors[mode].append(l1)
                    row=next(r for r in d['rows'] if r['index']==i and r['branch']==branch and r['mode']==mode)
                    assert math.isclose(l1,row['H']['relative_l1'],rel_tol=0,abs_tol=3e-15)
                    assert math.isclose(l2,row['H']['relative_l2'],rel_tol=0,abs_tol=3e-15)
                    assert math.isclose(rel,row['normal_zero_relative_MA'],rel_tol=1e-12,abs_tol=1e-18)
                    recomputed.append(dict(case_index=i,case_id=row['case_id'],branch=branch,mode=mode,
                        normal_zero_relative_MA=rel,H_L1=l1,H_L2=l2,output_mean=float(vol.mean()),
                        output_std=float(vol.std()),logit_mean=float(logit.mean()),logit_std=float(logit.std()),
                        logit_fraction_below_minus10=float((logit < -10).mean())))
            means={k:sum(v)/2 for k,v in errors.items()}
            summary.append(dict(case_index=i,case_id=row['case_id'],normal_zero_relative_MA=sum(rels)/2,
                dependency_pass=sum(rels)/2>=1e-3,normal_H_L1=means['normal'],zero_input_H_L1=means['zero'],
                wrong_H_L1=means['wrong_case'],wrong_minus_normal=means['wrong_case']-means['normal'],
                wrong_error_increases=means['wrong_case']>means['normal']))
    checks['18_full_fields_logits_H_metrics_independently_recomputed']=len(recomputed)==18
    checks['no_diagnostic_weight_updates']=d['model_hash_before']==d['model_hash_after'] and not d['optimizer_created'] and d['new_updates']==0
    checks['full_val_selector_replayed_exactly']=d['validation_replay']['measurement']==d['validation_logged']['measurement'] and d['validation_replay']['case_station_digest']==d['validation_logged']['case_station_digest']
    checks['no_forbidden_access']=all(d['access'][k]==0 for k in ['gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'])
    hist=[json.loads(s) for s in (HIST/'metrics.jsonl').read_text().splitlines()]
    checks['113_consecutive_epochs_83_steps_each']=len(hist)==113 and all(r['epoch']==j and r['optimizer_steps']==83*(j+1) and r['optimizer_steps_this_epoch']==83 for j,r in enumerate(hist))
    # Independent finite traversal of all numeric training-log leaves.
    def finite(x):
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        if isinstance(x,list):return all(finite(v) for v in x)
        return math.isfinite(x) if isinstance(x,(int,float)) else True
    checks['all_log_numbers_finite']=all(finite(r) for r in hist)
    checks['all_epochs_nonzero_finite_gradient']=all(0<r['post_clip_gradient_norm_max']<=1.00001 for r in hist)
    seed_everything(a['config']['training']['seed']);fresh=build_comparison_model(a['config'],'cpu')
    saved=torch.load(CK/'last.pt',map_location='cpu',weights_only=False)
    differences={k:(v-saved['model'][k]).double() for k,v in fresh.state_dict().items()}
    delta=float(torch.stack([v.square().sum() for v in differences.values()]).sum().sqrt())
    checks['finite_nonzero_initial_to_final_net_parameter_change']=math.isfinite(delta) and delta>0
    checks['optimizer_steps_independently_9379']=sorted({int(v['step']) for v in saved['optimizer']['state'].values()})==[9379]
    assert all(checks.values()),checks
    dependent=sum(r['dependency_pass'] for r in summary)
    wrong_increases=sum(r['wrong_error_increases'] for r in summary)
    val_l1=d['validation_replay']['measurement']['heldout']['relative_l1']
    # Descriptive collapse check uses the historical diagnostic's 1e-6 level;
    # it is not a new training stopping rule. Classify from actual saved values.
    collapse=all(r['val']['output']['mean']<1e-6 for r in hist[-5:]) and all(
        r['logit_fraction_below_minus10']==1 for r in recomputed if r['mode']=='normal')
    necessary_failure=dependent<2 or wrong_increases<2 or val_l1>=1 or collapse
    gates=dict(normal_zero_cases_passing=dependent,required_cases=2,
        wrong_error_increases_cases=wrong_increases,
        val_H_L1=val_l1,val_below_zero_baseline=val_l1<1,
        sustained_near_zero_output=collapse,epoch0_full_val_improvement='NOT_VERIFIABLE_FROM_HISTORY',
        consecutive_passes='NOT_ESTABLISHED',
        outcome='NO_GO_EXISTING_SAME_PROTOCOL_FAILED' if necessary_failure else 'INCONCLUSIVE_MISSING_GATE_EVIDENCE')
    write(ART/'independent_review.json',dict(checks=checks,all_passed=True,gates=gates,case_summary=summary,
        initial_to_final_net_parameter_change_L2=delta,
        norm_scope='net change over 9379 historical updates, not individual optimizer-step norm; per-step norms not in original full-run logs',
        raw_arrays_sha256=d['saved_arrays_sha256'],new_updates=0,
        review_method='separate CPU NumPy reductions from saved volumes/logits/Pdiff-H/reference/mask; no call to diagnostic metric functions; same agent, not external reviewer'))
    for name,rows in [('case_summary.csv',summary),('interventions.csv',recomputed)]:
        with (ART/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    scheduling=ROOT/'scheduling/zero_update_diagnostic.json'
    assert read(scheduling)['state']=='complete'
    write(ART/'resource_and_execution.json',dict(scheduling=read(scheduling),
        gpu_snapshots=[json.loads(s) for s in (ROOT/'gpu_snapshots.jsonl').read_text().splitlines()],
        events=[json.loads(s) for s in (ROOT/'scheduling/events.jsonl').read_text().splitlines()],
        runtime_seconds=d['seconds'],peak_allocated_bytes=d['peak_gpu_allocated_bytes'],
        controller_log=str(ROOT/'scheduling/zero_update_diagnostic_attempt1.log'),
        queue_unchanged=True,no_oom=True,no_new_training=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(1,114)
    fig,ax=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    ax[0,0].semilogy(x,[r['val']['measurement']['heldout']['relative_l1']-1 for r in hist]);ax[0,0].set_ylabel('Val H relative L1 minus 1 (zero baseline)')
    for k in ['mean','std']:
        ax[0,1].semilogy(x,[r['val']['output'][k] for r in hist],label=k)
    ax[0,1].set_ylabel('Val output mass fraction');ax[0,1].legend()
    for k in ['train','val']:
        ax[1,0].plot(x,[r[k]['loss']['total'] for r in hist],label=k)
    ax[1,0].set_yscale('log');ax[1,0].set_ylabel('Frozen self-supervised objective');ax[1,0].legend()
    ax[1,1].semilogy(x,[r['post_clip_gradient_norm_max'] for r in hist]);ax[1,1].set_ylabel('Max post-clip gradient norm')
    for panel in ax.flat:
        panel.axvspan(.5,30.5,color='orange',alpha=.13);panel.axvline(30.5,color='gray',ls='--',lw=.8)
        panel.set_xlabel('Completed historical epochs');panel.grid(alpha=.2)
    fig.suptitle('Senseiver self-supervised history: 113 epochs / 9,379 updates\nShaded prefix: same conditions as proposed 30-epoch pilot; NO-GO')
    fig.savefig(ART/'training_curves.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(checks=len(checks),all_passed=True,gates=gates,net_change=delta)))

if __name__=='__main__':
    main()
