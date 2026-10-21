"""Authorized CPU-only scoring with official functions and case paired statistics."""
from analyze import *
import evaluation as official
from scipy.stats import spearmanr


def authorize():
    # This command only records the explicit reply in the active session. The
    # literal reply and scope are retained; no historic authorization is reused.
    seal=verified(ART/'predictions_sealed.json')
    assert seal['all_prediction_hashes_shapes_ranges_verified']
    freeze(ART/'GT64_authorization.json',dict(at=now(),
        authority='explicit user reply in this session to request_user_input_async',
        user_reply='授权本阶段限定的 GT64 离线评分',
        question='锁定本阶段协议和预测清单后，仅读取同一35案例的 GT64，对已保存的 A/B/C/D 及两种子结构消融做离线配对评分；GT128始终禁止，不训练、不选模型。',
        instruction='/home/djc/.codex/attachments/a0f8df48-eebc-444e-aa98-6e118a8bd9fe/已粘贴的文本.txt',
        scope='only registered35test cases, Val-A-V4/clustered-V4 draw0, frozen A/B/C/D and matched structural deletion two seeds',
        protocol_sha256=sha(ART/'protocol.json'), prediction_seal_sha256=sha(ART/'predictions_sealed.json'),
        cases=verified(ART/'cases.json'),GT64_allowed=True,GT128_allowed=False,training_allowed=False,
        checkpoint_selection_allowed=False,model_tuning_allowed=False,independent_confirmation=False))


def score():
    protocol=verified(ART/'protocol.json'); verify_hashes(protocol['sources'])
    auth=verified(ART/'GT64_authorization.json'); seal=verified(ART/'predictions_sealed.json')
    assert auth['GT64_allowed'] and not auth['GT128_allowed']
    assert auth['protocol_sha256']==seal['protocol_sha256']==sha(ART/'protocol.json')
    assert auth['prediction_seal_sha256']==sha(ART/'predictions_sealed.json')
    manifest=verified(ART/'prediction_manifest.json');assert sha(ART/'prediction_manifest.json')==seal['manifest_sha256']
    cases=verified(ART/'cases.json');assert auth['cases']==cases
    # An execution lock captures analysis files added after the immutable design.
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    lock=dict(at=now(),sources=sources,authorization_sha256=sha(ART/'GT64_authorization.json'),
        protocol_sha256=sha(ART/'protocol.json'),seal_sha256=sha(ART/'predictions_sealed.json'),
        new_forwards=0,GT64_opened=False)
    freeze(ART/'scoring_lock.json',lock)
    sources_index={r['case_id']:r for r in rows(DATA/'source_index.jsonl')}
    historic_gt=read(ROOT/'docs/reports/artifacts/mrgr_uniform_ablations_v1/gt_access.json')['GT64_sha256_after_seal']
    oldmetrics=list(csv.DictReader((ART/'historical_case_metrics_draw0.csv').open()))
    oldlookup={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in oldmetrics}
    results=[]; hashes={}; max_history_diff=0.; first_gt=None
    with guard_for(manifest,gt=True) as guard:
        # Reverify every prediction after seal and before the first target read.
        for r in manifest:assert sha(r['prediction'])==r['prediction_sha256']
        for case in cases:
            source=sources_index[case];assert source['split']=='test'
            gtpath=DATA/source['gt_volume_64_path']
            if first_gt is None:first_gt=now()
            target=np.load(gtpath,allow_pickle=False).astype(np.float64).squeeze()
            assert target.shape==(64,)*3 and np.isfinite(target).all() and target.min()>=0 and target.max()<=1
            hashes[str(gtpath)]=sha(gtpath);assert hashes[str(gtpath)]==historic_gt[str(gtpath)]
            for r in [x for x in manifest if x['case_id']==case]:
                for stage,p in fields(r['prediction']).items():
                    volume=official.volume_metrics(p,target)
                    morphology=official.safe_shape(p,target)
                    morphology['absolute_peak_ratio_error']=abs(morphology['peak_ratio']-1)
                    flat=flatten(dict(volume=volume,morphology=morphology))
                    assert all(v is not None and np.isfinite(v) for v in flat.values())
                    key='uniform_fusion_seed20261011' if r['model']=='A0' else r['model']
                    old=oldlookup.get((case,r['condition'],stage,key))
                    if old:
                        for metric,value in flat.items():
                            diff=abs(value-float(old[metric])); max_history_diff=max(max_history_diff,diff)
                            assert diff<1e-11,(case,key,stage,metric,diff)
                    results.append(dict(case_id=case,condition=r['condition'],draw=0,model=r['model'],seed=r['seed'],stage=stage,**flat))
            print(f'Scored {len(hashes)}/35 cases; {len(results)}/980 fields',flush=True)
        assert not guard.denied and len(hashes)==35
    save_csv(ART/'case_metrics.csv',results)
    freeze(ART/'GT64_access.json',dict(at=now(),first_GT64_at=first_gt,prediction_seal_at=seal['at'],
        authorization_sha256=sha(ART/'GT64_authorization.json'),scoring_lock_sha256=sha(ART/'scoring_lock.json'),
        unique_GT64_files=35,GT64_sha256=hashes,GT128_read=False,new_forwards=0,training=False,
        official_archived_metrics_fields_verified=560,maximum_history_metric_difference=max_history_diff,audit=guard.summary()))
    print('Scoring complete; archived metric max difference',max_history_diff,flush=True)


def summaries():
    cases=verified(ART/'cases.json')
    raw=list(csv.DictReader((ART/'case_metrics.csv').open()))
    for r in raw:
        for k in METRICS:r[k]=float(r[k])
    lookup={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in raw}
    changes=list(csv.DictReader((ART/'prediction_changes.csv').open()))
    change={(r['case_id'],r['condition'],r['stage'],r['model']):r for r in changes}
    rng=np.random.default_rng(SEED);indices=rng.integers(0,35,(RESAMPLES,35))
    save_csv(ART/'bootstrap_indices.csv',[dict(replicate=i,**{f'i{j}':int(x) for j,x in enumerate(rr)}) for i,rr in enumerate(indices)])
    pairs=[];summary=[];correlations=[];patterns=[]
    comparisons=[('frozen',g,'A0',20261011) for g in GROUPS if g!='A0']
    comparisons += [('structural',f'uniform_pruned_observability_seed{s}',
        'A0' if s==20261011 else f'uniform_fusion_seed{s}',s) for s in (20261011,20261019)]
    for condition in CONDITIONS:
        for stage in ('C0','C1'):
            for kind,model,reference,seed in comparisons:
                vector={};d=[]
                for case in cases:
                    a=lookup[case,condition,stage,reference];b=lookup[case,condition,stage,model]
                    dd=float(change[case,condition,stage,model]['D']);d.append(dd)
                    row=dict(case_id=case,case_index=cases.index(case),condition=condition,draw=0,kind=kind,model=model,reference=reference,seed=seed,stage=stage,D=dd)
                    for k in METRICS:
                        delta=b[k]-a[k];row[k]=b[k];row['baseline_'+k]=a[k];row['delta_'+k]=delta
                        if k in ('relative_l1','relative_l2','absolute_peak_ratio_error','Dice50'):row['delta_'+k+'_pp']=100*delta
                        vector.setdefault(k,[]).append(delta)
                    pairs.append(row)
                for metric,values in vector.items():
                    astats=stats([lookup[c,condition,stage,reference][metric] for c in cases],indices)
                    bstats=stats([lookup[c,condition,stage,model][metric] for c in cases],indices)
                    summary.append(dict(kind=kind,model=model,reference=reference,seed=seed,condition=condition,draw=0,stage=stage,metric=metric,
                        **stats(values,indices),baseline_mean=astats['mean'],intervention_mean=bstats['mean'],
                        baseline_median=astats['median'],intervention_median=bstats['median'],baseline_sd=astats['sd'],intervention_sd=bstats['sd'],
                        increased=sum(x>TIE for x in values),decreased=sum(x < -TIE for x in values),unchanged=sum(abs(x)<=TIE for x in values),
                        direction='positive improves' if metric=='Dice50' else 'no monotone quality direction' if metric=='peak_ratio' else 'positive worsens'))
                summary.append(dict(kind=kind,model=model,reference=reference,seed=seed,condition=condition,draw=0,stage=stage,metric='D',**stats(d,indices)))
                for metric in ('relative_l1','relative_l2','absolute_peak_ratio_error','peak_location_error_mm','centroid_error_mm','Dice50'):
                    values=vector[metric]
                    constant=np.ptp(d)==0 or np.ptp(values)==0
                    correlations.append(dict(kind=kind,model=model,seed=seed,condition=condition,draw=0,stage=stage,metric=metric,n_cases=35,
                        pearson=None if constant else float(np.corrcoef(d,values)[0,1]),
                        spearman=None if constant else float(spearmanr(d,values).statistic),
                        undefined_reason='constant input or delta' if constant else None,exploratory=True))
                patterns.append(dict(kind=kind,model=model,seed=seed,condition=condition,draw=0,stage=stage,n_cases=35,
                    large_D_small_L1_cases=[c for i,c in enumerate(cases) if d[i]>=.05 and abs(vector['relative_l1'][i])<=.005],
                    small_D_peak_worse_cases=[c for i,c in enumerate(cases) if d[i]<=.01 and vector['absolute_peak_ratio_error'][i]>=.1],
                    small_D_location_worse_cases=[c for i,c in enumerate(cases) if d[i]<=.01 and vector['peak_location_error_mm'][i]>=600/63]))
    save_csv(ART/'paired_case_metrics.csv',pairs);save_csv(ART/'summary.csv',summary)
    save_csv(ART/'correlations.csv',correlations);freeze(ART/'case_patterns.json',patterns)
    freeze(ART/'statistics.json',dict(at=now(),seed=SEED,resamples=RESAMPLES,n_cases=35,draws=[0],
        summary=summary,correlations=correlations,interval_scope='unadjusted exploratory case-mean intervals; not training-seed population'))
    for r in summary:
        if r['stage']=='C1' and r['metric']=='relative_l1':print(r)


if __name__=='__main__':
    backend(); p=argparse.ArgumentParser();p.add_argument('action',choices=['authorize','score','summaries']);a=p.parse_args()
    try:{'authorize':authorize,'score':score,'summaries':summaries}[a.action]()
    except Exception:
        import traceback
        (ART/f'failure_{a.action}_{datetime.now().strftime("%Y%m%dT%H%M%S%f")}.log').write_text(traceback.format_exc());raise
