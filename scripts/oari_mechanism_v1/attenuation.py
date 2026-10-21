"""B: reuse sealed scalars, pair stages, retain draw/permutation hierarchy."""
from runtime import *


def run():
    verify_hashes(verified(BASE/'protocol.json')['sources'])
    audit=verified(BASE/'audit.json');verify_hashes(audit['source_metrics'])
    rows_old=[r for r in csv_read(HIST.BASE/'paired_metrics.csv') if r['experiment'] in ('P1','P2')]
    lookup={(r['case_id'],r['condition'],int(r['draw']),r['experiment'],int(r['permutation_index']),r['stage']):r for r in rows_old}
    assert len(lookup)==len(rows_old)==11340
    raw=csv_read(HIST.BASE/'per_case_metrics.csv');absolutes={(r['case_id'],r['condition'],int(r['draw']),r['experiment'],int(r['permutation_index']),r['stage']):r for r in raw if r['experiment'] in ('P1','P2')}
    a={(r['case_id'],r['condition'],int(r['draw']),r['stage']):r for r in raw if r['experiment']=='A'}
    plans=verified(HIST.BASE/'permutation_maps.json');cases=verified(HIST.BASE/'cases.json')
    # Map each scalar pair to the same manifest and frozen checkpoint, no volume
    # labels or metric recomputation. File bytes were sealed and checked earlier.
    manifest={}
    for li in range(17):
        root=HIST.BASE/'predictions'/f'layout{li:02d}'
        for r in rows(root/'manifest.jsonl'):
            if r['experiment'] in ('P1','P2'):
                manifest[r['case_id'],r['condition'],r['draw'],r['experiment'],r['permutation_index']]=r
    detailed=[]
    for key,c0 in lookup.items():
        case,condition,draw,mode,pi,stage=key
        if stage!='C0':continue
        c1=lookup[case,condition,draw,mode,pi,'C1'];mr=manifest[case,condition,draw,mode,pi]
        rr=dict(case_id=case,condition=condition,draw=draw,experiment=mode,permutation_index=pi,permutation=c0['permutation'],prediction_sha256=mr['sha256'],checkpoint_sha256=HIST.EXPECTED)
        for st,row in [('C0',c0),('C1',c1)]:
            absolute=absolutes[case,condition,draw,mode,pi,st]
            assert absolute['checkpoint_sha256']==HIST.EXPECTED and absolute['prediction_sha256']==mr['sha256']
            assert mr['input_ids']==next(p for p in plans if (p['case_id'],p['condition'],p['draw'])==(case,condition,draw))['input_ids']
            for metric in METRICS:
                val=None if row[metric]=='' else float(row[metric]);rr[f'delta_{st}_{metric}']=val
                base=a[case,condition,draw,st][metric];value=absolute[metric]
                assert (val is None and (base=='' or value=='')) or abs(float(value)-float(base)-val)<1e-12
        for metric in METRICS:
            x=rr[f'delta_C0_{metric}'];y=rr[f'delta_C1_{metric}']
            rr[f'attenuation_{metric}']=None if x is None or y is None else x-y
            rr[f'absolute_attenuation_{metric}']=None if x is None or y is None else abs(x)-abs(y)
        detailed.append(rr)
    assert len(detailed)==5670
    percase=[];panels=[];idx=np.random.default_rng(SEED).integers(0,35,(20000,35))
    for condition in CONDITIONS:
        nd=1 if condition in ('Val-A-V4','spread-V4') else 5
        for mode in ('P1','P2'):
            for case in cases:
                selected=[r for r in detailed if (r['case_id'],r['condition'],r['experiment'])==(case,condition,mode)]
                groups=[[r for r in selected if r['draw']==draw] for draw in range(nd)]
                expected_perm=(5 if condition=='V3' else 10) if mode=='P1' else 1
                assert all(len(g)==expected_perm for g in groups)
                record=dict(case_id=case,condition=condition,experiment=mode,draw_count=nd,permutations_per_draw=expected_perm)
                for metric in METRICS:
                    for label in ('delta_C0','delta_C1','attenuation','absolute_attenuation'):
                        key=f'{label}_{metric}';vv=[[r[key] for r in g] for g in groups]
                        record[key]=None if any(v is None for g in vv for v in g) else float(np.mean([np.mean(g) for g in vv]))
                percase.append(record)
            selected=[r for r in percase if (r['condition'],r['experiment'])==(condition,mode)]
            for metric in METRICS:
                panel=dict(condition=condition,experiment=mode,metric=metric,n_cases=35,draws=nd,exploratory=True)
                for label in ('delta_C0','delta_C1','attenuation','absolute_attenuation'):
                    vv=[r[f'{label}_{metric}'] for r in selected];defined=all(v is not None for v in vv)
                    aa=np.array(vv,dtype=float) if defined else None
                    panel.update({label+'_mean':float(aa.mean()) if defined else None,label+'_median':float(np.median(aa)) if defined else None,
                                  label+'_ci95':np.quantile(aa[idx].mean(1),[.025,.975]).tolist() if defined else None,label+'_case_values':vv})
                panels.append(panel)
    csv_write(BASE/'attenuation_per_draw_permutation.csv',detailed);csv_write(BASE/'per_case_attenuation.csv',percase)
    csv_write(BASE/'c0_c1_attenuation_statistics.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in panels])
    freeze(BASE/'attenuation_summary.json',dict(at=now(),completed=True,panels=panels,case_count=35,seed=SEED,resamples=20000,
        forwards=0,metric_rescoring=0,GT64_read=False,GT128_read=False,source_metrics_sha256=audit['source_metrics'],
        meaning='signed attenuation=deltaC0-deltaC1; for Dice larger is better, so signed shrinkage is not universally improvement; absolute_attenuation also reported',
        identity_checks='C0/C1 matched case/layout/draw/permutation, identical prediction SHA and checkpoint; stored paired differences matched stored absolute scalars'))
    print('B COMPLETE: reused11340 stage rows,5670 matched pairs; zero inference/GT',flush=True)


if __name__=='__main__':
    with Guard():run()
