#!/usr/bin/env python3
"""Independent CPU audit of sealed parameter screens; never reads GT."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import mrgr_parameter_optimization_v2 as run
read=run.read

def close(a,b):assert math.isclose(a,b,rel_tol=3e-6,abs_tol=3e-7),(a,b)
def verify_seed(out):
    p=run.verify(out);r=read(out/'comparison.json');logs={};arrays={};hard={};times={};checks=[]
    for arm in p['arms']:
        folder=out/arm;config=read(folder/'effective_config.json');setting=p['settings'][arm]
        assert config['optimizer']['lr']==setting['lr'] and config['loss']['consistency']==setting['consistency']
        assert config['training']['seed']==p['seed']
        logs[arm]=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()];assert len(logs[arm])==20
        for i,row in enumerate(logs[arm]):
            assert row['epoch']==i+1;close(row['learning_rate'],setting['lr']*.5*(1+math.cos(math.pi*i/300)))
            assert row['peak_allocated_bytes']>0
            for split in ('train','val'):
                q=row[split];assert q['sample_count']==(166 if split=='train' else 36) and not q['gt_used']
                l=q['loss'];standard=l['hold']+.5*l['cross']+.1*l['observed']+.05*l['consistency']
                close(l['standard_total'],standard);close(l['total'],standard+(setting['consistency']-.05)*l['consistency'])
                close(l['weighted_consistency_actual'],setting['consistency']*l['consistency'])
        checkpoint=folder/'checkpoints/last.pt';summary=read(folder/'summary.json')
        assert run.base.digest(checkpoint)==summary['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==20 and saved['updates']==1660 and saved['arm']==arm
        assert saved['protocol_sha256']==run.base.digest(out/'protocol.json') and not saved['volumetric_gt_used']
        assert saved['config']==config;assert sum(x.numel() for x in saved['model'].values())==344678
        assert all(torch.isfinite(x).all() for x in saved['model'].values())
        cases=[json.loads(x) for x in (folder/'epoch20_per_case.jsonl').read_text().splitlines()]
        assert [x['case_id'] for x in cases]==p['validation_case_ids']
        for row in cases:
            assert row['stations']==p['station_map'][row['case_id']]
            close(row['heldout_relative_l1'],.5*(row['A']['relative_l1']+row['B']['relative_l1']))
            for q in row['output_sanity'].values():assert q['std']>1e-8 and q['min']>=0 and q['max']<=1
        arrays[arm]=np.array([q['heldout_relative_l1'] for q in cases]);cluster=[json.loads(x) for x in (folder/'cluster5_per_case.jsonl').read_text().splitlines()]
        assert len(cluster)==180 and len({(x['case_id'],x['draw']) for x in cluster})==180
        for q in cluster:
            source=[s for s in p['cluster_settings'] if s['draw']==q['draw']][0]
            assert q['input_ids']==source['input_ids'][q['case_id']] and q['heldout_ids']==source['heldout_ids'][q['case_id']]
        hard[arm]=np.array([np.mean([q['relative_l1'] for q in cluster if q['case_id']==case]) for case in p['validation_case_ids']])
        times[arm]=np.mean([q['seconds'] for q in logs[arm][2:]])
        probe=read(folder/'late_gradient_probe.json')
        assert probe['actual_weight']==setting['consistency']
        close(probe['original_consistency_to_total_gradient_ratio'],probe['all_parameter_original_consistency_gradient_norm']/probe['all_parameter_total_gradient_norm'])
        close(probe['weighted_consistency_candidate'],setting['consistency']*probe['parts']['consistency'])
        close(probe['adjusted_total'],probe['original_total']+(setting['consistency']-.05)*probe['parts']['consistency'])
        assert -1.00001<=probe['cosine_consistency_with_total']<=1.00001
        for name in ('gt_access_audit.json','cluster5_audit.json'):
            audit=read(folder/name);assert all(audit[k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        for key,values in (('ordinary',arrays[arm]),('cluster5',hard[arm])):
            close(r[key][arm]['mean'],values.mean());close(r[key][arm]['p90'],np.quantile(values,.9))
        close(r['warm_epoch_seconds'][arm],times[arm]);del saved
    for arm in p['arms'][1:]:
        for x,y in zip(logs['control'],logs[arm]):
            assert x['loader_rng_sha256']==y['loader_rng_sha256']
            for split in ('train','val'):assert x[split]['case_station_digest']==y[split]['case_station_digest']
        c=r['contrasts'][arm];o=arrays[arm];b=arrays['control'];h=hard[arm];bh=hard['control']
        ratios=dict(ordinary=o.mean()/b.mean(),p90=np.quantile(o,.9)/np.quantile(b,.9),cluster=h.mean()/bh.mean(),runtime=times[arm]/times['control'])
        g=p['gate'];expected=dict(ordinary=ratios['ordinary']<=g['ordinary_mean_ratio_max'],p90=ratios['p90']<=g['p90_ratio_max'],cluster=ratios['cluster']<=g['cluster5_ratio_max'],runtime=ratios['runtime']<=g['runtime_ratio_max'],five_hour=times[arm]*300/3600<=g['hours300_max'],smoke=read(out/'smoke.json')['passed'])
        assert expected==c['checks'];assert all(expected.values())==c['gate_passed']
        for kind,delta in (('ordinary',o-b),('cluster5',h-bh)):
            rng=np.random.default_rng(20261017);interval=np.quantile(delta[rng.integers(0,36,(20000,36))].mean(1),[.025,.975])
            for x,y in zip(interval,c[kind]['paired_ci95']):close(x,y)
        close(c['ordinary']['mean_ratio'],ratios['ordinary']);close(c['ordinary']['p90_ratio'],ratios['p90']);close(c['cluster5']['mean_ratio'],ratios['cluster']);close(c['runtime_ratio'],ratios['runtime'])
    result=dict(passed=True,seed=p['seed'],arms=p['arms'],epochs=20*len(p['arms']),updates=1660*len(p['arms']),checks=['source/data/protocol/config/checkpoint hashes','LR analytic cosine schedule','standard and actual weighted objective algebra per epoch','same seed/case order/stations/shuffle','output finite/range/nonconstant','36 ordinary and180 cluster records per arm','all paired bootstrap intervals/gates independently recomputed','GT and foreign checkpoint guards zero'],checked_at=run.base.now(),verifier_sha256=run.base.digest(Path(__file__)))
    run.write(out/'independent_verification.json',result);print(json.dumps(result))

if __name__=='__main__':
    for out in sorted(run.OUT.glob('seed*')):
        if (out/'comparison.json').exists():verify_seed(out)
