#!/usr/bin/env python3
"""Independent scalar/count/checkpoint audit; no GT volume or predictions read."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src')]
import numpy as np
import torch
from geophy_set3d.training.deep_comparison import load_comparison_config,sha256
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training.utils import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/train_deepsets_selfsup_v1.yaml');a=ap.parse_args();c=load_comparison_config(a.config);out=Path(c['comparison']['output'])
    complete=json.loads((out/'training_complete.json').read_text());freeze=json.loads((out/'frozen_best.json').read_text())
    # Read the experiment's own last/best only; runtime guard denies all volumetric GT.
    with GTReadGuard(c['dataset']['root'],Path(freeze['checkpoint']).parent) as guard:
        last=torch.load(Path(freeze['checkpoint']).parent/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(freeze['checkpoint'],map_location='cpu',weights_only=False)
        assert last['epoch']==299 and last['optimizer_steps']==24900 and last['scheduler']['last_epoch']==300
        assert best['epoch']+1==freeze['best_epoch'] and best['config']==last['config']==c
        steps={int(s['step']) for s in last['optimizer']['state'].values()};assert steps=={24900}
        assert sha256(freeze['checkpoint'])==freeze['sha256']
        epochs=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];assert len(epochs)==300
        assert all(r['train']['sample_count']==166 and r['val']['sample_count']==36 for r in epochs)
        selected=min(epochs,key=lambda x:x['val']['measurement']['heldout']['relative_l1'])
        assert selected['epoch']==best['epoch'] and selected['val']['measurement']['heldout']['relative_l1']==best['best_val_metric']
        traces=[json.loads(l) for l in (out/'actual_training_inputs.jsonl').read_text().splitlines()];assert len(traces)==300
        assert [r['epoch'] for r in traces]==list(range(300))
        for epoch in traces:
            assert len(epoch['batches'])==166 # A/B each 83 batches.
            order=[x for b in epoch['batches'] if b['subset']=='A' for x in b['case_ids']]
            assert len(order)==len(set(order))==166
        rows=list(csv.DictReader((out/'paired_900.csv').open()));stats=list(csv.DictReader((out/'paired_statistics.csv').open()));assert len(rows)==900 and len(stats)==50
        identities={(r['case_id'],r['group'],int(r['draw'])) for r in rows};assert len(identities)==900
        groups=defaultdict(list)
        for r in rows:groups[(r['case_id'],r['group'])].append(r)
        assert len(groups)==180 and all(sorted(int(r['draw']) for r in v)==list(range(5)) for v in groups.values())
        cases=sorted({r['case_id'] for r in rows});assert len(cases)==36
        ix=np.random.default_rng(c['comparison']['bootstrap_seed']).integers(0,36,(10000,36));checks=0
        for s in stats:
            m=s['metric'];g=s['group'];va=[];vb=[]
            for case in cases:
                v=groups[(case,g)];va.append(math.fsum(float(r['Ours_'+m]) for r in v)/5);vb.append(math.fsum(float(r['DeepSets_'+m]) for r in v)/5)
            delta=np.asarray(va)-np.asarray(vb);ci=np.quantile(np.mean(delta[ix],axis=1),[.025,.975])
            expected=dict(ours_mean=math.fsum(va)/36,deepsets_mean=math.fsum(vb)/36,mean_delta=math.fsum(delta)/36,median_delta=float(np.median(delta)),ci_low=ci[0],ci_high=ci[1])
            for name,val in expected.items():assert math.isclose(float(s[name]),val,rel_tol=1e-11,abs_tol=1e-14),(g,m,name);checks+=1
            assert int(s['ours_wins'])==sum(x < -1e-8 for x in delta)
            assert int(s['deepsets_wins'])==sum(x > 1e-8 for x in delta)
            assert int(s['ties'])==sum(abs(x)<=1e-8 for x in delta)
        audit=guard.summary();assert audit['gt_open_attempts']==audit['forbidden_npy_attempts']==audit['supervised_checkpoint_open_attempts']==0
    write_json(out/'independent_audit.json',dict(passed=True,epochs=300,optimizer_steps=24900,optimizer_state_steps=sorted(steps),best_selected_only_from_heldout=True,
         exact_epoch_case_batch_counts=True,paired_rows=900,case_groups=180,scalar_and_bootstrap_checks=checks,all50_win_loss_tie_counts_checked=True,access=audit))
    print('Independent checkpoint/count/selector/scalar/bootstrap audit passed',flush=True)

if __name__=='__main__':main()
