#!/usr/bin/env python3
"""Fresh, GT-free matched feedback-gate screen on shared GPU3; never formal300."""
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

import probe_mrgr_multiscale_v1 as base
from probe_mrgr_multiscale_hard_panel_v1 import Replay, heldout
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.data.measurement_only import MeasurementOnlyPoolDataset
from geophy_set3d.training.utils import move_batch

OUT = base.ROOT / 'outputs/research/mrgr_feedback_gate_v2'
LAYOUT_SOURCE = base.ROOT / 'outputs/research/mrgr_multiscale_formal_v1_seed20261011/evaluation/frozen_model_test_v1/registration.json'
ARMS = ('multiscale', 'feedback_gate')
DEVICE = 'cuda:3'


class GatedAdapter(nn.Module):
    """Bounded per-view spatial gain; shared adapter keys preserved under original."""
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.gate = nn.Sequential(nn.Conv3d(48, 8, 1), nn.SiLU(), nn.Conv3d(8, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.record = False
        self.measurements = []
        self.force_identity = False

    def forward(self, x):
        delta = self.original(x)
        gain = 1 + .5 * torch.tanh(self.gate(x))
        if self.force_identity:
            gain = torch.ones_like(gain)
        if self.record:
            with torch.no_grad():
                change = delta * (gain - 1)
                self.measurements.append(dict(gain_mean=float(gain.mean()),
                    gain_std=float(gain.std(unbiased=False)), gain_min=float(gain.min()), gain_max=float(gain.max()),
                    mean_abs_gain_minus_one=float((gain-1).abs().mean()),
                    raw_delta_norm=float(delta.norm()), intervention_norm=float(change.norm()),
                    relative_intervention=float(change.norm()/delta.norm().clamp_min(1e-12))))
        return delta * gain


def model_for(config, arm, device=DEVICE):
    model = base.model_for(config, 'multiscale', 'cpu')
    before = torch.get_rng_state()
    if arm == 'feedback_gate':
        torch.manual_seed(20261013)
        model.residual_adapter = GatedAdapter(model.residual_adapter)
    torch.set_rng_state(before)
    return model.to(device)


def source_hashes():
    paths = [*sorted((base.ROOT/'src').rglob('*.py')),
             base.ROOT/'scripts/probe_mrgr_multiscale_v1.py',
             base.ROOT/'scripts/probe_mrgr_multiscale_hard_panel_v1.py',
             Path(__file__)]
    return {str(p.relative_to(base.ROOT)):base.digest(p) for p in paths}


def verify():
    p = json.loads((OUT/'protocol.json').read_text())
    assert base.digest(OUT/'protocol.json') == (OUT/'protocol.sha256').read_text().strip()
    assert p['sources'] == source_hashes()
    assert p['layout_source_sha256'] == base.digest(LAYOUT_SOURCE)
    for path, sha in p['data_hashes'].items():
        assert base.digest(Path(p['config']['dataset']['root'])/path) == sha
    return p


def register():
    OUT.mkdir(exist_ok=False)
    config = base.load_config(base.ROOT/'configs/train_selfsup_oari_mrgr_v1.yaml')
    config = copy.deepcopy(config)
    config['training'].update(device=DEVICE, batch_size=1, gradient_accumulation=2)
    config['loader'].update(num_workers=2)
    base.selfsup.validate_selfsup_config(config)
    with base.GTReadGuard(config['dataset']['root'], OUT/'checkpoints') as guard:
        train = base.selfsup.make_dataset(config)
        val = base.selfsup.make_dataset(config, validation=True)
        rng = np.random.default_rng(20261016)
        ti = sorted(rng.choice(len(train),16,replace=False).tolist())
        vi = sorted(rng.choice(len(val),8,replace=False).tolist())
        models = {arm:model_for(config,arm,'cpu') for arm in ARMS}
        left, right = models['multiscale'].state_dict(), models['feedback_gate'].state_dict()
        for key, value in left.items():
            rightkey = key.replace('residual_adapter.', 'residual_adapter.original.') if key.startswith('residual_adapter.') else key
            assert torch.equal(value, right[rightkey])
        layout = json.loads(LAYOUT_SOURCE.read_text())
        settings = [{k:s[k] for k in ('condition','draw','views','input_ids','heldout_ids')}
                    for s in layout['settings'] if s['condition']=='clustered-V4']
        p = dict(registered_at=base.now(),sources=source_hashes(), config=config,epochs=20,
            train_indices=ti,val_indices=vi,train_case_ids=[train.records[i]['case_id'] for i in ti],
            val_case_ids=[val.records[i]['case_id'] for i in vi],seed=20261011,
            parameters={arm:sum(x.numel() for x in m.parameters()) for arm,m in models.items()},
            shared_initial_tensors_exact=True,initialization='fresh; no historical or screened weights',
            architecture='multiscale shared decoder plus residual_adapter(x)*(1+0.5*tanh(pointwise48to8to1(x))); last gate layer zero',
            schedule='first20 epochs of T_max300 cosine; AdamW2e-4 wd1e-4; batch1 accumulation2 for BOTH arms',
            selection='fixed final20 only; no checkpoint or hyperparameter selection',
            endpoints=['ordinary fixed A/B to H4 projection L1 on8val', 'clustered V4 five draws, average draws within8val cases'],
            intervention=['gain mean/std/min/max','mean abs gain-1','delta norm and relative gate intervention','final gate forced1 counterfactual'],
            layout_source=str(LAYOUT_SOURCE),layout_source_sha256=base.digest(LAYOUT_SOURCE),settings=settings,
            data_hashes={name:base.digest(Path(config['dataset']['root'])/name) for name in ['source_index.jsonl','train_statistics.json','geometry/train_geometry.npz']},
            max_peak_memory_bytes=9*1024**3,estimated_runtime_gate_seconds=3600,
            no_gt=True,no_final_test=True,no_formal300=True,
            limitation='16train8val single-seed measurement-only prefix; no converged or3D claim; different budget from main detail branch',
            registration_audit=guard.summary())
        base.write_json(OUT/'protocol.json',p)
        (OUT/'protocol.sha256').write_text(base.digest(OUT/'protocol.json')+'\n')
        for arm,m in models.items():
            path=OUT/'checkpoints'/arm;path.mkdir(parents=True)
            base.atomic_checkpoint(path/'initial.pt',dict(model=m.state_dict(),epoch=0))
    print(json.dumps(dict(stage='registered',parameters=p['parameters'],train=16,val=8)),flush=True)
    return p


def smoke(p):
    config=p['config']
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        ds=base.selfsup.make_dataset(config)
        raw=base.dual_measurement_collate([ds[p['train_indices'][0]]])
        batch=base.selfsup.move_dual_batch(raw,DEVICE)
        preds={}; records={}
        for arm in ARMS:
            model=model_for(config,arm)
            model.eval()
            with torch.no_grad():preds[arm]=model(batch['A']).cpu()
            model.train();torch.cuda.reset_peak_memory_stats(DEVICE);times=[]
            for k in range(3):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize(DEVICE);start=time.perf_counter()
                loss,parts,volumes,projections=base.selfsup.objective(model,batch)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                torch.cuda.synchronize(DEVICE);times.append(time.perf_counter()-start)
                del loss,parts,volumes,projections
            records[arm]=dict(microbatch_seconds=times,mean_seconds=float(np.mean(times[1:])),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),norm=float(norm))
            assert records[arm]['peak_allocated_bytes']<=p['max_peak_memory_bytes']
            del model;gc.collect();torch.cuda.empty_cache()
        assert torch.equal(preds[ARMS[0]],preds[ARMS[1]])
        # 16 training microbatches, validation 8 at~half backward cost; allow2x overhead.
        estimated=sum(v['mean_seconds'] for v in records.values())*20*20*2
        result=dict(passed=True,initial_full_outputs_bitwise_equal=True,arms=records,
            estimated_two_arm_seconds=estimated,within_budget=estimated<=p['estimated_runtime_gate_seconds'],gt_access=guard.summary())
        base.write_json(OUT/'smoke.json',result)
    print(json.dumps(dict(stage='smoke',**result)),flush=True)
    return result


def train_arm(p,arm):
    config=p['config'];folder=OUT/arm;folder.mkdir(exist_ok=False)
    with base.GTReadGuard(config['dataset']['root'],OUT/'checkpoints') as guard:
        model=model_for(config,arm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)
        generator=torch.Generator().manual_seed(20261011)
        ds=base.selfsup.make_dataset(config);vd=base.selfsup.make_dataset(config,validation=True)
        tl=base.selfsup.make_loader(Subset(ds,p['train_indices']),config,True,generator)
        vl=base.selfsup.make_loader(Subset(vd,p['val_indices']),config)
        start=time.perf_counter();rows=[]
        for epoch in range(p['epochs']):
            ds.set_epoch(epoch);vd.set_epoch(0);torch.cuda.reset_peak_memory_stats(DEVICE)
            lr=optimizer.param_groups[0]['lr'];began=time.perf_counter()
            tr=base.selfsup.run_epoch(model,tl,config,optimizer)
            va=base.selfsup.run_epoch(model,vl,config)
            schedule.step()
            row=dict(epoch=epoch+1,train=tr,val=va,lr=lr,seconds=time.perf_counter()-began,
                loader_rng_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE))
            assert row['peak_allocated_bytes']<=p['max_peak_memory_bytes']
            rows.append(row);base.append(folder/'metrics.jsonl',row)
            base.atomic_checkpoint(OUT/'checkpoints'/arm/'last.pt',dict(model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),epoch=epoch+1,
                updates=(epoch+1)*8,rng=base.rng_state(generator),config=config,
                protocol_sha256=base.digest(OUT/'protocol.json'),volumetric_gt_used=False))
            base.write_json(folder/'status.json',dict(epoch=epoch+1,epochs=p['epochs'],state='running',
                val_H=va['measurement']['heldout']['relative_l1'],seconds=time.perf_counter()-start))
            base.write_json(folder/'gt_access_audit.json',guard.summary())
            print(f'{arm} epoch{epoch+1}/20 H={va["measurement"]["heldout"]["relative_l1"]:.7f} seconds={row["seconds"]:.2f}',flush=True)
        model.eval()
        if arm=='feedback_gate':model.residual_adapter.record=True
        ordinary=base.final_cases(model,vl,config)
        base.write_json(folder/'ordinary.json',ordinary)
        if arm=='feedback_gate':
            model.residual_adapter.force_identity=True
            counterfactual=base.final_cases(model,vl,config)
            model.residual_adapter.force_identity=False
            base.write_json(folder/'ordinary_force_gain_one.json',counterfactual)
            # Clear forced-identity records; record normal application only below.
            model.residual_adapter.measurements=[]
        hard=[]
        with torch.no_grad():
            for setting in p['settings']:
                hd=MeasurementOnlyPoolDataset(config['dataset']['root'],split='val',geometry_pool='train',
                    evaluation_only=True,min_input_views=4,max_input_views=4,hold_views=4,base_seed=20261011,
                    normalization='linear_p99_5',statistics_file='train_statistics.json')
                hd.sampler=Replay(setting)
                for idx in p['val_indices']:
                    sample=hd[idx];b=move_batch(projection_pool_collate([sample]),DEVICE)
                    inp={k:v for k,v in b.items() if not k.startswith('hold_')};h=heldout(b)
                    assert sample['case_id'] in p['val_case_ids']
                    field=model(inp)
                    measures={}
                    for name,target in [('heldout',h),('observed',inp)]:
                        pred=base.project_input(field,target)
                        valid=target['input_valid_ray'].bool()&target['view_mask'][:,:,None,None,None]
                        measures[name]=base.relative_projection_metrics(pred[0],target['input_projection_raw'][0],valid[0])
                    hard.append(dict(case_id=sample['case_id'],draw=setting['draw'],measurement=measures,
                        input_ids=sample['input_station_indices'].tolist(),heldout_ids=sample['hold_station_indices'].tolist()))
        base.write_json(folder/'hard_panel.json',hard)
        if arm=='feedback_gate':
            base.write_json(folder/'gate_interventions.json',model.residual_adapter.measurements)
        summary=dict(completed=True,epochs=p['epochs'],updates=p['epochs']*8,elapsed_seconds=time.perf_counter()-start,
            final_H=rows[-1]['val']['measurement']['heldout']['relative_l1'],gt_access=guard.summary(),
            final_checkpoint_sha256=base.digest(OUT/'checkpoints'/arm/'last.pt'))
        assert all(summary['gt_access'][k]==0 for k in ('gt_open_attempts','forbidden_npy_attempts','supervised_checkpoint_open_attempts'))
        base.write_json(folder/'summary.json',summary)
        base.write_json(folder/'status.json',dict(state='complete',**summary))
        del model,optimizer,tl,vl;gc.collect();torch.cuda.empty_cache()


def compare(p):
    logs={a:[json.loads(s) for s in (OUT/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    for x,y in zip(logs[ARMS[0]],logs[ARMS[1]]):
        assert x['lr']==y['lr'] and x['loader_rng_sha256']==y['loader_rng_sha256']
        for split in ('train','val'):
            assert x[split]['case_station_digest']==y[split]['case_station_digest']
    panels={};means={}
    for panel in ('ordinary','clustered'):
        values={}
        for arm in ARMS:
            if panel=='ordinary':
                rows=json.loads((OUT/arm/'ordinary.json').read_text())
                values[arm]={r['case_id']:r['heldout_relative_l1'] for r in rows}
            else:
                rows=json.loads((OUT/arm/'hard_panel.json').read_text())
                values[arm]={case:np.mean([r['measurement']['heldout']['relative_l1'] for r in rows if r['case_id']==case]) for case in p['val_case_ids']}
            means.setdefault(arm,{})[panel]=float(np.mean(list(values[arm].values())))
        cases=sorted(values[ARMS[0]]);x=np.array([values[ARMS[0]][c] for c in cases]);y=np.array([values[ARMS[1]][c] for c in cases]);delta=y-x
        rng=np.random.default_rng(20261016);boot=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
        panels[panel]=dict(reference=float(x.mean()),treatment=float(y.mean()),relative_improvement=float(1-y.mean()/x.mean()),
            delta_mean=float(delta.mean()),paired_ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta<0).sum()),cases=cases)
    interventions=json.loads((OUT/'feedback_gate/gate_interventions.json').read_text())
    actual=np.mean([r['mean_abs_gain_minus_one'] for r in interventions]);rel=np.mean([r['relative_intervention'] for r in interventions])
    enabled=json.loads((OUT/'feedback_gate/ordinary.json').read_text())
    disabled=json.loads((OUT/'feedback_gate/ordinary_force_gain_one.json').read_text())
    counterfactual=np.mean([r['heldout_relative_l1'] for r in disabled])-np.mean([r['heldout_relative_l1'] for r in enabled])
    result=dict(completed=True,matched_initial_weights_and_forward=True,matched_lr_loader_and_station_digests=True,
        epochs=p['epochs'],train_cases=16,val_cases=8,panels=panels,
        gate_mean_abs_gain_minus_one=float(actual),gate_relative_delta_intervention=float(rel),
        gain_one_minus_enabled_ordinary_H=float(counterfactual),sources_unchanged=p['sources']==source_hashes(),
        no_gt=True,no_final_test=True,no_formal300=True,
        interpretation='Small GT-free measurement prefix screen; no claim of converged or3D superiority')
    base.write_json(OUT/'comparison.json',result)
    print(json.dumps(result),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','smoke','train','compare','all'));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(2)
    p=register() if args.action in ('register','all') else verify()
    if args.action in ('smoke','all'):
        result=smoke(p)
        if not result['within_budget']:raise RuntimeError('Pre-update runtime budget gate failed')
    if args.action in ('train','all'):
        result=json.loads((OUT/'smoke.json').read_text());assert result['passed'] and result['within_budget']
        for arm in ([args.arm] if args.arm else ARMS):train_arm(p,arm)
    if args.action in ('compare','all'):compare(p)


if __name__=='__main__':main()
