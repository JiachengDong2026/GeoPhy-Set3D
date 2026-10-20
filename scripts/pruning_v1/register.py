"""Pre-register a fixed development-only factorial design before new outcomes."""
from runtime import *
import copy
import torch
from pruned_models import model_for
import train_worker as worker
import probe_mrgr_multiscale_v1 as probe

def setup():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'design.json').exists():
        verified(BASE/'design.json');return
    original=verified(HISTORY/'model_manifest.json')
    manifest=copy.deepcopy(original);manifest['models']=manifest['models'][:1]
    manifest['split_cases']={k:v for k,v in manifest['split_cases'].items() if k in ('train','val')}
    # Metadata references are retained, but no test arrays or GT are opened.
    freeze(BASE/'model_manifest.json',manifest)
    layouts=verified(HISTORY/'layouts.json')
    freeze(BASE/'layouts.json',{'val':layouts['val']})
    reg=verified(HISTORY/'registration.json')
    freeze(BASE/'registration.json',dict(registered_at=now(),cases={k:reg['cases'][k] for k in ('train','val')},
        statistics=dict(resamples=20000,seed=20261019,unit='case after within-case draw averaging',ci='percentile 95%',training_seed_uncertainty=False),
        settings_sha256=sha(BASE/'layouts.json'),test_allowed=False,GT128_allowed=False))
    initial=[];runs=[]
    for seed in SEEDS:
        config=copy.deepcopy(manifest['models'][0]['config']);config['training']['seed']=seed
        full=model_for(config,'multiscale','cpu').state_dict()
        for arm in ARMS:
            m=model_for(config,arm,'cpu');state=m.state_dict();mapping=[]
            for key,value in state.items():
                reference=full[key];rule='exact'
                if key=='interaction.token_projection.weight':reference=reference[:,:48];rule='columns [0:48]'
                if key=='interaction.pairwise_bias.0.weight':reference=reference[:,:3];rule='columns [0:3]'
                assert torch.equal(value,reference),key
                mapping.append(dict(key=key,shape=list(value.shape),rule=rule))
            target=worker.engine.register(arm,seed);runs.append(str(target))
            record=dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in m.parameters()),
                full_initial_sha256=probe.state_hash(full),initial_sha256=probe.state_hash(state),
                retained_mapping=mapping,deleted_keys=sorted(set(full)-set(state)),
                removed_elements=sum(v.numel() for v in full.values())-sum(v.numel() for v in state.values()),
                bias_policy='retained biases unchanged; no absorption of zero-descriptor embedding bias')
            freeze(target/'initialization.json',record);initial.append(record)
    freeze(BASE/'design.json',dict(registered_at=now(),worktree=str(PROJECT),historical_root=str(HISTORICAL),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        sources=source_identity(),runs=runs,arms=list(ARMS),seeds=list(SEEDS),epochs=300,updates=24900,
        reuse=['multiscale_seed20261011','multiscale_seed20261019','uniform_fusion_seed20261011','uniform_fusion_seed20261019','no_observability_seed20261011','no_observability_seed20261019'],
        hypothesis='Uniform valid-view fusion and structural descriptor deletion may trade reconstruction quality for cost; no direction assumed. Zero-input ablation is diagnostic only.',
        selection='strict minimum mean validation H4 relative L1 over all300 epochs; no GT or clustered metric selection',
        comparisons=['B-A','C-A','D-B','D-C','(D-C)-(B-A)','C-zero_input'],
        statistics=dict(resamples=20000,seed=20261019,paired_by=['training_seed','case_id'],draws='average five draws inside each case before case resampling',
            seed_means='descriptive; paired case bootstrap conditional on the two observed seeds, no seed-population uncertainty',
            claims='all development comparisons exploratory; no confirmatory noninferiority margin or significance claim'),
        fixed_figure_case_indices=[0,18,35],grid=[64,64,64],forbidden=['test projections','test GT64','all GT128'],
        resource_policy=dict(poll_seconds=15,min_free_MiB=20000,max_used_MiB=100,max_utilization=5,require_no_compute_process=True),
        limitations=['prior main model and final35 remain frozen','36 development cases already used in earlier exploration','two seeds insufficient to quantify seed population','new independent data needed for confirmation'] ))
    freeze(BASE/'initialization_manifest.json',initial)
    print('Registered',runs,flush=True)

if __name__=='__main__':setup()
