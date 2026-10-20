"""Freeze three fresh matched-budget runs before pilot outcomes exist."""
from runtime import *
import copy
import worker

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'design.json').exists():return verified(BASE/'design.json')
    manifest=copy.deepcopy(verified(HISTORY/'model_manifest.json'));manifest['models']=manifest['models'][:1]
    manifest['split_cases']={k:v for k,v in manifest['split_cases'].items() if k in ('train','val')}
    freeze(BASE/'model_manifest.json',manifest)
    reg=verified(HISTORY/'registration.json');freeze(BASE/'registration.json',dict(registered_at=now(),cases={k:reg['cases'][k] for k in ('train','val')},statistics=dict(seed=20261019,resamples=20000),test_allowed=False))
    freeze(BASE/'layouts.json',{'val':verified(HISTORY/'layouts.json')['val']})
    runs=[str(worker.register(arm)) for arm in ARMS]
    design=dict(at=now(),runs=runs,arms=list(ARMS),seed=SEED,epochs=EPOCHS,updates_per_run=1660,fresh_initialization=True,
        git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),sources=identity(),
        hypothesis='Residual OARI may improve cross-view corrections; compare against equal20-epoch baseline and parameter-matched zero-residual OARI control. No direction assumed.',
        structure='Lift(Enc(signed Y-P(C0), mask, path)) concatenated with original context0; original48→32+SiLU; independent OARI; original32→32 zero-initialized output; add to context0; original GCA/decoder C1.',
        initialization='retained baseline tensors exact; new OARI copied from same-seed initialized first-stage OARI into independent parameters; no added RNG consumption',
        protocol='166train/36val; batch2 FP32, canonical loss/P(C)/AdamW/clip and cosineT300; stop exactly after first20 epochs; select H4 minimum only',
        analysis='all17 layouts; ordinaryV4 and per-case average clustered5draw are screening endpoints; L1/L2, peak, GT-peak ratio, centroid, Dice10/50, observed/H4 projection, runtime/memory; case-paired bootstrap20000 seed20261019, descriptive uncorrected95% intervals',
        no_early_stopping=True,no_GT_for_training_or_selection=True,no_test_or_GT128=True,
        limitations=['one training seed','short early-training prefix may not predict converged300-epoch ranking','new OARI increases parameters versus baseline','zero-residual control preserves added parameters and internal P(C0), but removes residual information only','no capacity-matched non-attention block in this small pilot','no declaration of final-model superiority'],
        scheduling='reserve at most one dynamic UUID via shared gpu_locks honored by C/D; wait for no GPU compute processes/free>=20000MiB/used<100MiB/util<=5% twice, recheck before every launch; never signal other processes; release reservation on completion/failure',
        promote='only a preliminary signal if ordinary and clustered L1 improve without worse mean peak absolute error or centroid, and true residual beats zero residual; still requires longer matched-budget/multi-seed confirmation')
    freeze(BASE/'design.json',design);return design

if __name__=='__main__':main()
