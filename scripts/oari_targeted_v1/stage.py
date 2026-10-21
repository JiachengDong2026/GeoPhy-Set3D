"""Registered, isolated execution stage using the unchanged formal train engine."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'scripts/uniform_final_v1'))
import runtime as canonical_runtime
from runtime import *
import worker as canonical_worker
import copy
import numpy as np
import torch
import argparse
import importlib.util
sys.path.insert(0, str(HERE))
from variants import model_for, PLAIN

BASE = canonical_runtime.BASE.parent/'mrgr_oari_targeted_v1'
ART = PROJECT/'docs/reports/artifacts'/BASE.name
REPORT = PROJECT/'docs/reports/mrgr_oari_targeted_v1_zh.md'
TASKS = (('uniform_no_ray_relations',20261019),(PLAIN,20261011),(PLAIN,20261019))
SEED = 20261026


def load_unique(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def training_sources():
    result=canonical_runtime.training_identity()
    result.update({str(HERE/n):sha(HERE/n) for n in ('stage.py','variants.py','checks.py')})
    return result


def tensor_hash(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def input_hash(batch):
    h=hashlib.sha256()
    for k,v in sorted(batch.items()):
        h.update(k.encode());h.update(tensor_hash(v).encode() if isinstance(v,torch.Tensor) else repr(v).encode())
    return h.hexdigest()


def register():
    backend();BASE.mkdir(exist_ok=True)
    if (BASE/'protocol.json').exists():
        verify_hashes(verified(BASE/'protocol.json')['sources']);return
    references=verified(canonical_runtime.BASE/'reuse_methods.json')
    references=[m for m in references if m['key'] in ('uniform_fusion_seed20261011','uniform_fusion_seed20261019')]
    old=BASE.parent/'mrgr_uniform_ablations_v1'
    no_ray=next(m for m in verified(old/'final_lock.json')['methods'] if m['key']=='uniform_no_ray_relations_seed20261011')
    references.append(dict(no_ray,prediction_root=str(old/'evaluation/test'/no_ray['key']),source_metrics=str(old/'evaluation/test/metrics.jsonl')))
    assert len(references)==3
    identities=[]
    for m in references:
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
        assert ck['epoch']==m['best_epoch'] and ck['config']==m['config'] and not ck['volumetric_gt_used']
        model=model_for(m['config'],'uniform' if m['key'].startswith('uniform_fusion') else 'uniform_no_ray_relations','cpu')
        model.load_state_dict(ck['model'],strict=True)
        assert sum(p.numel() for p in model.parameters())==m['parameters']==342725
        root=Path(m['prediction_root']);mp=root/'manifest.jsonl';complete=read(root/'complete.json')
        sealed=verified(root.parent/'all_predictions_sealed.json')
        assert sha(mp)==complete['manifest_sha256']==sealed['manifests'][str(mp)]
        assert complete['checkpoint_sha256']==m['checkpoint_sha256']
        selected=[r for r in rows(mp) if r['condition'] in ('Val-A-V4','clustered-V4')]
        assert len(selected)==210
        for r in selected:assert sha(root/r['prediction'])==r['sha256']
        identities.append(dict(key=m['key'],checkpoint_sha256=m['checkpoint_sha256'],best_epoch=m['best_epoch'],parameters=m['parameters'],manifest_sha256=sha(mp)))
    main=next(m for m in references if m['key']=='uniform_fusion_seed20261011')
    assert main['checkpoint_sha256']=='64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc'
    config=main['config']
    old_no=verified(Path(no_ray['run'])/'protocol.json')
    def normalized(c):
        c=copy.deepcopy(c);c.pop('experiment');c['training'].pop('seed');return c
    assert normalized(old_no['config'])==normalized(config)
    # Search actual run directories for accidental reuse/restart candidates.
    candidates=[]
    for p in BASE.parent.glob('*/training/*'):
        if any(p.name==f'{arm}_seed{s}' for arm,s in TASKS):
            candidates.append(dict(path=str(p),last=(p/'checkpoints/last.pt').exists(),best=(p/'checkpoints/best.pt').exists()))
    assert all(not c['last'] and not c['best'] for c in candidates),candidates
    freeze(BASE/'candidate_audit.json',dict(at=now(),candidates=candidates,checkpoint_audit=identities))
    freeze(BASE/'references.json',references)
    historical=verified(HISTORY/'registration.json')
    settings=verified(HISTORY/'layouts.json')['test']
    settings=[s for s in settings if s['condition'] in ('Val-A-V4','clustered-V4')]
    assert len(settings)==6
    freeze(BASE/'cases.json',historical['cases']['test']);freeze(BASE/'layouts.json',settings)
    common_sources=training_sources()
    for arm,seed in TASKS:
        target=BASE/'training'/f'{arm}_seed{seed}';cfg=copy.deepcopy(config)
        cfg['training']['seed']=seed;cfg['experiment']=dict(name=target.name,output_root=str(target))
        model=model_for(cfg,arm,'cpu')
        freeze(target/'protocol.json',dict(registered_at=now(),arm=arm,seed=seed,config=cfg,epochs=300,
            planned_updates=24900,parameters=sum(p.numel() for p in model.parameters()),sources=common_sources,
            train_cases=historical['cases']['train'],validation_cases=historical['cases']['val'],subset_smoke=False,
            data_files=verified(HISTORY/'model_manifest.json')['training_data_sha256'],fresh_initialization=True,
            initial_model_sha256=canonical_worker.probe.state_hash(model.state_dict()),
            selection='strict minimum validation projectionH4 over all300epochs; GT forbidden',
            structure=str(model),fusion='UniformFusion'))
    freeze(BASE/'protocol.json',dict(at=now(),sources=common_sources,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        tasks=[dict(arm=a,seed=s) for a,s in TASKS],epochs=300,updates=24900,
        training_config_source=dict(path=main['checkpoint'],sha256=main['checkpoint_sha256']),
        test_cases=35,trained_model_layouts='ordinarydraw0 + clusteredfive draws,210 predictions per model',
        frozen_probe='official seed20261019 checkpoint, same OARIv2 obs_token, two layouts draw0,70 predictions; reuse if verified available',
        stats=dict(seed=SEED,resamples=20000,unit='pairedcase; clustered five draws averaged within case; draw0 separately',
            delta='treatment minus matched full model',interval='percentile95% mean difference, exploratory unadjusted',tie=1e-8),
        primary_metrics=['C1relative_l1'],secondary_metrics=['C0/C1L1/L2','peak ratio/absolute error','peak/centroid mm','Dice50','D','inference time','peakGPU memory'],
        authorization='current user explicitly requests three trainings and frozen seed19 evaluation in existing35case GT64 scope; all new predictions sealed before current-phase GT64',
        GT128_forbidden=True,training_GT64_forbidden=True,test_previously_used=True,independent_confirmation=False,
        model_promotion='review all layouts/seeds/metrics/costs; no automatic checkpoint/config change'))
    print(BASE,flush=True)


def train(target):
    assert verified(BASE/'checks/cpu.json')['passed'] and verified(BASE/'checks/gpu.json')['passed']
    target=Path(target);p=verified(target/'protocol.json')
    assert (p['arm'],p['seed']) in TASKS
    engine=canonical_worker.engine;engine.OUT=BASE;engine.model_for=model_for
    with AccessGuard(splits=['train','val'],output=BASE) as guard:
        engine.train(target,300)
        assert not guard.denied
        write(target/'boundary_audit.json',guard.summary())


def scheduler():
    s=load_unique('targeted_canonical_scheduler',PROJECT/'scripts/uniform_final_v1/controller.py')
    s.BASE=BASE;s.GPU_LOCKS=BASE.parent/'gpu_locks';s.progress=lambda:None
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','train']);p.add_argument('--target');a=p.parse_args()
    if a.action=='register':register()
    else:train(a.target)
