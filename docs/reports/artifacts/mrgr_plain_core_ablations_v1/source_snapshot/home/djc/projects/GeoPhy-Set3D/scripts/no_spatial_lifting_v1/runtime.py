"""Isolated geometry-free feature lifting experiment on the original paper MRGR."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'scripts/paper_final_v1'))
from common import *
import copy
import numpy as np
import torch
from torch.nn import functional as F
import importlib.util

def _official_uniform_factory():
    # Execute the registered final factory with its actual runtime, avoiding our
    # identically named runtime module. No source substitution or GCA fallback.
    folder=PROJECT/'scripts/uniform_final_v1'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    previous=sys.modules.get('runtime')
    try:
        sys.modules['runtime']=load('registered_uniform_runtime',folder/'runtime.py')
        return load('registered_uniform_models',folder/'uniform_models.py').model_for
    finally:
        if previous is None:sys.modules.pop('runtime',None)
        else:sys.modules['runtime']=previous

_REGISTERED_FACTORY=_official_uniform_factory()
def official_model(config, arm, device):
    assert arm=='uniform'
    return _REGISTERED_FACTORY(config,'uniform',device)
from probe_mrgr_multiscale_v1 import state_hash
from geophy_set3d.models.lifting import GeometryLifting

BASE = Path('/scratch/djc/GeoPhy-Set3D/mrgr_no_spatial_lifting_v1')
ARTIFACT = PROJECT/'docs/reports/artifacts/mrgr_no_spatial_lifting_v1'
REPORT = PROJECT/'docs/reports/mrgr_no_spatial_lifting_v1_zh.md'
SEEDS = (20261011, 20261019)
ARM = 'no_spatial_lifting'
UNIFORM_BASE = BASE.parent/'mrgr_uniform_final_v1'
EXPECTED = {20261011:'64c1c4a41e66b518e415243be5796b89d36f3c27e5c4165e98b77af49de381fc',20261019:'19bb5f698349c976358be782e9d43301afa9e1eec4a89ddb4e1ceeb8d9bd1424'}
CONDITIONS = ('Val-A-V4','spread-V4','V3','V6','clustered-V4')


class PlanarExtrusionLifting(GeometryLifting):
    """Retain detector-local features, but no geometry-conditioned resampling.

    Resize [vertical,horizontal] to32x32, put horizontal along Y and vertical
    along Z, and repeat along X. The same geometry-free convention applies to
    every view and both main/residual calls. Original angular_grid is used
    exclusively to retain original visibility; inherited descriptors are intact.
    """
    def forward(self, features, batch):
        b,v,c,_,_ = features.shape
        _, visible = self.angular_grid(batch)
        r = self.resolution
        plane = F.interpolate(features.flatten(0,1), size=(r,r), mode='bilinear', align_corners=True)
        xyz = plane.transpose(-1,-2).unsqueeze(2).expand(-1,-1,r,-1,-1).contiguous()
        return xyz.reshape(b,v,c,r,r,r), visible.reshape(b,v,r,r,r)


def model_for(config, arm, device):
    assert arm in (ARM, 'uniform')
    model = official_model(config, 'uniform', 'cpu')
    if arm == ARM:
        # No construction/RNG draw, no new parameters or buffers, identical state keys.
        model.lifting.__class__ = PlanarExtrusionLifting
    return model.to(device)


def training_identity():
    import train
    d = train.source_identity()
    d.update({str(HERE/n):sha(HERE/n) for n in ('runtime.py','audit.py','checks.py','worker.py','controller.py')})
    d.update({str(PROJECT/'scripts/uniform_final_v1'/n):sha(PROJECT/'scripts/uniform_final_v1'/n)
              for n in ('controller.py','scope.py','runtime.py','uniform_models.py')})
    return d


class TrainGuard(AccessGuard):
    def __init__(self):
        super().__init__(splits=['train','val'], output=BASE)
    def audit(self,event,args):
        if self.active and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(OUT/'evaluation/test'):
                self.denied.append(dict(path=str(p),reason='historical test predictions/metrics forbidden in training'))
                raise PermissionError('Test artifacts forbidden in training: '+str(p))
        super().audit(event,args)


def run_protocol(seed, smoke=None):
    root=BASE/('training' if smoke is None else 'resume_smoke')/f'{ARM}_seed{seed}'
    return root if smoke is None else root/smoke


def register_run(seed, smoke=None):
    target=run_protocol(seed,smoke)
    assert not (target/'protocol.json').exists()
    control=next(m for m in verified(BASE/'controls.json') if m['seed']==seed)
    config=copy.deepcopy(control['config'])
    config['training']['device']='cuda:0'
    config['experiment']=dict(name=f'{ARM}_seed{seed}',output_root=str(target))
    model=model_for(config,ARM,'cpu')
    original=verified(OUT/'registration.json')
    freeze(target/'protocol.json',dict(registered_at=now(),arm=ARM,seed=seed,config=config,
        epochs=300,parameters=342725,trainable_parameters=342725,planned_updates=24900 if smoke is None else 4,
        sources=training_identity(),train_cases=original['cases']['train'],validation_cases=original['cases']['val'],
        subset_smoke=smoke is not None,data_files=verified(OUT/'model_manifest.json')['training_data_sha256'],
        fresh_initialization=True,initial_model_sha256=state_hash(model.state_dict()),
        initialization='official same-seed factory; replace parameter-free lifting class without drawing RNG; all state tensors exactly matched',
        mapping='geometry-free fixed detector-plane extrusion; both C0 and residual lifting; original visibility and angular_grid retained',
        selection='strict lowest mean validation H4 projection relative L1 over all300 epochs; no volume GT',
        resume='canonical paper_final_v1/train.py epoch-commit model/best/optimizer/scheduler/Python/NumPy/Torch/CUDA/loader RNG/metrics; prime persistent workers before RNG restore',
        loader='canonical selfsup.make_loader:4 fork persistent workers, torch DataLoader default worker seeding; deterministic case/epoch A/B/H sampling',
        nondeterminism='FP32/TF32off/cudnn benchmarkoff; original differentiable projector grid_sample CUDA backward may use nondeterministic atomics'))
    return target


def scheduler():
    import importlib.util
    folder=PROJECT/'scripts/uniform_final_v1';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('extrusion_uuid_scheduler',folder/'controller.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    s.BASE=BASE;s.PROJECT=PROJECT;s.GPU_LOCKS=BASE.parent/'gpu_locks';s.progress=lambda:None
    return s
