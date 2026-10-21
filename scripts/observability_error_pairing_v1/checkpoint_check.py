"""CPU-only strict loading through the registered constructors; never forward."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/paper_final_v1'))
from common import read, sha, verified, AccessGuard, backend
import torch

backend()
model_info = read(sys.argv[1])[int(sys.argv[2])]
m = model_info
branch = 'pruning_v1' if 'pruned' in m['key'] else 'uniform_final_v1'
sys.path.insert(0, str(ROOT / 'scripts' / branch))
if branch == 'pruning_v1':
    from pruned_models import model_for
    arm = 'uniform_pruned_observability'
else:
    from uniform_models import model_for
    arm = 'uniform'
with AccessGuard(checkpoints=[m['checkpoint']], output=Path(sys.argv[1]).parent) as guard:
    assert sha(m['checkpoint']) == m['checkpoint_sha256']
    ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
    assert ck['epoch'] == m['best_epoch'] and ck['config'] == m['config']
    assert ck['protocol_sha256'] == sha(Path(m['run']) / 'protocol.json')
    assert not ck['volumetric_gt_used']
    model = model_for(m['config'], arm, 'cpu').eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == m['parameters']
    assert model.fusion.__class__.__name__ == 'UniformFusion'
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not guard.denied
import json
print(json.dumps(dict(key=m['key'], checkpoint_sha256=sha(m['checkpoint']), epoch=ck['epoch'],
    parameters=m['parameters'], seed=m['seed'], factory=f'scripts/{branch}/' +
    ('pruned_models.py' if branch == 'pruning_v1' else 'uniform_models.py') + ':model_for',
    arm=arm, strict_load=True, fusion='UniformFusion', eval=True, gradients=False,
    forward_calls=0, audit=guard.summary())))
