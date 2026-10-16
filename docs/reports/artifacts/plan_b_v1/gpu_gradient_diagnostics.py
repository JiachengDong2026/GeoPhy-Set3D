import json
import runpy
from pathlib import Path
import torch

ns = runpy.run_path('/home/djc/projects/GeoPhy-Set3D/tests/research/test_plan_b.py')
torch.set_num_threads(2)
torch.cuda.set_device(0)
device = torch.device('cuda')
batch = {k: v.to(device) for k, v in ns['make_batch'](3).items()}

def run(model):
    model.zero_grad(set_to_none=True)
    ctx = model.encode(batch)
    p = model.integrate(ctx, batch)
    v = model.sample_grid(ctx, 4)
    loss = (p / 32284.894580078173).square().mean() + v.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    return {k: v.grad.detach().clone().cpu().double() for k, v in model.named_parameters()}, (p.detach().cpu(), v.detach().cpu())

def compare(a, b):
    rows = {}
    for key in a:
        da = a[key]
        db = b[key]
        delta = da - db
        rows[key] = dict(max_abs=float(delta.abs().max()),
                         relative_l2=float(delta.norm()/da.norm().clamp_min(1e-30)),
                         reference_l2=float(da.norm()), reference_max_abs=float(da.abs().max()),
                         failing_old_count=int((delta.abs() > 1e-6 + 1e-4 * db.abs()).sum()),
                         failing_strict_count=int((delta.abs() > 1e-7 + 1e-5 * db.abs()).sum()))
    return rows

result = {'physical_gpu_uuid': 'GPU-d3da1ea7-9229-8f8a-5054-854962d4b6b7',
          'torch_version': torch.__version__, 'runs': []}
for matmul_tf32, cudnn_tf32 in ((False, True), (True, False), (True, True), (False, False)):
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    plain = ns['small_model'](checkpoint_query=False).to(device)
    checked = ns['small_model'](checkpoint_query=True).to(device)
    checked.load_state_dict(plain.state_dict())
    reference, out = run(plain)
    for repeat in range(3):
        again, output = run(plain)
        chk, output_checked = run(checked)
        row = {'matmul_tf32': matmul_tf32, 'cudnn_tf32': cudnn_tf32, 'repeat': repeat,
               'plain_repeated': compare(reference, again),
               'checkpoint_vs_plain': compare(again, chk),
               'forward_max_abs': max(float((x-y).abs().max()) for x,y in zip(output, output_checked))}
        result['runs'].append(row)
        for label in ('plain_repeated', 'checkpoint_vs_plain'):
            values = row[label]
            worst_l2 = max(values.items(), key=lambda x:x[1]['relative_l2'])
            worst_abs = max(values.items(), key=lambda x:x[1]['max_abs'])
            fails = sum(v['failing_old_count'] for v in values.values())
            print(json.dumps(dict(matmul_tf32=matmul_tf32, cudnn_tf32=cudnn_tf32, repeat=repeat, comparison=label,
                                  worst_l2=worst_l2, worst_abs=worst_abs, failing_old_count=fails)), flush=True)
path = Path('/home/djc/projects/GeoPhy-Set3D/docs/reports/artifacts/plan_b_v1/gpu_gradient_diagnostics.json')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
try:
    run(plain)
except RuntimeError as exc:
    result['deterministic_algorithms_exception'] = str(exc)
    print(str(exc), flush=True)
finally:
    torch.use_deterministic_algorithms(False)
path.write_text(json.dumps(result, indent=2))
