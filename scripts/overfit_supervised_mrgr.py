#!/usr/bin/env python3
"""Single 4-case, frozen-OARI, 60-epoch functional MRGR gate (no tuning)."""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from torch.utils.data import Subset
from geophy_set3d.models import build_model
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.engine import make_dataset, make_loader
from geophy_set3d.training.mrgr import run_epoch
from geophy_set3d.training.utils import load_config, seed_everything, directories, write_json, move_batch, atomic_checkpoint


def main():
    config = load_config('configs/train_supervised_oari_mrgr_300.yaml')
    paths = directories(config, '_gate')
    if (paths['logs']/'gate_summary.json').exists():
        raise FileExistsError('Only one MRGR gate is authorized')
    seed_everything(config['training']['seed']); torch.set_num_threads(2)
    device = config['training']['device']; model = build_model(config).to(device)
    checkpoint = torch.load('/data/djc/GeoPhy-Set3D/outputs/checkpoints/supervised_oari_300/best.pt', map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint['model'], strict=False)
    assert not incompatible.unexpected_keys and all(k.startswith('residual_') for k in incompatible.missing_keys)
    shared_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('residual_')}
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('residual_'))
    dataset = make_dataset(config, fixed_views=4); dataset.set_epoch(0)
    subset = Subset(dataset, range(4)); loader = make_loader(subset, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    initial = run_epoch(model, loader, config); rows = []
    torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    for epoch in range(60):
        tr = run_epoch(model, loader, config, optimizer)
        row = dict(epoch=epoch, train=tr); rows.append(row)
        with (paths['logs']/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(f"gate {epoch+1}/60 loss={tr['loss']['total']:.7f} C0={tr['stages']['C0']['mean_per_case']['relative_l1']:.6f} C1={tr['metrics']['mean_per_case']['relative_l1']:.6f}", flush=True)
    final = run_epoch(model, loader, config)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in shared_before.items())
    # A frozen C0 has no autograd graph during the gate. Enable shared decoder for
    # this diagnostic only, without an optimizer step, to isolate feedback gradient.
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.eval(); batch = move_batch(projection_pool_collate([dataset[0]]), device)
    s = model(batch, return_stages=True)
    gy, = torch.autograd.grad((s['C1']-batch['gt_volume_64']).square().mean(), s['measurement_prediction_Y0'], retain_graph=True)
    gc, = torch.autograd.grad(s['measurement_prediction_Y0'], s['C0'], grad_outputs=gy)
    connectivity = dict(y0_gradient_finite=bool(torch.isfinite(gy).all()), y0_gradient_abs_sum=float(gy.abs().sum()),
                        feedback_c0_gradient_finite=bool(torch.isfinite(gc).all()), feedback_c0_gradient_abs_sum=float(gc.abs().sum()),
                        prediction_std=float(s['C1'].std()))
    passed = (final['loss']['total'] < initial['loss']['total']*.95 and
              final['metrics']['mean_per_case']['relative_l1'] < final['stages']['C0']['mean_per_case']['relative_l1'] and
              all(r['train']['residual_gradient_abs_sum_max'] > 0 for r in rows) and
              connectivity['y0_gradient_finite'] and connectivity['feedback_c0_gradient_finite'] and
              connectivity['feedback_c0_gradient_abs_sum'] > 0 and connectivity['prediction_std'] > 1e-6)
    result = dict(passed=passed, cases=[dataset[i]['case_id'] for i in range(4)],
                  station_ids=[dataset[i]['input_station_indices'].tolist() for i in range(4)], epochs=60,
                  initial=initial, final=final, connectivity=connectivity, shared_weights_unchanged=True,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device), seconds=time.perf_counter()-began,
                  formal_initialization='fresh random; never gate or OARI checkpoint')
    (paths['logs']/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_checkpoint(paths['checkpoints']/'gate.pt', dict(model=model.state_dict(), config=config, gate=result))
    write_json(paths['logs']/'gate_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
