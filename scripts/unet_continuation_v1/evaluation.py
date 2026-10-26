"""Original diagnostic evaluator, with explicit CPU array serialization for CUDA."""
from continuation_context import *

@torch.no_grad()
def evaluate(model, batch, extra, directory, step, save_arrays):
    model.eval()
    total, parts, volumes, projections = objective(model, batch)
    scores = projection_scores(projections, batch)
    row = dict(step=step, loss=float(total), parts={k: float(v) for k, v in parts.items()},
               measurement={k: float(np.mean([r['relative_l1'] for r in values])) for k, values in scores.items()},
               fields=[], extra_E_H=[])
    arrays = {}
    for label, volume in zip(['A', 'B'], volumes):
        p = project_input(volume, extra)
        mask = extra['input_valid_ray'].bool() & extra['view_mask'][:, :, None, None, None]
        row['extra_E_H'].append(relative_projection_metrics(p, extra['input_projection_raw'], mask)['relative_l1'])
        v = volume.double()
        row['fields'].append(dict(branch=label, mean=float(v.mean()), std=float(v.std(correction=0)),
                                  min=float(v.min()), max=float(v.max()), nearzero=float((v < 1e-10).double().mean())))
        if save_arrays:
            arrays[label + '_volume'] = volume.detach().cpu().numpy()
            for target in ['A', 'B', 'H']:
                arrays[label + '_' + target + '_projection'] = projections['C_' + label][target].detach().cpu().numpy()
            arrays[label + '_E_projection'] = p.detach().cpu().numpy()
    row['extra_E_H'] = float(np.mean(row['extra_E_H']))
    if save_arrays:
        for target in ['A', 'B', 'H', 'E']:
            b = extra if target == 'E' else batch[target]
            for key in ['input_station_origin_m', 'input_ray_direction', 'input_t_near', 'input_path_length_m', 'input_projection_raw']:
                arrays[target + '_' + key] = b[key].detach().cpu().numpy()
            arrays[target + '_mask'] = (b['input_valid_ray'].bool() & b['view_mask'][:, :, None, None, None]).detach().cpu().numpy()
        path = directory / f'fields_step{step}.npz'
        np.savez_compressed(path, **arrays)
        row['arrays'] = dict(path=str(path), sha256=sha(path))
    return row
