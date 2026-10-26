"""Read-only CPU parameter gradients on the U-Net's own saved checkpoints."""
from diagnose import *
from geophy_set3d.training.selfsup import measurement_discrepancy
from geophy_set3d.training.unet_baseline_v1 import build_model
from geophy_set3d.training.rvtc_matched import state_hash
from geophy_set3d.training.utils import seed_everything


def main():
    torch.set_num_threads(2)
    seed_everything(20261011)
    config = load_config()
    init = json.loads((PROJECT / 'docs/reports/artifacts/unet_baseline_v1/initialization.json').read_text())
    guard = GTReadGuard(DATA, OLD / 'unet')
    rows = []
    with guard:
        ds = make_dataset(config)
        batch = dual_measurement_collate([ds[0]])
        model = build_model(config, init['c0'], init['alpha'], device='cpu').eval()
        named = list(model.named_parameters())
        parameters = [p for _, p in named]
        for epoch in [0, 30]:
            ckpath = OLD / f'unet/pilot/epoch{epoch}.pt'
            checkpoint_hash = sha(ckpath)
            checkpoint = torch.load(ckpath, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            before = state_hash(model.state_dict())
            total, parts, volumes, projections = objective(model, batch)
            smooth = total * 0
            for label, key, weight in [('C_A', 'H', .5), ('C_B', 'H', .5),
                                       ('C_A', 'B', .25), ('C_B', 'A', .25),
                                       ('C_A', 'A', .05), ('C_B', 'B', .05)]:
                smooth = smooth + weight * measurement_discrepancy(projections[label][key], batch[key], relative_l1=0)
            consistency = .05 * parts['consistency']
            relative = parts['hold'] + .5*parts['cross'] + .1*parts['observed'] - smooth
            vectors = {}
            losses = dict(smooth=smooth, relative=relative, consistency=consistency, total=total)
            for name, loss in losses.items():
                pair = torch.autograd.grad(loss, parameters, retain_graph=True)
                vectors[name] = {group: torch.cat([g.detach().double().flatten() for (n, _), g in zip(named, pair)
                                                  if n.startswith(group)])
                                 for group in ['encoder', 'up', 'decoder', 'readout']}
                vectors[name]['all'] = torch.cat(list(vectors[name].values()))
                vectors[name]['readout_bias'] = torch.cat([g.detach().double().flatten() for (n, _), g in zip(named, pair)
                                                          if n == 'readout.bias'])
            row = dict(epoch=epoch, index=0, checkpoint_sha256=checkpoint_hash,
                       losses={k: float(v.detach()) for k, v in losses.items()},
                       gradient_norms={name: {g: float(v.norm()) for g, v in groups.items()}
                                       for name, groups in vectors.items()},
                       readout_bias_gradient={name: float(groups['readout_bias'].item()) for name, groups in vectors.items()},
                       cosines={})
            for x, y in [('smooth', 'relative'), ('smooth', 'consistency'), ('relative', 'consistency')]:
                row['cosines'][x+'__'+y] = {group: float(torch.dot(vectors[x][group], vectors[y][group]) /
                                                        (vectors[x][group].norm()*vectors[y][group].norm()).clamp_min(1e-30))
                                                       for group in ['all', 'encoder', 'readout']}
            assert all(p.grad is None for p in parameters)
            assert state_hash(model.state_dict()) == before
            assert sha(ckpath) == checkpoint_hash
            rows.append(row)
            print('CPU unchanged parameter probe', epoch, row['gradient_norms'], flush=True)
    write('parameter_probe.json', dict(rows=rows, optimizer_updates=0, parameter_hashes_unchanged=True,
          access=guard.summary(), caveat='Single train0 case, epoch0 A/B/H; parameter gradients at two checkpoints do not identify the full training trajectory.'))


if __name__ == '__main__':
    main()
