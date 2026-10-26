"""Read-only train measurement decomposition and saved-field gradient probes."""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
import numpy as np
import torch
from geophy_set3d.training.unet_baseline_v1 import load_config
from geophy_set3d.training.selfsup import make_dataset, objective
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.data.measurement_only import dual_measurement_collate

ART = PROJECT / 'docs/reports/artifacts/shared_failure_v1'
ROOT = Path('/scratch/djc/GeoPhy-Set3D/shared_failure_v1')
OLD = Path('/scratch/djc/GeoPhy-Set3D/unet_baseline_v1')
DATA = Path('/data/djc/GeoPhy-Set3D/data/processed/geophy_projection_pool_v1')
SCALE = 32284.894580078173


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (ART / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + '\n')


def scalar_terms(c, y, length):
    a, b = 1 / 16.043, 1 / 28.965
    factor = 1e6 * length / SCALE
    chi = a * c / (b + (a - b) * c)
    prime = a * b / (b + (a - b) * c) ** 2
    prediction = factor * chi
    residual = prediction - y
    absolute = np.abs(residual)
    huber = np.where(absolute < .01, .5 * residual ** 2 / .01, absolute - .005)
    denominator = max(float(np.abs(y).sum()), 1e-6)
    smooth_gradient = np.clip(residual / .01, -1, 1) * factor * prime / len(y)
    relative_gradient = .1 * np.sign(residual) * factor * prime / denominator
    return dict(loss=float(huber.mean() + .1 * absolute.sum() / denominator),
                huber_loss=float(huber.mean()), relative_loss=float(.1 * absolute.sum() / denominator),
                smooth_gradient=smooth_gradient, relative_gradient=relative_gradient,
                prediction=prediction, denominator=denominator)


def quantiles(values):
    return np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()


def measurement_decomposition(ds, registration):
    rows = []
    fd_checks = []
    pooled_huber_derivative_sum = 0.
    pooled_count = 0
    bin_names = ['zero', '(0,1e-8]', '(1e-8,1e-6]', '(1e-6,1e-4]', '>1e-4']
    for index in registration['train_indices']:
        sample = ds[index]
        for branch in ['A', 'B', 'H']:
            batch = sample[branch]
            mask = batch['input_valid_ray'].numpy().astype(bool)
            y = batch['input_projection_raw'].numpy()[mask].astype(np.float64) / SCALE
            length = batch['input_path_length_m'].numpy()[mask].astype(np.float64)
            assert len(y) and np.isfinite(y).all() and (y >= 0).all()
            bins = [y == 0, (y > 0) & (y <= 1e-8), (y > 1e-8) & (y <= 1e-6),
                    (y > 1e-6) & (y <= 1e-4), y > 1e-4]
            assert np.all(np.sum(bins, axis=0) == 1)
            for c in registration['constant_concentrations']:
                terms = scalar_terms(c, y, length)
                groups = {}
                for name, selected in zip(bin_names, bins):
                    groups[name] = dict(count=int(selected.sum()), length_sum=float(length[selected].sum()),
                                        huber_gradient=float(terms['smooth_gradient'][selected].sum()),
                                        relative_gradient=float(terms['relative_gradient'][selected].sum()))
                smooth = float(terms['smooth_gradient'].sum())
                relative = float(terms['relative_gradient'].sum())
                rows.append(dict(index=index, case_id=batch['case_id'], branch=branch, c=c,
                                 rays=len(y), target_sum=float(y.sum()), **{k: terms[k] for k in
                                 ['loss', 'huber_loss', 'relative_loss', 'denominator']},
                                 huber_gradient=smooth, relative_gradient=relative,
                                 total_gradient=smooth + relative, groups=groups))
                if c == registration['constant_concentrations'][-1] and branch in ['A', 'B']:
                    pooled_huber_derivative_sum += smooth * len(y)
                    pooled_count += len(y)
                if index in [0, 1, 2, 3, 82, 165] and c in [1e-8, registration['constant_concentrations'][-1]]:
                    step = c * 1e-3
                    numerical = (scalar_terms(c + step, y, length)['loss'] -
                                 scalar_terms(c - step, y, length)['loss']) / (2 * step)
                    analytic = smooth + relative
                    # L1 kinks inside the interval may change a small fraction of rays.
                    assert np.isclose(numerical, analytic, rtol=.01, atol=.002), (index, branch, c, numerical, analytic)
                    fd_checks.append(dict(index=index, branch=branch, c=c, numerical=numerical,
                                          analytic=analytic))
        if index % 25 == 0:
            print('Measurement decomposition', index, flush=True)
    summaries = []
    # Equal A/B outputs: hold H once, and cross+observed = .3*(A+B).
    for c in registration['constant_concentrations']:
        selected = [row for row in rows if row['c'] == c]
        per_case = []
        for index in registration['train_indices']:
            pair = [row for row in selected if row['index'] == index]
            weighted = lambda key: sum((1 if row['branch'] == 'H' else .3) * row[key] for row in pair)
            groups = {name: {key: sum((1 if row['branch'] == 'H' else .3) * row['groups'][name][key]
                                     for row in pair) for key in ['huber_gradient', 'relative_gradient']}
                      for name in bin_names}
            per_case.append(dict(index=index, loss=weighted('loss'),
                                 gradient=weighted('total_gradient'), huber_gradient=weighted('huber_gradient'),
                                 relative_gradient=weighted('relative_gradient'), groups=groups))
        summaries.append(dict(c=c, original_objective_mean=float(np.mean([r['loss'] for r in per_case])),
                              joint_constant_gradient=float(np.mean([r['gradient'] for r in per_case])),
                              huber_gradient=float(np.mean([r['huber_gradient'] for r in per_case])),
                              relative_gradient=float(np.mean([r['relative_gradient'] for r in per_case])),
                              decreasing_output_is_descent_cases=sum(r['gradient'] > 0 for r in per_case),
                              gradient_quantiles=quantiles([r['gradient'] for r in per_case]),
                              groups={name: {key: float(np.mean([r['groups'][name][key] for r in per_case]))
                                             for key in ['huber_gradient', 'relative_gradient']} for name in bin_names},
                              per_case=per_case))
    unique = [r for r in rows if r['c'] == 0]
    counts = {name: sum(r['groups'][name]['count'] for r in unique) for name in bin_names}
    result = dict(method=registration['method'], constants=summaries,
                  measurement_bin_counts=counts, finite_difference_checks=fd_checks,
                  pooled_AB_equal_ray_huber_gradient_at_c0=pooled_huber_derivative_sum / pooled_count,
                  AB_rays=pooled_count, branch_target_sum_quantiles=quantiles([r['target_sum'] for r in unique]),
                  optimizer_updates=0, cases=166, branches_per_case=3)
    write('measurement_decomposition.json', result)
    with (ROOT / 'per_case_measurements.json').open('w') as stream:
        json.dump(rows, stream, allow_nan=False)
    return result


class SavedFields(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.a if self.calls == 1 else self.b


def field_probes(ds):
    rows = []
    for epoch in [0, 30]:
        for index in [0, 1, 2]:
            path = OLD / f'unet/pilot/diagnostic_epoch{epoch}/train_{index}.npz'
            before = sha(path)
            with np.load(path, allow_pickle=False) as arrays:
                a = torch.from_numpy(arrays['A_normal_volume'].copy())
                b = torch.from_numpy(arrays['B_normal_volume'].copy())
                zero_a = arrays['A_zero_volume'].astype(np.float64)
                zero_b = arrays['B_zero_volume'].astype(np.float64)
            model = SavedFields(a, b)
            batch = dual_measurement_collate([ds[index]])
            total, parts, volumes, _ = objective(model, batch)
            gradients = {}
            for name, loss in dict(total=total, **parts).items():
                gradients[name] = torch.autograd.grad(loss, (model.a, model.b), retain_graph=True)
            row = dict(epoch=epoch, index=index, array_sha256=before,
                       loss=float(total.detach()), parts={k: float(v.detach()) for k, v in parts.items()},
                       normal_zero_relative_MA=.5 * (float(np.abs(a.numpy()-zero_a).sum()/a.numpy().sum()) +
                                                     float(np.abs(b.numpy()-zero_b).sum()/b.numpy().sum())),
                       gradient={})
            weights = dict(hold=1., cross=.5, observed=.1, consistency=.05)
            vectors = {}
            for name, pair in gradients.items():
                vectors[name] = torch.cat([g.flatten().double() for g in pair])
                row['gradient'][name] = dict(norm=float(vectors[name].norm()),
                    joint_amplitude_derivative=sum(float((g.double()*v.detach().double()).sum()) for g, v in zip(pair, volumes)),
                    joint_logit_bias_derivative=sum(float((g.double()*v.detach().double()*(1-v.detach().double())).sum())
                                                    for g, v in zip(pair, volumes)),
                    positive_voxel_fraction=[float((g > 0).double().mean()) for g in pair],
                    negative_voxel_fraction=[float((g < 0).double().mean()) for g in pair])
            measurement = sum(weights[k] * vectors[k] for k in ['hold', 'cross', 'observed'])
            consistency = .05 * vectors['consistency']
            row['weighted_measurement_norm'] = float(measurement.norm())
            row['weighted_consistency_norm'] = float(consistency.norm())
            row['measurement_consistency_cosine'] = float(torch.dot(measurement, consistency) /
                                                         (measurement.norm()*consistency.norm()).clamp_min(1e-30))
            # Gradient direction is checked with the unchanged objective on frozen
            # fields scaled by fixed, symmetric factors, never optimized.
            numerical_losses = []
            with torch.no_grad():
                for scale in [.99, 1.01]:
                    value, _, _, _ = objective(SavedFields(a*scale, b*scale), batch)
                    numerical_losses.append(float(value))
            row['amplitude_finite_difference'] = (numerical_losses[1]-numerical_losses[0])/.02
            row['scaled_field_losses'] = dict(scale_099=numerical_losses[0], scale_101=numerical_losses[1])
            assert np.isclose(row['amplitude_finite_difference'], row['gradient']['total']['joint_amplitude_derivative'],
                              rtol=.03, atol=2e-5), row
            assert sha(path) == before
            rows.append(row)
            print('Saved full-field gradient', epoch, index, 'loss', row['loss'],
                  'dL/damplitude', row['gradient']['total']['joint_amplitude_derivative'], flush=True)
    write('saved_field_gradients.json', dict(rows=rows, optimizer_updates=0,
          caveat='Gradients are in saved-field space, not network parameter space; saved train0..2 epoch0 sampler only.'))
    return rows


def main():
    torch.set_num_threads(2)
    registration = json.loads((ART / 'registration.json').read_text())
    for name, digest in registration['frozen_sources'].items():
        assert sha(PROJECT / name) == digest, name
    for name, digest in registration['frozen_inputs'].items():
        assert sha(name) == digest, name
    guard = GTReadGuard(DATA, ROOT)
    with guard:
        ds = make_dataset(load_config())
        decomposition = measurement_decomposition(ds, registration)
        fields = field_probes(ds)
    for name, digest in registration['frozen_sources'].items():
        assert sha(PROJECT / name) == digest, name
    for name, digest in registration['frozen_inputs'].items():
        assert sha(name) == digest, name
    access = guard.summary()
    assert not any(access[k] for k in ['gt_open_attempts', 'forbidden_npy_attempts',
                                       'supervised_checkpoint_open_attempts'])
    write('execution.json', dict(access=access, optimizer_updates=0, network_forwards=0,
          checkpoint_loads=0, full_field_gradient_conditions=len(fields),
          input_files_reverified=len(registration['frozen_inputs']),
          frozen_sources_reverified=len(registration['frozen_sources']),
          raw_per_case_measurements=str(ROOT / 'per_case_measurements.json'),
          raw_sha256=sha(ROOT / 'per_case_measurements.json')))
    print('Read-only diagnosis complete', flush=True)


if __name__ == '__main__':
    main()
