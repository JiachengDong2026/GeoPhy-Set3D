"""Cross-check independent NumPy terms against unchanged Torch discrepancy."""
from diagnose import *
from geophy_set3d.training.selfsup import measurement_discrepancy


def main():
    torch.set_num_threads(2)
    registration = json.loads((ART / 'registration.json').read_text())
    rows = json.loads((ROOT / 'per_case_measurements.json').read_text())
    lookup = {(r['index'], r['branch'], r['c']): r for r in rows}
    maximum_loss_error = maximum_gradient_error = 0.
    count = 0
    guard = GTReadGuard(DATA, ROOT)
    with guard:
        ds = make_dataset(load_config())
        for index in range(166):
            sample = ds[index]
            for branch in ['A', 'B', 'H']:
                target = dual_measurement_collate([sample])[branch]
                target = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                          for k, v in target.items()}
                length = target['input_path_length_m']
                for value in registration['constant_concentrations']:
                    c = torch.tensor(value, dtype=torch.float64, requires_grad=True)
                    a, b = 1 / 16.043, 1 / 28.965
                    prediction = 1e6 * length * a * c / (b + (a - b) * c)
                    loss = measurement_discrepancy(prediction, target)
                    gradient, = torch.autograd.grad(loss, c)
                    row = lookup[index, branch, value]
                    assert np.isclose(float(loss.detach()), row['loss'], rtol=1e-11, atol=1e-12)
                    assert np.isclose(float(gradient), row['total_gradient'], rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(sum(g['huber_gradient'] for g in row['groups'].values()),
                                               row['huber_gradient'], rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(sum(g['relative_gradient'] for g in row['groups'].values()),
                                               row['relative_gradient'], rtol=1e-12, atol=1e-10)
                    maximum_loss_error = max(maximum_loss_error, abs(float(loss.detach())-row['loss']))
                    maximum_gradient_error = max(maximum_gradient_error, abs(float(gradient)-row['total_gradient']))
                    count += 1
    write('independent_review.json', dict(passed=True, comparisons=count,
          method='Original unchanged Torch measurement_discrepancy FP64/AD versus NumPy closed-form ray contributions; constant projection uses analytic physics, not a new full K96 recheck',
          max_loss_absolute_error=maximum_loss_error, max_gradient_absolute_error=maximum_gradient_error,
          access=guard.summary(), optimizer_updates=0))
    print('Original discrepancy cross-check passed', count, 'conditions', flush=True)


if __name__ == '__main__':
    main()
