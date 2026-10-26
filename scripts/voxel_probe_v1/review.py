"""Independent NumPy K96 recomputation from all saved initial/final fields."""
from probe_context import *
sys.path.insert(0, str(PROJECT / 'scripts/unet_baseline_v1'))
from numpy_projector import project as numpy_project


def discrepancy(predicted, target, mask):
    p = predicted[mask] / 32284.894580078173
    y = target[mask].astype(np.float64) / 32284.894580078173
    residual = np.abs(p-y)
    return float(np.mean(np.where(residual < .01, .5*residual**2/.01, residual-.005)) +
                 .1*residual.sum()/max(np.abs(y).sum(), 1e-6))


def main():
    r = verify()
    summaries = []
    total_fields = 0
    max_projection_relative = 0.
    for index in r['indices']:
        for arm in r['arms']:
            result = read(ROOT / f'case{index}_{arm}/result.json')
            assert result['actual_updates'] == r['updates_per_run']
            final_review = None
            for metric in [result['metrics'][0], result['metrics'][-1]]:
                entry = metric['arrays']
                assert sha(entry['path']) == entry['sha256']
                independent = {}
                values = {}
                with np.load(entry['path'], allow_pickle=False) as arrays:
                    for label in ['A','B']:
                        volume = arrays[label+'_volume'].astype(np.float64)
                        assert np.isfinite(volume).all() and volume.min() >= 0 and volume.max() <= 1
                        stat = next(v for v in metric['fields'] if v['branch'] == label)
                        np.testing.assert_allclose([volume.mean(),volume.std()], [stat['mean'],stat['std']],rtol=1e-12,atol=1e-15)
                        total_fields += 1
                        for target in ['A','B','H','E']:
                            mask = arrays[target+'_mask'].astype(bool)
                            prediction = numpy_project(volume,
                                arrays[target+'_input_station_origin_m'], arrays[target+'_input_ray_direction'],
                                arrays[target+'_input_t_near'], arrays[target+'_input_path_length_m'], mask)
                            reference = arrays[target+'_input_projection_raw']
                            stored = arrays[label+'_'+target+'_projection'].astype(np.float64)
                            error = float(np.abs(prediction[mask]-stored[mask]).sum()/max(np.abs(reference[mask]).sum(),1e-6))
                            max_projection_relative = max(max_projection_relative,error)
                            assert error < 2e-5,(index,arm,metric['step'],label,target,error)
                            independent[label,target] = float(np.abs(prediction[mask]-reference[mask]).sum()/max(np.abs(reference[mask]).sum(),1e-6))
                            values[label,target] = discrepancy(prediction, reference, mask)
                    groups = dict(observed=[('A','A'),('B','B')],cross=[('A','B'),('B','A')],heldout=[('A','H'),('B','H')])
                    scores = {k: float(np.mean([independent[pair] for pair in pairs])) for k,pairs in groups.items()}
                    score_E = .5*(independent['A','E']+independent['B','E'])
                    for key,value in scores.items():
                        np.testing.assert_allclose(value, metric['measurement'][key],rtol=2e-5,atol=2e-6)
                    np.testing.assert_allclose(score_E,metric['extra_E_H'],rtol=2e-5,atol=2e-6)
                    a,b = arrays['A_volume'].astype(np.float64),arrays['B_volume'].astype(np.float64)
                    consistency = float(np.abs(a-b).mean()/(.5*(np.abs(a).mean()+np.abs(b).mean())+1e-6))
                    loss = .5*(values['A','H']+values['B','H']) + .25*(values['A','B']+values['B','A']) + .05*(values['A','A']+values['B','B']) + .05*consistency
                    np.testing.assert_allclose(loss,metric['loss'],rtol=2e-5,atol=2e-7)
                    final_review = dict(index=index,arm=arm,step=metric['step'],independent_loss=loss,
                        independent_measurement=scores, independent_E=score_E,
                        nontrivial_measurement_fit=all(v<=.9 for v in scores.values()) and loss<result['metrics'][0]['loss'],
                        independent_E_below099=score_E<=.99)
                print('Independent fields',index,arm,metric['step'],scores,'E',score_E,flush=True)
            summaries.append(final_review)
    write(ART / 'independent_review.json',dict(passed=True,full_fields=total_fields,physical_integrals=total_fields*4,
          max_projection_error_relative_to_target=max_projection_relative, final_results=summaries,
          implementation='independent NumPy FP64 trilinear K96; all 64-cubed fields; original physics and mask; original objective recomputed independently',
          new_optimizer_updates=0))


if __name__ == '__main__':
    main()
