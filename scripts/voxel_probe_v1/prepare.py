"""Seal inputs and fixed fresh starts before any optimizer update."""
from probe_context import *


def main():
    registration = verify()
    setup()
    output = ROOT / 'prepared'
    output.mkdir(exist_ok=False)
    ledger = []
    guard = GTReadGuard(DATA, ROOT)
    with guard:
        ds = make_dataset(load_config())
        for index in registration['indices']:
            sample = ds[index]
            batch = dual_measurement_collate([sample])
            used = set()
            for b in sample.values():
                used.update(b['input_station_indices'].tolist())
            ids = sorted(set(range(ds.station_count)) - used)[:4]
            assert len(ids) == 4 and not used.intersection(ids)
            record, raw = ds._projections(index)
            extra = projection_pool_collate([ds._measurement_sample(record, raw, np.asarray(ids, dtype=np.int64), np.empty(0, dtype=np.int64))])
            model = fresh().eval()
            with torch.no_grad():
                a, b = model(batch['A']), model(batch['B'])
                initial = .5 * (a + b)
            init_hash = state_hash(model.state_dict())
            p = output / f'case{index}.pt'
            torch.save(dict(batch=batch, extra=extra, initial_shared=initial), p)
            arms = {}
            for arm in registration['arms']:
                arm_model = model if arm == 'unet' else DirectField(initial, registration['c0'], arm)
                directory = output / f'case{index}_{arm}'
                directory.mkdir()
                arms[arm] = evaluate(arm_model, batch, extra, directory, 0, False)
            # Zero field E baseline, using a detached exact-zero shared field.
            zero = DirectField(torch.zeros_like(initial), registration['c0'], 'voxel_scaled')
            zero_scores = evaluate(zero, batch, extra, output, 0, False)
            assert all(abs(v-1) < 1e-6 for v in zero_scores['measurement'].values())
            assert abs(zero_scores['extra_E_H']-1) < 1e-6
            ledger.append(dict(index=index, case_id=record['case_id'], path=str(p), sha256=sha(p),
                fresh_state_sha256=init_hash, input_identity=state_hash({k+'_'+n: v for k, fields in batch.items() for n, v in fields.items() if isinstance(v, torch.Tensor)}),
                fitted_stations={k: sample[k]['input_station_indices'].tolist() for k in ['A','B','H']},
                independent_E_stations=ids, initial_metrics=arms, zero=zero_scores))
            print('Prepared case', index, 'E', ids, 'initial H', arms['unet']['measurement']['heldout'], flush=True)
    write(ART / 'prepared.json', dict(cases=ledger, access=guard.summary(), optimizer_updates=0))


if __name__ == '__main__':
    main()
