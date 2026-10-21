"""One frozen GPU; original model forward for new outputs, OARI-only old replay."""
from runtime import *
from interventions import Capture, Intervention
from measurements import analyze, output_metrics
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.training.utils import move_batch
import argparse
import time


class PrefixCaptured(Exception):
    pass


def get_prefix(model, batch):
    saved = []
    def stop(module, args):
        saved.append(args)
        raise PrefixCaptured()
    hook = model.interaction.register_forward_pre_hook(stop)
    try:
        try:
            model(batch, return_stages=True)
        except PrefixCaptured:
            pass
        else:
            raise AssertionError('Prefix stop did not run')
    finally:
        hook.remove()
    assert len(saved) == 1
    return saved[0]


def read_fields(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ('C0', 'C1')}


def compare_history(fields, reference):
    comparisons = {}
    for stage in ('C0', 'C1'):
        a, b = fields[stage].astype(np.float64), reference[stage].astype(np.float64)
        maximum = float(np.abs(a - b).max())
        relative = float(np.abs(a - b).mean() / (np.abs(b).mean() + EPS))
        assert maximum <= 1e-5 and relative <= 1e-4, (stage, maximum, relative)
        comparisons[stage] = dict(max_abs=maximum, relative_mean_abs=relative)
    return comparisons


def fields_from(stages):
    result = {key: stages[key][0, 0].cpu().numpy().copy() for key in ('C0', 'C1')}
    assert all(a.shape == (64, 64, 64) and a.dtype == np.float32 and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in result.values())
    return result


def log_forward(plan, group, kind):
    append(BASE / 'forward_log.jsonl', dict(at=now(), plan_id=plan['plan_id'], group=group, kind=kind,
                                          checkpoint_sha256=HIST.EXPECTED, gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))


def process(model, plan, ds, references, engineering):
    target = BASE / 'units' / f'p{plan["plan_id"]:03d}'
    target.mkdir(parents=True, exist_ok=True)
    complete = target / 'complete.json'
    if complete.exists() and not complete.with_suffix('.json.sha256').exists():
        recovery = BASE / 'interrupted_artifacts' / f'{time.time_ns()}_p{plan["plan_id"]:03d}_complete'
        recovery.mkdir(parents=True)
        complete.rename(recovery / complete.name)
        append(BASE / 'recovery_log.jsonl', dict(at=now(), path=str(complete), preserved=str(recovery / complete.name)))
    if complete.exists():
        verify_hashes(verified(complete)['files'])
        return
    sample = ds[plan['case_index']]
    assert sample['case_id'] == plan['case_id']
    batch = move_batch(projection_pool_collate([sample]), 'cuda:0')
    batch = {key: value for key, value in batch.items() if not key.startswith('hold_')}
    assert batch['input_station_indices'][0].tolist() == plan['input_ids']
    ih = input_hash(batch)
    with np.load(plan['map_path'], allow_pickle=False) as z:
        maps, registered_mask, eligible = z['indices'], z['mask'], z['eligible']
    assert sha(plan['map_path']) == plan['map_sha256']
    baseline_fields = read_fields(references['A0']['path'])
    if engineering:
        plain = fields_from(model(batch, return_stages=True))
        log_forward(plan, 'A0', 'engineering_unhooked_repeat_full')
        compare_history(plain, baseline_fields)
        with Capture(model) as capture:
            baseline_stages = model(batch, return_stages=True)
        log_forward(plan, 'A0', 'engineering_capture_noop_full')
        compare_history(fields_from(baseline_stages), plain)
        baseline_check = compare_history(fields_from(baseline_stages), baseline_fields)
        original_args = capture.args
        del plain, baseline_stages
    else:
        original_args = get_prefix(model, batch)
        log_forward(plan, 'A0', 'prefix_only_stops_before_OARI')
        with Capture(model) as capture:
            model.interaction(*original_args)
        log_forward(plan, 'A0', 'OARI_only_missing_intermediate')
        baseline_check = None
    baseline = capture.cpu()
    original_identity, original_qkv = capture.identity, capture.qkv_hash
    assert np.array_equal(baseline['mask'], registered_mask), 'CPU registration vs real CUDA mask mismatch'
    baseline_capture = capture
    source_hash = HIST.state_hash(model)
    for group in GROUPS:
        out = target / f'{group}.json'
        if out.exists() and out.with_suffix('.json.sha256').exists():
            record = verified(out)
            assert record['input_sha256'] == ih
            verify_hashes(record['files'])
            continue
        # Preserve interrupted, uncommitted files before a deterministic replay.
        for orphan in (out, out.with_suffix('.json.tmp'), out.with_suffix('.json.sha256'),
                       target / f'{group}_prediction.npz', target / f'{group}_probes.npz'):
            if orphan.exists():
                recovery = BASE / 'interrupted_artifacts' / f'{time.time_ns()}_p{plan["plan_id"]:03d}_{group}'
                recovery.mkdir(parents=True)
                orphan.rename(recovery / orphan.name)
                append(BASE / 'recovery_log.jsonl', dict(at=now(), path=str(orphan), preserved=str(recovery / orphan.name)))
        indices = maps[int(group.split('_')[1])] if group.startswith('A3_') else None
        check = None
        if group == 'A0':
            current_capture, data, changed_fields = baseline_capture, baseline, baseline_fields
            audit = dict(valid_pair_count=int((registered_mask[:, :, None] & registered_mask[:, None, :]).sum()),
                         changed_valid_pairs=0, source_contamination_fraction=0., valid_pairs_receiving_invalid_source=0)
            check = baseline_check
            prediction = references[group]['path']
            prediction_hash = references[group]['sha256']
            reused = True
        else:
            with Intervention(model, group, indices, plan['legacy_permutation']) as intervention, Capture(model) as current_capture:
                if group in ('A1_P2', 'A2_P1') and not engineering:
                    model.interaction(*original_args)
                    stages = None
                    kind = 'OARI_only_missing_intermediate'
                else:
                    stages = model(batch, return_stages=True)
                    kind = 'original_full_model_return_stages'
            log_forward(plan, group, kind)
            data = current_capture.cpu()
            audit = dict(intervention.audit)
            audit['source_contaminated_query'] = intervention.source_contaminated_query.cpu().numpy()
            assert current_capture.identity == original_identity
            if group not in ('obs_token', 'obs_both'):
                assert current_capture.qkv_hash == original_qkv
            if group in ('obs_token',):
                assert np.array_equal(data['bias'], baseline['bias'])
            if group.startswith('A3_'):
                actual_changed = np.max(np.abs(data['relations'] - baseline['relations']), axis=(1, 2, 3)) > 1e-7
                assert np.array_equal(actual_changed, eligible)
                assert audit['valid_pairs_receiving_invalid_source'] == 0 and audit['changed_diagonal_count'] == 0
            if group in ('A1_P2', 'A2_P1'):
                prediction = references[group]['path']
                prediction_hash = references[group]['sha256']
                changed_fields = read_fields(prediction)
                if stages is not None:
                    check = compare_history(fields_from(stages), changed_fields)
                reused = True
            else:
                changed_fields = fields_from(stages)
                path = target / f'{group}_prediction.npz'
                assert not path.exists()
                np.savez_compressed(path, **changed_fields)
                prediction, prediction_hash, reused = str(path), sha(path), False
            del stages
        assert input_hash(batch) == ih
        assert all(not module.training for module in model.modules())
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        bs, attention, context, qc = analyze(plan, group, baseline, data, audit)
        outputs = [dict(plan_id=plan['plan_id'], case_id=plan['case_id'], condition=plan['condition'], group=group, stage=stage,
                        **output_metrics(baseline_fields[stage], changed_fields[stage])) for stage in ('C0', 'C1')]
        ids = current_capture.sample_ids
        probes = {name: data[name][ids] for name in ('mask', 'relations', 'content', 'context', 'attention', 'bias')}
        probes.update(voxel_indices=ids, token_input=current_capture.sample_token_input, qkv=current_capture.sample_qkv,
                      observability=current_capture.args[5][ids].cpu().numpy())
        probe_path = target / f'{group}_probes.npz'
        assert not probe_path.exists()
        np.savez_compressed(probe_path, **probes)
        files = {str(probe_path): sha(probe_path), prediction: prediction_hash}
        audit.pop('source_contaminated_query', None)
        freeze(out, dict(at=now(), plan_id=plan['plan_id'], case_id=plan['case_id'], condition=plan['condition'], group=group,
                         checkpoint_sha256=HIST.EXPECTED, input_sha256=ih, identity=original_identity, qkv_sha256=current_capture.qkv_hash,
                         model_state_sha256=source_hash, no_grad=True, eval=True, intervention=audit, numerical_qc=qc,
                         historical_comparison=check, prediction=prediction, prediction_sha256=prediction_hash, prediction_reused=reused,
                         probe_path=str(probe_path), files=files, bias_rows=bs, attention_rows=attention, context_rows=context, output_rows=outputs))
        if group != 'A0':
            del data, current_capture, changed_fields
    if engineering and plan['plan_id'] == 0:
        padded = {}
        for key, value in batch.items():
            padded[key] = torch.cat((value, torch.zeros_like(value[:, :1])), 1) if key.startswith('input_') or key == 'view_mask' else value
        padded['input_projection'][:, -1] = float('nan')
        extra = np.concatenate((maps[0], np.full((len(maps[0]), 1), plan['views'], dtype=np.uint8)), 1)
        with Intervention(model, 'A3_0', extra):
            fields = fields_from(model(padded, return_stages=True))
        log_forward(plan, 'A3_0', 'engineering_padding_full')
        compare_history(fields, read_fields(target / 'A3_0_prediction.npz'))
        f, g, vis, vm, rel, obs, *rest = original_args
        with Intervention(model, 'A3_0', maps[0]):
            ctx, att, bias = model.interaction(f, g, torch.zeros_like(vis), vm, torch.zeros_like(rel), torch.zeros_like(obs), *rest)
        log_forward(plan, 'A3_0', 'engineering_all_invalid_OARI_only')
        assert torch.count_nonzero(ctx) == torch.count_nonzero(att) == torch.count_nonzero(bias) == 0
    assert HIST.state_hash(model) == source_hash
    files = {str(target / f'{group}.json'): sha(target / f'{group}.json') for group in GROUPS}
    freeze(complete, dict(at=now(), completed=True, plan_id=plan['plan_id'], groups=len(GROUPS), files=files,
                          engineering=engineering, eval=True, gradients_absent=True, model_state_unchanged=True))
    print(f'INPUT {plan["plan_id"]} {plan["condition"]} case{plan["case_index"]}:18 groups complete', flush=True)


def run(limit=None):
    backend()
    p = validate()
    plans = verified(BASE / 'plans.json')
    layouts = verified(BASE / 'layouts.json')
    references = verified(BASE / 'reuse_manifest.json')
    order = p['engineering_plan_ids'] + [r['plan_id'] for r in plans if r['plan_id'] not in p['engineering_plan_ids']]
    if limit:
        order = order[:limit]
    with Guard() as guard, torch.inference_mode():
        model = model_load('cuda:0')
        initial = HIST.state_hash(model)
        for pos, pid in enumerate(order):
            plan = plans[pid]
            ds = dataset('test', layouts[plan['layout_index']])
            refs = {r['group']: r for r in references if r['plan_id'] == pid}
            if pos >= 2:
                assert verified(BASE / 'engineering_checks.json')['passed']
            process(model, plan, ds, refs, pid in p['engineering_plan_ids'])
            if pos == 1 and not (BASE / 'engineering_checks.json').exists():
                freeze(BASE / 'engineering_checks.json', dict(at=now(), passed=True, plan_ids=p['engineering_plan_ids'],
                       all18groups_each=True, baseline_unhooked_repeat_and_noop=True, historical_P1_P2_match=True,
                       CPU_CUDA_mask_equal=True, controlled_symmetry_diagonal_mask_preserved=True,
                       padding_invariance=True, all_invalid_safe=True, no_grad=True, eval=True))
            if pos >= 2:
                assert verified(BASE / 'engineering_checks.json')['passed']
            completed = list((BASE / 'units').glob('*/complete.json'))
            write(BASE / 'status.json', dict(at=now(), state='frozen_inference', completed_inputs=len(completed), total_inputs=175))
        assert HIST.state_hash(model) == initial and not guard.denied
    if len(list((BASE / 'units').glob('*/complete.json'))) == 175 and not (BASE / 'inference_complete.json').exists():
        freeze(BASE / 'inference_complete.json', dict(at=now(), completed=True, inputs=175, group_records=3150,
               expected_new_predictions=2625, reused_predictions=525, model_state_sha256=initial, audit=guard.summary(),
               forward_log_sha256=sha(BASE / 'forward_log.jsonl'), GT64_read=False, GT128_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        run(args.limit)
    except BaseException:
        import traceback
        freeze(BASE / f'failure_worker_{os.getpid()}.json', dict(at=now(), traceback=traceback.format_exc()))
        raise
