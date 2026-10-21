"""Register -> engineering gate -> 525 frozen forwards -> seal -> GT64 analysis."""
from runtime import *
import argparse
import fcntl
import importlib.util
import platform
import time
import traceback


def register():
    backend()
    assert BASE.parent.is_dir() and Path(sys.executable).is_file()
    BASE.mkdir(exist_ok=False)
    manifest = verified(OUT / 'model_manifest.json')
    lock = verified(OUT / 'final_lock.json')
    m = next(m for m in lock['methods'] if m['key'] == 'MRGR')
    official = next(m for m in manifest['models'] if m['key'] == 'MRGR')
    for k in ('checkpoint', 'checkpoint_sha256', 'best_epoch', 'parameters', 'seed', 'run', 'config'):
        assert official[k] == m[k]
    assert m['checkpoint_sha256'] == EXPECTED and m['best_epoch'] == 278 and m['parameters'] == 344678
    assert m['arm'] == 'multiscale' and sha(m['checkpoint']) == EXPECTED
    assert Path(m['config']['dataset']['root']).resolve() == DATA.resolve() and DATA.is_dir()
    verify_hashes(lock['sources']); verify_hashes(lock['data_identity']); verify_hashes(lock['evidence'])
    training_protocol = Path(m['run']) / 'protocol.json'
    assert sha(training_protocol) == training_protocol.with_suffix('.sha256').read_text().strip()
    p = read(training_protocol)
    verify_hashes({str(PROJECT / path): digest for path, digest in p['sources'].items()})
    freeze(BASE / 'method.json', m)
    cases = verified(OUT / 'registration.json')['cases']['test']
    settings = verified(OUT / 'layouts.json')['test']
    assert len(cases) == len(set(cases)) == 35
    chosen = [next(s for s in settings if s['condition'] == c and s['draw'] == 0) for c in CONDITIONS]
    freeze(BASE / 'cases.json', cases); freeze(BASE / 'layouts.json', chosen)
    history = OUT / 'evaluation/test/MRGR'
    complete = read(history / 'complete.json')
    seal = verified(OUT / 'evaluation/test/all_predictions_sealed.json')
    assert complete['checkpoint_sha256'] == EXPECTED
    assert sha(history / 'manifest.jsonl') == complete['manifest_sha256'] == seal['manifests'][str(history / 'manifest.jsonl')]
    assert not complete['audit']['denied'] and not any(k.startswith('GT64') for k in complete['audit']['counts'])
    rr = rows(history / 'manifest.jsonl')
    import evaluation
    evaluation.validate_rows(cases, settings, rr)
    assert len(rr) == complete['reconstructions'] == 595
    for r in rr:
        assert sha(history / r['prediction']) == r['sha256']
    baseline = []
    for li, s in enumerate(chosen):
        for i, case in enumerate(cases):
            r = next(r for r in rr if (r['condition'], r['draw'], r['case_id']) == (s['condition'], 0, case))
            baseline.append(dict(r, path=str(history / r['prediction']), case_index=i, layout_index=li,
                                 checkpoint_sha256=EXPECTED, epoch=278, parameters=344678))
    freeze(BASE / 'baseline_manifest.json', baseline)
    data_files = {str(DATA / 'geometry/train_geometry.npz'): sha(DATA / 'geometry/train_geometry.npz'),
                  str(DATA / 'source_index.jsonl'): sha(DATA / 'source_index.jsonl'),
                  str(DATA / 'train_statistics.json'): sha(DATA / 'train_statistics.json')}
    source = {r['case_id']: r for r in rows(DATA / 'source_index.jsonl')}
    for c in cases:
        path = DATA / source[c]['train_projection_path']
        assert path.is_file(); data_files[str(path)] = sha(path)
    with Guard() as guard, torch.no_grad():
        for r in baseline:
            reference(r)
        model = load_model('cpu'); initial = frozen_state(model)
    deps = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                numpy=np.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                platform=platform.platform(), installed_packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    freeze(BASE / 'reuse_audit.json', dict(at=now(), passed=True, all_historical_files_hashed=595,
        reused_predictions=175, reused_fields=350, repeated_baseline_formal_forwards=0,
        historical_manifest_sha256=sha(history / 'manifest.jsonl'), checkpoint_sha256=EXPECTED,
        epoch=278, parameters=344678, state_sha256=initial, factory='module_models_v2.model_for(config, multiscale, device)',
        loading='official evaluation.predict_neural strict state_dict/config/epoch/protocol checks',
        historical_complete_sha256=sha(history / 'complete.json'),
        historical_seal_sha256=sha(OUT / 'evaluation/test/all_predictions_sealed.json'), audit=guard.summary()))
    freeze(BASE / 'data_identity.json', data_files); freeze(BASE / 'dependencies.json', deps)
    freeze(BASE / 'initial_resources.json', resources())
    freeze(BASE / 'protocol.json', dict(at=now(), experiment=BASE.name, checkpoint_sha256=EXPECTED,
        epoch=278, parameters=344678, state_sha256=initial,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        worktree=subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True), sources=identity(),
        methods=list(MODES), cases=35, layouts=5, formal_forwards=525, fields=1050,
        layout_draw_policy='Use registered draw0 for each of the five named conditions, fixed before current GT. No averaging over the other four historical draws; 35x5x3=525 exactly.',
        engineering_cases=[dict(layout_index=li, case_index=ci) for li, ci in ((0, 0), (2, 17), (4, 34))],
        engineering_forwards=12, engineering_modes=['identity', *MODES],
        identity_tolerances=dict(max_absolute=1e-5, relative_mean_absolute=1e-4),
        intervention='Clone only sampling grid in first GeometryLifting.forward; negate h/v/both. Second lifting uses original bound forward. angular_grid is never replaced.',
        sampling=dict(mode='bilinear', padding_mode='zeros', align_corners=True),
        unchanged=['input measurements and all metadata', 'visibility and view_mask', 'geometry_descriptor',
                   'ray relations and observability', 'path lengths', 'physical projector', 'decoder', 'weights', 'feedback spatial mapping'],
        primary=dict(stage='C1', metric='relative_l1', estimand='within-case mean of 3 paired intervention-minus-baseline differences, then within-case mean of 5 layouts, then mean of 35 cases',
                     unit='absolute ratio and percentage points', sample_size=35),
        statistics=dict(resamples=20000, seed=SEED, unit='case', interval='percentile 95% paired bootstrap, exploratory and unadjusted',
                        per_layout=True, per_intervention=True, summary=['mean', 'ci95', 'median', 'all case deltas'],
                        other_metrics_exploratory=True, interventions_not_independent=True),
        metrics=dict(relative_l1='sum(abs(pred-GT))/max(sum(abs(GT)),1e-6)',
                     relative_l2='sqrt(sum((pred-GT)^2)/max(sum(GT^2),1e-12))',
                     dice50='2*intersection/(pred_count+GT_count), both thresholded at 0.5*GT.max',
                     absolute_peak_ratio_error='abs(pred.max/GT.max-1)',
                     centroid_distance_mm='Euclidean distance between concentration-weighted XYZ centers on endpoint grid, mm',
                     prediction_mean_absolute_change='mean(abs(intervention-baseline))',
                     prediction_relative_change='mean(abs(intervention-baseline))/(mean(abs(baseline))+1e-6)'),
        output_range=[0, 1], no_training=True, no_backward=True, no_optimizer=True, no_model_selection=True,
        GT128_forbidden=True, sealed_before_GT64=True, previously_used_test=True, independent_confirmation=False,
        input_data_hashes_sha256=sha(BASE / 'data_identity.json'), dependencies_sha256=sha(BASE / 'dependencies.json')))
    (BASE / 'protocol.json').chmod(0o444)
    (BASE / 'protocol.json.sha256').chmod(0o444)
    write(BASE / 'status.json', dict(at=now(), state='registered'))
    print('REGISTERED paper MRGR epoch278; 175 baselines verified; 525 formal forwards; GT not opened', flush=True)


def assert_protocol():
    p = verified(BASE / 'protocol.json'); verify_hashes(p['sources'])
    assert sha(method()['checkpoint']) == EXPECTED
    return p


def get_batch(model, layout, i):
    ds = dataset('test', layout)
    assert [r['case_id'] for r in ds.records] == verified(BASE / 'cases.json')
    sample = ds[i]
    assert sample['input_station_indices'].tolist() == layout['input_ids'][sample['case_id']]
    batch = move_batch(projection_pool_collate([sample]), 'cuda:0')
    return {k: v for k, v in batch.items() if not k.startswith('hold_')}


def engineering():
    backend(); p = assert_protocol()
    target = BASE / 'engineering'; target.mkdir(exist_ok=False)
    records = []
    with Guard() as guard, torch.no_grad():
        model = load_model('cuda:0')
        assert frozen_state(model) == p['state_sha256']
        for selection in p['engineering_cases']:
            li, ci = selection['layout_index'], selection['case_index']
            setting = verified(BASE / 'layouts.json')[li]
            row = next(r for r in verified(BASE / 'baseline_manifest.json') if r['layout_index'] == li and r['case_index'] == ci)
            ref = reference(row); batch = get_batch(model, setting, ci); expected = prepare(model, batch)
            for mode in p['engineering_modes']:
                fields, audit = run_forward(model, batch, expected, mode, engineering=True)
                diff = {k: differences(fields[k], ref[k]) for k in fields}
                if mode == 'identity':
                    assert all(d['max_abs'] <= 1e-5 and d['relative_change'] <= 1e-4 for d in diff.values()), diff
                path = target / f'layout{li:02d}_case{ci:02d}_{mode}.npz'
                assert not path.exists(); np.savez_compressed(path, **fields)
                record = dict(selection, condition=setting['condition'], draw=0, case_id=row['case_id'], mode=mode,
                              prediction=str(path), sha256=sha(path), prediction_difference=diff, audit=audit)
                append(target / 'manifest.jsonl', record); records.append(record)
            del batch, expected
        assert not guard.denied
        assert frozen_state(model) == p['state_sha256']
    freeze(BASE / 'engineering_checks.json', dict(at=now(), passed=True, records=records,
        model_forwards=len(records), formal_forwards=0, GT64_read=False, GT128_read=False,
        audit=guard.summary(), manifest_sha256=sha(target / 'manifest.jsonl'),
        max_identity_absolute=max(d['max_abs'] for r in records if r['mode'] == 'identity' for d in r['prediction_difference'].values()),
        max_identity_relative=max(d['relative_change'] for r in records if r['mode'] == 'identity' for d in r['prediction_difference'].values())))
    print('ENGINEERING PASSED: identity, all3 flips, feedback mapping, descriptors, inputs and weights', flush=True)


def predict(li):
    backend(); p = assert_protocol(); assert verified(BASE / 'engineering_checks.json')['passed']
    setting = verified(BASE / 'layouts.json')[li]
    target = BASE / 'predictions' / f'layout{li:02d}'; target.mkdir(parents=True, exist_ok=False)
    old = [r for r in verified(BASE / 'baseline_manifest.json') if r['layout_index'] == li]
    started = time.monotonic(); total = 0
    with Guard() as guard, torch.no_grad():
        model = load_model('cuda:0'); assert frozen_state(model) == p['state_sha256']
        for row in old:
            ci = row['case_index']; batch = get_batch(model, setting, ci); expected = prepare(model, batch); ref = reference(row)
            for mode in MODES:
                t = time.monotonic(); fields, audit = run_forward(model, batch, expected, mode)
                path = target / f'case{ci:02d}_{mode}.npz'; assert not path.exists()
                np.savez_compressed(path, **fields)
                record = dict(case_id=row['case_id'], case_index=ci, condition=setting['condition'], draw=0,
                    layout_index=li, views=setting['views'], intervention=mode, input_ids=row['input_ids'], heldout_ids=row['heldout_ids'],
                    prediction=path.name, sha256=sha(path), baseline_path=row['path'], baseline_sha256=row['sha256'],
                    checkpoint_sha256=EXPECTED, epoch=278, parameters=344678, audit=audit,
                    prediction_difference={k: differences(fields[k], ref[k]) for k in fields},
                    gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), seconds=time.monotonic()-t)
                append(target / 'manifest.jsonl', record); total += 1
            if ci % 5 == 0 or ci == 34:
                print(f'{setting["condition"]} case {ci+1}/35, formal forwards {total}/105', flush=True)
            del batch, expected, ref
        assert frozen_state(model) == p['state_sha256'] and not guard.denied
    rr = rows(target / 'manifest.jsonl')
    assert len(rr) == total == 105
    assert len({(r['case_id'], r['intervention']) for r in rr}) == 105
    for r in rr:
        assert sha(target / r['prediction']) == r['sha256']
        with np.load(target / r['prediction'], allow_pickle=False) as z:
            valid_fields({k: z[k] for k in z.files})
    freeze(target / 'complete.json', dict(at=now(), completed=True, layout_index=li, condition=setting['condition'],
        draw=0, predictions=105, fields=210, model_forwards=105, checkpoint_sha256=EXPECTED,
        state_before=p['state_sha256'], state_after=p['state_sha256'], audit=guard.summary(),
        manifest_sha256=sha(target / 'manifest.jsonl'), seconds=time.monotonic()-started))


def seal():
    p = assert_protocol(); manifests = {}; total = 0
    with Guard() as guard:
        engineering_check = verified(BASE / 'engineering_checks.json')
        engineering_manifest = BASE / 'engineering/manifest.jsonl'
        assert sha(engineering_manifest) == engineering_check['manifest_sha256']
        for record in rows(engineering_manifest):
            assert sha(record['prediction']) == record['sha256']
            with np.load(record['prediction'], allow_pickle=False) as z:
                valid_fields({k: z[k] for k in z.files})
        manifests[str(engineering_manifest)] = sha(engineering_manifest)
        for li in range(5):
            target = BASE / 'predictions' / f'layout{li:02d}'
            complete = verified(target / 'complete.json')
            assert complete['state_before'] == complete['state_after'] == p['state_sha256']
            assert not complete['audit']['denied'] and not any(k.startswith('GT64') for k in complete['audit']['counts'])
            assert sha(target / 'manifest.jsonl') == complete['manifest_sha256']
            rr = rows(target / 'manifest.jsonl'); assert len(rr) == 105
            for r in rr:
                assert sha(target / r['prediction']) == r['sha256']
                with np.load(target / r['prediction'], allow_pickle=False) as z:
                    valid_fields({k: z[k] for k in z.files})
            total += len(rr); manifests[str(target / 'manifest.jsonl')] = sha(target / 'manifest.jsonl')
        assert total == 525 and not guard.denied
    freeze(BASE / 'all_predictions_sealed.json', dict(at=now(), formal_forwards=total, fields=2*total,
        baseline_reused=175, engineering_forwards=12, sealed_before_GT64=True, GT128_read=False,
        manifest_hashes=manifests, protocol_sha256=sha(BASE / 'protocol.json'), audit=guard.summary()))
    print('SEALED all525 formal forwards /1050 fields before GT64', flush=True)


def scheduler():
    folder = PROJECT / 'scripts/uniform_final_v1'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('lifting_uuid_scheduler', folder / 'controller.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BASE = BASE; module.PROJECT = PROJECT
    module.GPU_LOCKS = BASE.parent / 'gpu_locks'
    module.progress = lambda: None
    return module


def controller():
    (BASE / 'scheduling').mkdir(exist_ok=True)
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            s = scheduler()
            s.run_gpu_once('engineering', [sys.executable, str(HERE / 'diagnostic.py'), 'engineering'], BASE / 'engineering_checks.json')
            assert verified(BASE / 'engineering_checks.json')['passed']
            for li in range(5):
                s.run_gpu_once(f'predict_layout{li:02d}', [sys.executable, str(HERE / 'diagnostic.py'), 'predict', '--layout-index', str(li)],
                               BASE / 'predictions' / f'layout{li:02d}' / 'complete.json')
            if not (BASE / 'all_predictions_sealed.json').exists():
                seal()
            write(BASE / 'status.json', dict(at=now(), state='sealed_scoring'))
            with (BASE / 'scheduling/analysis.log').open('x') as log:
                subprocess.run([sys.executable, str(HERE / 'analysis.py')], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert verified(BASE / 'acceptance.json')['completed']
            write(BASE / 'status.json', dict(at=now(), state='complete', acceptance=str(BASE / 'acceptance.json')))
        except BaseException:
            write(BASE / 'status.json', dict(at=now(), state='failed', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['register', 'controller', 'engineering', 'predict', 'seal'])
    parser.add_argument('--layout-index', type=int); a = parser.parse_args()
    try:
        if a.action == 'predict':
            predict(a.layout_index)
        else:
            globals()[a.action]()
    except BaseException:
        if BASE.exists():
            append(BASE / 'errors.jsonl', dict(at=now(), action=a.action, traceback=traceback.format_exc()))
        raise
