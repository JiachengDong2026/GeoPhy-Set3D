"""Independent NumPy forward reconstruction and raw-record statistical audit."""
from runtime import *
from scipy.special import expit, rel_entr, softmax as scipy_softmax
import math
from measurements import distances


def check(actual, expected, name, maxima, atol=1e-5, rtol=1e-4):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    error = float(np.max(np.abs(actual - expected), initial=0))
    maxima[name] = max(maxima.get(name, 0), error)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (name, error)


def rebuild(z, weights, group):
    def affine(x, key):
        return x @ weights[key + '.weight'].T + weights[key + '.bias']
    def norm(x, key):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * weights[key + '.weight'] + weights[key + '.bias']
    mask = z['mask']
    pair = mask[:, None, :, None] & mask[:, None, None, :]
    tokens = np.where(mask[..., None], affine(z['token_input'], 'token_projection'), 0)
    qkv = affine(norm(tokens, 'norm1'), 'qkv')
    q, k, value = qkv.reshape(len(mask), mask.shape[1], 3, 4, 8).transpose(2, 0, 3, 1, 4)
    content = (q @ k.swapaxes(-1, -2)) / math.sqrt(8)
    obs = z['observability']
    obs = np.where(mask.any(1)[:, None], obs, 0)
    inp = np.concatenate((z['relations'], np.broadcast_to(obs[:, None, None], (*z['relations'].shape[:-1], 5))), -1)
    cols = [0, 1, 2] if group == 'A1_P2' else CHANNELS.get(group, [])
    inp[..., cols] = 0
    hidden = affine(inp, 'pairwise_bias.0')
    bias = affine(hidden * expit(hidden), 'pairwise_bias.2').transpose(0, 3, 1, 2)
    bias = np.where(pair, bias, 0)
    logits = np.where(mask[:, None, None, :], content + bias, -np.inf)
    logits = np.where(mask.any(1)[:, None, None, None], logits, 0)
    attention = np.where(pair, scipy_softmax(logits, axis=-1), 0)
    value = np.where(mask[:, None, :, None], value, 0)
    attended = (attention @ value).transpose(0, 2, 1, 3).reshape(*tokens.shape)
    tokens = np.where(mask[..., None], tokens + affine(attended, 'attention_output'), 0)
    hidden = affine(norm(tokens, 'norm2'), 'ffn.0')
    context = np.where(mask[..., None], tokens + affine(hidden * expit(hidden), 'ffn.2'), 0)
    return dict(qkv=qkv, content=content, bias=bias, attention=attention, context=context)


def output_independent(first, second):
    a, b = first.astype(np.float64).ravel(), second.astype(np.float64).ravel()
    diff = b - a
    pa, pb = np.array(np.unravel_index(np.argmax(a), first.shape)), np.array(np.unravel_index(np.argmax(b), second.shape))
    return dict(mean_abs=np.linalg.norm(diff, ord=1) / len(diff), max_abs=np.linalg.norm(diff, ord=np.inf),
                relative_l1=np.linalg.norm(diff, ord=1) / max(np.linalg.norm(a, ord=1), EPS),
                relative_l2=np.linalg.norm(diff) / max(np.linalg.norm(a), EPS),
                peak_amplitude_delta=b.max() - a.max(), peak_distance_mm=np.sqrt(np.sum((pb - pa) ** 2)) * (600 / 63),
                changed_voxel_fraction=np.count_nonzero(np.abs(diff) > CHANGE) / len(diff))


def run():
    protocol = validate()
    assert verified(BASE / 'engineering_checks.json')['passed']
    maxima = {}
    records, independent_rows = [], []
    prediction_count = probes_count = 0
    with Guard(projections=False) as guard:
        with np.load(BASE / 'interaction_parameter_evidence.npz', allow_pickle=False) as w:
            weights = {key: w[key].astype(np.float64) for key in w.files}
        for plan in verified(BASE / 'plans.json'):
            folder = BASE / 'units' / f'p{plan["plan_id"]:03d}'
            base_record = verified(folder / 'A0.json')
            with np.load(base_record['prediction'], allow_pickle=False) as a:
                baseline_outputs = {key: a[key].copy() for key in ('C0', 'C1')}
            with np.load(base_record['probe_path'], allow_pickle=False) as a:
                base_probes = {key: a[key].copy() for key in a.files}
            for group in GROUPS:
                r = verified(folder / f'{group}.json')
                verify_hashes(r['files'])
                with np.load(r['probe_path'], allow_pickle=False) as a:
                    probes = {key: a[key].astype(np.float64) if a[key].dtype.kind == 'f' else a[key].copy() for key in a.files}
                reconstructed = rebuild(probes, weights, group)
                for key, value in reconstructed.items():
                    check(probes[key], value, key, maxima)
                if group in ('obs_token', 'obs_both'):
                    obs = np.zeros_like(probes['observability'])
                else:
                    obs = np.where(probes['mask'].any(1)[:, None], probes['observability'], 0)
                emb = obs @ weights['observability_embedding.0.weight'].T + weights['observability_embedding.0.bias']
                emb *= expit(emb)
                check(probes['token_input'][..., -8:], np.broadcast_to(emb[:, None], probes['token_input'][..., -8:].shape), 'obs_embedding', maxima)
                check(probes['token_input'][..., :48], base_probes['token_input'][..., :48], 'feature_geometry_tokens', maxima, atol=0, rtol=0)
                assert np.array_equal(probes['mask'], base_probes['mask'])
                if group not in ('obs_token', 'obs_both'):
                    assert r['qkv_sha256'] == base_record['qkv_sha256']
                if group.startswith('A3_'):
                    rel = probes['relations']
                    assert np.array_equal(rel, rel.swapaxes(1, 2))
                    assert np.array_equal(np.diagonal(rel, axis1=1, axis2=2), np.diagonal(base_probes['relations'], axis1=1, axis2=2))
                    assert r['intervention']['valid_pairs_receiving_invalid_source'] == 0
                # Independent SciPy JS and entropy on the retained valid queries.
                a, b = base_probes['attention'].astype(np.float64), probes['attention']
                a /= np.maximum(a.sum(-1, keepdims=True), 1e-300)
                b = b / np.maximum(b.sum(-1, keepdims=True), 1e-300)
                middle = (a + b) / 2
                js = (rel_entr(a, middle).sum(-1) + rel_entr(b, middle).sum(-1)) / 2
                direct_distances = distances(base_probes['attention'], probes['attention'])
                check(direct_distances['js'], np.maximum(js, 0), 'JS_scipy', maxima, atol=1e-13, rtol=1e-8)
                valid = np.broadcast_to(probes['mask'][:, None], js.shape)
                assert np.isfinite(js[valid]).all() and np.min(js[valid], initial=0) >= -1e-14
                entropy_delta = -rel_entr(b, np.ones_like(b)).sum(-1) + rel_entr(a, np.ones_like(a)).sum(-1)
                assert np.isfinite(entropy_delta[valid]).all()
                check(direct_distances['entropy_delta'], entropy_delta, 'entropy_scipy', maxima, atol=1e-13, rtol=1e-8)
                with np.load(r['prediction'], allow_pickle=False) as out:
                    for stage in ('C0', 'C1'):
                        assert out[stage].shape == (64, 64, 64) and np.isfinite(out[stage]).all()
                        assert out[stage].min() >= 0 and out[stage].max() <= 1
                        recomputed = output_independent(baseline_outputs[stage], out[stage])
                        stored = next(x for x in r['output_rows'] if x['stage'] == stage)
                        for name, value in recomputed.items():
                            check(stored[name], value, 'output_' + name, maxima, atol=1e-11, rtol=1e-10)
                prediction_count += 1
                probes_count += len(probes['mask'])
                records.append(r)
            print(f'INDEPENDENT VERIFY {plan["plan_id"] + 1}/175', flush=True)
        # Independently regroup raw nested group records (no analysis helpers imported).
        by_key = {}
        for r in records:
            att = [x for x in r['attention_rows'] if x['valid_views'] == 'all' and x['query_scope'] == 'all']
            ctx = next(x for x in r['context_rows'] if x['valid_views'] == 'all')
            c1 = next(x for x in r['output_rows'] if x['stage'] == 'C1')
            group = 'A3_P1star_mean8' if r['group'].startswith('A3_') else r['group']
            row = dict(attention_js_mean=sum(x['js_mean'] for x in att) / 4,
                       context_relative_l1=ctx['relative_l1'], C1_relative_l1=c1['relative_l1'], C1_max_abs=c1['max_abs'])
            by_key.setdefault((r['case_id'], r['condition'], group), []).append(row)
        summary = verified(BASE / 'summary.json')['summaries']
        for stored in summary:
            metric = stored['metric']
            if metric not in ('attention_js_mean', 'context_relative_l1', 'C1_relative_l1', 'C1_affected_case_fraction'):
                continue
            field = 'C1_max_abs' if metric == 'C1_affected_case_fraction' else metric
            xs = []
            for case in verified(BASE / 'cases.json'):
                layout_values = []
                for condition in CONDITIONS:
                    if stored['condition'] != 'across5_within_case_mean' and condition != stored['condition']:
                        continue
                    subset = by_key[(case, condition, stored['group'])]
                    layout_values.append(sum(x[field] for x in subset) / len(subset))
                xs.append(sum(layout_values) / len(layout_values))
            if metric == 'C1_affected_case_fraction':
                check(stored['mean'], sum(x > CHANGE for x in xs) / 35, 'case_summary', maxima, atol=1e-13, rtol=0)
            else:
                for name, value in [('mean', np.mean(xs)), ('median', np.percentile(xs, 50)), ('p90', np.percentile(xs, 90)), ('maximum', max(xs))]:
                    check(stored[name], value, 'case_summary', maxima, atol=1e-13, rtol=0)
        # Verify analytic SiLU Jacobian by central differences, no autograd.
        w1, b1, w2 = [weights[k] for k in ('pairwise_bias.0.weight', 'pairwise_bias.0.bias', 'pairwise_bias.2.weight')]
        points = np.random.default_rng(SEED).uniform(0, 1, size=(32, 8))
        z = points @ w1.T + b1
        s = expit(z)
        exact = np.einsum('hj,nj,jc->nhc', w2, s * (1 + z * (1 - s)), w1)
        for col in range(8):
            delta = np.zeros_like(points); delta[:, col] = 1e-5
            lo, hi = (points - delta) @ w1.T + b1, (points + delta) @ w1.T + b1
            fd = ((hi * expit(hi) - lo * expit(lo)) @ w2.T) / 2e-5
            check(exact[..., col], fd, 'analytic_jacobian', maxima, atol=1e-8, rtol=1e-5)
        assert prediction_count == 3150 and not guard.denied
        path = BASE / 'independent_verification.json'
        if not path.exists():
            freeze(path, dict(at=now(), passed=True, method='independent NumPy float64 full OARI reconstruction on predetermined probes; SciPy softmax/JS; independent all-output norms and raw-record regrouping',
                prediction_records=prediction_count, retained_probe_voxel_records=probes_count, max_errors=maxima,
                scope_limit='OARI reconstruction only on retained probes; attention whole-volume normalization/softmax checked by worker',
                checkpoint_sha256=HIST.EXPECTED, no_training=True, no_backward=True, GT64_read=False, GT128_read=False,
                guard=guard.summary(), protocol_sha256=sha(BASE / 'protocol.json')))
    print('INDEPENDENT VERIFICATION PASSED', flush=True)


if __name__ == '__main__':
    run()
