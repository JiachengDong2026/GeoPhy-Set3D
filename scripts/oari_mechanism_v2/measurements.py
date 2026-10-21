"""Masked actual-tensor measurements; voxels and permutations are never samples."""
from runtime import *


def stats(x, prefix=''):
    a = np.asarray(x, dtype=np.float64).ravel()
    if not a.size:
        return {prefix + 'n': 0}
    assert np.isfinite(a).all()
    absolute = np.abs(a)
    return {prefix + 'n': int(a.size), prefix + 'mean': float(a.mean()),
            prefix + 'rms': float(np.sqrt(np.mean(a * a))), prefix + 'std': float(a.std()),
            prefix + 'mean_abs': float(absolute.mean()), prefix + 'median_abs': float(np.median(absolute)),
            prefix + 'p90_abs': float(np.quantile(absolute, .9)), prefix + 'max_abs': float(absolute.max())}


def quantiles(a, prefix):
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return {prefix + '_n': 0}
    assert np.isfinite(a).all()
    return {prefix + '_n': int(a.size), prefix + '_mean': float(a.mean()),
            prefix + '_median': float(np.median(a)), prefix + '_p90': float(np.quantile(a, .9)),
            prefix + '_max': float(a.max())}


def softmax(logits, mask):
    pair = mask[:, None, :, None] & mask[:, None, None, :]
    a = np.where(mask[:, None, None, :], logits.astype(np.float64), -np.inf)
    a = np.where(mask.any(1)[:, None, None, None], a, 0)
    a = np.exp(a - a.max(-1, keepdims=True))
    return np.where(pair, a / a.sum(-1, keepdims=True), 0)


def distances(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a = a / a.sum(-1, keepdims=True).clip(1e-300)
    b = b / b.sum(-1, keepdims=True).clip(1e-300)
    m = .5 * (a + b)
    def kl(x, y):
        return np.where(x > 0, x * np.log(x.clip(1e-300) / y.clip(1e-300)), 0).sum(-1)
    return dict(js=np.maximum(.5 * (kl(a, m) + kl(b, m)), 0),
                mean_abs=np.abs(b - a).mean(-1), max_abs=np.abs(b - a).max(-1),
                l1=np.abs(b - a).sum(-1), l2=np.sqrt(((b - a) ** 2).sum(-1)),
                entropy_delta=kl(a, np.ones_like(a)) - kl(b, np.ones_like(b)))


def strata(mask):
    count = mask.sum(1)
    return [('all', mask.any(1))] + [(str(k), count == k) for k in np.unique(count) if k > 0]


def analyze(plan, group, baseline, data, audit):
    mask = baseline['mask']
    assert np.array_equal(mask, data['mask'])
    pair = mask[:, :, None] & mask[:, None, :]
    valid_attention = np.broadcast_to(pair[:, None], data['attention'].shape)
    assert np.isfinite(data['attention']).all() and np.isfinite(data['context']).all()
    assert np.max(np.abs(data['attention'][~valid_attention]), initial=0) == 0
    norm_error = float(np.max(np.abs(data['attention'].sum(-1) - mask[:, None])))
    assert norm_error <= 1e-6
    reconstructed = softmax(data['content'] + data['bias'], mask)
    error = float(np.max(np.abs(reconstructed - data['attention'])))
    assert error <= 2e-6
    changes = distances(baseline['attention'], data['attention'])
    bias_rows, attention_rows, context_rows = [], [], []
    meta = dict(plan_id=plan['plan_id'], case_id=plan['case_id'], case_index=plan['case_index'],
                condition=plan['condition'], draw=0, group=group)
    number = mask.sum(1)
    changes['mean_abs'] = changes['l1'] / number[:, None, None].clip(1)
    centered = data['bias'].astype(np.float64) - np.where(pair[:, None], data['bias'], 0).sum(-1, keepdims=True) / number[:, None, None, None].clip(1)
    baseline_content = baseline['content'].astype(np.float64)
    delta_bias = data['bias'].astype(np.float64) - baseline['bias'].astype(np.float64)
    diagonal = np.eye(mask.shape[1], dtype=bool)[None]
    for label, voxels in strata(mask):
        pairs = pair & voxels[:, None, None]
        queries = mask & voxels[:, None]
        for head in range(4):
            bs = stats(data['bias'][:, head][pairs], 'bias_')
            cs = stats(data['content'][:, head][pairs], 'content_')
            row = dict(meta, head=head, valid_views=label, **bs, **cs,
                       centered_bias_rms=float(np.sqrt(np.mean(centered[:, head][pairs] ** 2))),
                       bias_content_rms_ratio=bs['bias_rms'] / max(cs['content_rms'], EPS),
                       content_delta_rms=float(np.sqrt(np.mean((data['content'][:, head][pairs] - baseline_content[:, head][pairs]) ** 2))),
                       **stats(delta_bias[:, head][pairs], 'bias_delta_'))
            for part, select in [('diagonal', pairs & diagonal), ('offdiagonal', pairs & ~diagonal)]:
                row.update(stats(data['bias'][:, head][select], part + '_'))
            bias_rows.append(row)
            change_row = dict(meta, head=head, valid_views=label, query_scope='all')
            for name, values in changes.items():
                change_row.update(quantiles(values[:, head][queries], name))
            # Each key's actual mean over valid query/key positions.
            change_row['mean_key_weight_delta'] = json.dumps([
                float((data['attention'][:, head, :, key] - baseline['attention'][:, head, :, key])[pairs[:, :, key]].mean())
                if pairs[:, :, key].any() else None for key in range(mask.shape[1])])
            attention_rows.append(change_row)
    if group == 'A2_P1':
        dirty = audit['source_contaminated_query']
        for name, selection in [('source_contaminated', dirty & mask), ('source_clean', ~dirty & mask)]:
            if selection.any():
                for head in range(4):
                    r = dict(meta, head=head, valid_views='all', query_scope=name)
                    for metric, values in changes.items():
                        r.update(quantiles(values[:, head][selection], metric))
                    attention_rows.append(r)
    difference = data['context'].astype(np.float64) - baseline['context'].astype(np.float64)
    base = baseline['context'].astype(np.float64)
    keep = mask[..., None]
    absolute = np.where(keep, np.abs(difference), 0).sum((1, 2))
    norm1 = np.where(keep, np.abs(base), 0).sum((1, 2))
    sqdiff = np.where(keep, difference ** 2, 0).sum((1, 2))
    sqbase = np.where(keep, base ** 2, 0).sum((1, 2))
    rel = absolute / np.maximum(norm1, EPS)
    for label, selection in strata(mask):
        n = int(selection.sum())
        context_rows.append(dict(meta, valid_views=label, voxels=n,
            relative_l1=float(absolute[selection].sum() / max(norm1[selection].sum(), EPS)),
            relative_l2=float(np.sqrt(sqdiff[selection].sum() / max(sqbase[selection].sum(), EPS ** 2))),
            mean_abs=float(absolute[selection].sum() / (int(mask[selection].sum()) * 32)),
            near_zero_baseline_fraction=float((norm1[selection] <= EPS).mean()),
            fraction_relative_above_threshold=float((rel[selection] > CHANGE).mean()),
            **quantiles(rel[selection], 'voxel_relative_l1')))
    head_centered = []
    for left in range(4):
        for right in range(left + 1, 4):
            head_centered.append(float(np.sqrt(np.mean((centered[:, left][pair] - centered[:, right][pair]) ** 2))))
    qc = dict(attention_normalization_max_error=norm_error, independent_float64_softmax_max_error=error,
              pairwise_centered_head_difference_rms=head_centered,
              all_invalid_context_zero=bool(np.all(data['context'][~mask] == 0)))
    assert qc['all_invalid_context_zero']
    return bias_rows, attention_rows, context_rows, qc


def output_metrics(before, after):
    a, b = before.astype(np.float64), after.astype(np.float64)
    d = b - a
    first, second = np.unravel_index(a.argmax(), a.shape), np.unravel_index(b.argmax(), b.shape)
    maximum = float(np.abs(d).max())
    return dict(mean_abs=float(np.abs(d).mean()), max_abs=maximum,
                relative_l1=float(np.abs(d).sum() / max(np.abs(a).sum(), EPS)),
                relative_l2=float(np.sqrt(np.sum(d * d) / max(np.sum(a * a), EPS ** 2))),
                peak_before=float(a.max()), peak_after=float(b.max()), peak_amplitude_delta=float(b.max() - a.max()),
                relative_peak_amplitude_delta=float((b.max() - a.max()) / max(a.max(), EPS)),
                peak_location_before=json.dumps(first, default=int), peak_location_after=json.dumps(second, default=int),
                peak_distance_mm=float(np.linalg.norm(np.subtract(second, first)) * 600 / 63),
                changed=maximum > CHANGE, changed_voxel_fraction=float((np.abs(d) > CHANGE).mean()))
