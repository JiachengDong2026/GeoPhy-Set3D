"""Pad variable input/held-out views while keeping view and ray masks distinct."""
import torch


def projection_pool_collate(samples):
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError('Batch samples must have the same fields (including optional GT128)')
    result = {}
    for prefix, mask_name in (('input', 'view_mask'), ('hold', 'hold_view_mask')):
        counts = [len(sample[f'{prefix}_station_indices']) for sample in samples]
        maximum = max(counts)
        result[mask_name] = torch.arange(maximum)[None] < torch.tensor(counts)[:, None]
        for key in sorted(k for k in keys if k.startswith(prefix + '_')):
            values = [sample[key] for sample in samples]
            if any(not isinstance(v, torch.Tensor) or v.shape[0] != n
                   or v.shape[1:] != values[0].shape[1:] for v, n in zip(values, counts)):
                raise ValueError(f'Inconsistent view field {key}')
            # -1 prevents station padding from accidentally indexing station zero.
            padding = -1 if key.endswith('_station_indices') else 0
            padded = values[0].new_full((len(samples), maximum, *values[0].shape[1:]), padding)
            for i, value in enumerate(values):
                padded[i, :len(value)] = value
            result[key] = padded
    for key in sorted(k for k in keys if not k.startswith(('input_', 'hold_'))):
        values = [sample[key] for sample in samples]
        result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return result
