"""Six small OARI examples and checks on the frozen trained interaction."""
import torch
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models.ray_relational_interaction import RayGeometryDescriptor, LocalObservabilityDescriptor
from geophy_set3d.training.utils import move_batch, permute_views, write_json
from geophy_set3d.training.losses import supervised_loss
from .protocols import evaluation_dataset


def branch_connectivity(model, device):
    """Hold tokens fixed; vary relation or diversity descriptors independently."""
    torch.manual_seed(81)
    f = torch.randn(1, 1, 32, 2, 2, 2, device=device).expand(-1, 4, -1, -1, -1, -1)
    g = torch.randn(1, 1, 4, 2, 2, 2, device=device).expand(-1, 4, -1, -1, -1, -1)
    visible = torch.ones(1, 4, 2, 2, 2, dtype=torch.bool, device=device)
    vm = torch.ones(1, 4, dtype=torch.bool, device=device)
    origins = torch.tensor([[[-6., 0., 0.], [0., -6., 1.], [6., 0., -1.], [1., 6., 2.]]], device=device)
    xyz = torch.rand(8, 3, device=device)*.6
    azi = torch.tensor([[0., 45., 90., 135.]], device=device)
    d, r = RayGeometryDescriptor()(origins, xyz, azi, visible)
    mask = visible.flatten(2).transpose(1, 2).reshape(-1, 4)
    count = torch.full((8,), 4, device=device)
    o = LocalObservabilityDescriptor()(d, mask, count)
    parallel = LocalObservabilityDescriptor()(d[:, :1].expand_as(d), mask, count)
    with torch.no_grad():
        c, a, bias = model.interaction(f, g, visible, vm, r, o, True)
        _, changed, _ = model.interaction(f, g, visible, vm, torch.zeros_like(r), o, True)
        other, _, _ = model.interaction(f, g, visible, vm, r, parallel, True)
        empty = torch.zeros_like(visible)
        context, weights, empty_bias = model.interaction(f, g, empty, vm, r, o, True)
        fused, pooling = model.fusion(context, g, empty, vm)
    attention_change = float((a-changed).abs().max())
    obs_change = float((c-other).abs().max())
    return dict(pairwise_bias_changes_attention=attention_change > 1e-8,
        pairwise_attention_max_change=attention_change,
        observability_changes_interaction=obs_change > 1e-8,
        observability_context_max_change=obs_change,
        all_invalid_safe=bool((context == 0).all() and (weights == 0).all() and
            (empty_bias == 0).all() and (fused == 0).all() and (pooling == 0).all()),
        synthetic_pairwise_bias_finite=bool(torch.isfinite(bias).all()))


def interaction_diagnostics(model, config, output):
    output.mkdir(parents=True, exist_ok=True)
    device = config['training']['device']
    examples = []; finite = True; mask_ok = True; bias_mask_ok = True; pooling_mask_ok = True
    permutation = 0.; padding = 0.; equivariance = 0.; row_error = 0.; pooling_error = 0.
    bias_max = 0.; bias_symmetry = 0.
    for selection in ('spread', 'clustered'):
        ds = evaluation_dataset(config, views=4, selection=selection, draw=0)
        for index in (0, 18, 35):
            sample = ds[index]
            batch = move_batch(projection_pool_collate([sample]), device)
            with torch.no_grad():
                p, diag = model(batch, True)
                a, m, w = diag['interaction_attention'], diag['visibility'], diag['attention_weights']
                bias = diag['pairwise_bias']; r = diag['pairwise_geometry']; o = diag['observability_descriptor']
                pair = m[:, None, :, None] & m[:, None, None, :]
                flat_mask = m.flatten(2).transpose(1, 2).reshape(-1, 4)
                flat_pair = flat_mask[:, :, None] & flat_mask[:, None, :]
                finite &= all(bool(torch.isfinite(t).all()) for t in (a, w, bias, r, o))
                mask_ok &= bool((a[~pair.expand_as(a)] == 0).all())
                bias_mask_ok &= bool((bias[~flat_pair[:, None].expand_as(bias)] == 0).all())
                pooling_mask_ok &= bool((w[~m] == 0).all())
                row_error = max(row_error, float((a.sum(3)-m[:, None].float()).abs().max()))
                pooling_error = max(pooling_error, float((w.sum(1)-m.any(1).float()).abs().max()))
                bias_max = max(bias_max, float(bias.abs().max()))
                bias_symmetry = max(bias_symmetry, float((bias-bias.transpose(-1, -2)).abs().max()))
                average = a.sum((4, 5, 6))/m.sum((2, 3, 4))[:, None, :, None].clamp_min(1)
                # Only summaries and one representative voxel, never full volumes.
                crossing = torch.acos(r[..., 0].clamp(0, 1))*180/torch.pi
                crossing_mean = (crossing*flat_pair).sum(0)/flat_pair.sum(0).clamp_min(1)
                center = 16*32*32+16*32+16
                active = flat_mask.any(1)
                example = dict(case_id=sample['case_id'], case_index=index, selection=selection,
                    station_ids=sample['input_station_indices'].tolist(),
                    station_azimuth_deg=sample['input_station_azimuth_deg'].tolist(),
                    pairwise_crossing_angle_mean_deg=crossing_mean.cpu().tolist(),
                    pairwise_crossing_angle_center_deg=crossing[center].cpu().tolist(),
                    center_xyz_m=model.lifting.voxel_xyz[center].cpu().tolist(),
                    center_valid_views=flat_mask[center].cpu().tolist(),
                    observability_mean=o[active].mean(0).cpu().tolist() if active.any() else [0.]*5,
                    observability_center=o[center].cpu().tolist(),
                    attention_matrix_mean_by_head=average[0].cpu().tolist(),
                    attention_matrix_mean=average.mean(1)[0].cpu().tolist(),
                    attention_matrix_center_by_head=a[0, :, :, :, 16, 16, 16].cpu().tolist(),
                    pairwise_bias_center_by_head=bias[center].cpu().tolist(),
                    pooling_mean_weights=w.mean((2, 3, 4))[0].cpu().tolist(),
                    pooling_center_weights=w[0, :, 16, 16, 16].cpu().tolist())
                examples.append(example)
                write_json(output/f'{selection}_val{index}.json', example)
                order = torch.tensor([3, 1, 0, 2], device=device)
                pp, dp = model(permute_views(batch, order), True)
                permutation = max(permutation, float((p-pp).abs().max()))
                equivariance = max(equivariance, float((diag['context_features'][:, order]-dp['context_features']).abs().max()))
                padded = dict(batch)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and (key.startswith('input_') or key == 'view_mask'):
                        padded[key] = torch.cat((value, torch.zeros_like(value[:, :1])), 1)
                padded['input_projection'][:, -1] = float('nan')
                pp, dp = model(padded, True)
                padding = max(padding, float((p-pp).abs().max()))
                mask_ok &= bool((dp['interaction_attention'][:, :, -1] == 0).all() and
                                (dp['interaction_attention'][:, :, :, -1] == 0).all())
                pooling_mask_ok &= bool((dp['attention_weights'][:, -1] == 0).all())
    model.zero_grad(set_to_none=True)
    pred = model(batch)
    supervised_loss(pred, batch['gt_volume_64'], **config['loss'])[0].backward()
    gradient_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    pairwise_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.interaction.pairwise_bias.parameters())
    observability_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.interaction.observability_embedding.parameters())
    model.zero_grad(set_to_none=True)
    result = dict(attention_finite=finite, interaction_mask_correct=mask_ok,
        pairwise_bias_mask_correct=bias_mask_ok, pooling_mask_correct=pooling_mask_ok,
        interaction_weight_sum_max_error=row_error, pooling_weight_sum_max_error=pooling_error,
        prediction_permutation_max_difference=permutation, padding_max_difference=padding,
        context_permutation_equivariance_max_difference=equivariance,
        pairwise_bias_abs_max=bias_max, pairwise_bias_symmetry_max_difference=bias_symmetry,
        all_gradients_finite=gradient_finite, pairwise_gradient_nonzero=pairwise_gradient,
        observability_gradient_nonzero=observability_gradient, examples=examples,
        **branch_connectivity(model, device))
    result['passed'] = all(result[k] for k in ('attention_finite', 'interaction_mask_correct',
        'pairwise_bias_mask_correct', 'pooling_mask_correct', 'all_gradients_finite',
        'pairwise_gradient_nonzero', 'observability_gradient_nonzero', 'all_invalid_safe',
        'pairwise_bias_changes_attention', 'observability_changes_interaction')) and \
        max(permutation, padding) <= 1e-5 and max(row_error, pooling_error) < 1e-6
    write_json(output/'summary.json', result)
    if not result['passed']:
        raise AssertionError('Frozen OARI interaction diagnostics failed')
    return result
