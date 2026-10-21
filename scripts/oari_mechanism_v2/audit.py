"""CPU-only identity, geometry algebra, mask audits and immutable registration."""
from runtime import *
from interventions import controlled_maps
from measurements import stats


def register():
    backend()
    BASE.mkdir(exist_ok=False)
    old_protocol = verified(OLD / 'protocol.json')
    m = old_protocol['model']
    final = verified(HIST.PRIOR / 'model_identity.json')
    assert final['main'] == m
    assert (m['seed'], m['best_epoch'], m['parameters']) == (20261011, 285, 342725)
    assert sha(m['checkpoint']) == HIST.EXPECTED == m['checkpoint_sha256']
    cases = verified(HIST.BASE / 'cases.json')
    settings = verified(HIST.BASE / 'layouts.json')['test']
    layouts = [next(dict(s, historical_layout_index=i) for i, s in enumerate(settings)
                    if s['condition'] == condition and s['draw'] == 0) for condition in CONDITIONS]
    assert len(cases) == 35
    reference = rows(Path(m['prediction_root']) / 'manifest.jsonl')
    reuse, plans, permutations, channels, legacy_rows, sensitivity = [], [], [], [], [], []
    data_identity = verified(HIST.BASE / 'experiment_registry.json')
    prior_acceptance = verified(OLD / 'acceptance.json')
    assert prior_acceptance['completed'] and verified(OLD / 'independent_verification.json')['passed']
    assert verified(HIST.BASE / 'acceptance.json')['completed']
    with AccessGuard(checkpoints=[m['checkpoint']], output=BASE) as guard, torch.inference_mode():
        ck = torch.load(m['checkpoint'], map_location='cpu', weights_only=False)
        assert ck['epoch'] == 285 and ck['config'] == m['config'] and not ck['volumetric_gt_used']
        model = HIST.model_for(m['config'], 'uniform', 'cpu').eval().requires_grad_(False)
        model.load_state_dict(ck['model'], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 342725
        state = HIST.state_hash(model)
        assert state == data_identity['main_state_sha256']
        weight_arrays = {key: value.cpu().numpy() for key, value in model.interaction.state_dict().items()}
        np.savez_compressed(BASE / 'interaction_parameter_evidence.npz', **weight_arrays)
        for li, layout in enumerate(layouts):
            ds = dataset('test', layout)  # constructor reads metadata/geometry, never projections/GT
            assert [r['case_id'] for r in ds.records] == cases
            oldroot = HIST.BASE / 'predictions' / f'layout{layout["historical_layout_index"]:02d}'
            oldrows = rows(oldroot / 'manifest.jsonl')
            sealed = verified(HIST.BASE / 'all_predictions_sealed.json')
            assert sha(oldroot / 'manifest.jsonl') == sealed['manifest_hashes'][str(oldroot / 'manifest.jsonl')]
            for ci, case in enumerate(cases):
                pid = li * 35 + ci
                originals = [r for r in reference if (r['condition'], r['draw'], r['case_id']) == (layout['condition'], 0, case)]
                assert len(originals) == 1
                selected = [('A0', originals[0], Path(m['prediction_root']) / originals[0]['prediction'])]
                for group, oldgroup in [('A1_P2', 'P2'), ('A2_P1', 'P1')]:
                    r = next(r for r in oldrows if r['case_id'] == case and r['experiment'] == oldgroup and r['permutation_index'] == 0)
                    assert r['checkpoint_sha256'] == HIST.EXPECTED
                    selected.append((group, r, oldroot / r['prediction']))
                for group, r, path in selected:
                    guard.allowed[str(path.resolve())] = 'sealed_historical_prediction'
                    assert sha(path) == r['sha256']
                    reuse.append(dict(plan_id=pid, case_id=case, condition=layout['condition'], draw=0, group=group,
                                      path=str(path.resolve()), sha256=r['sha256'], checkpoint_sha256=HIST.EXPECTED))
                batch = geometry_batch(ds, layout['input_ids'][case])
                visible = model.lifting.angular_grid(batch)[1].reshape(1, layout['views'], 32, 32, 32)
                mask = visible.permute(0, 2, 3, 4, 1).reshape(-1, layout['views'])
                directions, rel = model.ray_geometry(batch['input_station_origin_m'], model.lifting.voxel_xyz,
                                                   batch['input_station_azimuth_deg'], visible)
                obs = model.observability(directions, mask, torch.full((32 ** 3,), layout['views']))
                relation, valid = rel.numpy(), mask.numpy()
                pair = valid[:, :, None] & valid[:, None, :]
                identity_error = float(np.abs(relation[..., 0][pair] ** 2 + relation[..., 1][pair] ** 2 - 1).max())
                assert identity_error < 3e-7
                assert np.isfinite(relation).all() and np.all(relation[~pair] == 0)
                assert relation[..., 2][pair].min() >= 0 and relation[..., 2][pair].max() <= 1
                seed = SEED + pid * 1009
                maps, eligible = controlled_maps(relation, valid, seed)
                path = BASE / 'permutation_maps' / f'p{pid:03d}.npz'
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, indices=maps, mask=valid, eligible=eligible)
                legacy = selected[-1][1]['permutation']
                plan = dict(plan_id=pid, layout_index=li, historical_layout_index=layout['historical_layout_index'], case_index=ci,
                            case_id=case, condition=layout['condition'], draw=0, views=layout['views'], input_ids=layout['input_ids'][case],
                            legacy_permutation=legacy, map_path=str(path), map_sha256=sha(path), permutation_seed=seed)
                plans.append(plan)
                per_rep = []
                for rep, index in enumerate(maps):
                    changed = relation[np.arange(len(valid))[:, None, None], index[:, :, None], index[:, None, :]]
                    changed_pair = (np.max(np.abs(changed - relation), axis=-1) > 1e-7) & pair
                    changed_voxel = changed_pair.any((1, 2))
                    assert np.array_equal(changed_voxel, eligible)
                    assert np.array_equal(valid[np.arange(len(valid))[:, None], index], valid)
                    assert np.array_equal(changed, changed.transpose(0, 2, 1, 3))
                    assert np.array_equal(changed[:, np.arange(layout['views']), np.arange(layout['views'])], relation[:, np.arange(layout['views']), np.arange(layout['views'])])
                    per_rep.append(dict(rep=rep, changed_valid_pairs=int(changed_pair.sum()), changed_voxels=int(changed_voxel.sum()),
                                        eligible_mapping_sha256=hashlib.sha256(index[eligible].tobytes()).hexdigest(),
                                        changed_relation_sha256=hashlib.sha256(changed.tobytes()).hexdigest()))
                assert len({r['changed_relation_sha256'] for r in per_rep}) == 8
                permutations.append(dict(plan, valid_view_count_histogram=np.bincount(valid.sum(1), minlength=layout['views'] + 1).tolist(),
                    eligible_voxels=int(eligible.sum()), active_voxels=int(valid.any(1).sum()), ineligible_one_or_two=int(((valid.sum(1) > 0) & (valid.sum(1) < 3)).sum()),
                    ineligible_constant_relations=int(((valid.sum(1) >= 3) & ~eligible).sum()), checks=per_rep))
                base = dict(plan_id=pid, case_id=case, condition=layout['condition'], identity_max_error=identity_error)
                diagonal = np.eye(layout['views'], dtype=bool)[None]
                for scope, select in [('all', pair), ('diagonal', pair & diagonal), ('offdiagonal', pair & ~diagonal)]:
                    for channel in range(3):
                        x = relation[..., channel][select]
                        if not len(x):
                            continue
                        cos = relation[..., 0][select]
                        channels.append(dict(base, channel=channel, scope=scope, **stats(x), minimum=float(x.min()), maximum=float(x.max()),
                                             zero_fraction=float((x == 0).mean()), missing_fraction=float((~np.isfinite(x)).mean()), coverage_fraction=1.,
                                             correlation_with_cos=float(np.corrcoef(x, cos)[0, 1]) if x.std() > 1e-12 and cos.std() > 1e-12 else None))
                # Analytic local Jacobian only: no autograd/backward. Geometry-only,
                # evenly spaced valid pairs, fixed before any output is evaluated.
                ii = np.argwhere(pair)
                ii = ii[np.linspace(0, len(ii) - 1, min(512, len(ii)), dtype=int)]
                x = np.concatenate((relation[ii[:, 0], ii[:, 1], ii[:, 2]], obs.numpy()[ii[:, 0]]), 1).astype(np.float64)
                w1, b1, w2 = [weight_arrays[k].astype(np.float64) for k in ('pairwise_bias.0.weight', 'pairwise_bias.0.bias', 'pairwise_bias.2.weight')]
                z = x @ w1.T + b1
                sigmoid = 1 / (1 + np.exp(-z))
                derivative = sigmoid * (1 + z * (1 - sigmoid))
                jacobian = np.einsum('hj,nj,jc->nhc', w2, derivative, w1)
                for head in range(4):
                    for channel in range(8):
                        sensitivity.append(dict(base, head=head, channel=channel, **stats(jacobian[:, head, channel]),
                                                input_std=float(x[:, channel].std()), sample_pairs=len(x)))
                for r in oldrows:
                    if r['case_id'] != case or r['experiment'] != 'P1':
                        continue
                    p = np.array(r['permutation'])
                    dirty = pair & ~(valid[:, p, None] & valid[:, None, p])
                    moved = relation[:, p][:, :, p]
                    legacy_rows.append(dict(base, permutation_index=r['permutation_index'], permutation=json.dumps(p.tolist()),
                        valid_pairs=int(pair.sum()), contaminated_valid_pairs=int(dirty.sum()), contaminated_fraction=float(dirty.sum() / pair.sum()),
                        changed_valid_pairs=int(((np.max(np.abs(moved - relation), -1) > 1e-7) & pair).sum()),
                        diagonal_changed=int((np.max(np.abs(moved - relation), -1)[:, np.arange(layout['views']), np.arange(layout['views'])] > 1e-7).sum()),
                        old_C0_relative_change=r['prediction_difference']['C0']['relative_prediction_difference'],
                        old_C1_relative_change=r['prediction_difference']['C1']['relative_prediction_difference']))
                print(f'CPU geometry/channel/permutation registration {pid + 1}/175', flush=True)
        assert not guard.denied
    csv_write(BASE / 'relation_channel_distributions.csv', channels)
    csv_write(BASE / 'historical_P1_mask_audit.csv', legacy_rows)
    csv_write(BASE / 'analytic_bias_jacobian.csv', sensitivity)
    freeze(BASE / 'reuse_manifest.json', reuse)
    freeze(BASE / 'cases.json', cases)
    freeze(BASE / 'layouts.json', layouts)
    freeze(BASE / 'permutation_manifest.json', permutations)
    freeze(BASE / 'plans.json', plans)
    freeze(BASE / 'checkpoint_audit.json', dict(at=now(), passed=True, model=m, state_sha256=state,
        config_sha256=hashlib.sha256(json.dumps(m['config'], sort_keys=True).encode()).hexdigest(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        historical_acceptances={str(OLD / 'acceptance.json'): sha(OLD / 'acceptance.json'), str(HIST.BASE / 'acceptance.json'): sha(HIST.BASE / 'acceptance.json')},
        reused_predictions=525, full_attention_context_cache_available=False, geometry_only_audit=guard.summary(),
        original_diagnostics_not_executed='evaluation/oari_diagnostics.py includes supervised backward and GT paths; inspected only', resources=resources()))
    freeze(BASE / 'relation_channel_audit.json', dict(at=now(), passed=True, first_two_deterministically_redundant=True,
        max_cos_squared_plus_sin_squared_error=max(r['identity_max_error'] for r in channels),
        azimuth_range=[min(r['minimum'] for r in channels if r['channel'] == 2), max(r['maximum'] for r in channels if r['channel'] == 2)],
        distribution_csv_sha256=sha(BASE / 'relation_channel_distributions.csv'),
        weights={key: dict(shape=list(a.shape), l2=float(np.linalg.norm(a))) for key, a in weight_arrays.items() if key.startswith(('pairwise_bias', 'observability_embedding'))},
        input_scale_weight_norm_not_functional_contribution=True, GT64_read=False, GT128_read=False))
    freeze(BASE / 'protocol.json', dict(at=now(), id=BASE.name, model=m, state_sha256=state, sources=sources(),
        instruction_sha256=sha('/home/djc/.codex/attachments/e487121f-df64-4942-b452-105b1cf1caa8/已粘贴的文本.txt'),
        n_cases=35, layouts=CONDITIONS, draw=0, groups=GROUPS, planned_inputs=175, formal_group_records=3150,
        scope='single actual OARI before C0; no second OARI exists; feedback and all other geometry paths unchanged',
        baseline_reuse='A0/P1/P2 C0/C1 all525 predictions reused. Only missing OARI intermediate tensors recomputed. New full forwards for8 controlled fields+7 channel interventions.',
        engineering='first ordinary/clustered case0, all18groups; unhooked vs capture baseline repeats and history; P1/P2 vshistory; valid outputs cached as formal results, no duplicate inference',
        engineering_plan_ids=[0, 70], ordinary_per_group_n=35, permutation_n_not_independent=True,
        controlled='per voxel jointly permute valid view indices; identity for <=2views or invariant graph; reject automorphisms;8 distinct global fields; seed20261024+1009*plan_id',
        local_V3_limit='at most5 nonidentity local maps; eight spatial fields may reuse local choices across fields, never claimed8 independent per-voxel permutations',
        channel_semantics='remove selected columns additive contribution at first affine, retain learned bias; token obs removes five input contributions before its SiLU, retains learned bias',
        channel_columns=CHANNELS, A1='exact historical P2 first affine Wobs*obs+b; not final bias zero/logits subtraction/physical neutral',
        interventions_OOD='single-channel or joint angular masking can leave deterministic cos/sin manifold; sensitivity only; no retraining contribution claim',
        epsilon=EPS, output_changed_max_abs_threshold=CHANGE, context_relative_threshold=CHANGE,
        thresholds='fixed numerical sensitivity markers, not thresholds for OARI usefulness; calibrated identity tolerance maxabs1e-5 and relative1e-4',
        summary='case medians/P90/max and affected fraction; eightA3 fields averaged within case before across35cases; no voxel or permutation pseudo-replication; no GT ranking',
        probes='three fixed XYZ-flat indices0/16383/32767 plus up to8 geometry-only evenly spaced active voxels; no full attention stored',
        independent='NumPy MLP/LayerNorm/QKV/softmax/context reconstruction on retained probes; all output changes separately recomputed; raw-table regrouping independent',
        GPU='one exclusively locked UUID, existing double snapshot15s;2CPUthreads; resumable atomic input units',
        verification_tolerances=dict(atol=1e-5, rtol=1e-4, independent_output_atol=1e-11),
        training_allowed=False, backward_allowed=False, optimizer_allowed=False, GT64_read=False, GT128_read=False,
        test_previously_used=True, independent_confirmation=False, seed=SEED))
    (BASE / 'protocol.json').chmod(0o444)
    freeze(BASE / 'dependencies.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda))
    write(BASE / 'status.json', dict(at=now(), state='registered', inputs=175))


if __name__ == '__main__':
    register()
