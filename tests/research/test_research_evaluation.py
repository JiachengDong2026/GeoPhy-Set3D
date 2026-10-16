import json
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from geophy_set3d.evaluation.research import EvaluationWithHoldout, frozen_record, paired_case_bootstrap, volume_metrics
from geophy_set3d.evaluation.view_selection import EvaluationViewSampler


def test_endpoint_integral_and_empty_segmentation():
    metrics = volume_metrics(np.ones((5,)*3)*.02, np.ones((5,)*3)*.01)
    assert metrics['reference_concentration_integral_m3'] == pytest.approx(.01*.6**3)
    assert metrics['relative_l1'] == pytest.approx(1.)
    assert metrics['nmse'] == pytest.approx(metrics['relative_l2']**2)
    assert metrics['centroid_error_mm'] == pytest.approx(0., abs=1e-12)
    empty = volume_metrics(np.zeros((5,)*3), np.zeros((5,)*3))
    assert empty['dice_0.001'] == 1.
    assert empty['peak_location_error_mm'] is None


def test_case_bootstrap_preserves_case_layout_clusters():
    a = [dict(case_id=c, geometry_id=str(g), relative_l1=.3)
         for c in ('a', 'b') for g in range(3)]
    b = [dict(r, relative_l1=.2) for r in a]
    result = paired_case_bootstrap(a, b, draws=100)
    assert result['cases'] == 2
    assert result['mean_difference'] == pytest.approx(.1)
    assert result['confidence_interval_95'] == pytest.approx([.1, .1])
    with pytest.raises(ValueError, match='same nonempty'):
        paired_case_bootstrap(a, b[:-1], draws=10)


def test_gate_checkpoint_cannot_unlock_gt_evaluation(tmp_path):
    (tmp_path/'frozen_checkpoint.json').write_text(json.dumps(dict(
        stage='gate', completed_epochs=300, volumetric_gt_used=False,
        selection_metric='mean held-out measurement relative L1')))
    with pytest.raises(ValueError, match='formal run'):
        frozen_record(tmp_path)


def test_evaluation_holdout_does_not_change_input_layout():
    base = EvaluationViewSampler(np.linspace(0, 180, 16), 4, 0, 'train')
    sampler = EvaluationWithHoldout(base)
    settings = dict(epoch=0, sample_index=0, case_id='plume')
    inputs, heldout = sampler.sample(16, **settings)
    assert np.array_equal(inputs, base.select('plume'))
    assert not set(inputs) & set(heldout)
    assert len(heldout) == 4
    assert np.array_equal(heldout, sampler.sample(16, **settings)[1])


def test_prediction_pass_never_opens_gt_and_checks_all_hashes_before_scoring(tmp_path, monkeypatch):
    from pool_fixture import make_pool
    from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
    script = Path(__file__).resolve().parents[2]/'scripts/evaluate_research_selfsup.py'
    spec = importlib.util.spec_from_file_location('evaluate_research_selfsup_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temporary, root, records, _ = make_pool()
    audit = dict(active=True, attempts=0)
    labels = {str((root/r['gt_volume_64_path']).resolve()) for r in records}
    labels.update(str(Path(r['source_target_path']).resolve()) for r in records)
    def deny_labels(event, args):
        if (audit['active'] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike))
                and str(Path(os.fsdecode(args[0])).resolve()) in labels):
            audit['attempts'] += 1
            raise AssertionError('Evaluation opened a label before prediction verification')
    sys.addaudithook(deny_labels)
    try:
        run = tmp_path/'run'
        (run/'checkpoints').mkdir(parents=True)
        checkpoint = run/'checkpoints/best.pt'
        config = dict(dataset=dict(root=str(root)), normalization=dict(method='linear_p99_5',
                      statistics_file='train_statistics.json'), training=dict(seed=20261011),
                      model=dict(variant='A2'))
        torch.save(dict(config=config, model={}), checkpoint)
        from geophy_set3d.data.build_utils import digest
        frozen = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint))
        monkeypatch.setattr(module, 'frozen_record', lambda _: frozen)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.operator = ExplicitRayOperator(samples_per_ray=2, projection_scale_ppm_m=1000.)
            def forward(self, batch, return_diagnostics=False):
                assert not any(k.startswith('hold_') for k in batch)
                return torch.full((1, 1, 64, 64, 64), .01)
        monkeypatch.setattr(module, 'build_set_unrolled_model', lambda _: Model())
        output = module.generate_predictions(run, 'cpu')
        assert audit['attempts'] == 0
        rows = [json.loads(line) for line in (output/'predictions_manifest.jsonl').read_text().splitlines()]
        assert len(rows) == 2
        assert all(not set(r['selected_station_ids']) & set(r['heldout_station_ids']) for r in rows)
        # Corrupt the LAST prediction: scoring must reject before even the first GT open.
        with (output/rows[-1]['prediction']).open('ab') as stream:
            stream.write(b'corrupted')
        with pytest.raises(ValueError, match='prediction changed'):
            module.score_saved_predictions(output)
        assert audit['attempts'] == 0
    finally:
        audit['active'] = False
        temporary.cleanup()
