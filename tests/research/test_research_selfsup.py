"""Numerical training-boundary, accumulation, protocol and file-access checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
import yaml

from geophy_set3d.data.measurement_only import DualMeasurementDataset, dual_measurement_collate
from geophy_set3d.physics.explicit_ray_operator import ExplicitRayOperator
from geophy_set3d.training.gt_read_guard import GTReadGuard
from geophy_set3d.training import research_selfsup as research

ROOT = Path(__file__).resolve().parents[2]


def config():
    return research.load_research_config(ROOT / 'configs/research/scheme_a_synthetic_smoke.yaml')


class TinyModel(torch.nn.Module):
    """Cheap stochastic differentiable reconstruction for trainer replay tests."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()) * .01)

    def forward(self, batch, return_diagnostics=False):
        signal = batch['input_projection'].flatten(1).mean(1)
        noise = torch.rand_like(signal) * .01 if self.training else 0
        c = torch.sigmoid(-4 + self.weight + signal + noise)[:, None, None, None, None].expand(-1, 1, 4, 4, 4)
        return dict(volume=c, C0=c, steps=[]) if return_diagnostics else c


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_design_and_protocol_changes_rejected(self):
        original = config()
        changes = [('status', 'design_template_not_executable'), ('protocol', 'other')]
        for key, value in changes:
            candidate = copy.deepcopy(original)
            candidate[key] = value
            with self.assertRaises(ValueError):
                research.validate_research_config(candidate)
        for section, key, value in [('training', 'precision', 'fp16'), ('loss', 'hold', 2.),
            ('validation', 'split', 'test'), ('model', 'steps', 0), ('dataset', 'kind', 'projection_pool')]:
            candidate = copy.deepcopy(original)
            candidate[section][key] = value
            with self.assertRaises(ValueError):
                research.validate_research_config(candidate)
        candidate = copy.deepcopy(original)
        candidate['model'].update(variant='A0', steps=0)
        research.validate_research_config(candidate)
        candidate['model']['steps'] = 2
        with self.assertRaises(ValueError):
            research.validate_research_config(candidate)

    def test_synthetic_sampling_manifest_replays_and_is_disjoint(self):
        dataset = research.SyntheticMeasurementDataset(config())
        first = list(research._sampling_rows(dataset, 0, 'train'))
        self.assertEqual(first, list(research._sampling_rows(dataset, 0, 'train')))
        self.assertNotEqual(first, list(research._sampling_rows(dataset, 1, 'train')))
        for row in first:
            sets = [set(row[key]) for key in ('A', 'B', 'H')]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_heldout_never_enters_network(self):
        settings = config()
        batch = dual_measurement_collate([research.SyntheticMeasurementDataset(settings)[0]])
        seen = []
        class Observer(TinyModel):
            def forward(self, branch, **kwargs):
                seen.append(branch['input_station_indices'].clone())
                return super().forward(branch, **kwargs)
        model = Observer()
        loss, _, _, _, _ = research.research_objective(model, batch, ExplicitRayOperator(samples_per_ray=8))
        loss.backward()
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], batch['A']['input_station_indices']))
        self.assertTrue(torch.equal(seen[1], batch['B']['input_station_indices']))
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_accumulation_weights_actual_tail_case_count(self):
        samples = [dict(x=torch.tensor(float(i+1)), y=torch.tensor(.1*(i+1)), case_id=f'c{i}') for i in range(3)]
        def collate(rows):
            return dict(A=dict(case_id=[r['case_id'] for r in rows]), x=torch.stack([r['x'] for r in rows]), y=torch.stack([r['y'] for r in rows]))
        def objective(model, batch, operator, diagnostics=False):
            prediction = torch.sigmoid(model(batch['x'][:, None])).reshape(-1, 1, 1, 1, 1).expand(-1, 1, 2, 2, 2)
            loss = (prediction[:, 0, 0, 0, 0] - batch['y']).square().mean()
            parts = {k: loss * 0 for k in ('hold', 'cross', 'observed', 'consistency')}
            return loss, parts, (prediction, prediction), {}, []
        fake_scores = {key: [dict(relative_l1=.1, relative_l2=.2)] for key in ('heldout', 'cross', 'observed')}
        models = []
        for batch_size, accumulation in ((2, 1), (1, 2)):
            torch.manual_seed(1)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=.05)
            settings = config()
            settings['training'].update(batch_size=batch_size, gradient_accumulation=accumulation)
            loader = DataLoader(samples, batch_size=batch_size, collate_fn=collate)
            with patch.object(research, 'research_objective', side_effect=objective), patch.object(research, 'move_dual_batch', side_effect=lambda batch, _: batch), patch.object(research, 'projection_scores', return_value=fake_scores):
                result = research.run_research_epoch(model, loader, settings, None, optimizer)
            self.assertEqual(result['optimizer_updates'], 2)
            self.assertEqual(result['sample_count'], 3)
            models.append(model)
        for first, second in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(first, second, rtol=1e-7, atol=1e-8)

    def test_epoch_boundary_resume_matches_uninterrupted_rng_optimizer_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, split = config(), config()
            full['experiment'].update(output_root=temporary, name='full')
            split['experiment'].update(output_root=temporary, name='split')
            builder = 'geophy_set3d.models.set_unrolled3d.build_set_unrolled_model'
            with patch(builder, side_effect=lambda _: TinyModel()):
                uninterrupted = research.train_research(full, stop_after_epochs=3)
                interrupted = research.train_research(split, stop_after_epochs=1)
                resumed = research.train_research(split, resume=True, stop_after_epochs=3)
            a = torch.load(uninterrupted / 'checkpoints/last.pt', weights_only=False)
            b = torch.load(resumed / 'checkpoints/last.pt', weights_only=False)
            self.assertEqual(a['epoch'], b['epoch'])
            self.assertEqual(a['global_step'], b['global_step'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            for key in ('torch', 'loader'):
                self.assertTrue(torch.equal(a['rng'][key], b['rng'][key]))
            for key in a['optimizer']['state'][0]:
                torch.testing.assert_close(a['optimizer']['state'][0][key], b['optimizer']['state'][0][key], rtol=0, atol=0)
            candidate = copy.deepcopy(b)
            candidate['sampling_epoch_next'] += 1
            with self.assertRaises(ValueError):
                research.validate_resume(candidate, split, b['provenance'])
            changed = copy.deepcopy(b['provenance'])
            changed['data_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                research.validate_resume(b, split, changed)
            self.assertFalse((interrupted / 'frozen_checkpoint.json').exists())

    def test_real_measurement_epoch_and_validation_never_open_volumetric_files(self):
        import sys
        sys.path.insert(0, str(ROOT / 'tests'))
        from pool_fixture import make_pool
        temporary, root, records, _ = make_pool()
        try:
            settings = config()
            model = TinyModel()
            with tempfile.TemporaryDirectory() as own:
                guard = GTReadGuard(root, own)
                with guard:
                    dataset = DualMeasurementDataset(root, split='train', a_range=(3, 5), b_range=(3, 5), hold_views=2)
                    loader = DataLoader(dataset, batch_size=1, collate_fn=dual_measurement_collate)
                    operator = ExplicitRayOperator(samples_per_ray=4, ray_chunk=512)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
                    research.run_research_epoch(model, loader, settings, operator, optimizer)
                    research.run_research_epoch(model, loader, settings, operator)
                summary = guard.summary()
                self.assertFalse(summary['gt_file_opened'])
                self.assertEqual(summary['forbidden_npy_attempts'], 0)
                self.assertEqual(summary['supervised_checkpoint_open_attempts'], 0)
                self.assertGreater(summary['projection_open_count'], 0)
        finally:
            temporary.cleanup()

    def test_formal_training_requires_real_gate_evidence(self):
        settings = research.load_research_config(ROOT / 'configs/research/scheme_a_a2_formal.yaml')
        with self.assertRaises(ValueError):
            research._validate_acceptance(settings, {})

    def test_formal_rejects_same_root_changed_data_content(self):
        settings = research.load_research_config(ROOT / 'configs/research/scheme_a_a2_formal.yaml')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            settings['acceptance'] = dict(smoke_run=str(directory), gate_run=str(directory))
            (directory / 'acceptance_result.json').write_text(json.dumps(dict(stage='smoke', passed=True, synthetic=False)))
            (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(settings))
            previous = dict(source_sha256='source', git_commit='commit', data=dict(files={'source_index.jsonl': 'old'}))
            (directory / 'provenance.json').write_text(json.dumps(previous))
            (directory / 'gt_read_audit.json').write_text(json.dumps(dict(gt_open_attempts=0, forbidden_npy_attempts=0, supervised_checkpoint_open_attempts=0)))
            current = copy.deepcopy(previous)
            current['data']['files']['source_index.jsonl'] = 'new'
            with self.assertRaisesRegex(ValueError, 'changed data'):
                research._validate_acceptance(settings, current)

    def test_gate_budget_cannot_be_omitted_or_expanded(self):
        settings = research.load_research_config(ROOT / 'configs/research/scheme_a_gate.yaml')
        for budget in ({}, dict(max_updates=2001, max_seconds=1800), dict(max_updates=2000, max_seconds=1801)):
            candidate = copy.deepcopy(settings)
            candidate['budget'] = budget
            with self.assertRaises(ValueError):
                research.validate_research_config(candidate)


if __name__ == '__main__':
    unittest.main()
