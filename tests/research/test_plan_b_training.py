"""CPU trainer engineering tests, independent of real B accuracy/acceptance.

The small continuous-API double isolates optimizer accumulation and exact replay.
It is deliberately not evidence for passing the real-data smoke/gate.
"""
import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Subset

import research.plan_b.training as training
from geophy_set3d.training.selfsup import move_dual_batch

ROOT = Path(__file__).resolve().parents[2]


class ReplayField(torch.nn.Module):
    """Small differentiable context/query interface for trainer replay tests."""
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(-3.))
        self.retained_contexts = []

    def encode(self, batch):
        value = (self.bias + batch['input_projection'].flatten(1).mean(1)*.01).sigmoid().reshape(-1, 1, 1, 1, 1)
        if value.requires_grad:
            value.retain_grad()
            self.retained_contexts.append(value)
        return value

    def sample_grid(self, context, resolution):
        assert resolution == 64
        return context.expand(-1, 1, 3, 3, 3)

    def integrate(self, context, batch):
        chi = (context/16.043)/(context/16.043+(1-context)/28.965)
        return batch['input_path_length_m']*chi*1e6


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    result = yaml.safe_load((ROOT/'configs/research/scheme_b_synthetic_smoke.yaml').read_text())
    result['experiment'].update(output_root=str(tmp_path), name='engineering_only')
    return result


@pytest.mark.parametrize('change', ['Q', 'loss', 'design', 'effective_batch', 'workers'])
def test_config_rejects_protocol_drift(config, change):
    if change == 'Q':
        config['model']['samples'] = 48
    elif change == 'loss':
        config['loss']['hold'] = .5
    elif change == 'design':
        config['status'] = 'design_template_not_executable'
    elif change == 'effective_batch':
        config['training']['gradient_accumulation'] = 1
    elif change == 'workers':
        config['loader']['num_workers'] = 1
    with pytest.raises(ValueError):
        training.validate_config(config)


def test_dual_context_objective_gradients(config):
    dataset = training.make_research_dataset(config)
    loader = training.make_research_loader(dataset, config)
    batch = move_dual_batch(next(iter(loader)), 'cpu')
    model = ReplayField()
    loss, parts, _, projections = training.objective(model, batch)
    assert set(parts) == {'hold', 'cross', 'observed', 'consistency'}
    assert set(projections) == {'C_A', 'C_B'}
    loss.backward()
    assert len(model.retained_contexts) == 2
    for context in model.retained_contexts:
        assert context.grad is not None and torch.isfinite(context.grad).all()
        assert context.grad.abs().sum() > 0
    assert model.bias.grad is not None and torch.isfinite(model.bias.grad)


def test_actual_tail_case_average_matches_explicit_updates(config):
    dataset = training.make_research_dataset(config)
    # Three cases exercise two-case accumulation plus a one-case final tail.
    loader = training.make_research_loader(Subset(dataset, [0, 1, 0]), config)
    model = ReplayField()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    result = training.run_epoch(model, loader, config, optimizer)
    batches = [move_dual_batch(batch, 'cpu') for batch in loader]
    for group in (batches[:2], batches[2:]):
        expected_optimizer.zero_grad(set_to_none=True)
        for batch in group:
            loss, _, _, _ = training.objective(reference, batch)
            (loss/len(group)).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        expected_optimizer.step()
    assert result['optimizer_updates'] == 2
    assert result['sample_count'] == 3 and result['complete_epoch']
    torch.testing.assert_close(model.bias, reference.bias, rtol=0, atol=0)


def test_completed_epoch_resume_exactly_replays_all_training_state(config, monkeypatch):
    monkeypatch.setattr(training, 'build_model', lambda config: ReplayField())
    config['experiment']['name'] = 'full'
    full = training.train(config)
    resumed_config = copy.deepcopy(config)
    resumed_config['experiment']['name'] = 'resumed'
    resumed = training.train(resumed_config, stop_after_epochs=7)
    boundary = torch.load(resumed/'checkpoints/last.pt', weights_only=False)
    assert boundary['epoch'] == 6 and boundary['sampling_epoch_next'] == 7
    training.train(resumed_config, resume=True)
    a = torch.load(full/'checkpoints/last.pt', weights_only=False)
    b = torch.load(resumed/'checkpoints/last.pt', weights_only=False)
    assert a['global_step'] == b['global_step'] == 20
    assert a['scheduler'] == b['scheduler']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for key in ('loader', 'torch'):
        assert torch.equal(a['rng'][key], b['rng'][key])
    assert a['rng']['python'] == b['rng']['python']
    assert a['rng']['numpy'][0] == b['rng']['numpy'][0]
    assert (a['rng']['numpy'][1] == b['rng']['numpy'][1]).all()
    assert a['rng']['numpy'][2:] == b['rng']['numpy'][2:]
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for key, value in a['optimizer']['state'][0].items():
        torch.testing.assert_close(value, b['optimizer']['state'][0][key], rtol=0, atol=0)
    invalid = copy.deepcopy(b)
    invalid['boundary'] = 'partial_epoch'
    with pytest.raises(ValueError, match='completed epoch'):
        training.validate_resume(invalid, resumed_config, b['provenance'])
    invalid = copy.deepcopy(b)
    invalid['provenance']['source_sha256'] = 'changed'
    with pytest.raises(ValueError, match='provenance changed'):
        training.validate_resume(invalid, resumed_config, b['provenance'])


def test_train_projection_bias_initialization_preserves_all_other_parameters(tmp_path):
    import math
    from types import SimpleNamespace
    import numpy as np
    shape = (64,64,64)
    valid = np.zeros(shape, dtype=bool)
    valid[0,0,0] = valid[1,0,0] = True
    lengths = np.zeros(shape, dtype=np.float64)
    lengths[0,0,0], lengths[1,0,0] = .25, .75
    projection = np.zeros(shape, dtype=np.float32)
    projection[0,0,0], projection[1,0,0] = 1000., 3000.
    np.save(tmp_path/'measurements.npy', projection)
    (tmp_path/'geometry').mkdir()
    (tmp_path/'geometry/train_geometry.npz').write_bytes(b'fixture geometry hash')
    (tmp_path/'source_index.jsonl').write_text('fixture metadata hash\n')
    dataset = SimpleNamespace(split='train', geometry_pool='train', root=tmp_path,
        station_count=64, geometry=dict(valid_ray=valid,path_length_m=lengths),
        records=[dict(split='train',case_id=f'train_{i}',train_projection_path='measurements.npy') for i in range(166)])
    model = torch.nn.Module()
    model.decoder = torch.nn.Sequential(torch.nn.Linear(61,64),torch.nn.SiLU(),torch.nn.Linear(64,1),torch.nn.Sigmoid())
    previous = {k:v.clone() for k,v in model.state_dict().items()}
    result = training.initialize_from_train_measurements(model,dataset)
    assert result['valid_ray_count'] == 332 and result['projection_sum_ppm_m'] == 664000.
    assert result['path_length_sum_m'] == 166.
    assert result['pooled_mole_fraction'] == .004
    concentration = (.004*16.043)/(.004*16.043+.996*28.965)
    torch.testing.assert_close(model.decoder[-2].bias, torch.tensor([math.log(concentration/(1-concentration))]))
    for key,value in model.state_dict().items():
        if key != 'decoder.2.bias':
            torch.testing.assert_close(value,previous[key],rtol=0,atol=0)
    dataset.split = 'val'
    with pytest.raises(ValueError,match='full166 train'):
        training.initialize_from_train_measurements(model,dataset)
