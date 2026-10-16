"""Budget architectures retain set symmetry, physical gradients and FP32 physics."""
import copy
from pathlib import Path
import sys
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_a_budget_v4 import LightDecoder3D, MultiScalePrior, load_budget_config
from plan_a_runtime_candidates import configure_model
from research.test_set_unrolled3d import make_batch, small_model, permute, pad


def test_budget_configs_are_separate_and_keep_numerical_physics(tmp_path):
    from geophy_set3d.training.research_selfsup import load_research_config
    path = ROOT / 'configs/research/scheme_a_budget_v4_w16_light.yaml'
    c = load_budget_config(path)
    assert c['model']['steps'] == 4 and c['model']['samples'] == 96
    with pytest.raises(ValueError, match='executable'):
        load_research_config(path)
    for field, value in [('steps', 2), ('samples', 48), ('output_resolution', 32)]:
        bad = yaml.safe_load(path.read_text())
        bad['model'][field] = value
        file = tmp_path / f'{field}.yaml'
        file.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match='K4/Q96'):
            load_budget_config(file)


@pytest.mark.parametrize('multiscale', [False, True])
def test_light_decoder_complete_gradient_chain_set_symmetry_and_padding(multiscale):
    torch.set_num_threads(2)
    model = small_model()
    model.initializer.decoder = LightDecoder3D(8)
    # Exercise interior prior gradients as well as its zero-init output head.
    if multiscale:
        model.prior = MultiScalePrior(8)
    with torch.no_grad():
        if multiscale:
            model.prior.coarse[-1].weight.normal_(std=.01)
            model.prior.local[-1].weight.normal_(std=.01)
        else:
            model.prior[-1].weight.normal_(std=.01)
    model.eval()
    batch = make_batch()
    expected = model(batch)
    assert expected.shape == (1, 1, 8, 8, 8)
    assert expected.dtype == torch.float32 and torch.isfinite(expected).all()
    assert ((expected >= 0) & (expected <= 1)).all()
    torch.testing.assert_close(model(permute(batch, torch.tensor([1, 0]))), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model(pad(batch)), expected, atol=1e-6, rtol=1e-5)
    observed = model.operator(expected, batch)
    observed.square().mean().backward()
    for module in (model.initializer.encoder, model.initializer.decoder, model.prior):
        grads = [p.grad for p in module.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert model.step_logits.grad is not None and model.step_logits.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('light_decoder,width,local_width', [(True,8,8), (False,32,32)])
def test_budget_bf16_uses_fp32_geometry_and_differentiable_physics(light_decoder, width, local_width):
    torch.set_num_threads(2)
    model = small_model()
    if light_decoder:
        model.initializer.decoder = LightDecoder3D(8)
    model.prior = MultiScalePrior(width, local_width=local_width)
    model.cuda()
    configure_model(model, dict(network_layout='contiguous', network_precision='bf16', compile_prior=False))
    batch = {k: v.cuda() for k, v in make_batch().items()}
    dtypes = []
    handle = model.initializer.lifting.register_forward_pre_hook(lambda m, args: dtypes.append(args[0].dtype))
    original = model.operator.residual_gradient
    def checked(c, y, stencil):
        assert c.dtype == y.dtype == torch.float32
        prediction, gradient = original(c, y, stencil)
        assert prediction.dtype == gradient.dtype == torch.float32
        assert gradient.requires_grad
        return prediction, gradient
    model.operator.residual_gradient = checked
    output = model(batch)
    output.square().mean().backward()
    assert output.dtype == torch.float32 and dtypes == [torch.float32]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    handle.remove()


def test_formal_budget_rejects_evidence_from_another_architecture(tmp_path, monkeypatch):
    from train_plan_a_budget import install_budget
    import geophy_set3d.models.set_unrolled3d as models
    import geophy_set3d.training.research_selfsup as training
    monkeypatch.setattr(models, 'build_set_unrolled_model', models.build_set_unrolled_model)
    for name in ('PROTOCOL', 'seed_everything', '_provenance', '_validate_acceptance', 'write_json'):
        monkeypatch.setattr(training, name, getattr(training, name))
    config = load_budget_config(ROOT / 'configs/research/scheme_a_budget_v4_multiscale.yaml')
    install_budget(config)
    smoke = tmp_path / 'smoke'
    smoke.mkdir()
    old = copy.deepcopy(config)
    old['architecture'] = {'decoder': 'historical'}
    (smoke / 'config_resolved.yaml').write_text(yaml.safe_dump(old))
    (smoke / 'provenance.json').write_text('{}')
    config['acceptance'] = {'smoke_run': str(smoke)}
    with pytest.raises(ValueError, match='different budget'):
        training._validate_acceptance(config, {})
