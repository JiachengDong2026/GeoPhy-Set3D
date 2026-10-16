"""Execution candidates must preserve state identity and physical FP32 boundaries."""
import copy
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from plan_a_runtime_candidates import configure_model
from research.test_set_unrolled3d import make_batch, small_model


def test_runtime_configs_require_separate_launcher_and_fixed_fp32(tmp_path):
    import yaml
    from train_plan_a_runtime import load_runtime_config
    from geophy_set3d.training.research_selfsup import load_research_config
    root=Path(__file__).resolve().parents[2]
    path=root/'configs/research/scheme_a_smoke_runtime_v3.yaml'
    config=load_runtime_config(path)
    assert config['execution_runtime']['network_precision']=='fp32'
    with pytest.raises(ValueError,match='executable'):
        load_research_config(path)
    changed=yaml.safe_load(path.read_text())
    changed['execution_runtime']['network_precision']='bf16'
    altered=tmp_path/'bad.yaml'
    altered.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match='qualified FP32'):
        load_runtime_config(altered)


def test_formal_runtime_rejects_mismatched_execution_evidence(tmp_path,monkeypatch):
    import yaml
    from train_plan_a_runtime import install_runtime,RUNTIME
    import geophy_set3d.models.set_unrolled3d as models
    import geophy_set3d.training.research_selfsup as training
    # Register originals with monkeypatch so all process-local installs restore.
    monkeypatch.setattr(models,'build_set_unrolled_model',models.build_set_unrolled_model)
    for name in ('seed_everything','_provenance','_validate_acceptance'):
        monkeypatch.setattr(training,name,getattr(training,name))
    install_runtime()
    evidence=tmp_path/'smoke'
    evidence.mkdir()
    (evidence/'config_resolved.yaml').write_text(yaml.safe_dump({'execution_runtime':dict(RUNTIME,compile_prior=False)}))
    config=dict(stage='formal',execution_runtime=RUNTIME,acceptance={'smoke_run':str(evidence)})
    with pytest.raises(ValueError,match='different execution runtime'):
        training._validate_acceptance(config,{})


def test_layout_preserves_state_and_complete_small_model_gradients():
    torch.set_num_threads(2)
    original = small_model()
    # Exercise propagation through all prior layers, beyond zero-last-layer init.
    with torch.no_grad():
        original.prior[-1].weight.normal_(std=.01)
    candidate = copy.deepcopy(original)
    before = {k:v.clone() for k,v in candidate.state_dict().items()}
    configure_model(candidate, dict(network_layout='channels_last_3d',
                    network_precision='fp32', compile_prior=False))
    assert before.keys() == candidate.state_dict().keys()
    for k,v in candidate.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    batch = make_batch()
    first,second=original(batch),candidate(batch)
    torch.testing.assert_close(first,second,rtol=1e-4,atol=1e-6)
    first.square().mean().backward()
    second.square().mean().backward()
    for (k,a),(j,b) in zip(original.named_parameters(),candidate.named_parameters()):
        assert k==j and (a.grad is None)==(b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-3,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_bf16_full_chain_has_fp32_lifting_physics_and_parameter_gradients():
    torch.set_num_threads(2)
    model=small_model().cuda()
    configure_model(model,dict(network_layout='contiguous',network_precision='bf16',compile_prior=False))
    batch={k:v.cuda() for k,v in make_batch().items()}
    dtypes=[]
    hook=model.initializer.lifting.register_forward_pre_hook(lambda m,args:dtypes.append(args[0].dtype))
    residual=model.operator.residual_gradient
    def checked(c,target,stencil):
        assert c.dtype==target.dtype==torch.float32
        out=residual(c,target,stencil)
        assert all(v.dtype==torch.float32 for v in out)
        return out
    model.operator.residual_gradient=checked
    out=model(batch)
    assert out.dtype==torch.float32 and dtypes==[torch.float32]
    out.square().mean().backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
    hook.remove()
