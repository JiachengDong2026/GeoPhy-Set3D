"""CPU checks before queue activation; GPU/recovery checks remain mandatory."""
from common import *
import copy
import ast
import torch
from module_models_v2 import ARMS,model_for
from module_registry_v2 import BASE,ablation_arms
from geophy_set3d.data.collate import projection_pool_collate


def main():
    backend();reg=verified(BASE/'registration.json');verify_hashes(reg['protocol_hashes'])
    assert len(reg['runs'])==8 and len(ablation_arms())==7
    assert reg['final_methods']==17+4==21
    assert reg['final_reconstructions']==21*35*17==12495
    assert reg['final_fields']==(17*2+4)*35*17==22610
    assert reg['final_primary_comparators']==20
    assert not (OUT/'final_method_choices_closed.json').exists() and not (OUT/'final_lock.json').exists()
    assert not (OUT/'evaluation/test/gt_access.json').exists()
    for d in reg['runs']:
        p=verified(Path(d)/'protocol.json');verify_hashes(p['sources'])
        assert p['planned_updates']==24900 and p['epochs']==300 and not p['subset_smoke']
    for d in (OUT/'training').glob('*/protocol.json'):verify_hashes(verified(d)['sources'])
    for p in (ROOT/'scripts/paper_final_v1').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    records=[]
    with AccessGuard(splits=['val']) as guard:
        setting=verified(OUT/'layouts.json')['val'][0]
        batch=projection_pool_collate([dataset('val',setting)[0]])
        batch={k:v for k,v in batch.items() if not k.startswith('hold_')}
        for arm in ARMS:
            config=verified(OUT/'training'/f'{arm}_seed20261011'/'protocol.json')['config']
            reference=model_for(config,'multiscale','cpu').state_dict()
            model=model_for(config,arm,'cpu');state=model.state_dict()
            assert all(torch.equal(v,reference[k]) for k,v in state.items())
            assert not (state.keys()-reference.keys())
            removed=reference.keys()-state.keys()
            assert (not removed) if arm!='uniform_fusion' else all(k.startswith('fusion.') for k in removed)
            if arm in ('no_ray_relations','no_observability'):
                bias=model.interaction.pairwise_bias
                torch.nn.init.normal_(bias[-1].weight,std=.1)
                x=torch.randn(2,4,4,8,requires_grad=True);bias(x).square().sum().backward()
                drop=slice(0,3) if arm=='no_ray_relations' else slice(3,8)
                keep=slice(3,8) if arm=='no_ray_relations' else slice(0,3)
                assert not x.grad[...,drop].any() and x.grad[...,keep].abs().sum()>0
                if arm=='no_observability':
                    emb=model.interaction.observability_embedding
                    torch.testing.assert_close(emb(torch.randn(9,5)),emb(torch.randn(9,5)),rtol=0,atol=0)
                del model;model=model_for(config,arm,'cpu')
            model.eval()
            with torch.no_grad():
                st=model(batch,return_stages=True)
                assert all(st[k].shape==(1,1,64,64,64) and torch.isfinite(st[k]).all() and st[k].min()>=0 and st[k].max()<=1 for k in ('C0','C1'))
                if arm=='zero_residual':assert not st['residual_input'][:,:,0].any() and st['measurement_prediction_Y0'] is None
                perm=torch.arange(3,-1,-1)
                b={k:(v[:,perm] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==4 and k.startswith(('input_','view_mask')) else v) for k,v in batch.items()}
                torch.testing.assert_close(st['C1'],model(b),rtol=2e-4,atol=2e-6)
            records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),initial_tensors_exact=True,CPU_actual_forward=True,view_permutation=True))
            del model,reference,state,st,b
        audit=guard.summary();assert not audit['denied']
    denied=[]
    with AccessGuard() as boundary:
        source=next(r for r in boundary.records if r['split']=='test')
        for path in (DATA/source['gt_volume_64_path'],Path(source['source_target_path'])):
            try:
                with path.open('rb'):raise AssertionError('forbidden open succeeded')
            except PermissionError:denied.append(str(path))
    assert len(denied)==2 and not boundary.touched
    freeze(BASE/'integration_checks.json',dict(at=now(),passed=True,records=records,audit=audit,forbidden_reads_denied_before_bytes=denied,
        original_and_new_training_sources_verified=True,final_methods=21,comparators=20,reconstructions=12495,fields=22610,
        GPU_gradient_and_resume_checks='mandatory queued module_checks_v2; not yet claimed passed',source_sha256=sha(__file__)))
    print('CPU module integration and test-boundary checks PASSED',flush=True)


if __name__=='__main__':main()
