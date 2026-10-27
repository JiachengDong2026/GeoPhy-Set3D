"""Full-ray GNOT chunk equivalence at trained weights; no scientific updates."""
from feasible_context import *
from geophy_set3d.models.ray_senseiver_ss import ray_tokens


def main():
    setup();verify();guard=GTReadGuard(DATA,ROOT)
    with guard:
        model=build('gnot');checkpoint=ROOT/'debug/gnot/last.pt'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
        ds=make_dataset(config());batch=dual_measurement_collate([ds[0],ds[1]])['A']
        tokens,excluded,counts=ray_tokens(batch)
        coordinates=model.query_coordinates_m[::256]  # Engineering equivalence test only.
        outputs=[];gradients=[];hashes=[]
        for size in [128,1024]:
            hashes.append(state_hash(model.state_dict()));model.train();model.zero_grad(set_to_none=True)
            value=model.forward_tokens(tokens,excluded,counts,coordinates_m=coordinates,chunk_size=size)
            value.square().mean().backward()
            gradient=torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
            assert torch.isfinite(value).all() and torch.isfinite(gradient).all() and gradient.norm()>0
            outputs.append(value.detach());gradients.append(gradient.detach().clone())
        field_error=float((outputs[0]-outputs[1]).norm()/outputs[0].norm())
        gradient_error=float((gradients[0]-gradients[1]).norm()/gradients[0].norm())
        assert hashes[0]==hashes[1] and field_error<2e-5 and gradient_error<5e-4
    write(ART/'gnot_chunk_checks.json',dict(passed=True,batch_size=2,all_valid_input_rays=True,
        engineering_query_nodes=1024,test_chunk_sizes=[128,1024],output_relative_error=field_error,
        full_parameter_gradient_relative_error=gradient_error,state_sha256=hashes[0],
        checkpoint_sha256=sha(checkpoint),scientific_optimizer_updates=0,access=guard.summary(),
        formal_execution='65536 query chunk, complete262144 queries and exact global attention, FP32, unchanged parameters'))
    print('GNOT chunk output relative error',field_error,'gradient error',gradient_error,flush=True)


if __name__=='__main__':main()
