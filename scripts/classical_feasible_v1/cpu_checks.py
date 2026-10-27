"""Shape, fresh-state identity, masked inputs, backward and exact state roundtrip."""
from feasible_context import *


def main():
    setup();verify();ds=make_dataset(config())
    batch=dual_measurement_collate([ds[0]])
    guard=GTReadGuard(DATA,ROOT)
    with guard:
        model=build('unet');initial=state_hash(model.state_dict());model.train()
        target=torch.full((1,1,64,64,64),.001)
        out=model(batch['A']);loss=(out-target).square().mean()/target.square().mean()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt=TrackedAdam(model,ROOT/'cpu_engineering_updates.jsonl');opt.step()
        assert opt.updates==1 and norm>0 and state_hash(model.state_dict())!=initial
        model.eval()
        with torch.no_grad():out=model(batch['A']);repeat=model(batch['A']);assert torch.equal(out,repeat)
        checkpoint=ROOT/'cpu_engineering.pt';atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict()))
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);other=build('unet');other.load_state_dict(ck['model']);other.eval()
        with torch.no_grad():assert torch.equal(other(batch['A']),out)
        assert out.shape==(1,1,64,64,64) and torch.isfinite(out).all() and out.min()>=0 and out.max()<=1
        assert initial==read(PROJECT/'docs/reports/artifacts/unet_baseline_v1/initialization.json')['initialized_state_sha256']
    write(ART/'cpu_checks.json',dict(passed=True,engineering_updates=1,formal_updates=0,fresh_state_sha256=initial,
        checkpoint_roundtrip_bitwise=True,access=guard.summary()))
    print('CPU dense target learning, exact checkpoint, fresh initialization checks passed')


if __name__=='__main__':main()
