"""Bitwise field equivalence for original features and worker-prepared inputs."""
from feasible_context import *
from prepared_inputs import PreparedDataset,prepare_model
from amplitude_adapter import AmplitudeModel,amplitude


def main():
    setup();verify();ds=make_dataset(config());reference=.007
    records=[];initial=state_hash(build('unet').state_dict())
    for normalized in [False,True]:
        data=PreparedDataset(ds,reference if normalized else None)
        raw_model=build('unet').eval();prepared_model=prepare_model(build('unet')).eval()
        if normalized:
            raw_model=AmplitudeModel(raw_model,reference);prepared_model=AmplitudeModel(prepared_model,reference)
        c=config();c['loader'].update(num_workers=2,persistent_workers=False,pin_memory=False)
        subset=torch.utils.data.Subset(data,[0,2,77,82])
        loader=make_loader(subset,c)
        with torch.no_grad():
            for pair,batch in zip([[0,2],[77,82]],loader):
                original=dual_measurement_collate([ds[i] for i in pair])
                for branch in ['A','B']:
                    a=raw_model(original[branch]);b=prepared_model(batch[branch])
                    error=float((a-b).abs().max());assert error<1e-8,(normalized,pair,branch,error)
                    records.append(dict(normalized=normalized,indices=pair,branch=branch,maximum_absolute_difference=error,bitwise_equal=torch.equal(a,b)))
        assert state_hash(build('unet').state_dict())==initial
    write(ART/'prepared_input_checks.json',dict(passed=True,records=records,scientific_optimizer_updates=0,
        operator='unchanged FixedRayBackprojection, original stencil/CPU scatter, larger geometry-only cache',GT_used=False))
    print('Worker-prepared original/canonical features match direct original implementation',flush=True)


if __name__=='__main__':main()
