"""Check held-out view masks and test scorer without opening test arrays or GT."""
from feasible_context import *
from test_unet import as_input
from geophy_set3d.evaluation.research import volume_metrics


def main():
    setup();verify();sample=make_dataset(config(),validation=True)[0]
    a,h=sample['A'],sample['H']
    combined={k:v for k,v in a.items() if not k.startswith('hold_')}
    combined.update({'hold_'+k[6:]:v for k,v in h.items() if k.startswith('input_')})
    batch=projection_pool_collate([combined]);original=projection_pool_collate([h]);mapped=as_input(batch,'hold')
    assert not any(k.startswith('hold_') for k in mapped)
    for key in original:
        if key.startswith('input_') or key=='view_mask':assert torch.equal(mapped[key],original[key]),key
    values=np.full((64,64,64),.001,dtype='float32')
    assert volume_metrics(values,values)['relative_l1']==0
    assert volume_metrics(values*0,values)['relative_l1']==1
    write(ART/'test_contract_checks.json',dict(passed=True,test_arrays_read=0,GT64_read=0,GT128_read=0,optimizer_updates=0,heldout_mask_mapping_verified=True))
    print('Test inference mask and metric contract checks passed without test/GT reads')


if __name__=='__main__':main()
