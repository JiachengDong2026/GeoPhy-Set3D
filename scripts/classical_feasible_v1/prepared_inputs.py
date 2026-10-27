"""Exact original CPU backprojection prepared by loader workers; no learned changes."""
from feasible_context import *
from amplitude_adapter import canonical
from geophy_set3d.models.baseline_fno_v1 import FixedRayBackprojection


class PreparedGrid:
    def __init__(self,original):self.original=original
    def __call__(self,batch):
        if '_prepared_grid' in batch:return batch['_prepared_grid'].to(batch['input_projection_raw'].device)
        return self.original(batch)


def prepare_model(model):
    model.adapter=PreparedGrid(model.adapter)
    return model


class PreparedDataset(torch.utils.data.Dataset):
    def __init__(self,base,reference=None):self.base=base;self.reference=reference;self.adapter=None
    def __len__(self):return len(self.base)
    @property
    def records(self):return self.base.records
    @property
    def _epoch(self):return self.base._epoch
    @property
    def sampler(self):return self.base.sampler
    @property
    def station_count(self):return self.base.station_count
    def set_epoch(self,epoch):return self.base.set_epoch(epoch)
    def _projections(self,index):return self.base._projections(index)
    def _views(self,*args):return self.base._views(*args)
    def __getitem__(self,index):
        sample=self.base[index]
        if self.adapter is None:
            self.adapter=FixedRayBackprojection(cache_entries=64)
            self.adapter.operator.cache_budget_bytes=2*1024**3
        for branch in ['A','B']:
            batch=projection_pool_collate([sample[branch]])
            if self.reference is not None:batch,_=canonical(batch,self.reference)
            sample[branch]['_prepared_grid']=self.adapter(batch)[0]
        return sample
