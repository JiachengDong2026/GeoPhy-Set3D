"""Per-case and pooled CH4 mass-fraction errors, without case normalization."""
import math
import torch


class VolumeMetrics:
    def __init__(self):
        self.rows = []

    def update(self, prediction, target, case_ids=None):
        p, g = prediction.detach().double().flatten(1), target.detach().double().flatten(1)
        for i in range(len(p)):
            d = p[i]-g[i]
            a, s, ga, gs = d.abs().sum().item(), d.square().sum().item(), g[i].abs().sum().item(), g[i].square().sum().item()
            n = g[i].numel()
            self.rows.append(dict(case_id=case_ids[i] if case_ids else str(len(self.rows)),
                                 mae=a/n, rmse=math.sqrt(s/n), relative_l1=a/max(ga, 1e-6),
                                 relative_l2=math.sqrt(s/max(gs, 1e-12)),
                                 integrated_concentration_relative_error=abs(p[i].sum().item()-g[i].sum().item())/max(ga, 1e-6),
                                 peak_value_absolute_error=abs(p[i].max().item()-g[i].max().item()),
                                 absolute_sum=a, squared_sum=s, gt_absolute_sum=ga, gt_squared_sum=gs, voxel_count=n))

    def compute(self):
        if not self.rows:
            raise ValueError('No metric samples')
        names = ('mae', 'rmse', 'relative_l1', 'relative_l2',
                 'integrated_concentration_relative_error', 'peak_value_absolute_error')
        mean = {k: sum(r[k] for r in self.rows)/len(self.rows) for k in names}
        sums = {k: sum(r[k] for r in self.rows) for k in
                ('absolute_sum', 'squared_sum', 'gt_absolute_sum', 'gt_squared_sum', 'voxel_count')}
        pooled = dict(mae=sums['absolute_sum']/sums['voxel_count'],
                      rmse=math.sqrt(sums['squared_sum']/sums['voxel_count']),
                      relative_l1=sums['absolute_sum']/max(sums['gt_absolute_sum'], 1e-6),
                      relative_l2=math.sqrt(sums['squared_sum']/max(sums['gt_squared_sum'], 1e-12)))
        return dict(case_count=len(self.rows), mean_per_case=mean, pooled=pooled)
