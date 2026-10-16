"""One encoding per case; continuous reprojection never uses the exported grid."""
import time

import torch

from geophy_set3d.data.measurement_only import assert_no_gt


class ContinuousReconstructor:
    def __init__(self, model, *, checkpoint_hash=None, config_hash=None):
        self.model = model
        self.provenance = dict(method='SetConditionalField', protocol='scheme_b_continuous_measurement_only_v1',
                               checkpoint_hash=checkpoint_hash, config_hash=config_hash)

    @torch.no_grad()
    def reconstruct(self, input_batch, output_grid=64):
        assert_no_gt(input_batch)
        if output_grid not in (64, 128):
            raise ValueError('Export supports 64 or 128 XYZ endpoint samples')
        self.model.eval()
        device = input_batch['input_projection'].device

        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        sync()
        started = time.perf_counter()
        context = self.model.encode(input_batch)
        field = self.model.sample_grid(context, output_grid)
        sync()
        sampled = time.perf_counter()
        reprojection = self.model.integrate(context, input_batch)
        sync()
        if (not torch.isfinite(field).all() or not torch.isfinite(reprojection).all()
                or bool((field < 0).any()) or bool((field > 1).any())):
            raise FloatingPointError('Nonfinite or unbounded reconstruction')
        return dict(field_pred=field, field_variable='methane_mass_fraction', axes='XYZ',
                    domain_m=[[0., 0., 0.], [.6, .6, .6]],
                    input_reprojection_ppm_m=reprojection, coverage=context.coverage,
                    timing=dict(encode_and_sample_s=sampled-started,
                                continuous_project_s=time.perf_counter()-sampled),
                    provenance=self.provenance,
                    resolution_note='Endpoint samples of one continuous function; 128 does not establish added information')
