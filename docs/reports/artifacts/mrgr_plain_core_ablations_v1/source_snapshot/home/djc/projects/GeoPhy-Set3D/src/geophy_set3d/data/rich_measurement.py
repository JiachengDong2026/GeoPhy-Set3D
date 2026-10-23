"""Measurement-only rich input, constructed before padding from the exact A/B union."""
import numpy as np
from .measurement_only import DualMeasurementDataset, assert_no_gt
from .collate import projection_pool_collate


class RichMeasurementDataset(DualMeasurementDataset):
    def __getitem__(self, index):
        row, projections = self._projections(index)
        a, b, h = self.sampler.sample(self.station_count, epoch=int(self._epoch.item()),
                                     sample_index=index, case_id=row['case_id'])
        empty = np.empty(0, dtype=np.int64)
        # Use _views for every field: measurements, masks and geometry share IDs.
        return {key: self._measurement_sample(row, projections, ids, empty)
                for key, ids in (('A', a), ('B', b), ('H', h), ('T', np.concatenate((a, b))))}


def rich_measurement_collate(samples):
    assert_no_gt(samples)
    return {key: projection_pool_collate([sample[key] for sample in samples])
            for key in ('A', 'B', 'H', 'T')}
