import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/reconstruct_plan_a.py'
spec = importlib.util.spec_from_file_location('reconstruct_plan_a', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def measurements():
    shape = (1, 1, 1, 64, 64)
    values = {key: np.zeros(shape, np.float32) for key in
              ('input_projection_raw', 'input_path_length_m', 'input_t_near')}
    values['input_projection_raw'][0, 0, 0, 0, 0] = 85547.742
    values['input_projection_raw'][0, 0, 0, 0, 1] = np.nan
    values['input_valid_ray'] = np.ones(shape, bool)
    values['input_valid_ray'][0, 0, 0, 0, 1] = False
    values.update(view_mask=np.ones((1, 1), bool),
                  input_ray_direction=np.zeros((1, 1, 64, 64, 3), np.float32),
                  horizontal_angle_deg=np.linspace(-5, 5, 64, dtype=np.float32)[None],
                  vertical_angle_deg=np.linspace(-4, 4, 64, dtype=np.float32)[None])
    for key in ('input_station_origin_m', 'input_center_direction', 'input_right_direction', 'input_up_direction'):
        values[key] = np.zeros((1, 1, 3), np.float32)
    return values


def test_external_pic_stays_fp32_and_invalid_nan_is_removed(tmp_path):
    path = tmp_path/'observations.npz'
    np.savez(path, **measurements())
    batch = module.load_measurements(path, 'cpu', 32284.894580078173)
    assert batch['input_projection_raw'].dtype == torch.float32
    assert torch.isfinite(batch['input_projection']).all()
    assert batch['input_projection'][0, 0, 0, 0, 0] > 1
    assert batch['input_projection'][0, 0, 0, 0, 1] == 0
    assert batch['input_valid_ray'][0, 0, 0, 1, 0]  # a valid zero measurement


def test_external_labels_and_different_scale_are_rejected(tmp_path):
    path = tmp_path/'observations.npz'
    values = measurements()
    values['gt_volume_64'] = np.zeros((4,)*3)
    np.savez(path, **values)
    with pytest.raises(ValueError, match='ground truth'):
        module.load_measurements(path, 'cpu', 32284.894580078173)
    values.pop('gt_volume_64')
    values['projection_scale_ppm_m'] = np.array([1.])
    np.savez(path, **values)
    with pytest.raises(ValueError, match='scale differs'):
        module.load_measurements(path, 'cpu', 32284.894580078173)
