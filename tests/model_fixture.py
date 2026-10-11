"""Finite, physically oriented synthetic scanner views for B0 tests."""
import torch


def model_batch(batch_size=1, views=3):
    origin = torch.tensor([[-6., .3, .3]]).expand(batch_size, views, 3).clone()
    directions = {'center_direction': [1., 0., 0.], 'right_direction': [0., 1., 0.], 'up_direction': [0., 0., 1.]}
    result = dict(input_projection=torch.rand(batch_size, views, 1, 64, 64),
                  input_valid_ray=torch.ones(batch_size, views, 1, 64, 64, dtype=torch.bool),
                  input_path_length_m=torch.full((batch_size, views, 1, 64, 64), .6),
                  input_station_origin_m=origin,
                  horizontal_angle_deg=torch.linspace(-5, 5, 64).expand(batch_size, -1),
                  vertical_angle_deg=torch.linspace(-4, 4, 64).expand(batch_size, -1),
                  view_mask=torch.ones(batch_size, views, dtype=torch.bool),
                  gt_volume_64=torch.rand(batch_size, 1, 64, 64, 64)*.1)
    for key, value in directions.items():
        result['input_'+key] = torch.tensor(value).expand(batch_size, views, 3).clone()
    return result
