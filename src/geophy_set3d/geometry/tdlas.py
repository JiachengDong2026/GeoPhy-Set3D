"""Independent TDLAS beams: rows=elevation, columns=horizontal steering."""
import numpy as np


def ray_box_intersection(origins, directions, bounds=(0.0, 0.6)):
    """Forward-ray slab intersection; invalid rays have finite zero distances."""
    origins = np.broadcast_to(origins, directions.shape)
    parallel = np.abs(directions) < 1e-14
    outside = np.any(parallel & ((origins < bounds[0]) | (origins > bounds[1])), axis=-1)
    safe = np.where(parallel, 1.0, directions)
    a = (bounds[0] - origins) / safe
    b = (bounds[1] - origins) / safe
    lo = np.where(parallel, -np.inf, np.minimum(a, b))
    hi = np.where(parallel, np.inf, np.maximum(a, b))
    near = np.maximum(np.max(lo, axis=-1), 0.0)
    far = np.min(hi, axis=-1)
    valid = (~outside) & (far > near) & np.isfinite(near) & np.isfinite(far)
    near = np.where(valid, near, 0.0)
    far = np.where(valid, far, 0.0)
    return near, far, far - near, valid


def make_geometry(config, count, seed, azimuth=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(*config['azimuth_deg'], count) if azimuth is None else np.array(azimuth, dtype=float)
    radius = rng.uniform(*config['radius_m'], count) if azimuth is None else np.full(count, 6.0)
    height = rng.uniform(*config['height_m'], count) if azimuth is None else np.full(count, 0.3)
    center = np.array(config['target_m'], dtype=np.float64)
    origin = center + np.stack([radius * np.cos(np.deg2rad(theta)), height - center[1],
                                radius * np.sin(np.deg2rad(theta))], axis=-1)
    forward = center - origin
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(right, forward)
    h, w = config['projection_hw']
    alpha = np.linspace(-config['horizontal_fan_deg']/2, config['horizontal_fan_deg']/2, w)
    beta = np.linspace(-config['vertical_scan_deg']/2, config['vertical_scan_deg']/2, h)
    a, b = np.meshgrid(np.deg2rad(alpha), np.deg2rad(beta), indexing='xy')
    # Spherical yaw then elevation, in an orthonormal forward/right/up basis.
    direction = (np.cos(b)[None,...,None] * np.cos(a)[None,...,None] * forward[:,None,None,:]
                 + np.cos(b)[None,...,None] * np.sin(a)[None,...,None] * right[:,None,None,:]
                 + np.sin(b)[None,...,None] * up[:,None,None,:])
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    near, far, length, valid = ray_box_intersection(origin[:,None,None,:], direction)
    return dict(station_origin_m=origin, station_azimuth_deg=theta, station_radius_m=radius,
                station_height_m=height, center_direction=forward, right_direction=right,
                up_direction=up, horizontal_angle_deg=alpha, vertical_angle_deg=beta,
                ray_direction=direction, t_near=near, t_far=far, path_length_m=length, valid_ray=valid)


def subset_geometry(geometry, indices):
    count = len(geometry['station_origin_m'])
    return {k: (v[indices] if k not in ('horizontal_angle_deg', 'vertical_angle_deg')
                and v.shape[0] == count else v) for k, v in geometry.items()}
