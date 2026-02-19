"""Refractive triangulation for 3D reconstruction."""

import numpy as np

from aquacal.config.schema import CalibrationResult, Vec2, Vec3
from aquacal.core._aquakit_bridge import (
    _bridge_refractive_back_project,
    _make_interface_params,
)
from aquacal.core.camera import Camera


def triangulate_point(
    calibration: CalibrationResult,
    observations: dict[str, Vec2],
) -> Vec3 | None:
    """
    Triangulate a single 3D point from multi-camera observations.

    Args:
        calibration: Complete calibration result
        observations: Dict mapping camera_name to 2D pixel coordinates.
                     Must have at least 2 cameras.

    Returns:
        3D point in world coordinates, or None if triangulation fails.
        Failure occurs if: fewer than 2 observations, camera not in calibration,
        or refractive_back_project() fails for all cameras.

    Notes:
        - Uses _bridge_refractive_back_project() to get rays in water
        - Finds point minimizing sum of squared distances to all rays
    """
    if len(observations) < 2:
        return None

    rays = []
    for cam_name, pixel in observations.items():
        if cam_name not in calibration.cameras:
            continue

        cam_calib = calibration.cameras[cam_name]
        camera = Camera(cam_name, cam_calib.intrinsics, cam_calib.extrinsics)

        interface_aq = _make_interface_params(
            water_z=cam_calib.water_z,
            n_air=calibration.interface.n_air,
            n_water=calibration.interface.n_water,
        )
        result = _bridge_refractive_back_project(camera, interface_aq, pixel)
        if result[0] is not None:
            rays.append((result[0], result[1]))

    if len(rays) < 2:
        return None

    try:
        return triangulate_rays(rays)
    except ValueError:
        return None


def triangulate_rays(rays: list[tuple[Vec3, Vec3]]) -> Vec3:
    """
    Find 3D point minimizing sum of squared distances to all rays.

    Uses closed-form linear least squares solution.

    Args:
        rays: List of (origin, direction) tuples. Directions must be unit vectors.
              Must have at least 2 rays.

    Returns:
        3D point that minimizes sum of squared distances to all rays.

    Raises:
        ValueError: If fewer than 2 rays provided or system is degenerate.
    """
    if len(rays) < 2:
        raise ValueError("Need at least 2 rays")

    A_sum = np.zeros((3, 3), dtype=np.float64)
    b_sum = np.zeros(3, dtype=np.float64)

    for origin, direction in rays:
        d = direction / np.linalg.norm(direction)  # ensure unit
        I_minus_ddT = np.eye(3) - np.outer(d, d)
        A_sum += I_minus_ddT
        b_sum += I_minus_ddT @ origin

    # Solve A_sum @ P = b_sum
    try:
        P = np.linalg.solve(A_sum, b_sum)
    except np.linalg.LinAlgError:
        raise ValueError("Degenerate ray configuration")

    return P


def point_to_ray_distance(point: Vec3, ray_origin: Vec3, ray_direction: Vec3) -> float:
    """
    Compute perpendicular distance from point to ray.

    Args:
        point: 3D point
        ray_origin: Origin of ray
        ray_direction: Unit direction of ray

    Returns:
        Perpendicular distance from point to ray (always non-negative).
    """
    v = point - ray_origin
    # Project v onto ray direction
    proj_length = np.dot(v, ray_direction)
    proj = proj_length * ray_direction
    # Perpendicular component
    perp = v - proj
    return np.linalg.norm(perp)
