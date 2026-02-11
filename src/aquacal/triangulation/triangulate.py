"""Refractive triangulation for 3D reconstruction."""

import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import CalibrationResult, Vec3, Vec2
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_back_project


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
        - Uses refractive_back_project() to get rays in water
        - Finds point minimizing sum of squared distances to all rays
    """
    if len(observations) < 2:
        return None

    # Build camera_distances dict with ALL cameras
    camera_distances = {
        cam_name: calibration.cameras[cam_name].interface_distance
        for cam_name in calibration.cameras
    }

    # Create single shared interface with all cameras
    interface = Interface(
        normal=calibration.interface.normal,
        camera_distances=camera_distances,
        n_air=calibration.interface.n_air,
        n_water=calibration.interface.n_water,
    )

    rays = []
    for cam_name, pixel in observations.items():
        if cam_name not in calibration.cameras:
            continue

        cam_calib = calibration.cameras[cam_name]
        camera = Camera(cam_name, cam_calib.intrinsics, cam_calib.extrinsics)

        result = refractive_back_project(camera, interface, pixel)
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
