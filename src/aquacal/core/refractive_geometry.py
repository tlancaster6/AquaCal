"""Core refractive geometry operations.

This module provides ray tracing through the air-water interface using Snell's law.
It is used by both calibration and downstream triangulation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from aquacal.config.schema import Vec3, Vec2
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface, ray_plane_intersection


def snells_law_3d(
    incident_direction: Vec3,
    surface_normal: Vec3,
    n_ratio: float
) -> Vec3 | None:
    """
    Apply Snell's law in 3D to compute refracted ray direction.

    Args:
        incident_direction: Unit vector of incoming ray (toward interface)
        surface_normal: Unit normal of interface (always pass interface.normal,
                       which points from water toward air [0,0,-1])
        n_ratio: Ratio n1/n2 where ray goes from medium 1 to medium 2

    Returns:
        Unit vector of refracted ray direction, or None if total internal reflection.

    Notes:
        - Function handles normal orientation internally based on ray direction
        - For air-to-water: n_ratio = n_air / n_water ~= 0.75
        - For water-to-air: n_ratio = n_water / n_air ~= 1.33
        - TIR only possible when going from denser to less dense medium
    """
    # Normalize incident direction
    d = incident_direction / np.linalg.norm(incident_direction)

    # Compute cos of incident angle with surface normal
    cos_i = np.dot(d, surface_normal)

    # Determine normal orientation: n should point into the destination medium
    # If cos_i < 0, ray travels opposite to normal (e.g., air->water when normal points up)
    # If cos_i > 0, ray travels same direction as normal (e.g., water->air)
    if cos_i < 0:
        n = -surface_normal
        cos_i = -cos_i
    else:
        n = surface_normal

    # Apply Snell's law: n1*sin(theta1) = n2*sin(theta2)
    # sin^2(theta_t) = n_ratio^2 * sin^2(theta_i) = n_ratio^2 * (1 - cos^2(theta_i))
    sin_t_sq = n_ratio**2 * (1 - cos_i**2)

    # Check for total internal reflection
    if sin_t_sq > 1.0:
        return None

    cos_t = np.sqrt(1 - sin_t_sq)

    # Compute refracted direction using vector form of Snell's law
    # t = n_ratio * d + (cos_t - n_ratio * cos_i) * n
    t = n_ratio * d + (cos_t - n_ratio * cos_i) * n

    # Normalize (should already be unit, but ensure numerical stability)
    return t / np.linalg.norm(t)


def trace_ray_air_to_water(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """
    Trace ray from camera through air-water interface.

    Args:
        camera: Camera object (camera.name must be in interface.camera_offsets)
        interface: Interface object
        pixel: 2D pixel coordinates

    Returns:
        Tuple of (intersection_point, refracted_direction):
        - intersection_point: where ray hits interface (world coords)
        - refracted_direction: unit direction of ray in water (points +Z, into water)
        Returns (None, None) if ray doesn't hit interface or TIR occurs.
    """
    # Get ray from camera in world frame
    ray_origin, ray_direction = camera.pixel_to_ray_world(pixel)

    # Get interface plane point for this camera
    interface_point = interface.get_interface_point(camera.C, camera.name)

    # Intersect ray with interface plane
    result = ray_plane_intersection(
        ray_origin, ray_direction, interface_point, interface.normal
    )

    # Check for valid intersection (ray must hit interface in forward direction)
    if result[0] is None or result[1] is None or result[1] <= 0:
        return None, None

    intersection, t = result

    # Refract at interface (air to water)
    refracted = snells_law_3d(
        ray_direction,
        interface.normal,
        interface.n_ratio_air_to_water
    )

    if refracted is None:  # TIR (shouldn't happen for air->water at normal incidence)
        return None, None

    return intersection, refracted


def refractive_back_project(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """
    Back-project pixel to ray in water.

    This is a convenience wrapper around trace_ray_air_to_water, providing
    an API consistent with Camera.pixel_to_ray_world.

    Args:
        camera: Camera object
        interface: Interface object
        pixel: 2D pixel coordinates

    Returns:
        Tuple of (ray_origin, ray_direction):
        - ray_origin: point on interface where ray enters water
        - ray_direction: unit direction of ray in water
        Returns (None, None) if back-projection fails.
    """
    # Delegates to trace_ray_air_to_water
    return trace_ray_air_to_water(camera, interface, pixel)


def refractive_project(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3
) -> Vec2 | None:
    """
    Project 3D underwater point to 2D pixel through refractive interface.

    This is the forward projection used for computing reprojection error.

    Args:
        camera: Camera object
        interface: Interface object
        point_3d: 3D point in water (world coordinates, Z > interface_z)

    Returns:
        2D pixel coordinates, or None if projection fails.

    Notes:
        - Uses 1D optimization to find interface intersection point
        - Returns None if: point above interface, TIR, optimization fails,
          or refracted ray doesn't reach camera
    """
    C = camera.C
    Q = np.asarray(point_3d, dtype=np.float64)
    z_int = interface.get_interface_distance(camera.name)

    # Check point is below interface (in water)
    if Q[2] <= z_int:
        return None

    # Check if point is (nearly) on optical axis - use direct projection
    xy_dist = np.sqrt((Q[0] - C[0])**2 + (Q[1] - C[1])**2)
    if xy_dist < 1e-8:
        # Point is directly below camera - ray goes straight through
        P = np.array([C[0], C[1], z_int], dtype=np.float64)
        return camera.project(P, apply_distortion=True)

    # General case: use 1D optimization to find interface point
    # Parameterize interface point by angle from camera XY to point XY
    # P(t) = C_xy + t * direction, where t is distance in XY plane

    # Direction from camera XY toward point XY (unit vector in XY)
    dir_xy = np.array([Q[0] - C[0], Q[1] - C[1]], dtype=np.float64)
    dir_xy = dir_xy / np.linalg.norm(dir_xy)

    def get_interface_point(t: float) -> Vec3:
        """Get interface point at distance t from camera in XY plane."""
        return np.array([C[0] + t * dir_xy[0], C[1] + t * dir_xy[1], z_int],
                        dtype=np.float64)

    def compute_refracted_ray(P: Vec3) -> Vec3 | None:
        """Compute refracted ray direction in air from water point Q through P."""
        d_water = P - Q
        d_water_norm = np.linalg.norm(d_water)
        if d_water_norm < 1e-10:
            return None
        d_water = d_water / d_water_norm
        return snells_law_3d(d_water, interface.normal, interface.n_ratio_water_to_air)

    def error_function(t: float) -> float:
        """
        Signed error measuring deviation of refracted ray from camera direction.
        Uses the XY component of the cross product for sign.
        """
        P = get_interface_point(t)

        d_air = compute_refracted_ray(P)
        if d_air is None:  # TIR
            # Return large error with sign based on which side we're on
            return 1e6 if t > xy_dist else -1e6

        # Vector from P to camera
        to_camera = C - P
        to_camera_norm = np.linalg.norm(to_camera)
        if to_camera_norm < 1e-10:
            return 0.0
        to_camera = to_camera / to_camera_norm

        # Error: component of cross product perpendicular to XY direction
        # This gives a signed error that changes sign at the solution
        cross = np.cross(d_air, to_camera)

        # Project onto axis perpendicular to dir_xy in XY plane
        # perp_xy = [-dir_xy[1], dir_xy[0]] (90 degree rotation)
        # But we can just use a consistent signed component
        # Use the component that gives a sign change
        return cross[0] * dir_xy[1] - cross[1] * dir_xy[0]

    # Search range: from near camera (t=0) to beyond point projection (t=2*xy_dist)
    t_max = 2.0 * xy_dist

    try:
        # Sample to find bracket
        e0 = error_function(0.0)

        # Check if e0 is already zero (degenerate case)
        t_solution = None
        if abs(e0) < 1e-12:
            t_solution = 0.0
        else:
            # Find a bracket with sign change
            found_bracket = False
            t_lo, t_hi = 0.0, 0.0

            # Sample points to find sign change
            n_samples = 50
            t_samples = np.linspace(0, t_max, n_samples)
            e_prev = e0
            t_prev = 0.0

            for t_test in t_samples[1:]:
                e_test = error_function(t_test)
                if e_prev * e_test < 0:
                    t_lo, t_hi = t_prev, t_test
                    found_bracket = True
                    break
                # Also check if we found a root directly
                if abs(e_test) < 1e-12:
                    t_solution = t_test
                    found_bracket = True
                    break
                e_prev = e_test
                t_prev = t_test

            if not found_bracket:
                return None

            # If we didn't find t_solution directly, use brentq
            if t_solution is None:
                t_solution = brentq(error_function, t_lo, t_hi, xtol=1e-9)

    except (ValueError, RuntimeError):
        return None

    # Get the interface point
    P = get_interface_point(t_solution)

    # Verify we can compute the refracted ray
    d_air = compute_refracted_ray(P)
    if d_air is None:
        return None

    # Project by finding where the ray from C through P goes
    ray_to_interface = P - C
    ray_norm = np.linalg.norm(ray_to_interface)
    if ray_norm < 1e-10:
        # P is at camera - just project P directly
        return camera.project(P, apply_distortion=True)

    ray_to_interface = ray_to_interface / ray_norm

    # Use a point along this ray for projection (1 meter along ray from camera)
    point_for_projection = C + ray_to_interface

    return camera.project(point_for_projection, apply_distortion=True)
