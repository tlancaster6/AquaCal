"""Core refractive geometry operations.

This module provides ray tracing through the air-water interface using Snell's law.
It is used by both calibration and downstream triangulation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from aquacal.config.schema import Vec2, Vec3
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface, ray_plane_intersection


def snells_law_3d(
    incident_direction: Vec3, surface_normal: Vec3, n_ratio: float
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

    Example:
        >>> import numpy as np
        >>> incident = np.array([0.0, 0.1, -1.0])  # Ray going down into water
        >>> normal = np.array([0.0, 0.0, -1.0])    # Points up from water to air
        >>> refracted = snells_law_3d(incident, normal, n_ratio=0.75)  # Air to water
        >>> print(f"Refracted: {refracted}")

    Note:
        For a detailed explanation of the refractive geometry model,
        see the :doc:`Refractive Geometry </guide/refractive_geometry>` guide.

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
    camera: Camera, interface: Interface, pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """
    Trace ray from camera through air-water interface.

    Args:
        camera: Camera object (camera.name must be in interface.camera_distances)
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
        ray_direction, interface.normal, interface.n_ratio_air_to_water
    )

    if refracted is None:  # TIR (shouldn't happen for air->water at normal incidence)
        return None, None

    return intersection, refracted


def refractive_back_project(
    camera: Camera, interface: Interface, pixel: Vec2
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


def _refractive_project_brent(
    camera: Camera, interface: Interface, point_3d: Vec3
) -> Vec2 | None:
    """
    Project 3D underwater point to 2D pixel through refractive interface (Brent-search).

    This is the general-purpose projection that works for any interface normal.
    Uses 1D Brent-search optimization to find the interface intersection point.

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
    z_int = interface.get_water_z(camera.name)

    # Check point is below interface (in water)
    if Q[2] <= z_int:
        return None

    # Check if point is (nearly) on optical axis - use direct projection
    xy_dist = np.sqrt((Q[0] - C[0]) ** 2 + (Q[1] - C[1]) ** 2)
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
        return np.array(
            [C[0] + t * dir_xy[0], C[1] + t * dir_xy[1], z_int], dtype=np.float64
        )

    def compute_refracted_ray(P: Vec3) -> Vec3 | None:
        """Compute refracted ray direction in air from water point Q through P."""
        d_water = P - Q
        d_water_norm = np.linalg.norm(d_water)
        if d_water_norm < 1e-10:
            return None
        d_water = d_water / d_water_norm
        return snells_law_3d(d_water, interface.normal, interface.n_ratio_water_to_air)

    def error_function(t: float) -> float | None:
        """
        Signed error measuring deviation of refracted ray from camera direction.
        Uses the XY component of the cross product for sign.

        Returns:
            Error value, or None if TIR occurs (invalid configuration).
        """
        P = get_interface_point(t)

        d_air = compute_refracted_ray(P)
        if d_air is None:  # TIR
            return None

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

    def error_function_for_brent(t: float) -> float:
        """Wrapper for brentq that returns large values for TIR."""
        err = error_function(t)
        if err is None:
            # Return large value with sign based on position
            # (sign doesn't matter much since we bracket properly now)
            return 1e6 if t > xy_dist else -1e6
        return err

    # Search range: from near camera (t=0) to beyond point projection (t=2*xy_dist)
    t_max = 2.0 * xy_dist

    try:
        # Sample to find bracket, excluding TIR samples
        # Collect valid (non-TIR) samples first
        n_samples = 50
        t_samples = np.linspace(0, t_max, n_samples)
        valid_samples: list[tuple[float, float]] = []

        for t_test in t_samples:
            e_test = error_function(t_test)
            if e_test is not None:
                valid_samples.append((t_test, e_test))

        if len(valid_samples) < 2:
            # Not enough valid samples (mostly TIR)
            return None

        # Check if any sample is already zero (degenerate case)
        t_solution = None
        for t_test, e_test in valid_samples:
            if abs(e_test) < 1e-12:
                t_solution = t_test
                break

        if t_solution is None:
            # Find a bracket with sign change among valid samples
            found_bracket = False
            t_lo, t_hi = 0.0, 0.0

            for i in range(len(valid_samples) - 1):
                t_prev, e_prev = valid_samples[i]
                t_test, e_test = valid_samples[i + 1]
                if e_prev * e_test < 0:
                    t_lo, t_hi = t_prev, t_test
                    found_bracket = True
                    break

            if not found_bracket:
                return None

            # Use brentq to find exact root
            t_solution = brentq(error_function_for_brent, t_lo, t_hi, xtol=1e-9)

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


def _refractive_project_newton(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3,
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> Vec2 | None:
    """
    Project 3D underwater point to 2D pixel through flat refractive interface (Newton-Raphson).

    Uses Newton-Raphson iteration for fast convergence (typically 2-4 iterations).
    Only works for horizontal interface (normal = [0, 0, -1]).
    Caller must check interface orientation before calling.

    Args:
        camera: Camera object
        interface: Interface object (assumes horizontal normal)
        point_3d: 3D point in water (world coordinates, Z > interface_z)
        max_iterations: Maximum Newton iterations (default 10)
        tolerance: Convergence tolerance for r_p (default 1e-9 meters)

    Returns:
        2D pixel coordinates, or None if projection fails.
    """

    C = camera.C
    Q = np.asarray(point_3d, dtype=np.float64)
    z_int = interface.get_water_z(camera.name)
    n_air = interface.n_air
    n_water = interface.n_water

    # Camera should be above interface (smaller Z in Z-down coords)
    h_c = z_int - C[2]  # vertical distance camera to interface
    if h_c <= 0:
        return None  # Camera at or below interface

    # Point should be below interface (larger Z)
    h_q = Q[2] - z_int  # vertical distance interface to point
    if h_q <= 0:
        return None  # Point at or above interface

    # Horizontal distance from camera to point
    dx = Q[0] - C[0]
    dy = Q[1] - C[1]
    r_q = np.sqrt(dx * dx + dy * dy)

    # Special case: point directly below camera
    if r_q < 1e-10:
        P = np.array([C[0], C[1], z_int], dtype=np.float64)
        return camera.project(P, apply_distortion=True)

    # Direction unit vector in XY plane (from camera toward point)
    dir_x = dx / r_q
    dir_y = dy / r_q

    # Initial guess: pinhole projection (straight line intersection)
    r_p = r_q * h_c / (h_c + h_q)

    # Newton-Raphson iteration
    for _ in range(max_iterations):
        # Compute f(r_p) and f'(r_p)
        d_air_sq = r_p * r_p + h_c * h_c
        d_air = np.sqrt(d_air_sq)

        r_q_minus_r_p = r_q - r_p
        d_water_sq = r_q_minus_r_p * r_q_minus_r_p + h_q * h_q
        d_water = np.sqrt(d_water_sq)

        sin_air = r_p / d_air
        sin_water = r_q_minus_r_p / d_water

        f = n_air * sin_air - n_water * sin_water

        # Derivative: f' = n_air * h_c² / d_air³ + n_water * h_q² / d_water³
        f_prime = n_air * h_c * h_c / (d_air_sq * d_air) + n_water * h_q * h_q / (
            d_water_sq * d_water
        )

        # Newton step
        delta = f / f_prime
        r_p = r_p - delta

        # Clamp to valid range
        r_p = max(0.0, min(r_p, r_q))

        if abs(delta) < tolerance:
            break

    # Compute interface point P
    px = C[0] + r_p * dir_x
    py = C[1] + r_p * dir_y
    P = np.array([px, py, z_int], dtype=np.float64)

    # Project P to pixel (the ray from C through P is the observed ray)
    return camera.project(P, apply_distortion=True)


def _refractive_project_newton_batch(
    camera: Camera,
    interface: Interface,
    points_3d: NDArray[np.float64],
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    """
    Project multiple 3D underwater points to 2D pixels (vectorized Newton-Raphson).

    Only works for horizontal interface (normal = [0, 0, -1]).
    Caller must check interface orientation before calling.

    Args:
        camera: Camera object
        interface: Interface object (assumes horizontal normal)
        points_3d: Array of shape (N, 3) with 3D points
        max_iterations: Maximum Newton iterations
        tolerance: Convergence tolerance

    Returns:
        Array of shape (N, 2) with pixel coordinates.
        Invalid projections have NaN values.
    """

    points = np.asarray(points_3d, dtype=np.float64)
    n_points = len(points)
    result = np.full((n_points, 2), np.nan, dtype=np.float64)

    C = camera.C
    z_int = interface.get_water_z(camera.name)
    n_air = interface.n_air
    n_water = interface.n_water

    # Camera should be above interface
    h_c = z_int - C[2]
    if h_c <= 0:
        return result

    # Compute per-point values
    Q = points
    h_q = Q[:, 2] - z_int  # (N,)
    dx = Q[:, 0] - C[0]  # (N,)
    dy = Q[:, 1] - C[1]  # (N,)
    r_q = np.sqrt(dx * dx + dy * dy)  # (N,)

    # Valid points mask: below interface and not directly below camera
    valid = (h_q > 0) & (r_q >= 1e-10)

    # Handle points directly below camera separately
    on_axis = (h_q > 0) & (r_q < 1e-10)
    if np.any(on_axis):
        axis_indices = np.where(on_axis)[0]
        for idx in axis_indices:
            P = np.array([C[0], C[1], z_int], dtype=np.float64)
            px = camera.project(P, apply_distortion=True)
            if px is not None:
                result[idx] = px

    # Process valid off-axis points
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return result

    # Extract valid subset
    h_q_v = h_q[valid]
    dx_v = dx[valid]
    dy_v = dy[valid]
    r_q_v = r_q[valid]

    # Direction unit vectors
    dir_x_v = dx_v / r_q_v
    dir_y_v = dy_v / r_q_v

    # Initial guess: pinhole projection
    r_p_v = r_q_v * h_c / (h_c + h_q_v)

    # Newton-Raphson iteration (vectorized)
    for _ in range(max_iterations):
        d_air_sq = r_p_v * r_p_v + h_c * h_c
        d_air = np.sqrt(d_air_sq)

        r_diff = r_q_v - r_p_v
        d_water_sq = r_diff * r_diff + h_q_v * h_q_v
        d_water = np.sqrt(d_water_sq)

        sin_air = r_p_v / d_air
        sin_water = r_diff / d_water

        f = n_air * sin_air - n_water * sin_water
        f_prime = n_air * h_c * h_c / (d_air_sq * d_air) + n_water * h_q_v * h_q_v / (
            d_water_sq * d_water
        )

        delta = f / f_prime
        r_p_v = r_p_v - delta
        r_p_v = np.clip(r_p_v, 0.0, r_q_v)

        if np.all(np.abs(delta) < tolerance):
            break

    # Compute interface points
    px_v = C[0] + r_p_v * dir_x_v
    py_v = C[1] + r_p_v * dir_y_v

    # Project each point
    for i, idx in enumerate(valid_indices):
        P = np.array([px_v[i], py_v[i], z_int], dtype=np.float64)
        projected = camera.project(P, apply_distortion=True)
        if projected is not None:
            result[idx] = projected

    return result


def _is_flat_interface(normal: Vec3) -> bool:
    """Check if interface normal is approximately [0, 0, -1] (flat horizontal)."""
    return abs(normal[0]) < 1e-6 and abs(normal[1]) < 1e-6 and abs(normal[2] + 1) < 1e-6


def refractive_project(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3,
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> Vec2 | None:
    """
    Project 3D underwater point to 2D pixel through refractive interface.

    Auto-selects the fastest algorithm:
    - Flat interface (normal ≈ [0,0,-1]): Newton-Raphson (2-4 iterations, ~50x faster)
    - General interface: Brent-search fallback

    This is the forward projection used for computing reprojection error.

    Args:
        camera: Camera object
        interface: Interface object
        point_3d: 3D point in water (world coordinates, Z > interface_z)
        max_iterations: Maximum Newton iterations for flat interface (default 10)
        tolerance: Convergence tolerance for flat interface (default 1e-9 meters)

    Returns:
        2D pixel coordinates, or None if projection fails.

    Example:
        >>> import numpy as np
        >>> from aquacal.core.camera import Camera
        >>> from aquacal.core.interface_model import Interface
        >>> # Assuming camera and interface are set up
        >>> point_3d = np.array([0.5, 0.3, 0.8])  # Underwater point
        >>> pixel = refractive_project(camera, interface, point_3d)
        >>> if pixel is not None:
        >>>     print(f"Projected to pixel: {pixel}")

    Note:
        For a detailed explanation of the refractive geometry model,
        see the :doc:`Refractive Geometry </guide/refractive_geometry>` guide.

    Notes:
        - Returns None if: point above interface, TIR, optimization fails,
          or refracted ray doesn't reach camera
    """
    if _is_flat_interface(interface.normal):
        return _refractive_project_newton(
            camera, interface, point_3d, max_iterations, tolerance
        )
    else:
        return _refractive_project_brent(camera, interface, point_3d)


def refractive_project_batch(
    camera: Camera,
    interface: Interface,
    points_3d: NDArray[np.float64],
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    """
    Project multiple 3D underwater points to 2D pixels (vectorized).

    Currently only supports flat interfaces (normal ≈ [0,0,-1]).
    Uses vectorized Newton-Raphson for fast batch projection.

    Args:
        camera: Camera object
        interface: Interface object
        points_3d: Array of shape (N, 3) with 3D points
        max_iterations: Maximum Newton iterations (default 10)
        tolerance: Convergence tolerance (default 1e-9 meters)

    Returns:
        Array of shape (N, 2) with pixel coordinates.
        Invalid projections have NaN values.

    Raises:
        ValueError: If interface normal is not horizontal [0, 0, -1]

    Example:
        >>> import numpy as np
        >>> points = np.array([[0.5, 0.3, 0.8], [0.2, 0.4, 0.9], [0.1, 0.2, 1.0]])
        >>> pixels = refractive_project_batch(camera, interface, points)
        >>> valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]

    Note:
        For a detailed explanation of the refractive geometry model,
        see the :doc:`Refractive Geometry </guide/refractive_geometry>` guide.

    Notes:
        - Batch Brent-search is not implemented. For non-flat interfaces,
          use refractive_project() in a loop instead.
    """
    # Check for flat interface
    if not _is_flat_interface(interface.normal):
        raise ValueError(
            "refractive_project_batch currently only supports flat interfaces "
            "(normal = [0,0,-1]). For tilted interfaces, use refractive_project() "
            "in a loop instead."
        )

    return _refractive_project_newton_batch(
        camera, interface, points_3d, max_iterations, tolerance
    )


# Deprecated backward-compatibility shims
def refractive_project_fast(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3,
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> Vec2 | None:
    """
    Deprecated: use refractive_project() instead.

    refractive_project() auto-selects the fast Newton-Raphson path for flat
    interfaces, making this function redundant.
    """
    import warnings

    warnings.warn(
        "refractive_project_fast() is deprecated. Use refractive_project(), "
        "which auto-selects the fast path for flat interfaces.",
        DeprecationWarning,
        stacklevel=2,
    )
    return refractive_project(camera, interface, point_3d, max_iterations, tolerance)


def refractive_project_fast_batch(
    camera: Camera,
    interface: Interface,
    points_3d: NDArray[np.float64],
    max_iterations: int = 10,
    tolerance: float = 1e-9,
) -> NDArray[np.float64]:
    """
    Deprecated: use refractive_project_batch() instead.

    refractive_project_batch() provides the same functionality with a clearer name.
    """
    import warnings

    warnings.warn(
        "refractive_project_fast_batch() is deprecated. Use refractive_project_batch() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return refractive_project_batch(
        camera, interface, points_3d, max_iterations, tolerance
    )
