"""Refractive interface (water surface) model."""

import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import Vec3


class Interface:
    """
    Planar refractive interface (air-water boundary).

    The interface is a horizontal plane at a specific Z-coordinate in the world frame.
    Per-camera offsets allow for cameras at slightly different heights.

    Attributes:
        normal: Unit normal vector pointing from water toward air [0, 0, -1]
        base_height: Z-coordinate of interface plane for reference camera
        camera_offsets: Per-camera adjustments to base_height
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)
    """

    def __init__(
        self,
        normal: Vec3,
        base_height: float,
        camera_offsets: dict[str, float],
        n_air: float = 1.0,
        n_water: float = 1.333
    ):
        """
        Initialize interface.

        Args:
            normal: Unit normal vector pointing from water to air (typically [0,0,-1])
            base_height: Z-coordinate of interface plane in world frame (e.g., 0.15 means
                        interface is at Z=0.15m, which is 0.15m below a camera at Z=0)
            camera_offsets: Per-camera offset from base_height. Reference camera should
                           have offset=0. Positive offset means interface is further from
                           that camera.
            n_air: Refractive index of air
            n_water: Refractive index of water
        """
        self.normal = normal / np.linalg.norm(normal)  # Ensure unit vector
        self.base_height = base_height
        self.camera_offsets = camera_offsets
        self.n_air = n_air
        self.n_water = n_water

    def get_interface_distance(self, camera_name: str) -> float:
        """
        Get the effective interface Z-coordinate for a specific camera.

        This equals the distance from camera to interface IF the camera is at Z=0.

        Args:
            camera_name: Name of camera

        Returns:
            base_height + camera_offsets[camera_name]

        Raises:
            KeyError: If camera_name not in camera_offsets
        """
        return self.base_height + self.camera_offsets[camera_name]

    def get_interface_point(self, camera_center: Vec3, camera_name: str) -> Vec3:
        """
        Get the point on the interface directly below the camera center.

        Assumes cameras look straight down (+Z direction in Z-down world frame).

        Args:
            camera_center: Camera center in world coordinates [x, y, z]
            camera_name: Name of camera (for offset lookup)

        Returns:
            3D point on interface plane [x, y, z_interface]

        Note:
            Only uses camera_center[0] and camera_center[1] (XY position).
            The Z-coordinate is determined by base_height + offset, not by
            camera_center[2]. This is intentional — the interface is at a
            fixed world Z position.
        """
        z_interface = self.base_height + self.camera_offsets[camera_name]
        return np.array([camera_center[0], camera_center[1], z_interface], dtype=np.float64)

    @property
    def n_ratio_air_to_water(self) -> float:
        """Ratio n_air / n_water for Snell's law (air to water)."""
        return self.n_air / self.n_water

    @property
    def n_ratio_water_to_air(self) -> float:
        """Ratio n_water / n_air for Snell's law (water to air)."""
        return self.n_water / self.n_air


def ray_plane_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3
) -> tuple[Vec3, float] | tuple[None, None]:
    """
    Compute intersection of ray with plane.

    Uses the parametric ray equation: P = origin + t * direction
    And plane equation: (P - plane_point) · plane_normal = 0

    Solving: t = ((plane_point - origin) · normal) / (direction · normal)

    Args:
        ray_origin: Origin of ray, shape (3,)
        ray_direction: Direction of ray (need not be unit), shape (3,)
        plane_point: Any point on the plane, shape (3,)
        plane_normal: Normal vector of plane (need not be unit), shape (3,)

    Returns:
        Tuple of (intersection_point, t) where intersection = origin + t * direction.
        Returns (None, None) if ray is parallel to plane (direction · normal ≈ 0).

    Notes:
        - Returns intersection for ANY t value, including negative (behind ray origin)
        - Caller should check t > 0 if only forward intersections are desired
        - Uses tolerance of 1e-10 for parallel check
    """
    # Compute denominator: direction · normal
    denom = np.dot(ray_direction, plane_normal)

    # Check if ray is parallel to plane
    if abs(denom) < 1e-10:
        return None, None

    # Compute t: ((plane_point - origin) · normal) / (direction · normal)
    t = np.dot(plane_point - ray_origin, plane_normal) / denom

    # Compute intersection point
    intersection = ray_origin + t * ray_direction

    return intersection, t
