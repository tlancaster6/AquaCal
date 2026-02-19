"""Refractive interface (water surface) model."""

import numpy as np

from aquacal.config.schema import Vec3


class Interface:
    """
    Planar refractive interface (air-water boundary).

    The interface is a horizontal plane at a fixed Z-coordinate in the world frame.

    Attributes:
        normal: Unit normal vector pointing from water toward air [0, 0, -1]
        camera_distances: Per-camera Z-coordinate of the water surface in world frame.
            After optimization this is the same value (water_z) for all cameras.
            The physical camera-to-water gap is computed internally by projection
            functions as ``water_z - C_z``.
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)
    """

    def __init__(
        self,
        normal: Vec3,
        camera_distances: dict[str, float],
        n_air: float = 1.0,
        n_water: float = 1.333,
    ):
        """
        Initialize interface.

        Args:
            normal: Unit normal vector pointing from water to air (typically [0,0,-1])
            camera_distances: Per-camera Z-coordinate of the water surface in world frame.
                            Typically the same value (water_z) for all cameras.
            n_air: Refractive index of air
            n_water: Refractive index of water
        """
        self.normal = normal / np.linalg.norm(normal)  # Ensure unit vector
        self.camera_distances = camera_distances
        self.n_air = n_air
        self.n_water = n_water

    def get_water_z(self, camera_name: str) -> float:
        """
        Get the water surface Z-coordinate for a specific camera.

        This is the Z-coordinate of the interface plane in world frame.
        The physical camera-to-water gap is ``z_interface - C_z``, computed
        internally by the projection functions.

        Args:
            camera_name: Name of camera

        Returns:
            Water surface Z-coordinate for the specified camera

        Raises:
            KeyError: If camera_name not in camera_distances
        """
        return self.camera_distances[camera_name]

    def get_interface_point(self, camera_center: Vec3, camera_name: str) -> Vec3:
        """
        Get the point on the interface directly below the camera center.

        Assumes cameras look straight down (+Z direction in Z-down world frame).

        Args:
            camera_center: Camera center in world coordinates [x, y, z]
            camera_name: Name of camera (for distance lookup)

        Returns:
            3D point on interface plane [x, y, z_interface]

        Note:
            Only uses camera_center[0] and camera_center[1] (XY position).
            The Z-coordinate is determined by the camera's interface distance,
            not by camera_center[2]. This is intentional — the interface is at a
            fixed world Z position.
        """
        z_interface = self.camera_distances[camera_name]
        return np.array(
            [camera_center[0], camera_center[1], z_interface], dtype=np.float64
        )

    @property
    def n_ratio_air_to_water(self) -> float:
        """Ratio n_air / n_water for Snell's law (air to water)."""
        return self.n_air / self.n_water

    @property
    def n_ratio_water_to_air(self) -> float:
        """Ratio n_water / n_air for Snell's law (water to air)."""
        return self.n_water / self.n_air


# DEPRECATED: use _bridge_ray_plane_intersection from core._aquakit_bridge
def ray_plane_intersection(
    ray_origin: Vec3, ray_direction: Vec3, plane_point: Vec3, plane_normal: Vec3
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
