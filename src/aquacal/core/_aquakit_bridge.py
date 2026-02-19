"""AquaKit geometry bridge — numpy↔torch conversion layer for AquaKit geometry calls."""

import aquakit
import numpy as np
import torch
from numpy.typing import NDArray

from aquacal.config.schema import Vec2, Vec3
from aquacal.core.camera import Camera

__all__ = [
    "_bridge_snells_law_3d",
    "_bridge_trace_ray_air_to_water",
    "_bridge_refractive_project",
    "_bridge_refractive_back_project",
    "_bridge_ray_plane_intersection",
    "_make_interface_params",
]


# ---------------------------------------------------------------------------
# Private conversion helpers
# ---------------------------------------------------------------------------


def _to_torch(arr: NDArray[np.float64]) -> torch.Tensor:
    """Convert numpy float64 array to float32 torch tensor.

    Args:
        arr: Input numpy array of any shape.

    Returns:
        Float32 torch tensor with the same shape as the input.
    """
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def _to_numpy(t: torch.Tensor) -> NDArray[np.float64]:
    """Convert torch tensor to float64 numpy array.

    Args:
        t: Input torch tensor of any shape.

    Returns:
        Float64 numpy array with the same shape as the input.
    """
    return t.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Factory for AquaKit InterfaceParams
# ---------------------------------------------------------------------------


def _make_interface_params(
    water_z: float, n_air: float, n_water: float
) -> "aquakit.types.InterfaceParams":
    """Build an AquaKit InterfaceParams for the water surface.

    Call sites use this factory rather than importing aquakit.types.InterfaceParams
    directly, avoiding the name collision with AquaCal's own InterfaceParams type.

    Args:
        water_z: Z-coordinate of the water surface in world frame (meters).
        n_air: Refractive index of air.
        n_water: Refractive index of water.

    Returns:
        AquaKit InterfaceParams with a fixed upward-pointing normal [0, 0, -1].
    """
    from aquakit.types import InterfaceParams as _AquaKitInterfaceParams

    return _AquaKitInterfaceParams(
        normal=torch.tensor([0.0, 0.0, -1.0]),
        water_z=water_z,
        n_air=n_air,
        n_water=n_water,
    )


# ---------------------------------------------------------------------------
# Bridge wrappers
# ---------------------------------------------------------------------------


def _bridge_snells_law_3d(
    incident_direction: Vec3,
    surface_normal: Vec3,
    n_ratio: float,
) -> Vec3 | None:
    """Apply Snell's law in 3D via AquaKit, with numpy in / numpy out.

    Converts inputs to torch tensors, delegates to ``aquakit.snells_law_3d``,
    and returns a numpy Vec3 (or None for total internal reflection).

    Note:
        Converts numpy↔torch internally. Callers never interact with torch.

    Args:
        incident_direction: Unit vector of the incoming ray, shape (3,).
        surface_normal: Unit normal of the interface (pass [0, 0, -1] for
            a horizontal air-water interface pointing upward), shape (3,).
        n_ratio: Ratio n1/n2 where the ray travels from medium 1 to medium 2.

    Returns:
        Unit vector of the refracted ray direction, shape (3,), or None if
        total internal reflection occurs.
    """
    incident_t = _to_torch(incident_direction).unsqueeze(0)  # (1, 3)
    normal_t = _to_torch(surface_normal)  # (3,)

    dirs, valid = aquakit.snells_law_3d(incident_t, normal_t, n_ratio)

    if valid[0].item():
        return _to_numpy(dirs[0])
    return None


def _bridge_trace_ray_air_to_water(
    camera: Camera,
    interface_aq: "aquakit.types.InterfaceParams",
    pixel: Vec2,
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """Trace a camera ray through the air-water interface via AquaKit.

    Back-projects the pixel to a world-space ray using ``camera.pixel_to_ray_world``,
    then delegates to ``aquakit.trace_ray_air_to_water``.

    Note:
        Converts numpy↔torch internally. Callers never interact with torch.

    Args:
        camera: AquaCal Camera object used to back-project the pixel.
        interface_aq: AquaKit InterfaceParams describing the water surface.
            Build with ``_make_interface_params()``.
        pixel: 2D pixel coordinates, shape (2,).

    Returns:
        Tuple of (intersection_point, refracted_direction):

        - intersection_point: Where the ray hits the interface (world coords),
          shape (3,).
        - refracted_direction: Unit direction of the ray in water, shape (3,).

        Returns (None, None) if the ray does not hit the interface or TIR
        occurs.
    """
    ray_origin, ray_direction = camera.pixel_to_ray_world(pixel)

    origins_t = _to_torch(ray_origin).unsqueeze(0)  # (1, 3)
    dirs_t = _to_torch(ray_direction).unsqueeze(0)  # (1, 3)

    ipts, rdirs, valid = aquakit.trace_ray_air_to_water(origins_t, dirs_t, interface_aq)

    if valid[0].item():
        return _to_numpy(ipts[0]), _to_numpy(rdirs[0])
    return None, None


def _bridge_refractive_project(
    camera: Camera,
    interface_aq: "aquakit.types.InterfaceParams",
    point_3d: Vec3,
) -> Vec2 | None:
    """Project an underwater 3D point to a pixel via AquaKit (two-step).

    Step 1: ``aquakit.refractive_project`` finds the interface point.
    Step 2: ``camera.project`` maps the interface point to a pixel.

    Note:
        AquaKit's ``refractive_project`` returns interface points, not pixels.
        The pixel is computed by the subsequent ``camera.project`` call, which
        applies full lens distortion. Converts numpy↔torch internally.

    Args:
        camera: AquaCal Camera object used for the final pixel projection.
        interface_aq: AquaKit InterfaceParams describing the water surface.
            Build with ``_make_interface_params()``.
        point_3d: 3D point in water (world coordinates, Z > water_z), shape (3,).

    Returns:
        2D pixel coordinates, shape (2,), or None if the projection fails
        (point above interface, invalid geometry, or camera projects behind
        the image plane).
    """
    points_t = _to_torch(np.asarray(point_3d)).unsqueeze(0)  # (1, 3)
    cam_center_t = _to_torch(camera.C)  # (3,)

    ipts, valid = aquakit.refractive_project(points_t, cam_center_t, interface_aq)

    if not valid[0].item():
        return None

    interface_point = _to_numpy(ipts[0])
    return camera.project(interface_point, apply_distortion=True)


def _bridge_refractive_back_project(
    camera: Camera,
    interface_aq: "aquakit.types.InterfaceParams",
    pixel: Vec2,
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """Back-project a pixel to a ray in water via AquaKit.

    Back-projects the pixel to a world-space ray using
    ``camera.pixel_to_ray_world``, then delegates to
    ``aquakit.refractive_back_project``.

    Note:
        Converts numpy↔torch internally. Callers never interact with torch.

    Args:
        camera: AquaCal Camera object used to back-project the pixel.
        interface_aq: AquaKit InterfaceParams describing the water surface.
            Build with ``_make_interface_params()``.
        pixel: 2D pixel coordinates, shape (2,).

    Returns:
        Tuple of (ray_origin, ray_direction):

        - ray_origin: Point on the interface where the ray enters water,
          shape (3,).
        - ray_direction: Unit direction of the ray in water, shape (3,).

        Returns (None, None) if back-projection fails.
    """
    ray_origin, ray_direction = camera.pixel_to_ray_world(pixel)

    pixel_rays_t = _to_torch(ray_direction).unsqueeze(0)  # (1, 3)
    cam_centers_t = _to_torch(ray_origin).unsqueeze(0)  # (1, 3)

    ipts, wdirs, valid = aquakit.refractive_back_project(
        pixel_rays_t, cam_centers_t, interface_aq
    )

    if valid[0].item():
        return _to_numpy(ipts[0]), _to_numpy(wdirs[0])
    return None, None


def _bridge_ray_plane_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> tuple[Vec3, float] | tuple[None, None]:
    """Compute ray-plane intersection via AquaKit, with numpy in / numpy out.

    Converts the point+normal plane representation to the scalar form
    (plane_d = dot(plane_point, plane_normal)) required by AquaKit, then
    delegates to ``aquakit.ray_plane_intersection``.

    Note:
        Converts numpy↔torch internally. Callers never interact with torch.

    Args:
        ray_origin: Origin of the ray, shape (3,).
        ray_direction: Direction of the ray (need not be unit), shape (3,).
        plane_point: Any point on the plane, shape (3,).
        plane_normal: Normal vector of the plane (need not be unit), shape (3,).

    Returns:
        Tuple of (intersection_point, t) where
        ``intersection = ray_origin + t * ray_direction``, or (None, None) if
        the ray is parallel to the plane.
    """
    # Convert point+normal form to scalar form required by AquaKit.
    plane_d = float(np.dot(plane_point, plane_normal))

    origins_t = _to_torch(ray_origin).unsqueeze(0)  # (1, 3)
    dirs_t = _to_torch(ray_direction).unsqueeze(0)  # (1, 3)
    pn_t = _to_torch(plane_normal)  # (3,)

    pts, valid = aquakit.ray_plane_intersection(origins_t, dirs_t, pn_t, plane_d)

    if not valid[0].item():
        return None, None

    intersection = _to_numpy(pts[0])
    # Recover parametric t from the intersection point.
    denom = max(np.dot(ray_direction, ray_direction), 1e-30)
    t = float(np.dot(intersection - ray_origin, ray_direction) / denom)
    return intersection, t
