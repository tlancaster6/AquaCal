"""Core geometry modules."""

from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera, undistort_points
from aquacal.core.interface_model import Interface, ray_plane_intersection
from aquacal.core.refractive_geometry import (
    snells_law_3d,
    trace_ray_air_to_water,
    refractive_project,
    refractive_project_batch,
    refractive_back_project,
)

__all__ = [
    "BoardGeometry",
    "Camera",
    "undistort_points",
    "Interface",
    "ray_plane_intersection",
    "snells_law_3d",
    "trace_ray_air_to_water",
    "refractive_project",
    "refractive_project_batch",
    "refractive_back_project",
]
