"""Core geometry modules."""

from aquacal.core._aquakit_bridge import (
    _bridge_ray_plane_intersection as ray_plane_intersection,
)
from aquacal.core._aquakit_bridge import (
    _bridge_refractive_back_project as refractive_back_project,
)
from aquacal.core._aquakit_bridge import (
    _bridge_refractive_project as refractive_project,
)
from aquacal.core._aquakit_bridge import (
    _bridge_snells_law_3d as snells_law_3d,
)
from aquacal.core._aquakit_bridge import (
    _bridge_trace_ray_air_to_water as trace_ray_air_to_water,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera, undistort_points
from aquacal.core.interface_model import Interface

__all__ = [
    "BoardGeometry",
    "Camera",
    "undistort_points",
    "Interface",
    "ray_plane_intersection",
    "snells_law_3d",
    "trace_ray_air_to_water",
    "refractive_project",
    "refractive_back_project",
]
