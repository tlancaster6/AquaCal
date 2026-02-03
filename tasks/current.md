# Task: 2.3 Implement refractive geometry

## Objective

Implement `core/refractive_geometry.py` with Snell's law, ray tracing through the air-water interface, and refractive projection/back-projection functions.

## Context Files

Read these files before starting (in order):

1. `CLAUDE.md` — Project conventions (Z-down world frame, interface normal [0,0,-1])
2. `docs/agent_implementation_spec.md` (lines 761-924) — Function specifications
3. `docs/COORDINATES.md` — Coordinate frame definitions
4. `src/aquacal/core/camera.py` — Camera class (dependency)
5. `src/aquacal/core/interface_model.py` — Interface class, ray_plane_intersection (dependency)

## Dependencies

- Task 2.1 (`core/camera.py`) — Camera class
- Task 2.2 (`core/interface_model.py`) — Interface class, ray_plane_intersection

## Modify

Files to create:

- `src/aquacal/core/refractive_geometry.py`
- `tests/unit/test_refractive_geometry.py`

Files to update:

- `src/aquacal/core/__init__.py` — Add exports for all 4 functions

## Do Not Modify

- `config/schema.py`, `utils/transforms.py`, `core/board.py`, `core/camera.py`, `core/interface_model.py`

## Acceptance Criteria

- [ ] `snells_law_3d` implemented with automatic normal orientation handling
- [ ] `snells_law_3d` returns `None` for total internal reflection
- [ ] `trace_ray_air_to_water` traces pixel through interface into water
- [ ] `refractive_back_project` delegates to `trace_ray_air_to_water`
- [ ] `refractive_project` uses iterative optimization to find interface intersection
- [ ] Round-trip test: `refractive_project` then `refractive_back_project` recovers ray through original point
- [ ] Tests pass: `pytest tests/unit/test_refractive_geometry.py -v`
- [ ] Type check passes: `mypy src/aquacal/core/refractive_geometry.py --ignore-missing-imports`
- [ ] `core/__init__.py` exports all 4 functions

## Function Specifications

### snells_law_3d

```python
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
        - For air-to-water: n_ratio = n_air / n_water ≈ 0.75
        - For water-to-air: n_ratio = n_water / n_air ≈ 1.33
        - TIR only possible when going from denser to less dense medium
    """
```

### trace_ray_air_to_water

```python
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
```

### refractive_back_project

```python
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
```

### refractive_project

```python
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
```

## Implementation Details

### snells_law_3d

```python
def snells_law_3d(
    incident_direction: Vec3,
    surface_normal: Vec3,
    n_ratio: float
) -> Vec3 | None:
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

    # Apply Snell's law: n1*sin(θ1) = n2*sin(θ2)
    # sin²(θ_t) = n_ratio² * sin²(θ_i) = n_ratio² * (1 - cos²(θ_i))
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
```

### trace_ray_air_to_water

```python
def trace_ray_air_to_water(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    # Get ray from camera in world frame
    ray_origin, ray_direction = camera.pixel_to_ray_world(pixel)

    # Get interface plane point for this camera
    interface_point = interface.get_interface_point(camera.C, camera.name)

    # Intersect ray with interface plane
    intersection, t = ray_plane_intersection(
        ray_origin, ray_direction, interface_point, interface.normal
    )

    # Check for valid intersection (ray must hit interface in forward direction)
    if intersection is None or t <= 0:
        return None, None

    # Refract at interface (air to water)
    refracted = snells_law_3d(
        ray_direction,
        interface.normal,
        interface.n_ratio_air_to_water
    )

    if refracted is None:  # TIR (shouldn't happen for air->water at normal incidence)
        return None, None

    return intersection, refracted
```

### refractive_back_project

```python
def refractive_back_project(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    # Delegates to trace_ray_air_to_water
    return trace_ray_air_to_water(camera, interface, pixel)
```

### refractive_project

```python
from scipy.optimize import brentq

def refractive_project(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3
) -> Vec2 | None:
    """
    Algorithm:
    1. Parameterize interface point P along line from camera XY to point XY
    2. Find parameter s where refracted ray from point_3d through P points to camera
    3. Project along the air ray to get pixel coordinates
    """
    C = camera.C
    Q = point_3d
    z_int = interface.get_interface_distance(camera.name)

    # Check point is below interface (in water)
    if Q[2] <= z_int:
        return None

    # Parameterize interface point: P(s) interpolates XY from C to Q at z=z_int
    # s=0 -> directly below camera, s=1 -> directly above point
    def get_interface_point(s: float) -> Vec3:
        px = C[0] + s * (Q[0] - C[0])
        py = C[1] + s * (Q[1] - C[1])
        return np.array([px, py, z_int], dtype=np.float64)

    def error_function(s: float) -> float:
        """
        Returns signed error: positive if refracted ray is "left" of camera,
        negative if "right". Zero when ray points exactly at camera.
        """
        P = get_interface_point(s)

        # Direction from Q to P (in water, going up toward interface)
        d_water = P - Q
        d_water_norm = np.linalg.norm(d_water)
        if d_water_norm < 1e-10:
            return 0.0
        d_water = d_water / d_water_norm

        # Refract at interface (water to air)
        d_air = snells_law_3d(d_water, interface.normal, interface.n_ratio_water_to_air)
        if d_air is None:  # TIR
            # Return large error to push optimization away from this region
            return 1e6 if s > 0.5 else -1e6

        # Vector from P to camera
        to_camera = C - P
        to_camera_norm = np.linalg.norm(to_camera)
        if to_camera_norm < 1e-10:
            return 0.0
        to_camera = to_camera / to_camera_norm

        # Error: cross product magnitude (signed by z-component)
        # If d_air and to_camera are aligned, cross product is zero
        cross = np.cross(d_air, to_camera)
        return cross[2]  # Use Z component as signed error

    # Find bracket for root finding
    # Try s in [0, 2] to allow some extrapolation
    try:
        # Check if solution exists in bracket
        e0 = error_function(0.0)
        e1 = error_function(1.0)
        e2 = error_function(2.0)

        # Find bracket with sign change
        if e0 * e1 < 0:
            s_lo, s_hi = 0.0, 1.0
        elif e1 * e2 < 0:
            s_lo, s_hi = 1.0, 2.0
        elif e0 * e2 < 0:
            s_lo, s_hi = 0.0, 2.0
        else:
            # No sign change found, try to find one
            # Sample more points
            for s_test in np.linspace(0, 2, 21):
                e_test = error_function(s_test)
                if e0 * e_test < 0:
                    s_lo, s_hi = 0.0, s_test
                    break
            else:
                return None  # Could not find valid bracket

        # Solve for s where error = 0
        s_solution = brentq(error_function, s_lo, s_hi, xtol=1e-9)

    except (ValueError, RuntimeError):
        return None

    # Get the interface point and refracted ray
    P = get_interface_point(s_solution)
    d_water = P - Q
    d_water = d_water / np.linalg.norm(d_water)
    d_air = snells_law_3d(d_water, interface.normal, interface.n_ratio_water_to_air)

    if d_air is None:
        return None

    # Project a point along the air ray to get pixel
    # Use P + d_air (any point along ray works for projection direction)
    # But we need a point in front of camera, so go toward camera
    point_in_air = P - 0.01 * d_air  # Small step toward camera (d_air points away from camera)

    # Actually, d_air points from P toward... let's check
    # d_water points from Q toward P (upward, -Z direction)
    # After refraction, d_air continues upward toward camera
    # So to get a point between P and C, we add d_air * small_positive
    # But we need the ray direction toward camera, which is -d_air

    # Simpler: just project P onto the image (P is on the interface, visible to camera)
    # The pixel is where the ray from C through P lands... but that's without distortion consideration

    # Best approach: project the underwater point Q using the found interface point
    # We know the ray path: Q -> P -> C
    # The pixel is determined by the direction from C toward P
    ray_to_interface = P - C
    ray_to_interface = ray_to_interface / np.linalg.norm(ray_to_interface)

    # Find where this ray (from C in direction ray_to_interface) projects
    # Use a point along this ray for projection
    point_for_projection = C + ray_to_interface  # 1 meter along ray

    return camera.project(point_for_projection, apply_distortion=True)
```

## Test Cases

```python
import numpy as np
import pytest
from aquacal.config.schema import CameraIntrinsics, CameraExtrinsics
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import (
    snells_law_3d,
    trace_ray_air_to_water,
    refractive_back_project,
    refractive_project,
)


@pytest.fixture
def simple_camera():
    """Camera at origin looking down +Z."""
    intrinsics = CameraIntrinsics(
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(5),
        image_size=(640, 480)
    )
    extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
    return Camera("cam0", intrinsics, extrinsics)


@pytest.fixture
def simple_interface():
    """Horizontal interface at Z=0.15."""
    return Interface(
        normal=np.array([0, 0, -1]),
        base_height=0.15,
        camera_offsets={'cam0': 0.0},
        n_air=1.0,
        n_water=1.333
    )


class TestSnellsLaw3D:
    def test_normal_incidence(self):
        """Ray perpendicular to surface passes straight through."""
        incident = np.array([0, 0, 1])  # Down (+Z)
        normal = np.array([0, 0, -1])   # Up (water to air)
        n_ratio = 1.0 / 1.333

        refracted = snells_law_3d(incident, normal, n_ratio)

        assert refracted is not None
        np.testing.assert_allclose(refracted, np.array([0, 0, 1]), atol=1e-10)

    def test_bends_toward_normal_air_to_water(self):
        """Ray entering water bends toward normal (steeper)."""
        # 30 degrees from vertical
        incident = np.array([0.5, 0, np.sqrt(0.75)])
        incident = incident / np.linalg.norm(incident)
        normal = np.array([0, 0, -1])
        n_ratio = 1.0 / 1.333

        refracted = snells_law_3d(incident, normal, n_ratio)

        assert refracted is not None
        # Refracted ray should be more vertical (larger Z component)
        assert abs(refracted[2]) > abs(incident[2])
        # X component should be smaller (bent toward normal)
        assert abs(refracted[0]) < abs(incident[0])

    def test_bends_away_from_normal_water_to_air(self):
        """Ray exiting water bends away from normal."""
        # Ray going up from water (negative Z direction)
        incident = np.array([0.2, 0, -np.sqrt(1 - 0.04)])
        incident = incident / np.linalg.norm(incident)
        normal = np.array([0, 0, -1])
        n_ratio = 1.333 / 1.0

        refracted = snells_law_3d(incident, normal, n_ratio)

        assert refracted is not None
        # X component should be larger (bent away from normal)
        assert abs(refracted[0]) > abs(incident[0])

    def test_total_internal_reflection(self):
        """Steep angle from water to air causes TIR."""
        # Critical angle ~48.6 degrees, use 60 degrees
        angle = np.radians(60)
        incident = np.array([np.sin(angle), 0, -np.cos(angle)])
        normal = np.array([0, 0, -1])
        n_ratio = 1.333 / 1.0

        refracted = snells_law_3d(incident, normal, n_ratio)

        assert refracted is None

    def test_no_tir_air_to_water(self):
        """TIR cannot occur when entering denser medium."""
        # Even at grazing angle, should not get TIR
        incident = np.array([0.99, 0, 0.14])  # Very steep
        incident = incident / np.linalg.norm(incident)
        normal = np.array([0, 0, -1])
        n_ratio = 1.0 / 1.333

        refracted = snells_law_3d(incident, normal, n_ratio)

        assert refracted is not None

    def test_output_is_unit_vector(self):
        """Refracted direction should be unit vector."""
        incident = np.array([0.3, 0.2, 0.8])
        incident = incident / np.linalg.norm(incident)
        normal = np.array([0, 0, -1])

        refracted = snells_law_3d(incident, normal, 1.0 / 1.333)

        assert refracted is not None
        np.testing.assert_allclose(np.linalg.norm(refracted), 1.0, atol=1e-10)


class TestTraceRayAirToWater:
    def test_center_pixel_goes_straight_down(self, simple_camera, simple_interface):
        """Principal point ray goes straight down through interface."""
        pixel = np.array([320, 240])

        intersection, direction = trace_ray_air_to_water(
            simple_camera, simple_interface, pixel
        )

        assert intersection is not None
        # Intersection at interface height
        np.testing.assert_allclose(intersection[2], 0.15, atol=1e-10)
        # Intersection directly below camera (camera at origin)
        np.testing.assert_allclose(intersection[:2], [0, 0], atol=1e-10)
        # Direction straight down
        np.testing.assert_allclose(direction, [0, 0, 1], atol=1e-10)

    def test_offset_pixel_refracts(self, simple_camera, simple_interface):
        """Off-center pixel refracts at interface."""
        pixel = np.array([420, 240])  # Right of center

        intersection, direction = trace_ray_air_to_water(
            simple_camera, simple_interface, pixel
        )

        assert intersection is not None
        # Intersection should be right of center
        assert intersection[0] > 0
        # Direction should point down-right, but more vertical than air ray
        assert direction[0] > 0  # Has rightward component
        assert direction[2] > 0  # Points into water (+Z)

    def test_returns_unit_direction(self, simple_camera, simple_interface):
        """Refracted direction should be unit vector."""
        pixel = np.array([400, 300])

        intersection, direction = trace_ray_air_to_water(
            simple_camera, simple_interface, pixel
        )

        assert direction is not None
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-10)


class TestRefractiveBackProject:
    def test_same_as_trace_ray(self, simple_camera, simple_interface):
        """refractive_back_project should match trace_ray_air_to_water."""
        pixel = np.array([350, 260])

        trace_result = trace_ray_air_to_water(simple_camera, simple_interface, pixel)
        back_result = refractive_back_project(simple_camera, simple_interface, pixel)

        if trace_result[0] is not None:
            np.testing.assert_allclose(back_result[0], trace_result[0])
            np.testing.assert_allclose(back_result[1], trace_result[1])


class TestRefractiveProject:
    def test_point_on_optical_axis(self, simple_camera, simple_interface):
        """Point directly below camera projects to principal point."""
        point = np.array([0, 0, 0.5])  # Below interface at z=0.15

        pixel = refractive_project(simple_camera, simple_interface, point)

        assert pixel is not None
        np.testing.assert_allclose(pixel, [320, 240], atol=0.1)

    def test_offset_point(self, simple_camera, simple_interface):
        """Offset underwater point projects away from principal point."""
        point = np.array([0.1, 0, 0.5])  # Right of center, underwater

        pixel = refractive_project(simple_camera, simple_interface, point)

        assert pixel is not None
        assert pixel[0] > 320  # Projects right of center

    def test_point_above_interface_returns_none(self, simple_camera, simple_interface):
        """Point above interface (in air) returns None."""
        point = np.array([0, 0, 0.1])  # Above interface at z=0.15

        pixel = refractive_project(simple_camera, simple_interface, point)

        assert pixel is None

    def test_round_trip_consistency(self, simple_camera, simple_interface):
        """Project then back-project should give ray through original point."""
        # Underwater point
        point = np.array([0.05, 0.03, 0.4])

        # Project to pixel
        pixel = refractive_project(simple_camera, simple_interface, point)
        assert pixel is not None

        # Back-project to ray
        origin, direction = refractive_back_project(
            simple_camera, simple_interface, pixel
        )
        assert origin is not None

        # Ray should pass near original point
        # Find closest point on ray to original point
        t = np.dot(point - origin, direction)
        closest = origin + t * direction

        np.testing.assert_allclose(closest, point, atol=1e-4)


class TestRefractiveProjectEdgeCases:
    def test_point_at_various_depths(self, simple_camera, simple_interface):
        """Test projection at various water depths."""
        for depth in [0.2, 0.5, 1.0, 2.0]:
            point = np.array([0.05, 0.02, depth])
            pixel = refractive_project(simple_camera, simple_interface, point)
            assert pixel is not None, f"Failed at depth {depth}"

    def test_point_at_various_offsets(self, simple_camera, simple_interface):
        """Test projection at various lateral offsets."""
        for offset in [0.0, 0.05, 0.1, 0.2]:
            point = np.array([offset, 0, 0.5])
            pixel = refractive_project(simple_camera, simple_interface, point)
            assert pixel is not None, f"Failed at offset {offset}"
```

## Import Structure

Update `core/__init__.py`:

```python
"""Core geometry modules."""

from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera, undistort_points
from aquacal.core.interface_model import Interface, ray_plane_intersection
from aquacal.core.refractive_geometry import (
    snells_law_3d,
    trace_ray_air_to_water,
    refractive_project,
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
    "refractive_back_project",
]
```

## Notes

### Coordinate Frame Reminder (Z-down)

- World frame: +Z points down (into water)
- Camera at Z ≈ 0, interface at Z = 0.15, underwater points at Z > 0.15
- Interface normal `[0, 0, -1]` points up (from water toward air)
- Ray going into water has `direction[2] > 0`

### refractive_project Algorithm Details

The algorithm uses Brent's method (1D root finding) because:
1. Cameras look nearly straight down, so the interface intersection is approximately along the line from camera to point
2. The error function (cross product of refracted ray and camera direction) is smooth and monotonic in the valid region
3. Brent's method is robust and fast for 1D problems

The parameterization `P(s) = lerp(C_xy, Q_xy, s)` at `z=z_interface` means:
- s=0: interface point directly below camera
- s=1: interface point directly above underwater point
- s∈(0,1): typical valid range
- s>1: extrapolation for wide-angle views
