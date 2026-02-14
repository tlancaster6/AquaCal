# Coding Conventions

**Analysis Date:** 2026-02-14

## Naming Patterns

**Files:**
- Snake case: `interface_estimation.py`, `refractive_geometry.py`, `_optim_common.py`
- Leading underscore for internal modules: `_optim_common.py` (private, used only by calibration stages)
- All lowercase, no hyphens

**Functions:**
- Snake case: `pack_params`, `unpack_params`, `compute_residuals`, `snells_law_3d`
- Leading underscore for internal helpers: `_compute_initial_board_poses`, `_refractive_project_brent`, `_multi_frame_pnp_init`
- Clear, verb-first when performing actions: `trace_ray_air_to_water`, `refractive_back_project`

**Variables:**
- Snake case for all local and module-level variables: `board_poses`, `camera_order`, `frame_order`, `jac_sparsity`, `residuals`
- Abbreviations preserved when meaningful: `tvec`, `rvec`, `K` (intrinsic matrix), `R` (rotation), `t` (translation), `C` (camera center), `n_ratio`, `cos_i`
- Coordinate subscripts for domain clarity: `point_world`, `point_3d`, `ray_origin`, `ray_direction`, `interface_point`

**Types/Classes:**
- PascalCase: `Camera`, `CameraIntrinsics`, `CameraExtrinsics`, `BoardGeometry`, `Interface`, `Detection`, `FrameDetections`
- Dataclasses: `BoardPose`, `DetectionResult`, `CalibrationResult`, `CameraCalibration`, `InterfaceParams`
- Exception classes: `CalibrationError`, `InsufficientDataError`, `ConvergenceError`

**Constants:**
- UPPER_SNAKE_CASE: Not commonly seen in codebase (mostly in type aliases like `Vec3 = NDArray[np.float64]`)

## Code Style

**Formatting:**
- Tool: Black (configured but settings not in repo root; defaults applied)
- Line length: Default Black (88 characters)
- Imports formatted: stdlib, third-party, local with blank lines between groups

**Linting:**
- Tool: mypy (type checking)
- Config: `types-PyYAML` included for type stubs
- Enforced in dependencies via `pyproject.toml`

**Module-level docstrings:**
- Required on every `.py` file as first statement
- One-line summary of module purpose
- Example: `"""Stage 3 refractive optimization for interface estimation."""`
- Multi-line docstrings include fuller explanation after blank line

## Import Organization

**Order (observed throughout codebase):**
1. Standard library: `import numpy`, `import cv2`, `from pathlib import Path`
2. Third-party: `import numpy as np`, `from scipy.optimize import least_squares`
3. Local (aquacal): `from aquacal.config.schema import ...`

**Path aliases:**
- None used; all imports use full `aquacal.*` module paths
- Type aliases defined in `aquacal/config/schema.py`: `Vec3`, `Mat3`, `Vec2`
- Imports these type aliases directly where needed: `from aquacal.config.schema import Vec3, Vec2, Mat3`

**Circular dependency avoidance:**
- Core modules (`config/schema.py`, `core/camera.py`, `core/refractive_geometry.py`) are foundational
- Calibration stages import from core, not vice versa
- Test files use relative imports with `sys.path.insert(0, ".")` when needed: See `tests/unit/test_interface_estimation.py` line 31

## Error Handling

**Patterns:**
- Specific exception types defined in `aquacal/config/schema.py`: `CalibrationError`, `InsufficientDataError`, `ConvergenceError`
- Catch and re-raise with context:
  ```python
  try:
      result = cv2.calibrateCamera(...)
  except cv2.error as e:
      raise ValueError(f"OpenCV calibration failed: {e}")
  ```
  (See `src/aquacal/calibration/intrinsics.py` lines 296-306)

- Defensive checks before expensive operations:
  ```python
  if result is None:
      return None, None
  if det.num_corners < min_corners:
      continue
  ```
  (Seen in `interface_estimation.py`, `refractive_geometry.py`)

- Validation at entry points:
  ```python
  if not video_paths:
      raise ValueError("video_paths cannot be empty")
  if len(results) < 2:
      raise ValueError(f"Need at least 2 results to compare, got {len(results)}")
  ```

## Logging

**Framework:** No centralized logging library used; relies on print() and CLI feedback
- Standard error handling via exceptions
- CLI (`cli.py`) catches exceptions and prints user-facing messages
- Verbose flag (`-v`) passed through pipeline for debug output (mechanism in place but not widely used)
- Examples in `cli.py` lines 152-180: try-except blocks with print error messages

## Comments

**When to Comment:**
- Not heavily commented; code is self-documenting through function/variable names
- Comments appear for:
  - Algorithm explanations (e.g., Snell's law derivation in `refractive_geometry.py`)
  - Non-obvious mathematical operations (e.g., Rodrigues rotation vector handling)
  - Coordinate system clarifications (e.g., interface normal orientation)

**JSDoc/TSDoc:**
- Google-style docstrings on all public functions and classes
- All parameters documented with type and semantics
- Return type and shape information included
- Examples provided for utility functions

**Example (from `transforms.py` lines 12-28):**
```python
def rvec_to_matrix(rvec: Vec3) -> Mat3:
    """
    Convert Rodrigues vector to rotation matrix.

    Args:
        rvec: Rotation vector, shape (3,). The axis is rvec/||rvec|| and
            the angle is ||rvec|| in radians.

    Returns:
        R: Rotation matrix, shape (3, 3)

    Example:
        >>> rvec = np.array([0.0, 0.0, np.pi/2])
        >>> R = rvec_to_matrix(rvec)
        >>> np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]))
        True
    """
```

## Function Design

**Size:** Most functions 20-80 lines; optimization helpers in `_optim_common.py` reach 150+ lines
- Monolithic functions acceptable when logically cohesive (e.g., `pack_params`, `unpack_params` handle structured transformation)
- Helpers split out when reused: `_compute_initial_board_poses`, `make_sparse_jacobian_func`

**Parameters:**
- Type hints on all parameters: `def pack_params(extrinsics: dict[str, CameraExtrinsics], water_z: float, ...)`
- Return type annotations on all functions
- Use dict for mappings: `dict[str, CameraIntrinsics]` (camera names to objects)
- Use list for ordered sequences: `list[str]` (camera order), `list[int]` (frame indices)
- Optional parameters default to None: `board_poses: dict[int, BoardPose] | None = None`

**Return Values:**
- Explicit tuple types: `tuple[Vec3, Vec3]` for ray origin/direction pairs
- Single-value returns unnested: `def num_corners(self) -> int:` not `-> tuple[int]`
- Union types for conditional returns: `Vec2 | None` when projection may fail
- Multiple returns as tuples: `return R_combined, t_combined` (composed poses)

## Module Design

**Exports:**
- Every package has `__init__.py` with explicit `__all__` list
- Public API in `aquacal/calibration/__init__.py` line 3-17: Imports key functions like `calibrate_intrinsics_single`, `optimize_interface`, `joint_refinement`
- Internal modules (prefixed `_`) not exported: `_optim_common`, `_numdiff` behavior internal to package

**Barrel Files:**
- Pattern used: `src/aquacal/calibration/__init__.py` re-exports public calibration functions
- Pattern used: `src/aquacal/config/__init__.py` re-exports all schema types

**Example (`src/aquacal/calibration/__init__.py` lines 1-17):**
```python
"""Calibration pipeline modules."""

from aquacal.calibration.intrinsics import (
    calibrate_intrinsics_single,
    calibrate_intrinsics_all,
)
from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    estimate_board_pose,
    refractive_solve_pnp,
    build_pose_graph,
    estimate_extrinsics,
)
from aquacal.calibration.interface_estimation import (
    optimize_interface,
)
from aquacal.calibration.refinement import (
    joint_refinement,
)

__all__ = [
    "calibrate_intrinsics_single",
    "calibrate_intrinsics_all",
    "Observation",
    "PoseGraph",
    "estimate_board_pose",
    "refractive_solve_pnp",
    "build_pose_graph",
    "estimate_extrinsics",
    "optimize_interface",
    "joint_refinement",
]
```

## Type Hints

**Usage:**
- Type hints on all public function signatures
- Numpy typing via `numpy.typing.NDArray` with shapes in docstrings: `NDArray[np.float64]  # shape (3,)`
- Type aliases at module top for readability: `Vec3 = NDArray[np.float64]  # shape (3,)`
- Collections use modern syntax: `dict[str, CameraIntrinsics]` not `Dict[str, ...]`

**Shape documentation:**
- Shapes provided in docstrings, not type hints (to match numpy convention)
- Example (`camera.py` line 81-91):
```python
def world_to_camera(self, point_world: Vec3) -> Vec3:
    """
    Transform point from world to camera coordinates.

    Args:
        point_world: 3D point in world frame, shape (3,)

    Returns:
        3D point in camera frame, shape (3,)

    Formula: p_cam = R @ p_world + t
    """
```

---

*Convention analysis: 2026-02-14*
