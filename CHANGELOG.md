# Changelog

All notable changes to this project will be documented in this file.

Format: Agents append entries at the top (below this header) with the date, files modified, and a brief summary.

---

<!-- Agents: add new entries below this line, above previous entries -->

## 2026-02-03
### [src/aquacal/core/refractive_geometry.py]
- Implemented snells_law_3d: 3D Snell's law with automatic normal orientation handling, returns None for TIR
- Implemented trace_ray_air_to_water: traces ray from camera pixel through air-water interface into water
- Implemented refractive_back_project: convenience wrapper for trace_ray_air_to_water with consistent API
- Implemented refractive_project: forward projection using 1D Brent's method optimization to find interface point
- Special handling for optical axis case (point directly below camera) to avoid degenerate parameterization

### [tests/unit/test_refractive_geometry.py]
- Created comprehensive test suite with 26 tests covering all functions
- Tests verify: Snell's law physics (normal incidence, bending toward/away from normal, TIR, unit vectors, symmetry)
- Tests verify: trace_ray_air_to_water (center pixel, offset pixel refraction, intersection on interface)
- Tests verify: round-trip consistency (project then back-project recovers original point)
- Tests verify: edge cases (various depths, offsets, interface boundary, grid of points)

### [src/aquacal/core/__init__.py]
- Added snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project imports and exports

## 2026-02-03
### [src/aquacal/core/interface_model.py]
- Implemented Interface class for planar refractive interface (air-water boundary)
- Added __init__ that normalizes normal vector and stores parameters (normal, base_height, camera_offsets, n_air, n_water)
- Implemented get_interface_distance: returns base_height + camera_offset for a given camera
- Implemented get_interface_point: returns 3D point on interface (uses XY from camera_center, Z from base_height + offset)
- Added properties: n_ratio_air_to_water (n_air/n_water) and n_ratio_water_to_air (n_water/n_air) for Snell's law
- Implemented ray_plane_intersection: computes ray-plane intersection using parametric equation, returns (point, t) or (None, None) if parallel
- Function returns intersection for ANY t value (including negative), uses tolerance of 1e-10 for parallel check

### [tests/unit/test_interface_model.py]
- Created comprehensive test suite with 19 tests covering all Interface class and ray_plane_intersection functionality
- Tests verify: normal normalization, parameter storage, interface distances (reference camera, positive/negative offsets), unknown camera KeyError
- Tests verify: interface point XY matching, Z coordinate calculation, camera Z ignored, per-camera offsets
- Tests verify: refractive index ratios (air-to-water and water-to-air)
- Tests verify: ray_plane_intersection basic/angled intersections, negative t, parallel rays, non-unit direction/normal, offset planes
- All tests pass with strict numerical tolerances

### [src/aquacal/core/__init__.py]
- Added Interface and ray_plane_intersection imports and exports
- Cleaned up __all__ to remove outdated comments and organize exports alphabetically

## 2026-02-03
### [src/aquacal/core/camera.py]
- Implemented Camera class with intrinsics and extrinsics parameters
- Added properties: K, dist_coeffs, R, t, C (camera center), image_size, P (3x4 projection matrix)
- Implemented world_to_camera: transforms 3D points from world to camera frame
- Implemented project: projects 3D world points to 2D pixels with optional distortion, returns None if point behind camera (Z ≤ 0)
- Implemented pixel_to_ray: back-projects pixels to unit rays in camera frame with optional undistortion
- Implemented pixel_to_ray_world: back-projects pixels to rays in world frame (returns origin and direction)
- Added undistort_points standalone function: wrapper around cv2.undistortPoints that preserves pixel coordinates
- All OpenCV calls use proper float64 type casting for mypy compatibility

### [tests/unit/test_camera.py]
- Created comprehensive test suite with 16 tests covering all Camera class functionality
- Tests verify: properties (camera center, projection matrix), world-to-camera transform, projection (with/without distortion, behind camera), back-projection (pixel-to-ray), round-trip consistency, undistort_points function
- All tests pass with strict numerical tolerances (atol=1e-10)

### [src/aquacal/core/__init__.py]
- Added Camera and undistort_points imports and exports
- Module now properly exports Camera class and undistort_points function

## 2026-02-03
### [src/aquacal/core/board.py]
- Implemented BoardGeometry class for ChArUco board 3D geometry
- Includes __init__, corner_positions property, num_corners property, get_opencv_board(), transform_corners(), and get_corner_array() methods
- Board frame convention: origin at top-left corner (ID 0), X right, Y down, Z into board (OpenCV 4.6+ CharucoBoard convention)
- Corner positions computed as [col * square_size, row * square_size, 0.0] for planar board geometry
- transform_corners applies rigid transform (rvec, tvec) using cv2.Rodrigues for rotation

### [tests/unit/test_board.py]
- Created comprehensive test suite with 18 tests covering all BoardGeometry functionality
- Tests verify: corner count, positions (origin, grid layout, planarity), transform operations (identity, translation, rotation), get_corner_array ordering
- Tests verify OpenCV board creation and size matching
- All tests pass with strict numerical tolerances

### [src/aquacal/core/__init__.py]
- Uncommented BoardGeometry export and import
- Module now properly exports BoardGeometry for use by other modules

## 2026-02-03
### [src/aquacal/utils/transforms.py]
- Implemented 5 rotation and transform utility functions: rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center
- Used cv2.Rodrigues for rotation conversions with proper type casting for mypy compatibility
- All functions include Google-style docstrings with Args, Returns, and Example sections
- Local type aliases (Vec3, Mat3) defined to avoid circular imports with schema.py

### [tests/unit/test_transforms.py]
- Created comprehensive test suite with 21 tests covering all functions
- Tests include: basic functionality, round-trip conversions, compose-with-inverse identity check, edge cases (180° rotation, zero translation, small angles)
- All tests pass with strict numerical tolerances (atol=1e-10)

### [src/aquacal/utils/__init__.py]
- Uncommented imports and exports for all 5 transform functions
- Module now properly exports: rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center

## 2026-02-03
### [src/aquacal/config/schema.py]
- Created complete schema module with all type definitions, dataclasses, and custom exceptions
- Implemented 3 type aliases: Vec2, Vec3, Mat3 for numpy array type hints
- Implemented 13 dataclasses: BoardConfig, CameraIntrinsics, CameraExtrinsics, CameraCalibration, InterfaceParams, CalibrationResult, DiagnosticsData, CalibrationMetadata, CalibrationConfig, BoardPose, Detection, FrameDetections, DetectionResult
- Implemented 4 custom exceptions: CalibrationError (base), InsufficientDataError, ConvergenceError, ConnectivityError
- Added computed properties: CameraExtrinsics.C (camera center), Detection.num_corners, FrameDetections.cameras_with_detections/num_cameras
- Added method: DetectionResult.get_frames_with_min_cameras()

### [tests/unit/test_schema.py]
- Created comprehensive test suite with 31 tests covering all dataclasses, properties, methods, and exception hierarchy
- All tests pass, type checking passes with mypy

### [src/aquacal/config/__init__.py]
- Uncommented all exports for schema types, dataclasses, and exceptions
- Verified all imports work correctly

## 2026-02-03
### [requirements.txt]
- Created requirements.txt mirroring dependencies from pyproject.toml
- Includes core dependencies (numpy, scipy, opencv-python>=4.6, pyyaml)
- Includes development dependencies (pytest, pytest-cov, mypy, black)
- Includes visualization dependencies (matplotlib, pandas)
- Added header comments explaining relationship to pyproject.toml and directing users to prefer pip install -e ".[dev,viz]"

## 2026-02-03
### [All __init__.py files]
- Added __all__ declarations to all package __init__.py files documenting public API
- Top-level src/aquacal/__init__.py: Added __all__ with __version__ and commented convenience re-exports
- config/__init__.py: Documented all schema types (Vec2/3, Mat3, dataclasses, exceptions)
- utils/__init__.py: Documented transform functions (rvec_to_matrix, compose_poses, etc.)
- core/__init__.py: Documented Camera, Interface, BoardGeometry, and refractive functions
- io/__init__.py: Documented VideoSet, detection, and serialization functions
- calibration/__init__.py: Documented all calibration pipeline functions
- validation/__init__.py: Documented error computation and diagnostic functions
- triangulation/__init__.py: Documented triangulation functions
- All imports commented out until modules are implemented (no import errors)

## 2026-02-03
### [pyproject.toml]
- Created pyproject.toml with PEP 621 format and setuptools backend
- Package name: aquacal, version: 0.1.0, Python >=3.10
- Required dependencies: numpy, scipy, opencv-python>=4.6, pyyaml
- Optional dependencies: [dev] (pytest, pytest-cov, mypy, black), [viz] (matplotlib, pandas)
- Configured src layout with setuptools.packages.find
- Verified installation and import work correctly

## 2026-02-03
### Refactor to src Layout

Reorganized the project from a flat layout to the standard Python `src/` layout.

#### Directory Structure Changes
- Created `src/aquacal/` package structure with all subpackages
- Added `__init__.py` files to all packages (aquacal, calibration, config, core, io, triangulation, utils, validation)
- Moved `config/example_config.yaml` to `src/aquacal/config/example_config.yaml`
- Deleted old top-level package directories (calibration/, config/, core/, io/, triangulation/, utils/, validation/)

#### [DEPENDENCIES.yaml]
- Updated all module paths from flat layout (e.g., `config/schema.py`) to src layout (e.g., `src/aquacal/config/schema.py`)

#### [docs/development_plan.md]
- Updated architecture diagram to reflect new src/aquacal/ structure

#### [docs/agent_implementation_spec.md]
- Updated implementation order diagram paths
- Updated all section headings to new paths
- Updated all import statements to use `aquacal.` prefix

#### [STRUCTURE.md]
- Completely rewrote to reflect new src layout
- Added import convention documentation

#### [CLAUDE.md]
- Updated test file path example to new src layout

---

## 2026-02-03
### Documentation Review and Consistency Fixes

#### [docs/agent_implementation_spec.md]
- Fixed world Z direction: "Z points down (into water)" (was incorrectly "up")
- Fixed interface normal to [0,0,-1] in all examples and specs (was [0,0,1])
- Updated Snell's law examples and tests for Z-down convention
- Added note that TASKS.md is authoritative for phase numbering
- Added InterfaceParams docstring clarifying per-camera distances stored in CameraCalibration
- Updated detect_all_frames to accept VideoSet in addition to dict
- Fixed utils/transforms.py to import Vec3/Mat3 from schema instead of redefining
- Added custom exception classes to schema section
- Confirmed board frame convention matches OpenCV 4.6+ (Z into board)
- Added TODO for refractive_project algorithm details
- Added holdout_fraction comment specifying random frame selection

#### [docs/development_plan.md]
- Fixed interface normal references from [0,0,1] to [0,0,-1]
- Added note that TASKS.md is authoritative for phase numbering
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [docs/COORDINATES.md]
- No changes needed (was already correct)

#### [TASKS.md]
- Added header noting this is authoritative source for task IDs
- Added Phase 0 (Project Setup) with tasks 0.1-0.3 for pyproject.toml, __init__.py, requirements.txt
- Updated interface file references to new names

#### [DEPENDENCIES.yaml]
- Added Phase 0 comment about project setup files
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [STRUCTURE.md]
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [CLAUDE.md]
- Added Testing Conventions section (test file naming, classes, fixtures)

#### [config/example_config.yaml] (new file)
- Created example configuration template with all options documented

#### [tasks/current.md]
- Cleaned up duplicate template text

#### [tests/synthetic/README.md] (new file)
- Added TODO specification for synthetic data generation requirements
