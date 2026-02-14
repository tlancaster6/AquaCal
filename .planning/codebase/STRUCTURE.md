# Codebase Structure

**Analysis Date:** 2026-02-14

## Directory Layout

```
AquaCal/
├── src/aquacal/
│   ├── __init__.py                # Public API exports
│   ├── __main__.py                # Entry point for `python -m aquacal`
│   ├── cli.py                     # CLI subcommands (calibrate, init, compare)
│   ├── config/
│   │   ├── __init__.py            # Config package exports
│   │   └── schema.py              # All dataclasses, types, exceptions
│   ├── core/
│   │   ├── __init__.py            # Core module exports
│   │   ├── board.py               # ChArUco board geometry
│   │   ├── camera.py              # Camera models (pinhole, fisheye)
│   │   ├── interface_model.py      # Water surface plane model
│   │   └── refractive_geometry.py  # Snell's law, projection, ray tracing
│   ├── io/
│   │   ├── __init__.py            # IO module exports
│   │   ├── detection.py           # ChArUco corner detection
│   │   ├── serialization.py       # JSON save/load CalibrationResult
│   │   └── video.py               # Synchronized multi-camera video loading
│   ├── calibration/
│   │   ├── __init__.py            # Calibration package exports
│   │   ├── _optim_common.py       # Shared optimization utilities
│   │   ├── extrinsics.py          # Stage 2: extrinsic initialization
│   │   ├── interface_estimation.py # Stage 3: joint optimization + aux cameras
│   │   ├── intrinsics.py          # Stage 1: in-air calibration
│   │   ├── pipeline.py            # End-to-end orchestration
│   │   └── refinement.py          # Stage 4: intrinsics refinement
│   ├── triangulation/
│   │   ├── __init__.py            # Triangulation exports
│   │   └── triangulate.py         # Ray-based 3D reconstruction
│   ├── utils/
│   │   ├── __init__.py            # Utils package exports
│   │   └── transforms.py          # Rotation/pose utilities
│   └── validation/
│       ├── __init__.py            # Validation module exports
│       ├── comparison.py          # Cross-run comparison (metrics, plots)
│       ├── diagnostics.py         # Report generation, visualizations
│       ├── reconstruction.py      # 3D distance errors, spatial analysis
│       └── reprojection.py        # Reprojection error computation
├── tests/
│   ├── __init__.py
│   ├── unit/                      # Unit tests (one test file per module)
│   │   ├── test_board.py
│   │   ├── test_camera.py
│   │   ├── test_cli.py
│   │   ├── test_comparison.py
│   │   ├── test_detection.py
│   │   ├── test_diagnostics.py
│   │   ├── test_extrinsics.py
│   │   ├── test_interface_estimation.py
│   │   ├── test_interface_model.py
│   │   ├── test_intrinsics.py
│   │   ├── test_optim_common.py
│   │   ├── test_pipeline.py
│   │   ├── test_reconstruction.py
│   │   ├── test_refinement.py
│   │   ├── test_reprojection.py
│   │   ├── test_refractive_geometry.py
│   │   ├── test_schema.py
│   │   ├── test_serialization.py
│   │   ├── test_transforms.py
│   │   ├── test_triangulate.py
│   │   └── test_video.py
│   ├── synthetic/                 # Integration/synthetic tests
│   │   ├── __init__.py
│   │   ├── conftest.py            # Pytest fixtures
│   │   ├── compare_refractive.py   # Script: run refractive comparison experiments
│   │   ├── experiment_helpers.py   # Shared experiment logic
│   │   ├── experiments.py          # Experiment implementations (fidelity, depth, scaling)
│   │   ├── ground_truth.py         # Synthetic rig generation, test data
│   │   └── test_refractive_comparison.py # Pytest-based experiments
│   └── integration/                # Placeholder for future integration tests
├── dev/                            # Development and documentation (gitignored)
│   ├── DESIGN.md                   # Architecture and design decisions
│   ├── GEOMETRY.md                 # Coordinate systems and transforms
│   ├── KNOWLEDGE_BASE.md           # Accumulated insights and gotchas
│   ├── CHANGELOG.md                # Chronological implementation log
│   ├── TASKS.md                    # Task status and orchestration notes
│   ├── tasks/                      # Task implementation files
│   │   └── archive/                # Completed tasks
│   ├── handoffs/                   # Agent handoff documents
│   └── tmp/                        # Temporary scripts and debugging
├── .planning/                      # Planning documents (gitignored)
│   └── codebase/                   # Generated codebase analysis
│       ├── ARCHITECTURE.md         # This architecture doc
│       ├── STRUCTURE.md            # This structure doc
│       ├── STACK.md                # Technology stack
│       ├── INTEGRATIONS.md         # External integrations
│       ├── CONVENTIONS.md          # Coding conventions
│       ├── TESTING.md              # Testing patterns
│       └── CONCERNS.md             # Technical debt
├── .claude/                        # Claude agent configuration
│   ├── agents/                     # Agent definitions (planner, executor, etc.)
│   ├── rules/                      # Code style and workflow rules
│   └── hooks/                      # Git hooks (auto-format, protect-tasks)
├── results/                        # Output directory for experimental results
├── pyproject.toml                  # Project metadata, dependencies, build config
├── README.md                       # Project overview
├── CLAUDE.md                       # Project instructions for Claude agents
└── LICENSE                         # Project license
```

## Directory Purposes

**src/aquacal:**
- Purpose: Main library code
- Contains: All implementation modules organized by responsibility
- Key files: `__init__.py` (public API), `__main__.py` (CLI entry), `cli.py` (subcommands)

**src/aquacal/config:**
- Purpose: Configuration and type definitions
- Contains: `schema.py` with all dataclasses, type aliases, custom exceptions
- Not a traditional config directory; no YAML files here (YAML parsed into `CalibrationConfig` dataclass)

**src/aquacal/core:**
- Purpose: Pure geometry and physics (no optimization, no IO)
- Contains: Camera models, board geometry, refractive geometry, interface modeling
- Key exports: `Camera`, `FisheyeCamera`, `BoardGeometry`, `Interface`, `refractive_project()`, `snells_law_3d()`

**src/aquacal/io:**
- Purpose: Data input/output
- Contains: Video loading (`VideoSet`), ChArUco detection, JSON serialization
- Key exports: `detect_all_frames()`, `load_calibration()`, `save_calibration()`, `VideoSet`

**src/aquacal/calibration:**
- Purpose: Multi-stage calibration pipeline and optimization
- Contains: 4 calibration stages, shared optimization utilities, pipeline orchestration
- Key files:
  - `_optim_common.py`: Parameter packing/unpacking, sparsity, bounds, residuals
  - `intrinsics.py`: Stage 1 (in-air calibration via OpenCV)
  - `extrinsics.py`: Stage 2 (pose graph initialization)
  - `interface_estimation.py`: Stage 3 (joint refractive optimization)
  - `refinement.py`: Stage 4 (intrinsics refinement)
  - `pipeline.py`: Orchestration, config loading, detection splitting, validation

**src/aquacal/triangulation:**
- Purpose: Refractive 3D reconstruction
- Contains: Ray tracing through water surface, linear intersection
- Key exports: `triangulate_points()` (takes 2D detections + calibration, returns 3D positions)

**src/aquacal/utils:**
- Purpose: Shared utilities
- Contains: Rotation matrix ↔ rotation vector conversion, pose composition
- Key exports: `rvec_to_matrix()`, `matrix_to_rvec()`, pose utilities

**src/aquacal/validation:**
- Purpose: Error computation, diagnostics, comparison
- Contains: Reprojection errors, 3D reconstruction errors, report generation, multi-run comparison
- Key exports: `compute_reprojection_errors()`, `compute_3d_distance_errors()`, `generate_diagnostic_report()`, `compare_calibrations()`

**tests/unit:**
- Purpose: Unit tests (one file per source module)
- Naming: `test_<module>.py` mirrors `src/aquacal/<module>.py`
- Scope: Individual functions and classes in isolation (mocked dependencies where needed)
- Run: `pytest tests/unit/`

**tests/synthetic:**
- Purpose: Integration tests with ground truth
- Contains: Synthetic rig generation, full pipeline tests, refractive comparison experiments
- Run: `pytest tests/synthetic/` or `python tests/synthetic/compare_refractive.py`
- Marked `@pytest.mark.slow` for experiments (skip with `pytest -m "not slow"`)

**dev:**
- Purpose: Development notes and task tracking (gitignored)
- Contains: Architecture docs (`DESIGN.md`), geometry reference (`GEOMETRY.md`), task status, implementation logs
- Key files: `DESIGN.md` (detailed architecture), `KNOWLEDGE_BASE.md` (accumulated gotchas), `TASKS.md` (orchestrator-maintained task list)

**.planning/codebase:**
- Purpose: Generated codebase analysis for other GSD agents
- Contains: ARCHITECTURE.md, STRUCTURE.md, STACK.md, INTEGRATIONS.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Generated by: `/gsd:map-codebase` orchestrator command

**.claude/agents, .claude/rules:**
- Purpose: Claude agent infrastructure
- Contains: Agent definitions, code style rules, workflow definitions
- Reference: `.claude/rules/code-style.md`, `.claude/rules/source-code.md`

**results/:**
- Purpose: Output directory for experimental runs
- Contains: Comparison reports, plots, CSV tables from `aquacal compare` and synthetic experiments
- Temporary; not committed to git

## Key File Locations

**Entry Points:**
- `src/aquacal/__init__.py`: Public API (`run_calibration`, `load_config`, `load_calibration`, `save_calibration`, core types)
- `src/aquacal/__main__.py`: CLI entry point
- `src/aquacal/cli.py`: Subcommand handlers (`cmd_calibrate`, `cmd_init`, `cmd_compare`)

**Configuration & Types:**
- `src/aquacal/config/schema.py`: All dataclasses and exceptions
  - `CalibrationConfig`: Input configuration from YAML
  - `CalibrationResult`: Pipeline output
  - `CameraIntrinsics`, `CameraExtrinsics`, `CameraCalibration`: Camera parameters
  - `DetectionResult`, `FrameDetections`, `Detection`: Detection containers
  - `BoardConfig`, `BoardPose`, `InterfaceParams`: Board and interface types
  - `DiagnosticsData`, `CalibrationMetadata`: Result metadata
  - Exceptions: `CalibrationError`, `InsufficientDataError`, `ConnectivityError`

**Core Geometry:**
- `src/aquacal/core/refractive_geometry.py`: Snell's law, projection
  - `snells_law_3d()`: Vector Snell's law application
  - `trace_ray_air_to_water()`: Ray tracing through interface
  - `refractive_project()`: Single-point projection (Newton-Raphson or Brent)
  - `refractive_project_batch()`: Vectorized projection
  - `refractive_project_fast()`: Optimized Newton-Raphson projection
- `src/aquacal/core/camera.py`: Camera models
  - `Camera` class: Pinhole projection model (K, dist_coeffs, image_size)
  - `FisheyeCamera` class: Equidistant fisheye model
  - `create_camera()`: Factory function
- `src/aquacal/core/board.py`: Board geometry
  - `BoardGeometry`: ChArUco board with corner management
  - `transform_corners()`: Apply board pose to 3D corners
- `src/aquacal/core/interface_model.py`: Water surface
  - `Interface`: Plane with normal, camera distances, refractive indices
  - `get_interface_point()`: Camera's plane intersection

**Detection & IO:**
- `src/aquacal/io/detection.py`: ChArUco detection
  - `detect_all_frames()`: Multi-camera ChArUco detection with filtering
  - `_detect_charuco_frame()`: Per-frame detection
  - `_filter_detections()`: Corner filtering (min count, collinearity)
- `src/aquacal/io/video.py`: Video loading
  - `VideoSet`: Synchronized multi-camera video with lazy frame loading
- `src/aquacal/io/serialization.py`: JSON round-trip
  - `load_calibration()`, `save_calibration()`: CalibrationResult ↔ JSON

**Calibration Stages:**
- `src/aquacal/calibration/intrinsics.py`: Stage 1
  - `calibrate_intrinsics_all()`: OpenCV intrinsic calibration per camera
  - `validate_intrinsics()`: Sanity checks on K, dist_coeffs
- `src/aquacal/calibration/extrinsics.py`: Stage 2
  - `build_pose_graph()`: Camera connectivity via shared board observations
  - `estimate_extrinsics()`: BFS pose chain, refractive PnP, rotation averaging
- `src/aquacal/calibration/interface_estimation.py`: Stage 3 + 3b
  - `optimize_interface()`: Joint refractive bundle adjustment
  - `register_auxiliary_camera()`: Post-hoc auxiliary camera registration
  - `_compute_initial_board_poses()`: PnP initialization for board poses
- `src/aquacal/calibration/refinement.py`: Stage 4
  - `joint_refinement()`: Joint optimization with intrinsics refinement
- `src/aquacal/calibration/_optim_common.py`: Shared utilities
  - `pack_params()`, `unpack_params()`: Parameter vector ↔ structured objects
  - `build_jac_sparsity()`: Jacobian sparsity pattern
  - `build_jacobian_callable()`: Sparse finite differences + dense solver
  - `compute_residuals()`: Cost function evaluation

**Pipeline & Config:**
- `src/aquacal/calibration/pipeline.py`: End-to-end orchestration
  - `load_config()`: YAML → `CalibrationConfig`
  - `run_calibration()`, `run_calibration_from_config()`: Full pipeline
  - `split_detections()`: Train/validation split
  - `_build_calibration_result()`: Assemble final result

**Validation & Diagnostics:**
- `src/aquacal/validation/reprojection.py`: Reprojection error
  - `compute_reprojection_errors()`: Per-camera, per-frame RMS
- `src/aquacal/validation/reconstruction.py`: 3D errors and spatial analysis
  - `compute_3d_distance_errors()`: Adjacent-corner distance comparison
  - `SpatialMeasurement`: Per-measurement spatial records (x, y, z, signed_error)
  - `bin_by_depth()`: Depth binning for stratified analysis
  - `save_spatial_measurements()`: CSV export
- `src/aquacal/validation/diagnostics.py`: Report generation
  - `generate_diagnostic_report()`: Statistical summary + recommendations
  - `save_diagnostic_report()`: Write CSV, PNG plots
  - `plot_camera_rig()`, `plot_reprojection_errors()`, `plot_depth_stratified_error()`: Visualizations
- `src/aquacal/validation/comparison.py`: Multi-run comparison
  - `compare_calibrations()`: Load N results, compute deltas
  - `ComparisonResult`: Metrics and parameter differences
  - `write_comparison_report()`: CSV tables + PNG plots
  - Visualizations: Per-camera RMS bar chart, position overlay, Z dumbbell, depth-stratified error, XY heatmaps

**Triangulation:**
- `src/aquacal/triangulation/triangulate.py`: Ray-based 3D reconstruction
  - `triangulate_points()`: Back-project 2D detections, intersect rays through interface
  - `_create_interface_from_calibration()`: Build Interface from CalibrationResult

**Utilities:**
- `src/aquacal/utils/transforms.py`: Rotation/pose utilities
  - `rvec_to_matrix()`, `matrix_to_rvec()`: Rodrigues conversion
  - `compose_poses()`, `invert_pose()`: Pose composition

## Naming Conventions

**Files:**
- `snake_case.py`: All module files
- Test files: `test_<module>.py` (e.g., `test_camera.py` for `camera.py`)
- Temporary/script files: descriptive snake_case (e.g., `compare_refractive.py`)
- Config/generated files: descriptive uppercase (e.g., `ARCHITECTURE.md`)

**Directories:**
- `snake_case/`: All directories (core, io, calibration, validation, utils)
- Special: `.claude/`, `.planning/`, `dev/` (dotfiles for infrastructure)

**Classes:**
- `PascalCase`: All classes (e.g., `Camera`, `BoardGeometry`, `CalibrationResult`)
- Subclasses follow pattern: `FisheyeCamera` (Camera subclass)

**Functions & Methods:**
- `snake_case`: All functions and methods
- Prefixes: `_private_function()` for internal helpers
- Factories: `create_camera()` (dispatch based on model)
- Getters: `get_interface_point()` (retrieve object property)
- Builders: `build_pose_graph()`, `build_jac_sparsity()` (construct complex object)

**Constants:**
- `UPPER_SNAKE_CASE`: All module-level constants (none yet, add if needed)

**Type Aliases:**
- `Vec3 = NDArray[np.float64]`: 3-element vectors
- `Mat3 = NDArray[np.float64]`: 3x3 matrices
- `Vec2 = NDArray[np.float64]`: 2-element vectors
- Defined in: `src/aquacal/config/schema.py`

## Where to Add New Code

**New Feature (e.g., intrinsics refinement improvement):**
- Primary code: `src/aquacal/calibration/refinement.py` (or new stage module)
- Tests: `tests/unit/test_refinement.py`
- Integration tests: `tests/synthetic/test_refractive_comparison.py` (if impacts validation metrics)
- Documentation: Update `dev/DESIGN.md` (architecture section), `CLAUDE.md` (if CLI interface changes)

**New Camera Model (e.g., rational distortion):**
- Implementation: `src/aquacal/core/camera.py` (new Camera subclass)
- Factory update: `create_camera()` in same file
- Tests: `tests/unit/test_camera.py` (new test class for model)
- Config: Add flag to `CalibrationConfig` schema (e.g., `rational_model_cameras`)
- Pipeline: Update `pipeline.py` to pass flag to intrinsics stage

**New Validation Metric:**
- Implementation: `src/aquacal/validation/<metric>.py` (new module) or extend existing
- Dataclass: Add field to `DiagnosticsData` (schema.py)
- Integration: Call metric function in `pipeline.py` validation section
- Export: Update `src/aquacal/validation/__init__.py`
- Tests: `tests/unit/test_<metric>.py`

**New Utility Function:**
- Implementation: `src/aquacal/utils/transforms.py` (or new utils module)
- Tests: `tests/unit/test_transforms.py`
- Export: Add to `src/aquacal/utils/__init__.py` and `__all__`

**New CLI Subcommand:**
- Handler: `cmd_<subcommand>()` in `src/aquacal/cli.py`
- Argument parser: Add subparser in `create_parser()`
- Tests: `tests/unit/test_cli.py` (new test for subcommand)
- Documentation: Update `CLAUDE.md` (CLI section) if interface changes

**Integration Test / Synthetic Experiment:**
- Test file: `tests/synthetic/test_<experiment>.py`
- Helpers: Add to `tests/synthetic/experiment_helpers.py` or `ground_truth.py` (reusable)
- Run: Mark with `@pytest.mark.slow` if long-running
- Results: Output to `results/` directory via CLI or script

## Special Directories

**src/aquacal/calibration/_optim_common.py:**
- Purpose: Shared optimization backend for Stages 3 and 4
- Generated: No (hand-written, core module)
- Committed: Yes
- Importance: Critical path; any changes require thorough testing

**dev/:**
- Purpose: Development notes, task tracking, design docs
- Generated: No (hand-edited, architecture documentation)
- Committed: No (gitignored)
- Structure:
  - `DESIGN.md`: Detailed architecture (see ARCHITECTURE.md for summary)
  - `GEOMETRY.md`: Coordinate systems, transforms, refractive geometry reference
  - `KNOWLEDGE_BASE.md`: Accumulated insights and known gotchas (maintained by orchestrator)
  - `CHANGELOG.md`: Chronological implementation log (append-only)
  - `TASKS.md`: Task list (maintained by orchestrator)
  - `tasks/`: Task implementation files (one per task)
  - `handoffs/`: Agent handoff documents (created by orchestrator when needed)

**.planning/codebase/:**
- Purpose: Generated codebase analysis for other agents
- Generated: Yes (by `/gsd:map-codebase` command)
- Committed: No (gitignored)
- Contents: ARCHITECTURE.md, STRUCTURE.md, STACK.md, INTEGRATIONS.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

**results/:**
- Purpose: Experimental output (plots, CSV, JSON)
- Generated: Yes (by `aquacal compare` and synthetic experiments)
- Committed: No (gitignored)
- Contents: Timestamped subdirectories with results

**tests/synthetic/:**
- Purpose: Ground-truth-based integration tests
- Generated: No (hand-written)
- Committed: Yes
- Key distinction: Uses synthetic rig generation to ensure pixel-perfect test data, allowing validation of algorithm correctness independent of real camera hardware

---

*Structure analysis: 2026-02-14*
