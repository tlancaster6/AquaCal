# Task List

**This file is the authoritative source for task IDs and phase numbering.**
Other documents (`development_plan.md`, `agent_implementation_spec.md`) may organize content by conceptual phases, but task assignments should reference the IDs below.

Status key: `[ ]` not started | `[~]` in progress | `[x]` complete

---

## Phase 0: Project Setup

- [x] **0.1** Create `pyproject.toml` with dependencies and package config
- [x] **0.2** Add `__init__.py` files to all packages with explicit exports
- [x] **0.3** Add `requirements.txt` for pip-only users

## Phase 1: Foundation

- [x] **1.1** Define configuration and output schemas (`config/schema.py`)
- [x] **1.2** Implement rotation/transform utilities (`utils/transforms.py`)
- [x] **1.3** Implement board geometry (`core/board.py`)

## Phase 2: Core Geometry

- [x] **2.1** Implement camera model (`core/camera.py`)
- [x] **2.2** Implement interface model (`core/interface_model.py`)
- [x] **2.3** Implement refractive geometry (`core/refractive_geometry.py`)

## Phase 3: Data Pipeline

- [x] **3.1** Implement video loading (`io/video.py`)
- [x] **3.2** Implement ChArUco detection (`io/detection.py`)
- [x] **3.3** Implement serialization (`io/serialization.py`)

## Phase 4: Calibration Stages

- [x] **4.1** Implement intrinsic calibration (`calibration/intrinsics.py`)
- [x] **4.2** Implement extrinsic initialization (`calibration/extrinsics.py`)
- [x] **4.3** Implement interface/pose optimization (`calibration/interface_estimation.py`)
- [x] **4.4** Implement joint refinement (`calibration/refinement.py`)

## Phase 5: Validation

- [x] **5.1** Implement reprojection error computation (`validation/reprojection.py`)
- [x] **5.2** Implement 3D reconstruction metrics (`validation/reconstruction.py`)
- [x] **5.3** Implement diagnostics (`validation/diagnostics.py`)

## Phase 6: Integration

- [x] **6.1** Implement triangulation module (`triangulation/triangulate.py`)
- [x] **6.2** Implement pipeline orchestration (`calibration/pipeline.py`)
- [x] **6.3** Add CLI entry point

## Phase 7: Testing & Documentation

- [x] **7.1** Synthetic data tests (full pipeline with known ground truth)
- [ ] **7.2** Real data validation
- [ ] **7.3** Documentation and examples

---

## Post-MVP: Simplifications & Refactoring

- [ ] **P.1** Simplify `Interface` class: remove `base_height` parameter, store only per-camera distances directly. Currently the calibration stages set `base_height=0` and put the full distance in `camera_offsets`, making `base_height` redundant. Consider whether shared base + offsets is ever needed, or if per-camera distances are always independent.

- [ ] **P.2** Consolidate synthetic data generation: Refactor existing unit tests (`test_interface_estimation.py`, `test_refinement.py`, `test_reprojection.py`) to use the centralized `tests/synthetic/ground_truth.py` module instead of duplicating `generate_synthetic_detections()` in each file.

- [x] **P.3** Performance optimization for refractive projection: Implement closed-form Newton-Raphson projection for flat interfaces (~10-20x speedup) and sparse Jacobian structure for `optimize_interface()` (~5-10x speedup). Required for practical calibration of large camera arrays (13+ cameras, 100+ frames).

- [ ] **P.4** Intelligent frame selection: Add a frame selection step before Stage 3 optimization that caps the number of frames at a configurable budget (~50-80) while maximizing camera coverage, pose graph connectivity, and spatial/angular diversity of board poses. Ensures bounded optimization time regardless of input video length.

---

## Future: Advanced Optimization

- [ ] **F.1** Ceres Solver integration: Replace scipy `least_squares` with Ceres Solver for Stage 3/4 optimization. Ceres provides automatic differentiation (exact Jacobians), built-in Schur complement solver (marginalizes board poses, reducing effective problem to ~85 camera/interface params regardless of frame count), and robust loss functions. Requires writing a custom `RefractiveCostFunction` implementing the flat-interface projection with Snell's law. Integration via pyceres or pybind11 wrapper, with scipy as fallback for users without C++ dependency.
