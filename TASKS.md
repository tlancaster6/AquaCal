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
- [ ] **2.3** Implement refractive geometry (`core/refractive_geometry.py`)

## Phase 3: Data Pipeline

- [ ] **3.1** Implement video loading (`io/video.py`)
- [ ] **3.2** Implement ChArUco detection (`io/detection.py`)
- [ ] **3.3** Implement serialization (`io/serialization.py`)

## Phase 4: Calibration Stages

- [ ] **4.1** Implement intrinsic calibration (`calibration/intrinsics.py`)
- [ ] **4.2** Implement extrinsic initialization (`calibration/extrinsics.py`)
- [ ] **4.3** Implement interface/pose optimization (`calibration/interface_estimation.py`)
- [ ] **4.4** Implement joint refinement (`calibration/refinement.py`)

## Phase 5: Validation

- [ ] **5.1** Implement reprojection error computation (`validation/reprojection.py`)
- [ ] **5.2** Implement 3D reconstruction metrics (`validation/reconstruction.py`)
- [ ] **5.3** Implement diagnostics (`validation/diagnostics.py`)

## Phase 6: Integration

- [ ] **6.1** Implement triangulation module (`triangulation/triangulate.py`)
- [ ] **6.2** Implement pipeline orchestration (`calibration/pipeline.py`)
- [ ] **6.3** Add CLI entry point

## Phase 7: Testing & Documentation

- [ ] **7.1** Synthetic data tests (full pipeline with known ground truth)
- [ ] **7.2** Real data validation
- [ ] **7.3** Documentation and examples
