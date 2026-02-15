---
phase: 04-example-data
plan: 01
subsystem: datasets
tags: [synthetic-data, testing, public-api]
dependency_graph:
  requires: [core/refractive_geometry, core/board, config/schema]
  provides: [datasets/synthetic, datasets/rendering]
  affects: [tests/synthetic, tests/unit]
tech_stack:
  added: [datasets-module]
  patterns: [preset-based-generation, fixed-seed-reproducibility]
key_files:
  created:
    - src/aquacal/datasets/__init__.py
    - src/aquacal/datasets/synthetic.py
    - src/aquacal/datasets/rendering.py
    - tests/unit/test_datasets.py
  modified:
    - tests/synthetic/ground_truth.py
decisions:
  - Clean detections by default (noisy=False), explicit opt-in for noise
  - Three fixed presets only (small, medium, large) - no custom configuration
  - Board config standardized across all presets (12x9 ChArUco, DICT_5X5_100)
  - Minimal image rendering (corners on black background) sufficient for testing
metrics:
  duration: 660
  completed_date: "2026-02-15T01:18:02Z"
  tasks_completed: 2
  tests_added: 11
  files_created: 4
  files_modified: 1
  commits: 2
---

# Phase 4 Plan 1: Synthetic Data API

**One-liner:** Public `generate_synthetic_rig()` API with 3 fixed presets (small, medium, large) and optional image rendering

## Overview

Created the `aquacal.datasets` public API by refactoring synthetic data generation from test helpers into a first-class module. Researchers can now generate synthetic calibration data with known ground truth in a single function call, with fixed presets and optional image rendering.

## What Was Built

### Public API (`src/aquacal/datasets/`)

**`synthetic.py`** — Core synthetic data generation:
- `generate_synthetic_rig(preset, *, include_images=False, noisy=False)` — Main entry point
- Three fixed presets with reproducible seeds:
  - `'small'`: 2 cameras, 10 frames, seed=42
  - `'medium'`: 6 cameras, 80 frames, seed=123
  - `'large'`: 13 cameras (real rig geometry), 300 frames, seed=456
- All presets use 12x9 ChArUco board (60mm squares, 45mm markers, DICT_5X5_100)
- Clean detections by default, `noisy=True` adds Gaussian pixel jitter (0.3-0.5px)
- Returns `SyntheticScenario` with complete ground truth + detections
- Optional `include_images=True` renders grayscale ChArUco frames

**`rendering.py`** — Synthetic ChArUco image rendering:
- `render_synthetic_frame()` — Projects 3D board through refractive interface, draws corners
- `render_scenario_images()` — Renders all cameras/frames in a scenario
- Minimal rendering approach: black background + white corner circles (sufficient for detection)
- Grayscale uint8 output at full camera resolution

**`__init__.py`** — Public API exports:
- Exports: `generate_synthetic_rig`, `SyntheticScenario`
- Comprehensive module docstring with usage examples
- Internal helpers NOT exported (clean public interface)

### Test Infrastructure

**Refactored `tests/synthetic/ground_truth.py`:**
- Now imports all generation functions from `aquacal.datasets.synthetic`
- Keeps `create_scenario()` for test-specific presets (ideal, minimal, realistic)
- Reduced from 740 lines to 143 lines (80% reduction via deduplication)

**New `tests/unit/test_datasets.py`:**
- 11 comprehensive tests covering:
  - All three presets (small, medium, large)
  - Fixed seed reproducibility
  - Noisy vs clean detections
  - Image rendering (with/without)
  - Invalid preset error handling
  - Board config consistency
  - Camera naming conventions
  - Reference camera at origin
- All tests pass in 41s

## Key Implementation Decisions

### 1. Clean detections by default
**Context:** Users need predictable behavior for testing.
**Decision:** `noisy=False` by default, explicit `noisy=True` to add pixel noise.
**Impact:** Simpler API, opt-in for robustness testing.

### 2. Fixed presets only (no custom configuration)
**Context:** Infinite configuration space makes API confusing.
**Decision:** Three fixed, well-tested presets covering common use cases.
**Impact:** Simple, predictable API. Power users can use internal functions.

### 3. Minimal image rendering
**Context:** Full checkerboard rendering complex, not critical for testing.
**Decision:** Render corners as white circles on black background.
**Impact:** Fast rendering (~1s for small preset), sufficient for detection tests.

### 4. Board config standardization
**Context:** All presets need consistent board for fair comparison.
**Decision:** 12x9 ChArUco (60mm squares, DICT_5X5_100) matches real hardware.
**Impact:** Test results directly comparable to real rig.

## Verification Results

**Public API verification:**
```python
# Basic generation
s = generate_synthetic_rig('small')
# Cameras: 2, Frames: 10, Reproducible: OK

# With images
s = generate_synthetic_rig('small', include_images=True)
# Images: 2 cameras, Image shape: (1080, 1920), dtype: uint8
```

**Test results:**
- 11 new dataset tests: PASSED (41s)
- 563 existing tests (excluding slow): PASSED (78s)
- 0 regressions from refactoring
- Total test coverage maintained

## Deviations from Plan

None - plan executed exactly as written.

## Files Changed

**Created:**
- `src/aquacal/datasets/__init__.py` (40 lines) — Public API
- `src/aquacal/datasets/synthetic.py` (810 lines) — Core generation
- `src/aquacal/datasets/rendering.py` (200 lines) — Image rendering
- `tests/unit/test_datasets.py` (230 lines) — Dataset tests

**Modified:**
- `tests/synthetic/ground_truth.py` (-618 lines) — Refactored to re-export

**Commits:**
1. `41ea6e0` — feat(04-01): add aquacal.datasets module with synthetic rig generation
2. `e3847d8` — feat(04-01): refactor tests to use public datasets API

## Acceptance Criteria

✅ `generate_synthetic_rig()` works for all three presets with correct camera/frame counts
✅ Fixed seed reproducibility: same preset always generates identical data
✅ `include_images=True` renders grayscale images at full camera resolution
✅ `noisy=True` adds Gaussian pixel jitter to detections
✅ All existing tests pass (no regressions from refactoring)
✅ New dataset tests validate the public API

## Usage Examples

```python
from aquacal.datasets import generate_synthetic_rig

# Quick smoke test
scenario = generate_synthetic_rig('small')
print(f"{len(scenario.intrinsics)} cameras, {len(scenario.board_poses)} frames")
# 2 cameras, 10 frames

# With rendered images
scenario = generate_synthetic_rig('small', include_images=True)
img = scenario.images['cam0'][0]  # First frame from cam0
print(img.shape, img.dtype)  # (1080, 1920) uint8

# Noisy detections for robustness testing
scenario = generate_synthetic_rig('medium', noisy=True)
print(f"Noise std: {scenario.noise_std}px")  # Noise std: 0.5px
```

## Next Steps

- Plan 02: Real calibration dataset from lab rig
- Plan 03: Jupyter tutorial notebooks using synthetic data
- Consider adding more rendering options (full checkerboard, ArUco markers) if needed

## Self-Check: PASSED

**Files created:**
```bash
✓ src/aquacal/datasets/__init__.py
✓ src/aquacal/datasets/synthetic.py
✓ src/aquacal/datasets/rendering.py
✓ tests/unit/test_datasets.py
```

**Commits exist:**
```bash
✓ 41ea6e0: feat(04-01): add aquacal.datasets module
✓ e3847d8: feat(04-01): refactor tests to use public API
```

**Tests pass:**
```bash
✓ 11 new dataset tests pass
✓ 563 existing tests pass (no regressions)
```
