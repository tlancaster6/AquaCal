---
phase: 10-documentation-audit
plan: 02
subsystem: config, calibration, core, validation, triangulation, datasets
tags: [refactoring, clarity, backward-compat]
dependency_graph:
  requires: []
  provides: [water_z_terminology]
  affects: [all-modules]
tech_stack:
  patterns: [backward-compatibility, deprecation-warning]
key_files:
  created: []
  modified:
    - src/aquacal/config/schema.py
    - src/aquacal/calibration/pipeline.py
    - src/aquacal/io/serialization.py
    - src/aquacal/calibration/_optim_common.py
    - src/aquacal/core/interface_model.py
    - src/aquacal/core/refractive_geometry.py
    - src/aquacal/validation/reprojection.py
    - src/aquacal/validation/comparison.py
    - src/aquacal/validation/diagnostics.py
    - src/aquacal/triangulation/triangulate.py
    - src/aquacal/datasets/synthetic.py
    - src/aquacal/datasets/rendering.py
    - src/aquacal/datasets/loader.py
    - src/aquacal/datasets/data/small/ground_truth.json
    - docs/guide/coordinates.md
    - docs/guide/refractive_geometry.md
    - docs/tutorials/01_full_pipeline.ipynb
    - docs/tutorials/02_diagnostics.ipynb
    - docs/tutorials/03_synthetic_validation.ipynb
    - tests/* (18 test files updated)
decisions:
  - Renamed CalibrationConfig.initial_interface_distances to initial_water_z
  - Renamed CameraCalibration.interface_distance field to water_z
  - Added backward compatibility in config loading (YAML) with deprecation warning
  - Added backward compatibility in JSON deserialization (old calibration files)
  - Chose water_z over interface_z for clarity (matches internal variable naming)
metrics:
  duration_minutes: 14
  completed_date: 2026-02-15
  tasks_completed: 2
  files_modified: 43
  tests_passing: 586
---

# Phase 10 Plan 02: Rename interface_distance to water_z Summary

**One-liner:** Renamed confusing `interface_distance` parameter to `water_z` across entire codebase with full backward compatibility

## What Was Done

### Task 1: Source Code Rename
- **Schema changes** (`config/schema.py`):
  - Renamed `CalibrationConfig.initial_interface_distances` → `initial_water_z`
  - Renamed `CameraCalibration.interface_distance` → `water_z`
  - Updated all docstrings and examples

- **Backward compatibility** (`calibration/pipeline.py`):
  - Config loading accepts both `initial_distances` (old) and `initial_water_z` (new)
  - Emits `DeprecationWarning` when old field used
  - Seamlessly maps old → new for smooth migration

- **JSON serialization** (`io/serialization.py`):
  - Writes `water_z` to new calibration files
  - Reads both `water_z` (new) and `interface_distance` (old)
  - Enables loading of old calibration results

- **Core modules updated**:
  - Renamed all internal variables and parameters
  - `Interface.get_interface_distance()` → `get_water_z()`
  - Updated 17 source files across calibration, core, validation, triangulation, datasets

### Task 2: Documentation and Tests
- **Documentation**:
  - Updated `docs/guide/coordinates.md` and `refractive_geometry.md`
  - Updated 3 Jupyter tutorial notebooks
  - All references now use `water_z` terminology

- **Tests**:
  - Updated 18 test files
  - Fixed `ground_truth.json` field name to match `SyntheticScenario.water_zs`
  - All 586 fast tests passing

## Verification

**Completeness check:**
```bash
grep -r "interface_distance" src/ tests/ docs/ --include="*.py" --include="*.md"
# Only 5 matches: backward-compatibility code in serialization.py
```

**Test results:**
- 586 tests passed
- 0 failures
- 7 deprecation warnings (expected from backward-compat tests)

**Backward compatibility verified:**
- Old YAML configs with `initial_distances` load successfully
- Old JSON calibrations with `interface_distance` deserialize correctly
- Deprecation warnings guide users to new field name

## Key Changes

| Component | Old Name | New Name |
|-----------|----------|----------|
| Config field | `initial_interface_distances` | `initial_water_z` |
| Camera calibration field | `interface_distance` | `water_z` |
| Interface method | `get_interface_distance()` | `get_water_z()` |
| Internal variables | `interface_distances` | `water_z_values` or `water_zs` |

## Technical Notes

**Why `water_z` instead of `interface_distance`?**
- The parameter is a Z-coordinate (world frame), not a distance
- "Interface distance" suggested per-camera physical gap
- Actually stores global water surface Z position
- `water_z` matches internal variable naming and is semantically accurate

**Backward compatibility strategy:**
- YAML config: accept both field names, warn on old
- JSON calibration: silent fallback (no warning needed for saved results)
- Tests verify both paths work

## Impact

**Benefits:**
- Clearer terminology reduces user confusion
- Matches internal variable naming conventions
- Documentation and code now aligned
- Smooth migration path for existing users

**No breaking changes:**
- All old configs and calibrations continue to work
- Deprecation warnings guide migration
- Full test coverage maintained

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

✓ All modified files exist and contain expected changes
✓ Commits 4487b84 and 5b172e9 exist in git log
✓ 586 tests passing
✓ No unintended `interface_distance` references (only backward-compat code)
✓ Backward compatibility verified with tests
