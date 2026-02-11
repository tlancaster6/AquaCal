# Task: 7.3c Top-Level Exports

## Objective

Update `src/aquacal/__init__.py` to re-export the essential public API so downstream consumers can write `from aquacal import load_calibration, CalibrationResult` without knowing the internal package structure.

## Background

Currently `src/aquacal/__init__.py` exports only `__version__`. All the re-exports are commented out with "uncomment as modules are implemented" — they're all implemented now.

**Prerequisites**: Tasks 7.3a (unified projection) and 7.3b (convenience methods) should be complete.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/__init__.py` — Current top-level exports (mostly commented out)
2. `tasks/public_api.md` — Section 1 "Tiered Top-Level Exports" defines the target API surface
3. `src/aquacal/io/serialization.py` — `load_calibration()`, `save_calibration()`
4. `src/aquacal/calibration/pipeline.py` — `run_calibration()`, `load_config()`
5. `src/aquacal/config/schema.py` — `CalibrationResult`, `CameraCalibration`, `CameraIntrinsics`, `CameraExtrinsics`

## Modify

- `src/aquacal/__init__.py`
- `tests/unit/test_schema.py` (add import smoke tests — file created by 7.3b)

## Do Not Modify

Everything not listed above. In particular:
- Subpackage `__init__.py` files — their exports stay as-is for power users
- `TASKS.md` — orchestrator maintains this

## Design

### Part 1: Update `src/aquacal/__init__.py`

Replace the commented-out stubs with actual imports. The tiered strategy from `public_api.md`:

**Tier 1 — Top-level** (`from aquacal import X`): The essentials every downstream consumer needs.

```python
"""AquaCal: Refractive multi-camera calibration library."""

__version__ = "0.1.0"

# Load/save calibration results
from aquacal.io.serialization import load_calibration, save_calibration

# Core types
from aquacal.config.schema import (
    CalibrationResult,
    CameraCalibration,
    CameraIntrinsics,
    CameraExtrinsics,
)

# Run calibration
from aquacal.calibration.pipeline import run_calibration, load_config

__all__ = [
    "__version__",
    # Load/save
    "load_calibration",
    "save_calibration",
    # Core types
    "CalibrationResult",
    "CameraCalibration",
    "CameraIntrinsics",
    "CameraExtrinsics",
    # Run calibration
    "run_calibration",
    "load_config",
]
```

**Tier 2 — Subpackage imports** (unchanged, already work for power users):
- `from aquacal.core import Camera, Interface, refractive_project, refractive_project_batch`
- `from aquacal.calibration import optimize_interface, build_pose_graph`
- `from aquacal.triangulation import triangulate_point`

These are not promoted to the top level — they stay as reach-in imports via their subpackages.

### Part 2: Import smoke tests

Add a small test class to `tests/unit/test_schema.py` (created in 7.3b) that verifies the public API is importable. These are not functional tests — just import checks that catch broken re-exports.

```python
class TestPublicAPI:
    """Verify top-level imports work."""

    def test_top_level_imports(self):
        """All tier-1 exports are importable from aquacal."""
        from aquacal import (
            load_calibration,
            save_calibration,
            CalibrationResult,
            CameraCalibration,
            CameraIntrinsics,
            CameraExtrinsics,
            run_calibration,
            load_config,
        )
        # Verify they're the real objects, not None
        assert callable(load_calibration)
        assert callable(save_calibration)
        assert callable(run_calibration)
        assert callable(load_config)

    def test_version_string(self):
        """__version__ is a non-empty string."""
        import aquacal
        assert isinstance(aquacal.__version__, str)
        assert len(aquacal.__version__) > 0

    def test_subpackage_imports(self):
        """Tier-2 subpackage imports still work."""
        from aquacal.core import Camera, Interface, refractive_project
        from aquacal.calibration import optimize_interface
        from aquacal.triangulation import triangulate_point
        assert callable(refractive_project)
```

## Acceptance Criteria

- [ ] `from aquacal import load_calibration, save_calibration` works
- [ ] `from aquacal import CalibrationResult, CameraCalibration, CameraIntrinsics, CameraExtrinsics` works
- [ ] `from aquacal import run_calibration, load_config` works
- [ ] `__all__` lists exactly the intended exports (no internal leakage)
- [ ] Subpackage imports (`from aquacal.core import ...`) still work
- [ ] Import smoke tests pass: `pytest tests/unit/test_schema.py -v`
- [ ] Full unit test suite still passes: `pytest tests/unit/ -v`
- [ ] No modifications to files outside "Modify" list

## Notes

1. **Why not export `Camera`, `Interface`, `refractive_project` at top level**: These are power-user primitives. The typical downstream workflow (load result → project/back-project) never needs them directly, thanks to the 7.3b convenience methods. Promoting them would clutter the top-level namespace and create a confusing "two ways to do it" situation for beginners.

2. **Import ordering**: The top-level `__init__.py` imports trigger loading of `io.serialization`, `config.schema`, and `calibration.pipeline` (and their transitive deps) at `import aquacal` time. This is fine — all these modules are lightweight. The heavy dependencies (OpenCV, scipy) are already imported by these modules at their own module level.

3. **No `InterfaceParams` at top level**: Users rarely need the raw interface parameters. They access them via `result.interface` if needed. Keeping the export list tight makes it more approachable.

## Model Recommendation

**Sonnet** — Trivial task: edit one `__init__.py` file and add a few import assertions to an existing test file.