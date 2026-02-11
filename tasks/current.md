# Task: P.5 Standardize Synthetic Pipeline Tests

## Objective

Make the test structure consistent across all calibration scenarios in `test_full_pipeline.py`. Each scenario should test the same metrics with scenario-appropriate thresholds. Remove misplaced ground-truth fixture tests from calibration test classes.

## Background

The three calibration scenario test classes (`TestIdealScenario`, `TestRealisticScenario`, `TestMinimalScenario`) currently have inconsistent structure:

- **`TestIdealScenario`** (good): Tests rotation, translation, interface distance errors AND RMS reprojection error.
- **`TestRealisticScenario`** (mixed): Tests rotation, translation, interface distance errors but NOT RMS error. Also contains `test_has_13_cameras` and `test_geometry` which test the ground-truth fixture properties, not calibration accuracy. These are duplicates of tests already in `TestGenerateRealRigArray`.
- **`TestMinimalScenario`** (weak): Only tests that 2 cameras exist in the result — no accuracy checks at all.

## Context Files

Read these files before starting (in order):

1. `tests/synthetic/test_full_pipeline.py` — The file to modify. Read the full file.
2. `tests/synthetic/ground_truth.py` (lines 605-686) — `compute_calibration_errors()` returns: `rotation_error_deg`, `translation_error_mm`, `interface_distance_error_mm`, `focal_length_error_percent`, `principal_point_error_px`.

## Modify

- `tests/synthetic/test_full_pipeline.py`
- `tests/synthetic/conftest.py` — change scenario fixture scopes from function to class

## Do Not Modify

Everything not listed above. In particular:
- `tests/synthetic/ground_truth.py` — test infrastructure stays as-is
- `TASKS.md` — orchestrator maintains this

## Design

### Part 1: Remove Misplaced Tests from `TestRealisticScenario`

Delete these two tests from `TestRealisticScenario`:

- `test_has_13_cameras` (lines 408-412): Duplicates `TestGenerateRealRigArray.test_creates_13_cameras`
- `test_geometry` (lines 414-431): Duplicates `TestGenerateRealRigArray.test_inner_ring_radius` and `test_outer_ring_radius`

These test the ground-truth fixture, not calibration accuracy. They already exist in `TestGenerateRealRigArray` (lines 226-298).

### Part 2: Add Class-Scoped Result Fixtures

Add three class-scoped fixtures that run the calibration once per scenario class, avoiding redundant optimization runs (especially important for the 13-camera realistic scenario):

```python
@pytest.fixture(scope="class")
def ideal_result(scenario_ideal):
    """Run calibration once for all ideal scenario tests."""
    result = _run_calibration_stages(scenario_ideal, noise_std=0.0)
    errors = compute_calibration_errors(result, scenario_ideal)
    return result, errors


@pytest.fixture(scope="class")
def realistic_result(scenario_realistic):
    """Run calibration once for all realistic scenario tests."""
    result = _run_calibration_stages(scenario_realistic)
    errors = compute_calibration_errors(result, scenario_realistic)
    return result, errors


@pytest.fixture(scope="class")
def minimal_result(scenario_minimal):
    """Run calibration once for all minimal scenario tests."""
    result = _run_calibration_stages(scenario_minimal)
    errors = compute_calibration_errors(result, scenario_minimal)
    return result, errors
```

Place these after `_run_calibration_stages()` and before the test classes.

**Important**: For class-scoped fixtures to work with session/module-scoped scenario fixtures, the `conftest.py` scenario fixtures need to be at least class scope too. Currently they are function-scoped (the default). Update `tests/synthetic/conftest.py` to use `scope="class"`:

```python
@pytest.fixture(scope="class")
def scenario_ideal() -> SyntheticScenario:
    ...
```

Do the same for `scenario_minimal` and `scenario_realistic`.

### Part 3: Standardize Test Structure

Each scenario class should have exactly 4 tests with scenario-appropriate thresholds. Tests use the class-scoped result fixtures to avoid redundant optimization runs:

```python
@pytest.mark.slow
class TestIdealScenario:
    """Test with zero noise - should recover ground truth exactly."""

    def test_rotation_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["rotation_error_deg"] < 0.5

    def test_translation_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["translation_error_mm"] < 5.0

    def test_interface_distance_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["interface_distance_error_mm"] < 10.0

    def test_rms_reprojection_error(self, ideal_result):
        result, errors = ideal_result
        assert result.diagnostics.reprojection_error_rms < 1.0


@pytest.mark.slow
class TestRealisticScenario:
    """Test with 13-camera rig matching actual hardware."""

    def test_rotation_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["rotation_error_deg"] < 1.5

    def test_translation_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["translation_error_mm"] < 15.0

    def test_interface_distance_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["interface_distance_error_mm"] < 20.0

    def test_rms_reprojection_error(self, realistic_result):
        result, errors = realistic_result
        assert result.diagnostics.reprojection_error_rms < 2.0


@pytest.mark.slow
class TestMinimalScenario:
    """Test edge case: minimum viable configuration (2 cameras)."""

    def test_rotation_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["rotation_error_deg"] < 3.0

    def test_translation_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["translation_error_mm"] < 30.0

    def test_interface_distance_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["interface_distance_error_mm"] < 30.0

    def test_rms_reprojection_error(self, minimal_result):
        result, errors = minimal_result
        assert result.diagnostics.reprojection_error_rms < 3.0
```

### Threshold Rationale

| Metric | Ideal (0px noise) | Realistic (0.5px noise, 13 cams) | Minimal (0.5px noise, 2 cams) |
|---|---|---|---|
| Rotation (deg) | < 0.5 | < 1.5 | < 3.0 |
| Translation (mm) | < 5.0 | < 15.0 | < 30.0 |
| Interface dist (mm) | < 10.0 | < 20.0 | < 30.0 |
| RMS (px) | < 1.0 | < 2.0 | < 3.0 |

Ideal thresholds are kept from the existing tests. Realistic thresholds are kept from the existing tests. Minimal thresholds are looser — 2 cameras provide much less geometric constraint.

### Part 4: `@pytest.mark.slow` Markers

All three calibration scenario test classes must be marked `@pytest.mark.slow`. This lets developers skip the optimization-heavy tests during quick iteration:

```bash
pytest tests/synthetic/ -m "not slow" -v   # fast fixture/utility tests only
pytest tests/synthetic/ -v                  # everything including slow calibration tests
```

The `slow` marker is already registered in `pyproject.toml`. Do NOT mark the non-calibration test classes (`TestGenerateCameraIntrinsics`, `TestGenerateCameraArray`, `TestGenerateRealRigArray`, `TestCreateScenario`, `TestGenerateSyntheticDetections`, `TestComputeCalibrationErrors`) — those are fast.

## Acceptance Criteria

- [ ] `TestIdealScenario` has 4 tests: rotation, translation, interface distance, RMS
- [ ] `TestRealisticScenario` has 4 tests: rotation, translation, interface distance, RMS
- [ ] `TestMinimalScenario` has 4 tests: rotation, translation, interface distance, RMS
- [ ] All three calibration scenario classes marked `@pytest.mark.slow`
- [ ] Class-scoped result fixtures (`ideal_result`, `realistic_result`, `minimal_result`) run optimization once per class
- [ ] `conftest.py` scenario fixtures use `scope="class"`
- [ ] No ground-truth fixture tests in calibration scenario classes (no camera count checks, no geometry checks)
- [ ] `TestGenerateRealRigArray` unchanged (still has fixture geometry tests)
- [ ] `TestGenerateSyntheticDetections` unchanged
- [ ] `TestCreateScenario` unchanged
- [ ] `TestComputeCalibrationErrors` unchanged
- [ ] `_run_calibration_stages()` helper unchanged
- [ ] All tests pass: `pytest tests/synthetic/test_full_pipeline.py -v`
- [ ] Slow marker works: `pytest tests/synthetic/ -m "not slow" -v` runs only fast tests
- [ ] No modifications to files outside "Modify" list

## Notes

1. **Minimal scenario thresholds are guesses**: The minimal scenario (2 cameras, 0.5px noise) has never been tested for accuracy before — only that it runs. The thresholds above (3 deg, 30mm, 30mm, 3px) are generous starting points. If tests fail, loosen them — the important thing is that *some* accuracy check exists, not that it's tight.

2. **`compute_calibration_errors` uses ground-truth intrinsics**: Since `_run_calibration_stages()` passes ground-truth intrinsics to the result (line 90), `focal_length_error_percent` and `principal_point_error_px` will always be 0. No need to test these — they're trivially zero. Only test the 3 estimated quantities: rotation, translation, interface distance.

3. **Existing `TestComputeCalibrationErrors.test_perfect_match_gives_zero_errors`** already tests all 5 error metrics with a perfect-match result. That test stays as-is.

## Model Recommendation

**Sonnet** — Straightforward restructuring. Delete 2 tests, add missing tests with specified thresholds, refactor existing tests into consistent 4-test pattern. No logic changes.