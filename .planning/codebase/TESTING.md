# Testing Patterns

**Analysis Date:** 2026-02-14

## Test Framework

**Runner:**
- Framework: pytest
- Version: Latest (specified in `pyproject.toml` dev dependencies)
- Config: `pyproject.toml` lines 54-57

**Assertion Library:**
- numpy.testing for array assertions: `np.testing.assert_allclose()`
- pytest assertions for boolean/equality: `assert result is not None`, `assert len(poses) == 3`

**Run Commands:**
```bash
python -m pytest tests/                        # Run all tests
python -m pytest tests/ -m "not slow"          # Skip slow optimization tests
python -m pytest tests/unit/test_camera.py -v  # Run single file with verbose output
python -m pytest tests/unit/ -k "TestProject"  # Run tests matching pattern
python -m pytest tests/ --cov                  # Generate coverage report
```

## Test File Organization

**Location:**
- Unit tests: `tests/unit/test_*.py` — one test file per source module
- Integration tests: `tests/synthetic/` — full pipeline tests with ground truth data
- Naming convention: `test_<module_name>.py` corresponds to `src/aquacal/<module_name>`

**Examples:**
- `tests/unit/test_camera.py` → `src/aquacal/core/camera.py`
- `tests/unit/test_interface_estimation.py` → `src/aquacal/calibration/interface_estimation.py`
- `tests/synthetic/test_full_pipeline.py` → End-to-end pipeline validation
- `docs/tutorials/02_synthetic_validation.ipynb` → Synthetic refractive geometry comparison (moved from tests to tutorial)

**Naming:**
- Test classes: `Test<Feature>` (e.g., `TestCameraProperties`, `TestPackUnpackParams`)
- Test methods: `test_<behavior>` (e.g., `test_principal_point`, `test_round_trip_no_distortion`)
- Descriptive names include expected outcome: `test_point_behind_camera_returns_none`, `test_no_distortion_unchanged`

## Test Structure

**Suite Organization:**
- Tests grouped into classes by feature/concern
- Example (`test_camera.py` lines 22-32):
```python
class TestCameraProperties:
    def test_camera_center_at_origin(self, simple_camera):
        np.testing.assert_allclose(simple_camera.C, np.zeros(3))

    def test_projection_matrix_shape(self, simple_camera):
        assert simple_camera.P.shape == (3, 4)

    def test_projection_matrix_values(self, simple_camera):
        expected = np.hstack([simple_camera.K, np.zeros((3, 1))])
        np.testing.assert_allclose(simple_camera.P, expected)
```

**Fixture Scope:**
- Function scope (default): Most fixtures, rebuilt per test
- Class scope: Expensive synthetic data generation
  - `scenario_ideal`, `scenario_minimal`, `scenario_realistic` in `tests/synthetic/conftest.py`
  - `ideal_result`, `minimal_result`, `realistic_result` in `test_full_pipeline.py`

**Patterns:**
- Setup via fixtures (preferred): `@pytest.fixture` decorators
- Teardown: Implicit (numpy arrays/objects garbage collected); no explicit cleanup needed
- Assertion pattern: `np.testing.assert_allclose(actual, expected, atol=1e-10)` for floats

**Example (`test_interface_estimation.py` lines 119-149):**
```python
class TestComputeInitialBoardPoses:
    def test_computes_poses_for_all_frames(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Computes board pose for each frame with valid detections."""
        detections = generate_synthetic_detections(...)

        poses = _compute_initial_board_poses(...)

        assert len(poses) == len(synthetic_board_poses)
        for frame_idx, pose in poses.items():
            assert isinstance(pose, BoardPose)
            assert pose.frame_idx == frame_idx
            assert pose.rvec.shape == (3,)
            assert pose.tvec.shape == (3,)
```

## Mocking

**Framework:** unittest.mock (implicit; not observed in test files)

**Patterns:**
- Minimal mocking; tests use real objects and synthetic data instead
- Ground truth data generated deterministically: `generate_synthetic_detections()` creates reproducible camera arrays and board poses
- Example (`test_full_pipeline.py` lines 29-83): `_run_calibration_stages()` helper function substitutes for test harness:
  ```python
  def _run_calibration_stages(
      scenario: SyntheticScenario,
      noise_std: float | None = None,
  ) -> CalibrationResult:
      """Run calibration stages 2-3 using synthetic detections."""
      actual_noise = noise_std if noise_std is not None else scenario.noise_std
      detections = generate_synthetic_detections(
          intrinsics=scenario.intrinsics,
          extrinsics=scenario.extrinsics,
          interface_distances=scenario.interface_distances,
          board=board,
          board_poses=scenario.board_poses,
          noise_std=actual_noise,
          seed=42,  # Fixed seed for reproducibility
      )
      # Run real calibration pipeline stages
      pose_graph = build_pose_graph(detections, min_cameras=2)
      initial_extrinsics = estimate_extrinsics(...)
      opt_extrinsics, opt_distances, opt_poses, rms = optimize_interface(...)
  ```

**What to Mock:**
- File I/O: Not needed; tests use generated data in memory
- External libraries: opencv-python, scipy used directly (not mocked)
- Video input: Replaced with synthetic detections from `ground_truth.py`

**What NOT to Mock:**
- Core algorithms: All geometry and optimization code tested with real implementations
- Camera models: Actual `Camera` and `FisheyeCamera` classes instantiated
- Optimization: Real `scipy.optimize.least_squares` runs

## Fixtures and Factories

**Test Data:**
- Factory pattern via fixture generators in conftest files
- Example (`test_interface_estimation.py` lines 52-60):
```python
@pytest.fixture
def intrinsics() -> dict[str, CameraIntrinsics]:
    """Intrinsics for 3 cameras."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return {
        "cam0": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam1": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam2": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
    }
```

- Synthetic scenario factory in `tests/synthetic/conftest.py` (lines 7-22):
```python
@pytest.fixture(scope="class")
def scenario_ideal() -> SyntheticScenario:
    """Ideal scenario: no noise, verify math."""
    return create_scenario("ideal")

@pytest.fixture(scope="class")
def scenario_minimal() -> SyntheticScenario:
    """Minimal scenario: 2 cameras, edge case."""
    return create_scenario("minimal")

@pytest.fixture(scope="class")
def scenario_realistic() -> SyntheticScenario:
    """Realistic scenario: 13 cameras matching actual hardware."""
    return create_scenario("realistic")
```

**Location:**
- Per-test fixtures: In test file or local conftest.py
- Shared fixtures: `tests/synthetic/conftest.py` for synthetic scenarios
- Fixture inheritance: Test files can define fixtures that override/extend parent conftest fixtures

**Ground Truth Generation:**
- Helper function: `generate_synthetic_detections()` in `tests/synthetic/ground_truth.py`
- Usage pattern: Called from within test helper functions (see `_run_calibration_stages`)
- Parameters: `intrinsics`, `extrinsics`, `interface_distances`, `board`, `board_poses`, `noise_std`, optional `seed`
- Returns: `DetectionResult` with deterministic pixel coordinates

**Example (test_interface_estimation.py lines 129-141):**
```python
detections = generate_synthetic_detections(
    intrinsics,
    ground_truth_extrinsics,
    ground_truth_distances,
    board,
    synthetic_board_poses,
    noise_std=0.0,  # No noise for unit tests
    min_corners=4,
)

poses = _compute_initial_board_poses(
    detections, intrinsics, ground_truth_extrinsics, board
)
```

## Coverage

**Requirements:** No explicit target (not enforced in CI/pre-commit)

**View Coverage:**
```bash
python -m pytest tests/ --cov=aquacal --cov-report=html
```

**Coverage observation:**
- Strong coverage in core modules (`core/`, `config/`)
- Good coverage in calibration stages (`calibration/`)
- Test data and validation modules well-tested via synthetic pipeline

## Test Types

**Unit Tests:**
- Location: `tests/unit/`
- Scope: Individual functions and classes in isolation
- Example: `TestCameraProperties` verifies intrinsics matrix construction, projection matrix formula
- Approach: Direct API calls with controlled inputs, numpy array assertions
- Expected runtime: < 1 second each

**Integration Tests:**
- Location: `tests/synthetic/`
- Scope: Full calibration pipeline (Stages 2-3) with known ground truth
- Example: `TestFullPipelineIdeal` runs synthetic detections through extrinsic init and interface optimization
- Approach: Generate synthetic scenario with ground truth geometry, run stages, verify outputs match expected errors
- Marked: `@pytest.mark.slow` (optimization can take seconds)
- Expected runtime: 5-30 seconds for realistic scenario (13 cameras, 100 frames)

**E2E Tests:**
- Not explicitly separate; Synthetic tests serve as E2E validation
- Pipeline orchestrated in `calibration/pipeline.py` tested indirectly through stages

## Common Patterns

**Async Testing:**
- Not applicable (no async code in codebase)

**Error Testing:**
```python
def test_skips_frames_with_insufficient_corners(
    self,
    board,
    intrinsics,
    ground_truth_extrinsics,
    ground_truth_distances,
    synthetic_board_poses,
):
    """Skips frames where no camera has enough corners."""
    detections = generate_synthetic_detections(...)

    # Request very high min_corners to filter everything
    poses = _compute_initial_board_poses(
        detections, intrinsics, ground_truth_extrinsics, board, min_corners=1000
    )

    assert len(poses) == 0  # No poses computed
```

**None Return Testing:**
```python
def test_point_behind_camera_returns_none(self, simple_camera):
    """Point with Z <= 0 in camera frame returns None."""
    assert simple_camera.project(np.array([0, 0, -1])) is None
    assert simple_camera.project(np.array([0, 0, 0])) is None
```

**Array Tolerance Testing:**
```python
def test_round_trip_no_distortion(self, simple_camera):
    """project -> pixel_to_ray_world should recover original ray direction."""
    p_world = np.array([0.5, -0.3, 2.0])
    pixel = simple_camera.project(p_world, apply_distortion=False)
    origin, direction = simple_camera.pixel_to_ray_world(pixel, undistort=False)

    expected_dir = p_world - origin
    expected_dir = expected_dir / np.linalg.norm(expected_dir)
    np.testing.assert_allclose(direction, expected_dir, atol=1e-10)
```

**Parametric Test Pattern (not heavily used, but available):**
```python
def test_different_board_sizes(self):
    """Test BoardGeometry with various board dimensions."""
    for squares_x in [4, 6, 8]:
        for squares_y in [3, 5, 7]:
            config = BoardConfig(
                squares_x=squares_x,
                squares_y=squares_y,
                square_size=0.04,
                marker_size=0.03,
                dictionary="DICT_4X4_50",
            )
            board = BoardGeometry(config)
            assert board.num_corners == squares_x * squares_y
```

## Slow Test Marker

**Purpose:** Distinguish optimization-heavy tests from fast unit tests

**Usage:**
```bash
python -m pytest tests/ -m "not slow"  # Skip slow tests during development
python -m pytest tests/ -m "slow"      # Run only slow tests
python -m pytest tests/                # Run all tests (default)
```

**Marked Tests:**
- `tests/synthetic/test_full_pipeline.py::TestFullPipelineIdeal::test_ideal_extrinsics` — `@pytest.mark.slow`
- `tests/synthetic/test_full_pipeline.py` — Multiple `@pytest.mark.slow` tests (refractive comparison moved to tutorial 02)

## Test Isolation

**Isolation approach:**
- Fixtures with `scope="function"` create fresh data per test
- Class-scoped fixtures (synthetic scenarios) computed once, shared across class tests
- No shared state across test classes

**Example (`test_full_pipeline.py` lines 125-146):**
```python
@pytest.fixture(scope="class")
def ideal_result(scenario_ideal):
    """Run calibration once for all ideal scenario tests."""
    result = _run_calibration_stages(scenario_ideal, noise_std=0.0)
    errors = compute_calibration_errors(result, scenario_ideal)
    return result, errors

class TestFullPipelineIdeal:
    def test_ideal_extrinsics(self, ideal_result):
        """Ideal scenario should recover extrinsics to machine precision."""
        result, errors = ideal_result
        assert errors.max_extrinsic_error < 1e-6
```

---

*Testing analysis: 2026-02-14*
