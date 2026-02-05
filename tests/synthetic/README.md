# Synthetic Test Data

This directory contains full pipeline tests using synthetic data with known ground truth.

## Overview

The synthetic testing infrastructure validates the calibration pipeline by:
1. Generating camera arrays with known intrinsics, extrinsics, and interface distances
2. Creating synthetic board trajectories
3. Projecting board corners through refractive interface to generate detections
4. Running calibration stages and comparing results to ground truth

## Modules

### `ground_truth.py`

Core module for synthetic data generation:

- **`SyntheticScenario`**: Dataclass containing complete ground truth
- **`generate_camera_intrinsics()`**: Create intrinsics from FOV and image size
- **`generate_camera_array()`**: Create camera arrays with grid/line/ring layouts
- **`generate_real_rig_array()`**: Create 13-camera array matching real hardware
- **`generate_board_trajectory()`**: Create board pose sequences
- **`generate_real_rig_trajectory()`**: Create trajectory for real rig geometry
- **`generate_synthetic_detections()`**: Project corners through interface
- **`create_scenario()`**: Factory for predefined test scenarios
- **`compute_calibration_errors()`**: Compare results to ground truth

### `conftest.py`

Pytest fixtures for test scenarios:
- `scenario_ideal`: 4 cameras, 0 noise - verify math correctness
- `scenario_minimal`: 2 cameras, edge case - minimum viable config
- `scenario_realistic`: 13 cameras matching real hardware, 0.5px noise

### `test_full_pipeline.py`

Integration tests covering:
- Generation function correctness
- Pipeline recovery of ground truth
- Real rig geometry validation
- Edge cases and error conditions

## Real Rig Geometry

The 13-camera rig has a specific layout:

```
      cam7 (600mm)
       |
cam12--cam1--cam8     (inner ring at 300mm)
  \    |    /
   cam6--cam0--cam2   (cam0 at center)
  /    |    \
cam11--cam5--cam9
       |
      cam10
```

- **cam0**: Center, reference camera (roll=0)
- **cam1-cam6**: Inner ring at 300mm radius, 60° apart
- **cam7-cam12**: Outer ring at 600mm radius, 60° apart
- **Roll angles**: θ + 90° where θ is angular position (X-axis tangent to circle)

## Running Tests

```bash
# Run all synthetic tests
pytest tests/synthetic/ -v

# Run specific test class
pytest tests/synthetic/test_full_pipeline.py::TestRealRigScenario -v

# Run with coverage
pytest tests/synthetic/ --cov=aquacal --cov-report=html
```

## Usage Notes

1. **Skipping Stage 1**: Synthetic tests use ground truth intrinsics directly.
   Stage 1 (intrinsic calibration) requires actual video files with ChArUco
   patterns, which we don't have in synthetic scenarios.

2. **Seed consistency**: All random generators use explicit seeds for
   reproducibility. The `seed` parameter propagates through all RNG calls.

3. **Coordinate conventions**:
   - World frame: Z-down (into water)
   - Board poses place the board underwater at positive Z values
   - Interface normal: [0, 0, -1] points from water toward air

4. **Error tolerances**: Tests use relaxed tolerances because:
   - Stage 2 (extrinsics initialization) uses PnP which ignores refraction
   - Stage 3 optimization must correct for this initialization error
   - Real-world noise propagates through the optimization

5. **Detection generation**: Uses `refractive_project()` to accurately
   simulate how corners appear after refraction at the water surface.
