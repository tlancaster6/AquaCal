# AquaCal

Refractive multi-camera calibration library for arrays of cameras in air viewing an underwater volume through the water surface. AquaCal jointly optimizes camera intrinsics, extrinsics, per-camera interface distances, and board poses to achieve accurate 3D reconstruction in refractive multi-camera systems. Designed for researchers and engineers working with multi-camera underwater imaging setups.

## Installation

AquaCal requires Python 3.10 or later.

Install from source:

```bash
pip install .
```

For development (includes pytest, mypy, black):

```bash
pip install .[dev]
```

## Quick Start

1. **Prepare calibration videos**: Record two sets of videos with all cameras:
   - In-air videos: Move a ChArUco board in front of the cameras above water
   - Underwater videos: Move the same (or different) board underwater

2. **Create a configuration file** (`config.yaml`). See also aquacal init in the CLI reference section below for semi-automated config file generation:

```yaml
board:
  squares_x: 7
  squares_y: 5
  square_size: 0.03
  marker_size: 0.022

cameras: [cam0, cam1, cam2]

paths:
  intrinsic_videos:
    cam0: videos/cam0_inair.mp4
    cam1: videos/cam1_inair.mp4
    cam2: videos/cam2_inair.mp4
  extrinsic_videos:
    cam0: videos/cam0_underwater.mp4
    cam1: videos/cam1_underwater.mp4
    cam2: videos/cam2_underwater.mp4
  output_dir: output/
```

3. **Run calibration**:

```bash
aquacal calibrate config.yaml
```

4. **Find results** in the output directory:
   - `calibration.json`: Camera parameters, interface distances, and diagnostics
   - Additional validation plots and per-frame error reports (if enabled)

## CLI Reference

### Commands

#### `aquacal calibrate <config_path>`

Run the complete calibration pipeline from a YAML configuration file.

**Arguments:**
- `config_path`: Path to configuration YAML file (required)

**Options:**
- `-v, --verbose`: Enable verbose output during calibration
- `-o, --output-dir <path>`: Override the output directory specified in config
- `--dry-run`: Validate configuration file without running calibration

**Examples:**

```bash
# Basic usage
aquacal calibrate config.yaml

# Verbose output
aquacal calibrate config.yaml --verbose

# Override output directory
aquacal calibrate config.yaml -o results/experiment1/

# Validate config without running
aquacal calibrate config.yaml --dry-run
```

#### `aquacal init`

Generate a configuration YAML file by scanning video directories. Extracts camera names from filenames using a regex pattern and writes a starter config with TODO placeholders for board measurements.

**Options:**
- `--intrinsic-dir <path>`: Directory containing in-air calibration videos (required)
- `--extrinsic-dir <path>`: Directory containing underwater calibration videos (required)
- `-o, --output <path>`: Output config file path (default: `config.yaml`)
- `--pattern <regex>`: Regex with one capture group to extract camera name from filename stem (default: `(.+)` = full filename)

**Examples:**

```bash
# Basic usage — camera names from full filename stems
aquacal init --intrinsic-dir videos/intrinsic/ --extrinsic-dir videos/extrinsic/

# Extract camera name from prefix (e.g., "cam0_recording.mp4" → "cam0")
aquacal init --intrinsic-dir inair/ --extrinsic-dir underwater/ --pattern "([^_]+)"

# Specify output path
aquacal init --intrinsic-dir inair/ --extrinsic-dir underwater/ -o my_config.yaml
```

The generated config file includes TODO placeholders for board dimensions that you must fill in before running calibration. Camera names found in one directory but not the other are reported as warnings; only cameras present in both directories are included.

#### `aquacal --version`

Display the installed version of AquaCal.

## Configuration File Reference

The configuration file is a YAML file with the following structure. All fields are shown with their default values (where applicable).

```yaml
# ChArUco board for underwater (extrinsic) calibration
board:
  squares_x: 7                 # Number of chessboard squares in X direction
  squares_y: 5                 # Number of chessboard squares in Y direction
  square_size: 0.03            # Size of each square in meters
  marker_size: 0.022           # Size of ArUco markers in meters
  dictionary: DICT_4X4_50      # ArUco dictionary (default: DICT_4X4_50)
  # legacy_pattern: false      # Set true if board has marker in top-left cell (pre-OpenCV 4.6)

# Optional: separate board for in-air intrinsic calibration
# If not specified, uses the same board as above
# intrinsic_board:
#   squares_x: 12
#   squares_y: 9
#   square_size: 0.025
#   marker_size: 0.018
#   dictionary: DICT_4X4_100

# List of camera identifiers
cameras: [cam0, cam1, cam2]

# Optional: cameras needing 8-coefficient rational distortion model
# rational_model_cameras: [cam2]

# Optional: auxiliary cameras (registered post-hoc, excluded from joint optimization)
# auxiliary_cameras: [overview_cam]

# Video file paths
paths:
  intrinsic_videos:            # In-air videos for intrinsic calibration
    cam0: path/to/cam0_inair.mp4
    cam1: path/to/cam1_inair.mp4
    cam2: path/to/cam2_inair.mp4
  extrinsic_videos:            # Underwater videos for extrinsic calibration
    cam0: path/to/cam0_underwater.mp4
    cam1: path/to/cam1_underwater.mp4
    cam2: path/to/cam2_underwater.mp4
  output_dir: output/          # Directory for calibration results

# Refractive interface (water surface) parameters
interface:
  n_air: 1.0                   # Refractive index of air (default: 1.0)
  n_water: 1.333               # Refractive index of water (default: 1.333, fresh water at 20°C)
  normal_fixed: false          # Estimate ref camera tilt (default); true = assume perpendicular
  # initial_distances:         # Approximate camera-to-water distances (meters, within 2-3x is fine)
  #   cam0: 0.20
  #   cam1: 0.20

# Optimization settings
optimization:
  robust_loss: huber           # Robust loss function: huber | soft_l1 | linear (default: huber)
  loss_scale: 1.0              # Scale parameter for robust loss in pixels (default: 1.0)
  # max_calibration_frames: 150  # Max frames for Stage 3/4 (null = no limit)
  # refine_intrinsics: false     # Stage 4: refine focal lengths and principal points (default: false)

# Detection filtering
detection:
  min_corners: 8               # Minimum corners required to use a detection (default: 8)
  min_cameras: 2               # Minimum cameras required to use a frame (default: 2)
  frame_step: 1                # Process every Nth frame (1 = all, 5 = every 5th)

# Validation settings
validation:
  holdout_fraction: 0.2        # Fraction of frames held out for validation (default: 0.2)
  save_detailed_residuals: true  # Save per-corner residuals to output (default: true)
```

## Technical Methodology

### Problem Setting

AquaCal calibrates a rigid array of cameras mounted in air, viewing downward through a flat water surface into an underwater volume. Standard multi-camera calibration ignores refraction and produces systematic errors when applied across an air-water interface. AquaCal explicitly models refraction at the water surface using Snell's law, treating the interface as a horizontal plane at a per-camera distance below each camera's optical center.

### Refractive Camera Model

Each camera's projection is modeled as a two-segment ray path:

1. **Air segment**: A ray from the camera center through the lens (standard pinhole + distortion model) travels in air to the water surface.
2. **Refraction**: At the air-water interface, the ray refracts according to Snell's law in 3D, bending toward the surface normal as it enters the denser medium.
3. **Water segment**: The refracted ray continues in a straight line to the target point underwater.

Forward projection (3D point to pixel) inverts this path using a Newton-Raphson solver to find the interface intersection point that connects the 3D target to the camera via a physically consistent refracted ray. This replaces the standard pinhole projection used in conventional calibration.

### Calibration Stages

The pipeline estimates camera parameters in four stages, progressing from simple per-camera estimates to a joint global optimization:

**Stage 1 — Intrinsic Calibration**: Each camera is calibrated independently using in-air ChArUco board observations (no refraction). This yields per-camera intrinsic matrices (focal length, principal point) and distortion coefficients via standard OpenCV calibration.

**Stage 2 — Extrinsic Initialization**: Underwater ChArUco detections are used to build a pose graph linking cameras that observe the same board frame. Camera extrinsics (rotation and translation relative to a reference camera) are initialized by chaining pairwise transforms through the graph. Board poses are initialized via refractive PnP — a 6-DOF least-squares refinement of standard PnP that accounts for refraction.

**Stage 3 — Joint Refractive Optimization**: The core calibration step. A nonlinear least-squares optimizer (Levenberg-Marquardt) jointly refines:
- Camera extrinsics (6 DOF per camera, reference camera fixed)
- Per-camera interface distances (1 parameter per camera)
- Board poses (6 DOF per observed frame)

The cost function minimizes reprojection error: for each detected corner, the known 3D board point is projected through the refractive model to a predicted pixel, and the residual against the detected pixel is computed. A Huber robust loss reduces sensitivity to outlier detections.

**Stage 4 — Optional Intrinsic Refinement**: Optionally re-refines focal lengths and principal points alongside extrinsics and interface distances. Useful when in-air intrinsics are not fully representative of the underwater imaging condition.

### Scalability

AquaCal scales to large camera arrays (10+ cameras) and many frames by exploiting the sparse structure of the optimization problem. A configurable frame budget (`max_calibration_frames`) allows uniform temporal subsampling of frames entering optimization while retaining all frames for the earlier initialization stages.

### Validation

A random fraction of detected frames (default 20%) is held out from optimization. Calibration quality is assessed on these held-out frames via:
- **Reprojection error**: RMS pixel distance between detected and predicted corner positions (per-camera and overall).
- **3D reconstruction error**: Adjacent ChArUco corners are triangulated from multi-camera observations and compared to the known board geometry (square size), providing a metric in physical units (mm).

For details on coordinate conventions (world frame, camera frame, interface normal direction), see [`docs/COORDINATES.md`](dev/docs/COORDINATES.md).

## Output

After successful calibration, the output directory contains:

- **`calibration.json`**: Complete calibration result including:
  - Camera intrinsics (K matrix, distortion coefficients, image size)
  - Camera extrinsics (rotation, translation, camera center in world coordinates)
  - Per-camera interface distances
  - Interface parameters (normal vector, refractive indices)
  - Board configuration
  - Diagnostics (RMS reprojection errors, validation 3D errors)
  - Metadata (calibration date, number of frames used, software version)

- **Per-corner residuals** (if `save_detailed_residuals: true`): Detailed reprojection errors for every detected corner, useful for identifying problematic frames or cameras.

The calibration can be loaded and used in downstream applications for 3D reconstruction of underwater scenes.

## Using Results in Python

After calibration, load and use results programmatically:

```python
import numpy as np
from aquacal import load_calibration

# Load a completed calibration
result = load_calibration("output/calibration.json")

# Project a 3D underwater point into a camera's image
pixel = result.project("cam0", np.array([0.1, 0.05, 0.4]))

# Back-project a pixel to a 3D ray in water
origin, direction = result.back_project("cam0", np.array([800.0, 600.0]))

# Access raw calibration data
cam = result.cameras["cam0"]
print(cam.intrinsics.K)                    # 3x3 intrinsic matrix
print(cam.extrinsics.R, cam.extrinsics.t)  # Rotation and translation
print(cam.interface_distance)              # Camera-to-water distance (m)
print(result.diagnostics.reprojection_error_rms)  # Overall RMS error (px)
```

### Top-level imports

The essential API is available directly from the top-level package:

```python
from aquacal import (
    load_calibration,   # Load calibration from JSON
    save_calibration,   # Save calibration to JSON
    CalibrationResult,  # Result type with project/back_project methods
    CameraCalibration,  # Per-camera calibration data
    CameraIntrinsics,   # Intrinsic parameters (K, distortion, image size)
    CameraExtrinsics,   # Extrinsic parameters (R, t)
    run_calibration,    # Run pipeline from config file path
    load_config,        # Load and validate YAML config
)
```

Internals can also be imported from subpackages:

```python
from aquacal.core import Camera, Interface, refractive_project
from aquacal.calibration import optimize_interface
from aquacal.triangulation import triangulate_point
```