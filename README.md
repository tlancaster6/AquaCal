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

2. **Create a configuration file** (`config.yaml`):

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
  normal_fixed: true           # Fix interface normal to [0, 0, -1] (default: true)

# Optimization settings
optimization:
  robust_loss: huber           # Robust loss function: huber | soft_l1 | linear (default: huber)
  loss_scale: 1.0              # Scale parameter for robust loss in pixels (default: 1.0)

# Detection filtering
detection:
  min_corners: 8               # Minimum corners required to use a detection (default: 8)
  min_cameras: 2               # Minimum cameras required to use a frame (default: 2)

# Validation settings
validation:
  holdout_fraction: 0.2        # Fraction of frames held out for validation (default: 0.2)
  save_detailed_residuals: true  # Save per-corner residuals to output (default: true)
```

### Field Descriptions

**board**: Defines the ChArUco calibration board used for underwater calibration. ChArUco boards combine a chessboard pattern with ArUco markers for robust detection.

**intrinsic_board** (optional): Separate board specification for in-air intrinsic calibration. Useful if you have a larger or different board for in-air calibration. If omitted, the `board` configuration is used for both stages.

**cameras**: List of camera names. These names are used as keys in the video path dictionaries and in the output calibration file.

**paths**: File paths for input videos and output directory. Each camera must have both an intrinsic (in-air) and extrinsic (underwater) video.

**interface**: Parameters of the refractive interface (water surface). `n_water` should be adjusted for salt water (≈1.34) if applicable. `normal_fixed` determines whether the interface normal is fixed to vertical or optimized.

**optimization**: Controls the optimization behavior. Robust loss functions reduce the influence of outliers. `huber` is recommended for most cases. `loss_scale` determines the threshold in pixels at which errors are considered outliers.

**detection**: Filtering criteria for detected corners. Frames with fewer than `min_corners` total corners or visible in fewer than `min_cameras` cameras are discarded.

**validation**: Controls held-out validation. A random selection of complete frames (across all cameras) is held out for validation. `save_detailed_residuals` includes per-corner reprojection errors in the output.

## Pipeline Overview

The calibration pipeline consists of the following stages:

1. **Intrinsic Calibration**: Standard in-air calibration using the ChArUco board to estimate camera intrinsic parameters (focal length, principal point, distortion coefficients) for each camera independently.

2. **Extrinsic Initialization**: Detects the ChArUco board in underwater videos and builds a pose graph to initialize camera extrinsics (rotation and translation relative to a reference camera) and board poses in 3D space.

3. **Interface and Pose Optimization**: Jointly optimizes camera extrinsics, per-camera interface distances (distance from each camera to the water surface), and all board poses to minimize reprojection error. This is the core refractive calibration stage that accounts for refraction at the water surface using Snell's law.

4. **Validation**: Evaluates calibration quality on held-out frames by computing reprojection errors and 3D reconstruction accuracy. Diagnostics include RMS reprojection error per camera and mean 3D reconstruction error.

For details on coordinate conventions (world frame, camera frame, interface normal direction), see [`docs/COORDINATES.md`](docs/COORDINATES.md).

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