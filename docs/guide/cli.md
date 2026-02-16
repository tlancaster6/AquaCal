# CLI Reference

AquaCal provides three command-line tools for calibration workflows. All values are in meters.

## aquacal calibrate

Run the complete calibration pipeline from a configuration file.

**Syntax:**
```bash
aquacal calibrate <config_path> [options]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `config_path` | Path | Path to configuration YAML file (required) |

**Options:**

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Enable verbose output showing optimizer iteration details |
| `-o`, `--output-dir PATH` | Override the output directory specified in config |
| `--dry-run` | Validate configuration file without running calibration |

**Examples:**

```bash
# Run calibration with default settings
aquacal calibrate config.yaml

# Run with verbose output to see optimizer progress
aquacal calibrate config.yaml -v

# Validate config without running calibration
aquacal calibrate config.yaml --dry-run

# Override output directory
aquacal calibrate config.yaml -o /path/to/output
```

**Exit Codes:**
- `0`: Success
- `1`: File not found or I/O error
- `2`: Invalid configuration
- `3`: Calibration failed (e.g., insufficient data, optimization divergence)
- `130`: User interrupt (Ctrl+C)

---

## aquacal init

Generate a configuration file by scanning video directories.

**Syntax:**
```bash
aquacal init --intrinsic-dir PATH --extrinsic-dir PATH [options]
```

**Required Options:**

| Flag | Type | Description |
|------|------|-------------|
| `--intrinsic-dir PATH` | Path | Directory containing in-air calibration videos |
| `--extrinsic-dir PATH` | Path | Directory containing underwater calibration videos |

**Optional Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output PATH` | `config.yaml` | Output path for generated config file |
| `--pattern REGEX` | `(.+)` | Regex with one capture group to extract camera name from filename stem |

**How It Works:**

1. Scans both directories for video files (`.mp4`, `.avi`, `.mov`, `.mkv`)
2. Extracts camera names from filenames using the regex pattern (default: entire filename stem)
3. Matches cameras between intrinsic and extrinsic directories
4. Generates a config YAML with placeholder board parameters and matched video paths

**Examples:**

```bash
# Basic usage with default pattern
aquacal init --intrinsic-dir videos/in_air --extrinsic-dir videos/underwater

# Extract camera names from filenames like "camera_cam0_inair.mp4" → "cam0"
aquacal init --intrinsic-dir in_air/ --extrinsic-dir underwater/ --pattern "camera_(.+)_"

# Custom output path
aquacal init --intrinsic-dir in_air/ --extrinsic-dir underwater/ -o my_config.yaml
```

**Camera Name Extraction:**

The `--pattern` regex must have exactly one capture group. For example:

| Filename | Pattern | Extracted Name |
|----------|---------|----------------|
| `cam0.mp4` | `(.+)` | `cam0` |
| `camera_cam0_inair.mp4` | `camera_(.+)_inair` | `cam0` |
| `2024-01-15_cam0.mp4` | `[0-9-]+_(.+)` | `cam0` |

**Exit Codes:**
- `0`: Success
- `1`: Directory not found, no videos found, no common cameras, or invalid regex pattern

---

## aquacal compare

Compare multiple calibration runs and generate diagnostic reports.

**Syntax:**
```bash
aquacal compare <dir1> <dir2> [dir3 ...] [options]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `directories` | Paths | Two or more directories containing `calibration.json` files (minimum 2) |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output-dir PATH` | `comparison_output` | Output directory for comparison results |
| `--no-plots` | False | Skip PNG plot generation (only generate CSV files) |

**Output Files:**

The command generates:
- `metrics.csv` — Overall metrics (RMS, camera counts) for each run
- `per_camera.csv` — Per-camera metrics across all runs
- `parameter_diffs.csv` — Pairwise differences in extrinsics and water_z
- `rms_bar_chart.png` — Bar chart comparing RMS reprojection errors
- `position_overlay.png` — 3D scatter plot of camera positions across runs
- `z_position_dumbbell.png` — Camera Z-positions with connecting lines between runs
- `depth_error_plot.png` — (if spatial_measurements.csv exists) Error vs. depth analysis
- `depth_binned.csv` — (if spatial data available) Binned error statistics

**Examples:**

```bash
# Compare two calibration runs
aquacal compare output_run1/ output_run2/

# Compare three runs with custom output directory
aquacal compare run1/ run2/ run3/ -o comparison_results/

# Generate only CSV files (no plots)
aquacal compare run1/ run2/ --no-plots
```

**Exit Codes:**
- `0`: Success
- `1`: Directory not found, calibration.json missing, or I/O error
- `2`: Comparison failed (e.g., incompatible camera sets)

---

## Global Options

| Flag | Description |
|------|-------------|
| `--version` | Display AquaCal version and exit |

**Example:**
```bash
aquacal --version
# Output: aquacal 0.1.0
```

---

## Configuration File Format

All three commands work with YAML configuration files. The `init` command generates a template config with:

- **board**: ChArUco board parameters (squares_x, squares_y, square_size, marker_size, dictionary)
- **cameras**: List of camera names (first camera is the reference camera, world origin)
- **paths**: Intrinsic and extrinsic video paths per camera
- **interface**: Refractive indices and optional initial water_z estimates
- **optimization**: Robust loss settings, max frames, intrinsic refinement flags
- **detection**: Corner detection thresholds and frame sampling
- **validation**: Holdout fraction for validation set

After running `init`, you must edit the generated config to:
1. Set accurate board dimensions (measure your physical board)
2. Optionally add `initial_water_z` estimates per camera (improves Stage 3 initialization)
3. Configure `auxiliary_cameras`, `rational_model_cameras`, or `fisheye_cameras` if needed

See the [User Guide](index.md) for detailed explanations of calibration theory and the [API Reference](../api/index.rst) for programmatic usage.
