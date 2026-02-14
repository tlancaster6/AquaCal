# Technology Stack

**Analysis Date:** 2026-02-14

## Languages

**Primary:**
- Python 3.10+ - All source code, tests, and utilities
  - Type hints via `numpy.typing.NDArray` with shape annotations in docstrings
  - Uses standard library: dataclasses, pathlib, json, argparse

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12 (specified in `pyproject.toml` classifiers)

**Package Manager:**
- pip (setuptools-based installation)
- Lockfile: Not enforced (users manage via virtual environments)

## Frameworks

**Core Computation:**
- NumPy (numerical arrays and linear algebra)
- SciPy (optimization: `scipy.optimize.least_squares`, `scipy.optimize._numdiff` for sparse derivatives)

**Computer Vision:**
- OpenCV (`opencv-python>=4.6`)
  - Used for: intrinsic calibration, ChArUco board detection, camera matrix operations
  - Modules: `cv2.calibrateCamera()`, `cv2.aruco.*` functions
  - File references: `src/aquacal/calibration/intrinsics.py`, `src/aquacal/io/detection.py`

**Data Processing & Visualization:**
- Matplotlib (plotting and diagnostics visualization)
  - Used in: `src/aquacal/validation/diagnostics.py`, `src/aquacal/validation/comparison.py`
  - Non-interactive backend (`matplotlib.use("Agg")`) for headless operation
- Pandas (data analysis, CSV output)
  - Used in: `src/aquacal/validation/reconstruction.py`, `src/aquacal/validation/comparison.py`

**Configuration:**
- PyYAML (YAML parsing)
  - Config file format: `src/aquacal/config/example_config.yaml`
  - Loaded via: `src/aquacal/calibration/pipeline.py:load_config()`

## Key Dependencies

**Critical (Core Pipeline):**
- `numpy` - Coordinate transforms, matrix operations, refractive geometry math
  - Used throughout: `src/aquacal/core/`, `src/aquacal/calibration/`, `src/aquacal/triangulation/`
- `scipy` - Nonlinear least-squares optimization with sparse Jacobians
  - Key: `scipy.optimize.least_squares()` in `src/aquacal/calibration/_optim_common.py`
  - Sparse FD via `scipy.optimize._numdiff.approx_derivative()` + `group_columns()`
- `opencv-python` - Camera calibration and marker detection
  - Key: `cv2.calibrateCamera()` in `src/aquacal/calibration/intrinsics.py`
  - Key: ChArUco detection in `src/aquacal/io/detection.py`

**Infrastructure:**
- `pyyaml` - Configuration loading
- `matplotlib` - Visualization (diagnostics, comparisons, camera rigs)
- `pandas` - Data export and analysis

**Testing (Optional):**
- `pytest` - Test runner
- `pytest-cov` - Coverage reporting
- `mypy` - Static type checking
- `black` - Code formatting

## Configuration

**Environment:**
- Configuration via YAML files (e.g., `config.yaml`)
- No environment variables required for normal operation
- Key config sections:
  - `board`: ChArUco board dimensions and ArUco dictionary
  - `cameras`: List of camera names
  - `paths`: Input video paths and output directory
  - `interface`: Water surface refractive indices (n_air, n_water)
  - `optimization`: Loss function, refinement options
  - `detection`: Board detection thresholds
  - `validation`: Holdout fraction and reporting options

**Build:**
- `pyproject.toml`: PEP 518 build system with setuptools backend
- Entry point: `aquacal = "aquacal.cli:main"` → `src/aquacal/cli.py`

## Platform Requirements

**Development:**
- Python 3.10+ interpreter
- C dependencies: OpenCV (pre-built wheels available)
- No GPU acceleration (NumPy/SciPy use CPU-only by default)

**Production:**
- Python 3.10+ interpreter
- Input: MP4 video files (any codec OpenCV can read)
- Output: JSON results, optional PNG plots (requires matplotlib + Agg backend)
- Disk space: Depends on calibration frame count and output diagnostics

## Notable Absence

**Not Detected:**
- No web framework (Flask, FastAPI, Django)
- No database connectivity (SQLite, PostgreSQL, etc.)
- No cloud SDKs (AWS, GCP, Azure)
- No network/API clients (requests library)
- No real-time streaming (gRPC, WebSockets)
- No containerization (Docker not in repo, but compatible)
- No CI/CD automation files (.github/workflows, .gitlab-ci.yml)

---

*Stack analysis: 2026-02-14*
