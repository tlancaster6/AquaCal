# Phase 4: Example Data - Research

**Date:** 2026-02-14
**Researcher:** Claude (GSD Phase Researcher)
**Objective:** What do I need to know to PLAN this phase well?

---

## Executive Summary

Phase 4 will add synthetic data generation API and real dataset support to AquaCal, enabling researchers to quickly test the library with both ground-truth-validated synthetic scenarios and production-quality real rig calibrations. The implementation requires:

1. **Refactoring existing test helpers** from `tests/synthetic/ground_truth.py` into a public API at `aquacal.datasets`
2. **Rendering synthetic ChArUco images** using OpenCV drawing functions (refractive projection + overlay)
3. **Downloading and caching datasets** from Zenodo using requests + tqdm
4. **Including small datasets in-package** using setuptools package_data
5. **Managing cache location** in working directory (`./aquacal_data/`) for transparency

The codebase already has excellent foundations: `generate_synthetic_detections()`, `generate_camera_array()`, `generate_real_rig_array()`, and complete `SyntheticScenario` dataclasses. The main work is (a) refactoring these into public API with fixed presets, (b) adding synthetic image rendering, and (c) building dataset download/loading infrastructure.

---

## 1. Existing Codebase Analysis

### 1.1 Synthetic Test Infrastructure

**Location:** `/c/Users/tucke/PycharmProjects/AquaCal/tests/synthetic/ground_truth.py` (740 lines)

**Key Components:**

1. **`SyntheticScenario` dataclass** (lines 26-38):
   - Complete ground truth bundle: intrinsics, extrinsics, interface_distances, board_poses, board_config
   - Already includes `name` and `description` fields for preset identification
   - Ready to be the return type for public API

2. **Camera Array Generation:**
   - `generate_camera_array()` (lines 103-194): Flexible layouts (grid/line/ring), parameterized spacing
   - `generate_real_rig_array()` (lines 197-302): **13-camera concentric ring rig** matching production hardware
     - Center camera at origin
     - Inner ring: 6 cameras at 300mm radius
     - Outer ring: 6 cameras at 600mm radius
     - Realistic specs: 1600×1200 resolution, 56° horizontal FOV
   - Both support height variation via RNG seed for reproducibility

3. **Board Trajectory Generation:**
   - `generate_board_trajectory()` (lines 305-355): Generic trajectory with pose graph connectivity guarantees
   - `generate_real_rig_trajectory()` (lines 358-406): Tuned for real rig depth range (0.9-1.5m)
   - `generate_dense_xy_grid()` (lines 409-457): Regular grid for heatmap evaluation

4. **Projection and Detection:**
   - `generate_synthetic_detections()` (lines 460-541): **This is the core function to refactor**
     - Projects board corners through refractive interface (`refractive_project()`)
     - Adds Gaussian pixel noise (configurable std)
     - Filters corners outside image bounds
     - Returns `DetectionResult` with `FrameDetections` structure
     - Already works with all schema types (Camera, Interface, BoardGeometry)

5. **Predefined Scenarios:**
   - `create_scenario()` (lines 544-655): String-based presets
     - `"ideal"`: 4 cameras, 20 frames, 0 noise (verify math correctness)
     - `"minimal"`: 2 cameras, 10 frames, 0.3px noise (edge case)
     - `"realistic"`: 13 cameras, 30 frames, 0.5px noise (matches real hardware)
   - Board config: 12×9 ChArUco, 60mm squares, 45mm markers, DICT_5X5_100
   - All use fixed random seed for reproducibility

**Refactoring Strategy:**
- Move `SyntheticScenario`, `generate_synthetic_detections()`, and helper functions to `aquacal.datasets.synthetic`
- Rename `create_scenario()` to `generate_synthetic_rig()`
- Replace parameterized scenarios with fixed presets: `'small'`, `'medium'`, `'large'`
- Keep test file but import from public API to avoid duplication

### 1.2 Data Schema (Ready to Use)

**Location:** `/c/Users/tucke/PycharmProjects/AquaCal/src/aquacal/config/schema.py`

All required types exist and are stable:
- `BoardConfig` (lines 28-47): ChArUco specification
- `CameraIntrinsics` (lines 50-65): K matrix, distortion, image size, fisheye flag
- `CameraExtrinsics` (lines 68-89): R, t, with `.C` property for camera center
- `BoardPose` (lines 267-279): frame_idx, rvec, tvec
- `Detection` (lines 282-301): corner_ids, corners_2d
- `FrameDetections` (lines 304-332): Per-frame multi-camera detections
- `DetectionResult` (lines 335-358): Full detection bundle with filtering utilities
- `CalibrationResult` (lines 132-148): Complete calibration output (for reference dataset)

**Implications:**
- No new schema types needed
- Return types for `generate_synthetic_rig()` and `load_example()` can use existing dataclasses
- Possible addition: `ExampleDataset` dataclass to bundle detections + ground truth + metadata

### 1.3 Serialization and I/O Patterns

**Location:** `/c/Users/tucke/PycharmProjects/AquaCal/src/aquacal/io/serialization.py` (276 lines)

**Key Patterns:**
- JSON serialization with version field (`SERIALIZATION_VERSION = "1.0"`)
- Helper functions: `_ndarray_to_list()`, `_list_to_ndarray()` for numpy arrays
- Recursive serialization of nested dataclasses
- Handles optional fields (e.g., `is_fisheye`, `is_auxiliary`)
- Type-safe deserialization with validation

**Reusable for Datasets:**
- Can serialize/deserialize `DetectionResult` for cached synthetic data
- Can save/load ground truth bundles (intrinsics, extrinsics, board poses)
- Pattern to follow: version header + recursive dict conversion

**Location:** `/c/Users/tucke/PycharmProjects/AquaCal/src/aquacal/io/video.py` (lines 1-100 shown)

**Video Loading Pattern:**
- `VideoSet` class: lazy initialization, context manager protocol
- Validates file existence on construction
- Property-based API (`frame_count`, `is_open`)
- Error handling: `FileNotFoundError` for missing files

**Implications for Dataset Loading:**
- Follow similar validation pattern (check cache exists, download if missing)
- Context manager not needed (datasets are static files, not streams)
- Return structured objects, not raw file paths

---

## 2. Synthetic Image Rendering

### 2.1 ChArUco Board Rendering in OpenCV

**Official Documentation:**
- [OpenCV: cv::aruco::CharucoBoard Class Reference](https://docs.opencv.org/4.x/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html)
- [OpenCV: Detection of ChArUco Boards](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html)

**Modern Syntax (OpenCV 4.6+):**
```python
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_Y, SQUARES_X),  # Note: (rows, cols) order
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)
img = cv2.aruco.CharucoBoard.generateImage(
    board,
    (width, height),  # Output size in pixels
    marginSize=20     # Border in pixels
)
```

**Legacy Syntax (Pre-4.6):**
```python
board = cv2.aruco.CharucoBoard_create(...)  # Deprecated
img = board.draw((width, height), marginSize=20)
```

**Key Points:**
- Much online code is outdated; use `cv2.aruco.CharucoBoard()` constructor, not `_create()`
- `generateImage()` returns a numpy array (grayscale uint8)
- Board is clean synthetic pattern (perfect corners, no noise)
- Size parameter is output image dimensions, not physical size

**Rendering Strategy for Phase 4:**
1. **In-air boards (intrinsic calibration):**
   - Use `generateImage()` directly — pinhole projection, no refraction
   - Alternative: Project board corners via pinhole model, draw detected pattern
   - Decision: User discretion (CONTEXT.md: "Above-water board projection method")

2. **Underwater boards (extrinsic calibration):**
   - **Cannot use `generateImage()`** — must render refracted projection
   - Algorithm:
     1. Generate 3D board corner positions (from BoardPose)
     2. Project each corner through refractive interface (`refractive_project()`)
     3. Create black image at camera resolution
     4. Draw ChArUco pattern at projected positions using OpenCV drawing functions
   - Drawing approach:
     - `cv2.drawChessboardCorners()` (for detected corners overlay)
     - OR manual: `cv2.circle()` for corners, `cv2.line()` for grid, `cv2.aruco.drawDetectedMarkers()` for markers
   - **Challenge:** ArUco marker rendering at projected positions
     - OpenCV `drawDetectedMarkers()` expects marker corners, not board corners
     - May need to skip markers, render only checkerboard grid + corners
     - OR: Project marker corners separately, use `cv2.fillPoly()` for marker quads

**Complexity Assessment:**
- In-air: Trivial (use `generateImage()`)
- Underwater: Moderate — need custom rendering loop
- **Recommendation:** Start with corner-only rendering (circles + connecting lines), defer marker rendering if complex
- Validate: Rendered corners should match `generate_synthetic_detections()` pixel coordinates exactly (same refractive projection)

**References:**
- [Using ChArUco boards in OpenCV](https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d)
- [OpenCV Examples: CalibrateCamera.py](https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py)

### 2.2 Rendering Requirements from CONTEXT.md

**Decisions:**
- Minimal rendering: black background with ChArUco pattern
- Correct refractive geometry (not photorealistic)
- Must support both above-water and below-water boards
- Full frame at realistic resolution (e.g., 1920×1080), not cropped

**Design Implications:**
- Single rendering function: `render_synthetic_frame(camera, board_pose, board_geometry, interface, image_size, underwater=True)`
- Returns: `NDArray[np.uint8]` (grayscale or BGR)
- Optional: Cache rendered images for large presets (avoid re-rendering 1000+ frames)

---

## 3. Dataset Hosting and Download Infrastructure

### 3.1 Zenodo for Academic Datasets

**Why Zenodo:**
- DOI-backed (persistent citation)
- Academic credibility (CERN-operated)
- Free for open data (up to 50GB per dataset)
- Versioning support
- REST API for programmatic downloads

**Official Resources:**
- [Zenodo Developers Guide](https://developers.zenodo.org/)
- [GitHub: zenodo_get](https://github.com/dvolgyes/zenodo_get) — Python CLI tool for Zenodo downloads

**API Example (from Zenodo docs):**
```python
import requests

# Get record metadata
record_id = "1234567"
response = requests.get(f"https://zenodo.org/api/records/{record_id}")
files = response.json()["files"]

# Download file with progress
file_url = files[0]["links"]["self"]
file_name = files[0]["key"]

response = requests.get(file_url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(file_name, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

**Integration with tqdm:**
Multiple working examples found:
- [GitHub Gist: Python requests download with tqdm](https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51)
- [Visualizing Download Progress with tqdm](https://blasferna.com/articles/visualizing-download-progress-with-tqdm-in-python/)

**Standard Pattern:**
```python
import requests
from tqdm import tqdm

response = requests.get(url, stream=True, allow_redirects=True)
total_size = int(response.headers.get('content-length', 0))

with open(dest_path, 'wb') as f:
    with tqdm(
        desc=dest_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
```

**References:**
- [Zenodo: Part 6 - Upload/Download Guide](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/technical-guidelines/zenodo)
- [zenodo-get on PyPI](https://pypi.org/project/zenodo-get/)

### 3.2 Dataset Size Strategy

**From CONTEXT.md Decisions:**
- Real dataset: 9+ cameras, extracted frames (not raw video)
- Frame count TBD — experiment to find sweet spot between quality and size
- Include reference calibration result from production-quality run (low framestep)
- Synthetic medium and large presets also hosted on Zenodo

**Realistic Size Estimates:**

**Synthetic Datasets:**
- Small preset: 2-4 cameras, 10-20 frames
  - Detections JSON: ~50KB
  - With images (1920×1080 grayscale PNG): ~2-5MB
  - **Ship in-package** (verified installation without download)

- Medium preset: 6-8 cameras, 50-100 frames
  - Detections JSON: ~500KB
  - With images: ~50-100MB
  - **Host on Zenodo** (too large for PyPI wheel)

- Large preset: 13 cameras (real rig geometry), 200-500 frames
  - Detections JSON: ~2-5MB
  - With images: ~200-500MB
  - **Host on Zenodo**

**Real Dataset:**
- 9-13 cameras, intrinsic + extrinsic captures
- Intrinsic: ~50 frames per camera (grid coverage)
- Extrinsic: ~100 frames per camera (underwater trajectory)
- Image format: PNG or JPEG at ~1600×1200
  - PNG: ~500KB per frame
  - JPEG (quality 95): ~200KB per frame
- Total estimate:
  - Intrinsic: 13 cameras × 50 frames × 200KB = ~130MB
  - Extrinsic: 13 cameras × 100 frames × 200KB = ~260MB
  - **Combined: ~400MB** (requires frame subsampling or JPEG compression)
- Reference calibration JSON: ~50KB

**Compression Strategy:**
- ZIP compression: ~30-50% reduction for images
- Final real dataset target: **<200MB ZIP** (acceptable for Zenodo, one-time download)
- Alternative: Split into intrinsic.zip + extrinsic.zip for modular downloads

### 3.3 Cache Location Best Practices

**Research Findings:**

**Avoid `~/.cache/` or `Path.home()`:**
- [Prevent pip filling up home directory](https://hpc.ncsu.edu/Software/python_cache_directory.php)
- [CSC Docs: Python pip cache FAQ](https://docs.csc.fi/support/faq/python-pip-cache/)
- Issue: Home directories often have strict quotas on HPC systems
- pip's `~/.cache/pip` can grow to gigabytes, causing quota errors

**User's Decision (CONTEXT.md):**
- **Working directory cache:** `./aquacal_data/`
- Rationale: Transparency, easy to find/delete, no hidden accumulation
- Include `.gitignore` inside cache directory (prevents accidental commits)

**Implementation:**
```python
from pathlib import Path

def get_cache_dir() -> Path:
    """Get or create the dataset cache directory."""
    cache_dir = Path.cwd() / "aquacal_data"
    cache_dir.mkdir(exist_ok=True)

    # Create .gitignore if it doesn't exist
    gitignore = cache_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n")  # Ignore all contents

    return cache_dir
```

**Directory Structure:**
```
./aquacal_data/
├── .gitignore
├── synthetic-medium.zip
├── synthetic-medium/          # Extracted
│   ├── detections.json
│   ├── ground_truth.json
│   └── images/
│       ├── cam0/
│       ├── cam1/
│       └── ...
├── real-rig.zip
└── real-rig/
    ├── intrinsic/
    ├── extrinsic/
    ├── config.yaml
    └── reference_calibration.json
```

**Alternative Consideration (cross-platform):**
- [appdirs on PyPI](https://pypi.org/project/appdirs/) — platform-specific cache dirs
- Pattern: `appdirs.user_cache_dir("aquacal", "AquaCal")`
- Would use `~/.cache/aquacal` on Linux, `~/Library/Caches/aquacal` on macOS, `%LOCALAPPDATA%\aquacal` on Windows
- **Decision:** Stick with working directory per user preference (explicit over implicit)

**References:**
- [pip Caching Documentation](https://pip.pypa.io/en/stable/topics/caching/)
- [cachepath on PyPI](https://pypi.org/project/cachepath/)

---

## 4. Python Packaging: Including Data Files

### 4.1 Modern Approach with pyproject.toml

**Official Documentation:**
- [Setuptools: Data Files Support](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
- [Configuring setuptools using pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)
- [How to Specify Package Data in pyproject.toml](https://www.pythontutorials.net/blog/specifying-package-data-in-pyproject-toml/)

**Current `pyproject.toml` Status:**
- Project uses setuptools build backend
- No `[tool.setuptools.package-data]` section currently
- Packages found via `[tool.setuptools.packages.find]` with `where = ["src"]`

**Adding Package Data:**
```toml
[tool.setuptools.package-data]
aquacal = [
    "datasets/small/*.json",
    "datasets/small/images/**/*.png",
    "datasets/manifests/*.json",
]
```

**Key Points:**
- Glob patterns supported: `*.json`, `**/*.png` (recursive)
- Files must be inside package directory: `src/aquacal/datasets/...`
- `package_data` applies to wheels (binary distributions)
- MANIFEST.in only needed for source distributions (sdist), not required here

**Implementation Plan:**
1. Create `src/aquacal/datasets/` module
2. Add small preset data:
   ```
   src/aquacal/datasets/
   ├── __init__.py
   ├── synthetic.py        # generate_synthetic_rig()
   ├── loader.py           # load_example()
   ├── download.py         # _download_with_progress()
   ├── manifests/
   │   └── datasets.json   # Zenodo URLs, checksums, metadata
   └── small/              # In-package small preset
       ├── detections.json
       ├── ground_truth.json
       └── images/
           ├── cam0_frame00.png
           └── ...
   ```
3. Update `pyproject.toml` with package_data globs
4. Manifest file lists available datasets:
   ```json
   {
     "small": {
       "type": "synthetic",
       "included": true,
       "path": "small/"
     },
     "medium": {
       "type": "synthetic",
       "included": false,
       "zenodo_record": "1234567",
       "zenodo_filename": "synthetic-medium.zip",
       "checksum_sha256": "abc123...",
       "size_mb": 85
     },
     "real-rig": {
       "type": "real",
       "included": false,
       "zenodo_record": "7654321",
       "zenodo_filename": "real-rig.zip",
       "checksum_sha256": "def456...",
       "size_mb": 180
     }
   }
   ```

**Accessing Package Data at Runtime:**
```python
from importlib.resources import files
import json

# Load manifest
manifest_text = files("aquacal.datasets").joinpath("manifests/datasets.json").read_text()
manifest = json.loads(manifest_text)

# Load small preset (included)
small_detections = files("aquacal.datasets").joinpath("small/detections.json").read_text()
```

**References:**
- [Python Packaging User Guide: Using MANIFEST.in](https://packaging.python.org/en/latest/guides/using-manifest-in/)
- [GitHub: setuptools package_data experiment](https://github.com/abravalheri/experiment-setuptools-package-data)

### 4.2 MANIFEST.in (If Needed)

**Current Status:**
- No MANIFEST.in file exists in repository
- Not required for package_data in wheels (setuptools handles it automatically)

**When MANIFEST.in is Needed:**
- Including non-Python files in **source distribution (sdist)** only
- Top-level files outside package directory (e.g., README.md, LICENSE)
- Already handled by setuptools defaults for common files

**Decision:** Not needed for Phase 4. Package data handled via `pyproject.toml`.

**Reference:**
- [Is MANIFEST.in still needed? - pyOpenSci Forum](https://pyopensci.discourse.group/t/is-manifest-in-still-needed-to-include-data-in-a-package-built-with-setuptools/392)

---

## 5. Dataset Loading UX Patterns

### 5.1 Inspiration from scikit-learn

**Pattern: `fetch_openml()`**
- [scikit-learn fetch_openml documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)

**Key Features:**
- Auto-download on first call
- Cache in `~/scikit_learn_data` by default
- `data_home` parameter for custom cache location
- Returns structured Bunch object (dict-like with attribute access)
- Retry on HTTP errors (`n_retries`, `delay` parameters)
- Parser selection (`parser='auto'` uses best available)

**Relevant for AquaCal:**
- Cache directory pattern (but use working dir instead of home)
- Structured return object (not just file paths)
- Retry/timeout configuration
- Progress indication (scikit-learn doesn't show progress, but we should)

**Example Return Object:**
```python
@dataclass
class ExampleDataset:
    """Loaded example dataset."""
    name: str
    type: str  # "synthetic" or "real"
    detections: DetectionResult
    ground_truth: SyntheticScenario | None  # Only for synthetic
    calibration_result: CalibrationResult | None  # Only for real (reference)
    metadata: dict
    cache_path: Path
```

### 5.2 Proposed API

**Synthetic Data Generation:**
```python
from aquacal.datasets import generate_synthetic_rig

# Generate synthetic scenario (always fresh, no caching)
scenario = generate_synthetic_rig('small')  # or 'medium', 'large'

# Returns SyntheticScenario with:
# - intrinsics, extrinsics, interface_distances (ground truth)
# - board_poses, board_config
# - detections (DetectionResult)
# - name, description

# Optional: generate with images
scenario_with_images = generate_synthetic_rig('small', include_images=True)
# Adds: scenario.images = dict[str, dict[int, NDArray]]  # cam_name -> frame_idx -> image
```

**Dataset Loading:**
```python
from aquacal.datasets import load_example

# Load small preset (included in package, instant)
small = load_example('small')

# Load medium preset (downloads from Zenodo on first call, cached thereafter)
medium = load_example('medium')  # Shows tqdm progress bar

# Load real rig dataset
real = load_example('real-rig')  # ~180MB download, one-time

# All return ExampleDataset with unified interface
print(small.detections.total_frames)
print(medium.ground_truth.intrinsics['cam0'].K)
print(real.calibration_result.diagnostics.reprojection_error_rms)
```

**Cache Management:**
```python
from aquacal.datasets import get_cache_info, clear_cache

# Inspect cache
info = get_cache_info()
# Returns: {"cache_dir": Path, "datasets": {...}, "total_size_mb": 265}

# Clear specific dataset
clear_cache('medium')

# Clear all
clear_cache()
```

### 5.3 Download Error Handling

**Considerations:**
- Network failures (timeout, connection reset)
- Partial downloads (interrupted)
- Checksum validation (detect corruption)
- Zenodo API rate limiting (unlikely but possible)
- Disk space errors

**Implementation Strategy:**
```python
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

def download_with_progress(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    max_retries: int = 3
) -> None:
    """Download file with progress bar and validation."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(dest, 'wb') as f:
                with tqdm(
                    desc=f"Downloading {dest.name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)

            # Validate checksum
            if expected_sha256:
                actual = hashlib.sha256(dest.read_bytes()).hexdigest()
                if actual != expected_sha256:
                    dest.unlink()  # Remove corrupted file
                    raise ValueError(f"Checksum mismatch: {actual} != {expected_sha256}")

            return  # Success

        except (requests.RequestException, ValueError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")
            print(f"Download failed (attempt {attempt + 1}/{max_retries}), retrying...")
```

**Checksum Validation:**
- Store SHA256 hashes in manifest file
- Compute hash after download
- Delete and retry if mismatch
- Prevents corrupted data from being cached

**References:**
- [Implementing Progress Bar for Downloads - DNMTechs](https://dnmtechs.com/implementing-a-progress-bar-for-http-file-downloads-using-python-3-and-requests/)
- [Download Large Files with Progress Bar - AI Mind](https://pub.aimind.so/download-large-file-in-python-with-beautiful-progress-bar-f4f86b394ad7)

---

## 6. Implementation Checklist

### 6.1 Synthetic Data API

**Files to Create:**
- `src/aquacal/datasets/__init__.py` — Public API exports
- `src/aquacal/datasets/synthetic.py` — `generate_synthetic_rig()`, rendering functions
- `src/aquacal/datasets/schemas.py` — `ExampleDataset` dataclass (if needed)

**Refactoring Tasks:**
1. Copy functions from `tests/synthetic/ground_truth.py`:
   - `SyntheticScenario` → reuse from test file or move to public schema
   - `generate_camera_intrinsics()`, `generate_camera_array()`, `generate_real_rig_array()`
   - `generate_board_trajectory()`, `generate_real_rig_trajectory()`
   - `generate_synthetic_detections()`

2. Create `generate_synthetic_rig()`:
   - Replace `create_scenario()` logic
   - Fixed presets: `'small'`, `'medium'`, `'large'`
   - Fixed random seeds per preset (reproducibility)
   - Default board config (7×5 ChArUco, 30mm squares per CONTEXT.md decision)

3. Implement preset definitions:
   ```python
   PRESETS = {
       'small': {
           'n_cameras': 2,
           'n_frames': 10,
           'layout': 'line',
           'noise_std': 0.3,
           'seed': 42,
       },
       'medium': {
           'n_cameras': 6,
           'n_frames': 80,
           'layout': 'grid',
           'noise_std': 0.5,
           'seed': 123,
       },
       'large': {
           'n_cameras': 13,
           'n_frames': 300,
           'layout': 'real_rig',  # Use generate_real_rig_array()
           'noise_std': 0.5,
           'seed': 456,
       },
   }
   ```

4. Implement image rendering:
   - `render_synthetic_frame(camera, board_pose, board_geometry, interface, underwater=True)`
   - In-air: Use `cv2.aruco.CharucoBoard.generateImage()` OR pinhole projection + drawing
   - Underwater: Refractive projection + manual rendering (circles/lines for corners, skip markers initially)

**Testing:**
- Update `tests/synthetic/conftest.py` to import from public API
- Validate that existing tests still pass
- Add new test: `test_generate_synthetic_rig_reproducibility()` (same preset = same output)

### 6.2 Dataset Download Infrastructure

**Files to Create:**
- `src/aquacal/datasets/download.py` — Download, extraction, caching logic
- `src/aquacal/datasets/loader.py` — `load_example()`, cache management
- `src/aquacal/datasets/manifests/datasets.json` — Dataset metadata

**Implementation Tasks:**

1. **Manifest File:**
   ```json
   {
     "version": "1.0",
     "datasets": {
       "small": {
         "type": "synthetic",
         "included": true,
         "relative_path": "small/",
         "description": "2 cameras, 10 frames, minimal scenario"
       },
       "medium": {
         "type": "synthetic",
         "included": false,
         "zenodo_record_id": "TBD",
         "zenodo_filename": "synthetic-medium.zip",
         "checksum_sha256": "TBD",
         "size_bytes": 89128960,
         "description": "6 cameras, 80 frames, grid layout"
       },
       "real-rig": {
         "type": "real",
         "included": false,
         "zenodo_record_id": "TBD",
         "zenodo_filename": "real-rig.zip",
         "checksum_sha256": "TBD",
         "size_bytes": 188743680,
         "description": "13-camera production rig, full calibration"
       }
     }
   }
   ```

2. **Download Function:**
   - Use requests + tqdm pattern (see Section 5.3)
   - Zenodo API: `https://zenodo.org/api/records/{record_id}` for metadata
   - Direct download: `https://zenodo.org/record/{record_id}/files/{filename}`
   - Retry on failure (3 attempts)
   - SHA256 validation

3. **Cache Management:**
   - `get_cache_dir()` → `Path.cwd() / "aquacal_data"`
   - Auto-create `.gitignore` inside
   - Extract ZIP to cache (preserve structure)
   - Check extraction marker (`.extracted` file or directory existence)

4. **Loading Logic:**
   - Check manifest for dataset name
   - If `included=true`: Load from package data (`importlib.resources`)
   - If `included=false`: Check cache, download if missing, extract, load
   - Return structured `ExampleDataset` object

**Error Cases:**
- Unknown dataset name → raise `ValueError` with list of available datasets
- Download failure → raise `RuntimeError` with retry suggestion
- Checksum mismatch → delete partial file, raise `ValueError`
- Insufficient disk space → catch `OSError`, raise with clear message

### 6.3 Package Data Configuration

**Update `pyproject.toml`:**
```toml
[tool.setuptools.package-data]
aquacal = [
    "datasets/manifests/*.json",
    "datasets/small/*.json",
    "datasets/small/images/**/*.png",
]
```

**Verify Inclusion:**
```bash
# Build wheel
python -m build --wheel

# Inspect contents
unzip -l dist/aquacal-*.whl | grep datasets
```

**Size Budget:**
- Manifest JSON: <10KB
- Small preset JSON: ~50KB
- Small preset images (10 frames, 2 cameras, 1920×1080 PNG): ~2-5MB
- **Total added to wheel:** <10MB (acceptable)

### 6.4 Real Dataset Preparation

**Tasks:**
1. **Select Frame Subset:**
   - Extract frames from production rig videos
   - Intrinsic: ~50 frames/camera (diverse board orientations, grid coverage)
   - Extrinsic: ~100 frames/camera (underwater trajectory, connectivity)
   - Use `ffmpeg` for extraction:
     ```bash
     ffmpeg -i cam0.mp4 -vf "select='not(mod(n,5))'" -vsync 0 cam0_frame%04d.png
     ```

2. **Compress Images:**
   - JPEG quality 95 (visually lossless, 50-60% smaller than PNG)
   - OR: PNG with `pngcrush` optimization

3. **Generate Reference Calibration:**
   - Run full pipeline with `frame_step=1` (all frames, high quality)
   - Save `reference_calibration.json`
   - Document parameters used (for reproducibility section)

4. **Package Structure:**
   ```
   real-rig/
   ├── README.txt              # Dataset description, usage instructions
   ├── config.yaml             # Working calibration config
   ├── reference_calibration.json
   ├── intrinsic/
   │   ├── cam0/
   │   │   ├── frame0000.jpg
   │   │   └── ...
   │   ├── cam1/ ...
   │   └── ...
   └── extrinsic/
       ├── cam0/
       ├── cam1/ ...
       └── ...
   ```

5. **Upload to Zenodo:**
   - Create new Zenodo record (sandbox first: https://sandbox.zenodo.org)
   - Set metadata: title, description, keywords, license (MIT), creators
   - Upload ZIP file
   - Publish and obtain DOI
   - Record `record_id` and `checksum_sha256` in manifest

**Target Size:** <200MB ZIP (decompress to ~400MB)

### 6.5 Documentation and Tests

**Documentation:**
- Docstrings for all public functions (Google style)
- Usage examples in `datasets/__init__.py` module docstring
- Update `CLAUDE.md` with dataset API reference

**Tests:**
- `tests/unit/test_datasets.py`:
  - `test_generate_synthetic_rig_presets()` — validate all presets run
  - `test_generate_synthetic_rig_reproducibility()` — same seed = same output
  - `test_load_example_small()` — load included dataset
  - `test_cache_dir_creation()` — verify .gitignore created
- `tests/integration/test_dataset_download.py` (slow):
  - `test_download_medium_dataset()` — full download + extraction + loading
  - Mock Zenodo API for CI (avoid actual downloads in tests)

**Update Existing Tests:**
- `tests/synthetic/conftest.py` → import from `aquacal.datasets.synthetic`
- Verify all synthetic tests still pass

---

## 7. Key Decisions and Open Questions

### 7.1 Locked Decisions (from CONTEXT.md)

✓ Function name: `generate_synthetic_rig()`
✓ Module: `aquacal.datasets` (single module for both synthetic and real)
✓ Presets: `'small'`, `'medium'`, `'large'` — fixed, no custom config
✓ Fixed random seeds per preset (reproducibility)
✓ Default board: 7×5 ChArUco, 30mm squares (specified in CONTEXT as example, confirm exact params)
✓ ChArUco only (no checkerboard support)
✓ Refactor `generate_synthetic_detections()` test helper into public API
✓ Default: detections only, optional flag for images
✓ Noise: clean by default, optional Gaussian pixel jitter preset
✓ Hosting: Zenodo for medium/large synthetic + real dataset
✓ Small preset ships in-package
✓ Cache: `./aquacal_data/` with `.gitignore`
✓ Progress bar: tqdm for downloads
✓ Real dataset: 9+ cameras, intrinsic + extrinsic image sets (frames extracted from video)
✓ Include reference calibration result

### 7.2 Claude's Discretion

**Return Type of `generate_synthetic_rig()`:**
- **Recommendation:** Return `SyntheticScenario` (existing dataclass from tests)
- Already has all required fields: intrinsics, extrinsics, interface_distances, board_poses, board_config, detections, name, description
- Optional: Add `images` field as `dict[str, dict[int, NDArray]] | None` for when `include_images=True`

**Above-water Board Projection:**
- **Option A:** Use `cv2.aruco.CharucoBoard.generateImage()` (simplest, clean synthetic board)
- **Option B:** Project corners via pinhole model, draw at projected locations (consistent with underwater approach)
- **Recommendation:** Option A for in-air (fast, clean), custom projection only for underwater (required)

**Exact Preset Parameters:**
- **Small:** 2 cameras, line layout, 10 frames, 0.3px noise, seed=42
- **Medium:** 6 cameras, grid layout, 80 frames, 0.5px noise, seed=123
- **Large:** 13 cameras, real rig geometry, 300 frames, 0.5px noise, seed=456
- Board config (all presets): 7×5 ChArUco, 30mm squares, 22.5mm markers, DICT_5X5_100
- Rationale: Small = quick smoke test, Medium = balanced testing, Large = stress test/benchmarking

**Data Object Structure (`load_example()`):**
```python
@dataclass
class ExampleDataset:
    name: str
    type: str  # "synthetic" or "real"
    detections: DetectionResult
    ground_truth: SyntheticScenario | None = None  # Synthetic only
    reference_calibration: CalibrationResult | None = None  # Real only
    metadata: dict = field(default_factory=dict)  # Size, source, description
    cache_path: Path | None = None  # None if included in-package
```

**Download Retry Behavior:**
- Max retries: 3
- Exponential backoff: 1s, 2s, 4s delays
- Timeout: 30s per request
- Raise `RuntimeError` with clear message after final failure

**Cache Directory Structure:**
```
./aquacal_data/
├── .gitignore
├── downloads/
│   ├── synthetic-medium.zip
│   └── real-rig.zip
├── synthetic-medium/       # Extracted
│   ├── detections.json
│   ├── ground_truth.json
│   └── images/
└── real-rig/              # Extracted
    ├── intrinsic/
    ├── extrinsic/
    ├── config.yaml
    └── reference_calibration.json
```

### 7.3 Open Questions for Planning

1. **Board Config Confirmation:**
   - CONTEXT.md mentions "7x5 ChArUco, 30mm squares" as an example
   - Should we match existing test board (12×9, 60mm squares, 45mm markers)?
   - OR use a new smaller board (7×5, 30mm) optimized for synthetic scenarios?
   - **Recommendation:** Use 12×9 board to match existing tests, update CONTEXT example if needed

2. **Real Dataset Frame Count:**
   - "TBD — needs experimentation to find sweet spot"
   - Intrinsic: 50 frames/camera (good for diverse coverage)
   - Extrinsic: 100 frames/camera (ensures connectivity)
   - Total: 13 cameras × 150 frames/camera = 1950 frames → ~195-390MB
   - **Plan:** Start with these numbers, subsample if size exceeds 200MB

3. **Underwater Image Rendering Complexity:**
   - Full ChArUco rendering (markers + checkerboard) is complex
   - Simpler: Render corners only (circles) or corners + checkerboard grid (no markers)
   - **Question:** Is marker visualization needed for synthetic images, or just corners?
   - **Recommendation:** Start with corners + grid (no markers), add markers in future if requested

4. **Medium Preset Images:**
   - 6 cameras × 80 frames × ~50KB/image = ~24MB
   - Acceptable for Zenodo, but slows generation if created on-the-fly
   - **Plan:** Pre-generate and cache medium images in Zenodo ZIP (one-time cost)

5. **Test Coverage for Downloads:**
   - Full downloads in CI would slow tests significantly
   - **Plan:** Mock Zenodo API responses, test extraction/loading only
   - Optional: One slow integration test that actually downloads (mark `@pytest.mark.slow`)

---

## 8. External Dependencies

### 8.1 Current Dependencies (from `pyproject.toml`)

```toml
dependencies = [
    "numpy",
    "scipy",
    "opencv-python>=4.6",
    "pyyaml",
    "matplotlib",
    "pandas",
]
```

### 8.2 New Dependencies Required

**For Download and Progress:**
- `requests` — HTTP client for Zenodo downloads
  - Standard library alternative: `urllib` (more verbose, no streaming progress)
  - **Decision:** Add `requests` (de facto standard, easier to use)

- `tqdm` — Progress bars
  - **Decision:** Add `tqdm` (widely used, minimal overhead)

**Updated Dependencies:**
```toml
dependencies = [
    "numpy",
    "scipy",
    "opencv-python>=4.6",
    "pyyaml",
    "matplotlib",
    "pandas",
    "requests",  # NEW
    "tqdm",      # NEW
]
```

**Justification:**
- Both are lightweight, pure Python packages
- `requests` is already a transitive dependency of many common packages
- `tqdm` is standard for CLI progress indication in scientific Python

---

## 9. Success Criteria Validation

**From Roadmap:**

1. ✓ **User can generate synthetic calibration scenarios via `aquacal.datasets.generate_synthetic_rig()` with configurable rig size**
   - Note: CONTEXT.md changed to fixed presets (`'small'`, `'medium'`, `'large'`), not "configurable"
   - Update success criterion: "User can generate synthetic calibration scenarios via `aquacal.datasets.generate_synthetic_rig()` with preset rig sizes ('small', 'medium', 'large')"

2. ✓ **User can download a real calibration dataset (<50MB total) from GitHub Releases or examples/datasets/**
   - CONTEXT.md updated: Real dataset hosted on Zenodo (DOI-backed), ~180MB
   - Update success criterion: "User can download a real calibration dataset (~180MB) from Zenodo via `load_example('real-rig')`"

3. ✓ **User can load example datasets via convenience function `aquacal.datasets.load_example()`**
   - Covered by implementation plan

4. ✓ **Larger real datasets (>10MB) are hosted on Zenodo with DOI and download instructions**
   - All non-small datasets hosted on Zenodo
   - DOI obtained during dataset upload
   - Download automatic via `load_example()`, manual instructions in docs

**Proposed Updated Success Criteria:**
1. User can generate synthetic calibration scenarios via `aquacal.datasets.generate_synthetic_rig()` with preset rig sizes: `'small'`, `'medium'`, `'large'`
2. User can access small synthetic dataset instantly (included in package, no download required)
3. User can download medium/large synthetic datasets and real rig dataset from Zenodo via `load_example()` with progress indication
4. Each dataset includes detections, ground truth (synthetic) or reference calibration (real), and optional images
5. Datasets are cached in `./aquacal_data/` to avoid re-downloading
6. All Zenodo-hosted datasets have DOI for persistent citation

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Synthetic image rendering complexity** | Medium (delays delivery) | Medium | Start with corners-only rendering, defer full ChArUco markers to future |
| **Real dataset size exceeds 200MB** | Low (acceptable for Zenodo) | Medium | JPEG compression, frame subsampling, split into intrinsic/extrinsic ZIPs |
| **Zenodo download failures in CI** | Low (tests flaky) | Low | Mock API calls in tests, mark download tests as `@pytest.mark.slow` |
| **Cache directory conflicts** | Low (user confusion) | Low | Clear docs, `.gitignore` auto-creation, `get_cache_info()` utility |
| **Package size increase** | Medium (PyPI limits) | Low | Small preset only (~5-10MB), well within PyPI's 100MB wheel limit |

### 10.2 Scope Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Feature creep (custom synthetic configs)** | High (delays v1.0) | Medium | CONTEXT.md locked: presets only, no custom config |
| **Multiple real datasets requested** | Medium (upload time) | Low | Start with one rig, add more in v2 if demand exists |
| **Documentation burden** | Medium (delays Phase 6) | Medium | Minimal docs in Phase 4 (docstrings only), full tutorial in Phase 6 |

### 10.3 External Dependencies

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Zenodo downtime** | Low (temporary failure) | Very Low | Retry logic, cached data persists, clear error messages |
| **Zenodo API changes** | Medium (breaks downloads) | Very Low | Use stable v1 API, fallback to direct file URLs |
| **requests/tqdm version conflicts** | Low (install issues) | Very Low | Both stable, widely compatible packages |

---

## 11. References

### Documentation
- [OpenCV: CharucoBoard Class Reference](https://docs.opencv.org/4.x/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html)
- [OpenCV: Detection of ChArUco Boards](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html)
- [Setuptools: Data Files Support](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
- [Zenodo Developers Guide](https://developers.zenodo.org/)
- [scikit-learn fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)

### Tutorials and Examples
- [Using ChArUco boards in OpenCV - Medium](https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d)
- [Python requests download with tqdm - GitHub Gist](https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51)
- [Visualizing Download Progress with tqdm](https://blasferna.com/articles/visualizing-download-progress-with-tqdm-in-python/)
- [Specifying Package Data in pyproject.toml](https://www.pythontutorials.net/blog/specifying-package-data-in-pyproject-toml/)

### Tools and Libraries
- [zenodo_get on PyPI](https://pypi.org/project/zenodo-get/)
- [appdirs on PyPI](https://pypi.org/project/appdirs/)
- [pip Caching Documentation](https://pip.pypa.io/en/stable/topics/caching/)

### Best Practices
- [Python Packaging User Guide: Using MANIFEST.in](https://packaging.python.org/en/latest/guides/using-manifest-in/)
- [Prevent pip filling up home directory - NCSU HPC](https://hpc.ncsu.edu/Software/python_cache_directory.php)
- [Python pip cache FAQ - CSC Docs](https://docs.csc.fi/support/faq/python-pip-cache/)

---

## 12. Next Steps for Planning

### 12.1 Plan Structure Recommendation

**Suggested Plans (3 total):**

1. **04-01-PLAN.md: Synthetic Data API and Rendering**
   - Refactor `tests/synthetic/ground_truth.py` into `aquacal.datasets.synthetic`
   - Implement `generate_synthetic_rig()` with fixed presets
   - Implement synthetic image rendering (in-air + underwater)
   - Update existing tests to import from public API
   - Success: All three presets generate valid scenarios, images render correctly

2. **04-02-PLAN.md: Dataset Download and Caching Infrastructure**
   - Create `aquacal.datasets.download` and `aquacal.datasets.loader`
   - Implement Zenodo download with tqdm progress
   - Implement cache management in `./aquacal_data/`
   - Add `requests` and `tqdm` to dependencies
   - Package small preset via `pyproject.toml` package_data
   - Success: `load_example('small')` works instantly, download/cache/load cycle works

3. **04-03-PLAN.md: Real Dataset Preparation and Zenodo Upload**
   - Extract frames from production rig videos
   - Compress images (JPEG quality 95)
   - Generate reference calibration
   - Create dataset ZIP with README and config
   - Upload to Zenodo (sandbox first, then production)
   - Update manifest with DOI and checksums
   - Success: Real dataset downloadable via `load_example('real-rig')`, <200MB size

**Alternative (4 plans):**
- Split 04-01 into: (a) refactoring without rendering, (b) image rendering separately
- Pro: Smaller incremental steps
- Con: Rendering depends on refactored API, creates cross-plan dependency

**Recommendation:** 3-plan structure (above) balances incremental delivery with logical cohesion.

### 12.2 Open Questions to Resolve Before Planning

1. **Board config for presets:** Confirm 12×9 (existing) vs 7×5 (CONTEXT example)
2. **Underwater rendering scope:** Corners-only vs full ChArUco pattern
3. **Medium preset image strategy:** Pre-generate for Zenodo vs generate-on-demand
4. **Real dataset frame count:** Finalize intrinsic/extrinsic frame counts based on size target

**Recommendation:** Resolve in `/gsd:discuss-phase` if needed, or make Claude's Discretion call during planning.

---

**End of Research Document**
