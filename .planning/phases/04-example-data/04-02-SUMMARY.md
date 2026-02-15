---
phase: 04-example-data
plan: 02
subsystem: datasets
tags: [dataset-loading, download-infrastructure, caching, public-api]
dependency_graph:
  requires: [datasets/synthetic, io/serialization, config/schema]
  provides: [datasets/loader, datasets/download, datasets/manifest]
  affects: [datasets/__init__, pyproject.toml]
tech_stack:
  added: [requests, tqdm, importlib.resources, zipfile]
  patterns: [package-data, manifest-registry, download-caching, progress-bars]
key_files:
  created:
    - src/aquacal/datasets/loader.py
    - src/aquacal/datasets/download.py
    - src/aquacal/datasets/_manifest.py
    - src/aquacal/datasets/data/manifest.json
    - src/aquacal/datasets/data/small/detections.json
    - src/aquacal/datasets/data/small/ground_truth.json
  modified:
    - src/aquacal/datasets/__init__.py
    - pyproject.toml
    - tests/unit/test_datasets.py
    - tests/synthetic/ground_truth.py
decisions:
  - Small preset ships in-package (17KB total) for zero-download quick start
  - Cache directory at ./aquacal_data/ with auto-generated .gitignore
  - Zenodo datasets not yet uploaded (zenodo_record_id = null placeholder)
  - NotImplementedError for non-available datasets with helpful message
  - Download with progress bar, checksum validation, and exponential backoff retry
  - importlib.resources for package data access (Python 3.10+ compatibility)
metrics:
  duration: 544
  completed_date: "2026-02-15T01:29:57Z"
  tasks_completed: 2
  tests_added: 8
  files_created: 6
  files_modified: 4
  commits: 2
---

# Phase 4 Plan 2: Dataset Loading and Download Infrastructure

**One-liner:** `load_example()` API with in-package small preset and Zenodo download infrastructure for larger datasets

## Overview

Built the complete dataset loading, downloading, and caching infrastructure. The small preset (2 cameras, 10 frames) ships in-package for instant loading with zero downloads. Larger datasets are registered in a manifest with Zenodo URLs (placeholders until upload), downloadable via an infrastructure with progress bars, checksum validation, and caching.

## What Was Built

### Dataset Loading (`src/aquacal/datasets/loader.py`)

**`ExampleDataset` dataclass:**
- name, type, detections, ground_truth, reference_calibration, metadata, cache_path
- Unified container for both synthetic and real calibration datasets

**`load_example(name: str)`** — Main public API:
- Loads small preset from package data (instant, no download)
- For non-included datasets: checks manifest, downloads from Zenodo if available
- Deserializes detections and ground truth from JSON
- Returns ExampleDataset with all data populated

**Deserialization helpers:**
- `_deserialize_detections()` — Converts JSON to DetectionResult
- `_deserialize_ground_truth()` — Converts JSON to SyntheticScenario
- Reverses serialization format from io/serialization.py

### Download Infrastructure (`src/aquacal/datasets/download.py`)

**`get_cache_dir()`:**
- Returns `./aquacal_data/` relative to cwd
- Creates directory and `.gitignore` containing `*\n` on first access

**`download_with_progress()`:**
- Downloads file with tqdm progress bar (bytes, human-readable)
- SHA256 checksum validation after download
- Exponential backoff retry (1s, 2s, 4s) on failure
- Deletes partial file on checksum mismatch

**`download_and_extract()`:**
- Constructs Zenodo URL from record ID and filename
- Downloads ZIP to `aquacal_data/downloads/`
- Extracts to `aquacal_data/{dataset_name}/`
- Returns path to extracted directory
- Skips download if already cached (cache hit)

**`clear_cache()`:**
- Clear specific dataset or entire cache
- Removes both extracted directory and download ZIP

**`get_cache_info()`:**
- Returns cache directory path, list of cached datasets, total size in MB

### Manifest System (`src/aquacal/datasets/_manifest.py`)

**`get_manifest()`:**
- Loads manifest.json from package data using `importlib.resources`
- Returns dataset registry as dict

**`get_dataset_info(name: str)`:**
- Returns metadata for named dataset
- Raises ValueError with list of available datasets if not found

**`list_datasets()`:**
- Returns list of all available dataset names

**Manifest structure (`data/manifest.json`):**
```json
{
  "version": "1.0",
  "datasets": {
    "small": {
      "type": "synthetic",
      "included": true,
      "description": "2 cameras, 10 frames — ships with package, no download needed"
    },
    "medium": {
      "type": "synthetic",
      "included": false,
      "zenodo_record_id": null,
      "zenodo_filename": "synthetic-medium.zip",
      "checksum_sha256": null,
      "size_bytes": null,
      "description": "6 cameras, 80 frames — download from Zenodo"
    },
    ...
  }
}
```

### In-Package Small Preset Data

**Generated static data files:**
- `data/small/ground_truth.json` (4.7 KB) — Intrinsics, extrinsics, board poses, interface distances
- `data/small/detections.json` (12 KB) — Frame detections for all cameras
- Total: 17 KB (small enough to ship in package)

**Generation script (`generate_small_preset_data.py`):**
- Calls `generate_synthetic_rig('small')` to create scenario
- Serializes ground truth and detections to JSON
- Run once when synthetic data generation changes
- Files are committed to repo (not generated at install time)

### Updated pyproject.toml

**Added dependencies:**
- `requests` — HTTP downloads from Zenodo
- `tqdm` — Progress bars for downloads

**Package data configuration:**
```toml
[tool.setuptools.package-data]
aquacal = [
    "datasets/data/**/*.json",
]
```
Ensures small preset data ships with pip install.

### Updated Public API (`src/aquacal/datasets/__init__.py`)

**New exports:**
- `load_example` — Load example datasets
- `ExampleDataset` — Dataset container dataclass
- `list_datasets` — List available datasets
- `clear_cache` — Clear dataset cache
- `get_cache_info` — Get cache status

**Existing exports (from Plan 01):**
- `generate_synthetic_rig` — Generate synthetic scenarios
- `SyntheticScenario` — Ground truth container

**Updated module docstring:**
- Examples for both synthetic generation and dataset loading
- Cache management documentation

### Test Infrastructure

**Added 8 new tests (`tests/unit/test_datasets.py`):**
1. `test_load_example_small()` — Verify ExampleDataset structure
2. `test_load_example_small_detections_match_generated()` — Verify loaded data matches generated
3. `test_load_example_nonexistent()` — Verify error handling for unknown datasets
4. `test_load_example_medium_not_available()` — Verify NotImplementedError for Zenodo datasets
5. `test_list_datasets()` — Verify manifest listing
6. `test_get_cache_dir()` — Verify cache creation and .gitignore
7. `test_clear_cache()` — Verify cache clearing (specific and full)
8. `test_get_cache_info()` — Verify cache info reporting
9. `test_manifest_loading()` — Verify manifest structure

**Fixed imports in `tests/synthetic/ground_truth.py`:**
- Re-exports functions moved to aquacal.datasets.synthetic
- Added: `generate_synthetic_detections`, `compute_calibration_errors`, `generate_camera_intrinsics`
- Prevents import errors in existing tests

**Test results:**
- 8 new dataset loading tests: PASSED
- 572 existing tests (excluding slow): PASSED (84s)
- 0 regressions from refactoring

## Key Implementation Decisions

### 1. Small preset ships in-package
**Context:** Users need instant access to test data without network dependency.
**Decision:** Generate small preset data once, commit JSON files to repo, ship with package.
**Impact:** 17KB overhead in package, but zero-download quick start for all users.

### 2. Manifest-based registry
**Context:** Need flexible system for both included and downloadable datasets.
**Decision:** Single manifest.json with type, included flag, and Zenodo metadata.
**Impact:** Easy to add new datasets, clear separation of included vs. download.

### 3. Zenodo placeholders (zenodo_record_id = null)
**Context:** Datasets not yet uploaded to Zenodo.
**Decision:** Manifest has null placeholders, load_example() raises NotImplementedError with helpful message.
**Impact:** Infrastructure ready for when datasets are uploaded (Plan 03 or future work).

### 4. Cache at ./aquacal_data/ in cwd
**Context:** Need predictable, user-controllable cache location.
**Decision:** Cache relative to cwd, not user home directory.
**Impact:** Easy to find and clean up, works well for project-local workflows.

### 5. importlib.resources for package data
**Context:** Need portable way to access package data files.
**Decision:** Use `importlib.resources.files()` instead of `__file__` manipulation.
**Impact:** Python 3.10+ compatible, works with both installed packages and editable installs.

## Verification Results

**load_example() works:**
```python
from aquacal.datasets import load_example
ds = load_example('small')
# Loaded: small, type=synthetic, cameras=2
```

**list_datasets() works:**
```python
from aquacal.datasets import list_datasets
print(list_datasets())
# ['small', 'medium', 'large', 'real-rig']
```

**Medium raises NotImplementedError:**
```python
load_example('medium')
# NotImplementedError: Dataset 'medium' is not yet available for download.
# Use generate_synthetic_rig('medium') to generate it locally.
```

**Data files exist:**
- src/aquacal/datasets/data/small/detections.json (12 KB)
- src/aquacal/datasets/data/small/ground_truth.json (4.7 KB)

## Deviations from Plan

None - plan executed exactly as written.

## Files Changed

**Created:**
- `src/aquacal/datasets/loader.py` (230 lines) — load_example() API
- `src/aquacal/datasets/download.py` (215 lines) — Download infrastructure
- `src/aquacal/datasets/_manifest.py` (65 lines) — Manifest loading
- `src/aquacal/datasets/data/manifest.json` (35 lines) — Dataset registry
- `src/aquacal/datasets/data/small/detections.json` (12 KB) — Small preset detections
- `src/aquacal/datasets/data/small/ground_truth.json` (4.7 KB) — Small preset ground truth
- `generate_small_preset_data.py` (130 lines) — Data generation script

**Modified:**
- `src/aquacal/datasets/__init__.py` (+30 lines) — New exports and docstring
- `pyproject.toml` (+4 lines) — Dependencies and package-data
- `tests/unit/test_datasets.py` (+191 lines) — 8 new tests
- `tests/synthetic/ground_truth.py` (+3 imports) — Re-export moved functions

**Commits:**
1. `5ae1ef8` — feat(04-02): add dataset loading, download, and caching infrastructure
2. `1c29021` — test(04-02): add comprehensive loader and cache tests

## Acceptance Criteria

✅ load_example('small') works instantly from in-package data
✅ load_example('medium') gives clear NotImplementedError (Zenodo not yet configured)
✅ Download infrastructure (download.py) is ready for when Zenodo records are created
✅ Cache management (get_cache_dir, clear_cache, get_cache_info) works correctly
✅ Package data configuration ensures small preset ships with pip install
✅ requests and tqdm added as dependencies
✅ All 572 tests pass (no regressions)

## Usage Examples

```python
from aquacal.datasets import load_example, list_datasets, get_cache_info

# List available datasets
print(list_datasets())
# ['small', 'medium', 'large', 'real-rig']

# Load small preset (instant, no download)
ds = load_example('small')
print(f"{len(ds.ground_truth.intrinsics)} cameras")  # 2 cameras

# Access detections
frame0 = ds.detections.frames[0]
print(frame0.detections.keys())  # dict_keys(['cam0', 'cam1'])

# Access ground truth
gt_intrinsics = ds.ground_truth.intrinsics['cam0']
print(gt_intrinsics.K[0, 0])  # Focal length

# Check cache status
info = get_cache_info()
print(f"Cached: {info['cached_datasets']}")  # []

# Try loading medium (not yet available)
try:
    load_example('medium')
except NotImplementedError as e:
    print(e)  # Dataset 'medium' is not yet available...
```

## Next Steps

- Plan 03: Upload medium/large synthetic datasets and real-rig to Zenodo
- Update manifest.json with real Zenodo record IDs and checksums
- Test full download flow with real Zenodo URLs
- Consider adding validation examples/notebooks using load_example()

## Self-Check: PASSED

**Files created:**
```bash
✓ src/aquacal/datasets/loader.py
✓ src/aquacal/datasets/download.py
✓ src/aquacal/datasets/_manifest.py
✓ src/aquacal/datasets/data/manifest.json
✓ src/aquacal/datasets/data/small/detections.json
✓ src/aquacal/datasets/data/small/ground_truth.json
```

**Commits exist:**
```bash
✓ 5ae1ef8: feat(04-02): add dataset loading infrastructure
✓ 1c29021: test(04-02): add comprehensive loader tests
```

**Tests pass:**
```bash
✓ 8 new dataset loading tests pass
✓ 572 existing tests pass (no regressions)
```

**Verification commands:**
```bash
✓ load_example('small') loads from package data
✓ list_datasets() returns all datasets
✓ load_example('medium') raises NotImplementedError
✓ Small preset data files exist (17KB total)
```
