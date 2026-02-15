---
phase: 04-example-data
plan: 03
subsystem: datasets
tags: [zenodo, real-data, download, manifest]
dependency_graph:
  requires: [04-02]
  provides: [real-rig-dataset, md5-checksum-support]
  affects: [datasets.loader, datasets.download, datasets.manifest]
tech_stack:
  added: []
  patterns: [algorithm:hash checksum format, nested ZIP handling, real dataset loading]
key_files:
  created: []
  modified:
    - src/aquacal/datasets/data/manifest.json
    - src/aquacal/datasets/download.py
    - src/aquacal/datasets/loader.py
decisions:
  - Use algorithm:hash format (md5:... or sha256:...) for checksums to support Zenodo's MD5-only API
  - Real datasets return empty DetectionResult with metadata and cache_path for user to process
  - Handle nested directory structure automatically (Zenodo archives often have top-level folder)
metrics:
  duration_seconds: 267
  tasks_completed: 2
  files_modified: 3
  test_status: passing
  completed_date: 2026-02-14
---

# Phase 04 Plan 03: Real Dataset Upload and Manifest Update Summary

**One-liner:** Real production rig dataset (12 cameras, 156 MB) uploaded to Zenodo and integrated with download infrastructure supporting MD5 checksums.

## Objective

Prepare and upload real calibration dataset to Zenodo, then update the manifest so `load_example('real-rig')` works end-to-end. This provides researchers with access to a real-world calibration dataset from a production rig to validate AquaCal against non-synthetic data.

## What Was Built

### Task 1: Prepare and upload real calibration dataset to Zenodo (CHECKPOINT)

**Status:** COMPLETE (user action)

The user successfully uploaded the real calibration dataset to Zenodo:
- **Zenodo Record ID:** 18645385
- **DOI:** 10.5281/zenodo.18645385
- **Filename:** real-rig-calib.zip (user renamed from real-rig.zip)
- **Size:** 164,023,590 bytes (156.4 MB)
- **MD5 checksum:** c66380aaa8cbca6bc04a3157baacbee8
- **Contents:** 12 cameras, intrinsic/extrinsic image directories, config.yaml, reference_calibration.json

**Note:** User tweaked config.yaml inside ZIP to set n_air=1.0, n_water=1.333 (original had both at 1.0 from non-refractive test run).

### Task 2: Update manifest with Zenodo record IDs and verify end-to-end download

**Status:** COMPLETE

**Files modified:**
- `src/aquacal/datasets/data/manifest.json` — Added real-rig Zenodo metadata
- `src/aquacal/datasets/download.py` — Enhanced checksum validation
- `src/aquacal/datasets/loader.py` — Implemented real dataset loading

**Changes:**

1. **Manifest updates:**
   - Changed `checksum_sha256` field to `checksum` with `algorithm:hash` format
   - Added real-rig entry: record_id=18645385, checksum=md5:c66380..., size_bytes=164023590
   - Filename updated to match user's upload: real-rig-calib.zip

2. **Download module enhancements:**
   - `download_with_progress()` now accepts `expected_checksum` parameter
   - Parses algorithm:hash format (e.g., "md5:abc..." or "sha256:def...")
   - Supports both MD5 and SHA256 verification
   - Validates checksum format and raises helpful errors for unsupported algorithms

3. **Loader module enhancements:**
   - Handles nested directory structure (Zenodo ZIP has top-level folder)
   - Real dataset loader reads config.yaml for camera metadata
   - Loads reference_calibration.json if present (stored as raw dict for now)
   - Returns ExampleDataset with empty DetectionResult but full metadata and cache_path
   - Synthetic datasets still use detections.json/ground_truth.json serialization

**Verification:**
```python
from aquacal.datasets import load_example
ds = load_example('real-rig')
# Loaded real-rig: type=real, cameras=12, has_ref_calib=True
```

**End-to-end flow:**
1. `load_example('real-rig')` checks manifest
2. Downloads from Zenodo (156 MB) with progress bar
3. Verifies MD5 checksum
4. Extracts to `./aquacal_data/real-rig/`
5. Handles nested directory (real-rig/real-rig/)
6. Loads config.yaml and reference_calibration.json
7. Returns ExampleDataset with metadata

**Cache info:**
- Cache directory: `./aquacal_data/`
- Cached datasets: [real-rig]
- Total size: 192 MB (after extraction)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Changed checksum format from SHA256-only to algorithm:hash**
- **Found during:** Task 2
- **Issue:** Plan assumed SHA256 checksums, but Zenodo API only provides MD5 checksums. The download code expected `checksum_sha256` field.
- **Fix:**
  - Renamed manifest field from `checksum_sha256` to `checksum`
  - Changed format to `algorithm:hash` (e.g., "md5:c66380..." or "sha256:def...")
  - Updated `download_with_progress()` to parse algorithm and select appropriate hash function
  - This makes the system future-proof for other hash algorithms
- **Files modified:** manifest.json, download.py
- **Commit:** 5eba1d8

**2. [Rule 3 - Blocking] Implemented real dataset loading logic**
- **Found during:** Task 2
- **Issue:** Plan mentioned "verify end-to-end download" but loader had NotImplementedError for cache loading. Real datasets have different structure than synthetic (no pre-serialized detections.json).
- **Fix:**
  - Added type-specific loading in `load_example()`
  - Real datasets: load config.yaml + reference_calibration.json, return empty DetectionResult
  - Synthetic datasets: load detections.json + ground_truth.json (existing logic)
  - Added nested directory handling (Zenodo ZIP has top-level folder)
  - Added yaml import (PyYAML already in dependencies)
- **Files modified:** loader.py
- **Commit:** 5eba1d8

## Decisions Made

1. **Checksum format:** Use `algorithm:hash` format instead of separate fields per algorithm. This is more flexible and follows common convention (e.g., `md5:abc...`).

2. **Real dataset loader behavior:** Return ExampleDataset with empty DetectionResult but full metadata and cache_path. Users can process the raw image directories themselves. This avoids pre-computing detections for large datasets.

3. **Nested directory handling:** Automatically detect and unwrap nested directories in ZIP archives. Zenodo archives often have a top-level folder matching the dataset name.

4. **Reference calibration storage:** Store as raw dict for now rather than deserializing to CalibrationResult. Full deserialization can be added later if needed.

## Testing

**Dataset tests:** All 20 tests in `tests/unit/test_datasets.py` pass.

**Manual verification:**
```bash
python -c "from aquacal.datasets import load_example; ds = load_example('real-rig'); print(f'Loaded {ds.name}: type={ds.type}, cameras={len(ds.detections.camera_names)}, has_ref_calib={ds.metadata.get(\"has_reference_calibration\", False)}')"
# Output: Loaded real-rig: type=real, cameras=12, has_ref_calib=True
```

**Download performance:** 156 MB file downloaded in ~33 seconds with progress bar and MD5 verification.

**No regressions:** All dataset tests pass, including existing tests for small synthetic dataset.

## Known Limitations

1. **Reference calibration deserialization:** Currently stored as raw JSON dict. Full `CalibrationResult` deserialization not implemented (not needed for this plan's scope).

2. **Real dataset detections:** Not pre-computed or serialized. Users must run detection on raw images. This is intentional for large datasets.

3. **Synthetic medium/large datasets:** Not yet uploaded to Zenodo (zenodo_record_id=null). Users see NotImplementedError with helpful message.

## Impact

**Users can now:**
- Call `load_example('real-rig')` to download a real production rig dataset
- Access 12-camera calibration data with reference results
- Validate AquaCal on non-synthetic data
- Inspect config.yaml and reference_calibration.json for real-world examples

**Dataset infrastructure:**
- Supports both MD5 and SHA256 checksums (Zenodo compatibility)
- Handles nested ZIP structures automatically
- Differentiates real vs synthetic dataset loading
- Provides clear error messages for unavailable datasets

## Next Steps

After this plan, Phase 4 is complete. The example data infrastructure is ready:
- Small synthetic dataset ships in-package (no download)
- Real production rig dataset available via Zenodo
- Download infrastructure with checksums, progress, and caching
- Clear loader API with metadata support

Future enhancements (not in this phase):
- Upload synthetic-medium and synthetic-large to Zenodo
- Deserialize reference_calibration to CalibrationResult
- Add dataset validation utilities
- Create dataset generation CLI tool

## Self-Check: PASSED

**Created files exist:**
- ✓ .planning/phases/04-example-data/04-03-SUMMARY.md (this file)

**Modified files exist:**
- ✓ src/aquacal/datasets/data/manifest.json
- ✓ src/aquacal/datasets/download.py
- ✓ src/aquacal/datasets/loader.py

**Commits exist:**
- ✓ 5eba1d8: feat(04-03): add real-rig dataset to manifest and support MD5 checksums

**Functionality verified:**
- ✓ `load_example('real-rig')` downloads and loads successfully
- ✓ MD5 checksum verification works
- ✓ Cache directory created at ./aquacal_data/
- ✓ All dataset tests pass
