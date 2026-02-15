---
phase: 04-example-data
verified: 2026-02-14T23:30:00Z
status: gaps_found
score: 3/4
gaps:
  - truth: "Existing synthetic and unit tests pass without modification to test logic"
    status: failed
    reason: "Import error in tests - generate_synthetic_detections and compute_calibration_errors not re-exported from ground_truth.py"
    artifacts:
      - path: "tests/synthetic/ground_truth.py"
        issue: "Missing re-exports for generate_synthetic_detections, compute_calibration_errors, generate_camera_intrinsics"
    missing:
      - "Add missing imports to ground_truth.py from aquacal.datasets.synthetic"
      - "Re-export: generate_synthetic_detections, compute_calibration_errors, generate_camera_intrinsics"
---

# Phase 04: Example Data Verification Report

**Phase Goal:** Researchers have access to both synthetic calibration datasets with known ground truth and real-world example data

**Verified:** 2026-02-14T23:30:00Z

**Status:** gaps_found

**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can generate synthetic calibration scenarios via aquacal.datasets.generate_synthetic_rig() with preset rig sizes | ✓ VERIFIED | All three presets work: small (2 cameras, 10 frames), medium (6 cameras, 80 frames), large (13 cameras, 300 frames). Reproducible with fixed seeds. |
| 2 | User can download a real calibration dataset from Zenodo via load_example('real-rig') | ✓ VERIFIED | Real-rig dataset (Zenodo record 18645385) downloads successfully, MD5 checksum verified, loads 12 cameras with reference calibration. |
| 3 | User can load example datasets via convenience function aquacal.datasets.load_example() | ✓ VERIFIED | load_example('small') works instantly from package data, load_example('real-rig') downloads from Zenodo with progress bar and caching. |
| 4 | Larger real datasets (>10MB) are hosted on Zenodo with DOI and download instructions | ✓ VERIFIED | Real-rig dataset (156 MB) hosted at DOI 10.5281/zenodo.18645385, manifest includes size, checksum, and Zenodo record ID. |

**Score:** 4/4 phase-level truths verified


### Plan-Level Must-Haves

**Plan 04-01 Must-Haves:**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | generate_synthetic_rig('small') returns 2 cameras and 10 frames | ✓ VERIFIED | Returns 2 cameras, 10 frames |
| 2 | generate_synthetic_rig('medium') returns 6 cameras and 80 frames | ✓ VERIFIED | Returns 6 cameras, 80 frames |
| 3 | generate_synthetic_rig('large') returns 13 cameras and 300 frames | ✓ VERIFIED | Returns 13 cameras, 300 frames |
| 4 | Calling same preset twice produces identical results | ✓ VERIFIED | Intrinsics matrices match exactly on repeated calls |
| 5 | generate_synthetic_rig with include_images=True returns rendered images | ✓ VERIFIED | Returns images dict, grayscale uint8 (1080, 1920) |
| 6 | Existing synthetic and unit tests pass without modification | ✗ FAILED | Import errors in 5 test files |

**Plan 04-02 Must-Haves:**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | load_example('small') returns ExampleDataset instantly without download | ✓ VERIFIED | Loads from package data, no download |
| 2 | load_example('medium') raises clear error about Zenodo | ✓ VERIFIED | NotImplementedError with helpful message |
| 3 | Small preset data included in package | ✓ VERIFIED | 17 KB data files in src/aquacal/datasets/data/small/ |
| 4 | Cache directory created with .gitignore | ✓ VERIFIED | ./aquacal_data/ exists with subdirectories |
| 5 | requests and tqdm in dependencies | ✓ VERIFIED | Both in pyproject.toml dependencies |

**Plan 04-03 Must-Haves:**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Real dataset uploaded to Zenodo with DOI | ✓ VERIFIED | DOI 10.5281/zenodo.18645385 |
| 2 | Manifest updated with Zenodo metadata | ✓ VERIFIED | Has record_id, checksum (MD5), size_bytes |
| 3 | load_example('real-rig') downloads and loads successfully | ✓ VERIFIED | Downloads 156 MB, verifies MD5, loads 12 cameras |


### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| src/aquacal/datasets/__init__.py | ✓ VERIFIED | Exports all public API: generate_synthetic_rig, load_example, etc. |
| src/aquacal/datasets/synthetic.py | ✓ VERIFIED | 810 lines, has generate_synthetic_rig with 3 presets |
| src/aquacal/datasets/rendering.py | ✓ VERIFIED | 200 lines, image rendering functions |
| src/aquacal/datasets/loader.py | ✓ VERIFIED | 280 lines, load_example and deserializers |
| src/aquacal/datasets/download.py | ✓ VERIFIED | 215 lines, download with progress and checksums |
| src/aquacal/datasets/_manifest.py | ✓ VERIFIED | 65 lines, manifest loading functions |
| src/aquacal/datasets/data/manifest.json | ✓ VERIFIED | Has all 4 datasets with Zenodo metadata |
| src/aquacal/datasets/data/small/*.json | ✓ VERIFIED | detections.json (12 KB), ground_truth.json (4.7 KB) |
| pyproject.toml | ✓ VERIFIED | Has requests, tqdm; package-data configured |
| tests/unit/test_datasets.py | ✓ VERIFIED | 20 tests, all pass in 45.66s |
| tests/synthetic/ground_truth.py | ⚠️ PARTIAL | Missing 3 re-exports (see gaps) |

### Key Link Verification

| From | To | Status | Details |
|------|-----|--------|---------|
| synthetic.py | refractive_geometry | ✓ WIRED | Import and usage verified |
| rendering.py | refractive_geometry | ✓ WIRED | Import and usage verified |
| loader.py | _manifest.py | ✓ WIRED | get_dataset_info imported and used |
| loader.py | download.py | ✓ WIRED | download_and_extract imported and used |
| __init__.py | loader.py | ✓ WIRED | Exports load_example, ExampleDataset |
| ground_truth.py | synthetic.py | ⚠️ PARTIAL | Some functions re-exported, 3 missing |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| loader.py | 241 | TODO comment | ℹ️ Info | Reference calibration stored as dict, not blocker |
| ground_truth.py | N/A | Missing re-exports | 🛑 Blocker | 5 test files fail to import |


### Gaps Summary

**Gap: Missing re-exports in tests/synthetic/ground_truth.py**

Plan 04-01 moved synthetic data generation from tests/synthetic/ground_truth.py to src/aquacal/datasets/synthetic.py. The plan required ground_truth.py to re-export all functions so existing test imports wouldn't break.

However, three critical functions are missing from ground_truth.py:
- generate_synthetic_detections
- compute_calibration_errors
- generate_camera_intrinsics

**Impact:** 5 test files fail to import:
- tests/unit/test_interface_estimation.py
- tests/unit/test_reprojection.py
- tests/unit/test_refinement.py
- tests/synthetic/test_full_pipeline.py
- tests/synthetic/test_refractive_comparison.py

**Fix needed:** Add to tests/synthetic/ground_truth.py:
```python
from aquacal.datasets.synthetic import (
    # ... existing imports ...
    generate_synthetic_detections,
    compute_calibration_errors,
    generate_camera_intrinsics,
)
```

**Violates success criteria:** Plan 04-01 explicitly stated "Existing synthetic and unit tests pass without modification to test logic." This gap directly violates that criteria.

---

**Overall Assessment:**

The phase goal IS achieved - researchers have access to synthetic data generation (working perfectly) and real-world example data (156 MB dataset on Zenodo, loads successfully). The public API works correctly for all intended use cases.

However, the refactoring broke internal test infrastructure. While the new dataset tests pass (20/20), the broader test suite cannot run due to missing re-exports. This is a technical debt issue that blocks phase completion.

**Recommendation:** Fix the missing re-exports before marking phase complete.

---

_Verified: 2026-02-14T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
