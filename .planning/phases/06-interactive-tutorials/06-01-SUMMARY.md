---
phase: 06-interactive-tutorials
plan: 01
subsystem: io
tags: [frameset, imageset, auto-detection, tdd]
dependency_graph:
  requires: [VideoSet, detect_all_frames, BoardGeometry]
  provides: [FrameSet, ImageSet, _create_frame_source]
  affects: [detect_all_frames, io.__init__]
tech_stack:
  added: [natsort]
  patterns: [Protocol, runtime_checkable, duck-typing]
key_files:
  created:
    - src/aquacal/io/frameset.py
    - src/aquacal/io/images.py
    - tests/unit/test_images.py
  modified:
    - src/aquacal/io/detection.py
    - src/aquacal/io/__init__.py
    - pyproject.toml
decisions:
  - ImageSet validates strict frame count equality (not min) for safety
  - Natural sort ordering via natsort library (industry standard)
  - Auto-detection based on Path.is_dir() vs Path.is_file()
  - FrameSet as Protocol (not ABC) for structural subtyping
  - ImageSet has no resources to manage (context manager is no-op)
  - Step parameter supported in ImageSet but images assumed pre-curated
metrics:
  duration: 463s
  tasks_completed: 1
  files_created: 3
  files_modified: 3
  tests_added: 14
  tests_passing: 586
  completed_date: 2026-02-15
---

# Phase 6 Plan 1: FrameSet Protocol + ImageSet + Auto-Detection Summary

**One-liner:** Unified FrameSet protocol with ImageSet implementation enables image directory inputs alongside videos with automatic source detection.

## Objective

Implement abstraction layer allowing the detection pipeline to accept both video files (VideoSet) and image directories (ImageSet) transparently via Protocol-based duck typing and automatic source detection.

## What Was Built

### FrameSet Protocol (`frameset.py`)
- `@runtime_checkable` Protocol defining common interface for frame sources
- Properties: `camera_names`, `frame_count`, `is_open`
- Methods: `iterate_frames()`, `get_frame()`, `open()`, `close()`
- Context manager support: `__enter__`, `__exit__`
- Enables structural subtyping (duck typing) for VideoSet and ImageSet

### ImageSet Class (`images.py`)
- Loads synchronized image directories (JPEG/PNG, case-insensitive)
- Natural sort ordering via `natsort` library (img_1, img_2, img_10)
- Strict frame count validation (all cameras must have equal counts)
- Mirrors VideoSet API completely for drop-in compatibility
- Context manager is no-op (no resources like VideoCapture to manage)
- On-demand image loading via `cv2.imread()`

### Auto-Detection (`detection.py`)
- `_create_frame_source(paths: dict[str, str]) -> FrameSet`
- Detects directory vs file paths using `Path.is_dir()` / `Path.is_file()`
- Validates all paths are same type (no mixing directories and files)
- Returns ImageSet for directories, VideoSet for files
- Integrated into `detect_all_frames()` signature: `dict[str, str] | FrameSet`

### Dependency Addition
- Added `natsort>=8.4.0` to `pyproject.toml` dependencies
- Industry-standard natural sorting (not lexicographic)

### Public API Updates (`io/__init__.py`)
- Exported `FrameSet` and `ImageSet` to public API
- Updated `__all__` list with new symbols

## Implementation Details

### TDD Execution
**RED Phase (d1ce779):**
- Added natsort dependency to pyproject.toml
- Created comprehensive test suite (14 tests)
- Tests covered: basic initialization, natural sort ordering, frame count validation, mixed extensions, context manager, auto-detection, integration with detect_all_frames
- Tests failed (modules didn't exist yet)

**GREEN Phase (c73aa71):**
- Implemented FrameSet Protocol with full docstrings
- Implemented ImageSet class with natural sorting and validation
- Added _create_frame_source() auto-detection function
- Modified detect_all_frames() to accept FrameSet Protocol
- Updated io/__init__.py exports
- All 14 new tests passing, all 586 total tests passing

**REFACTOR Phase:**
- Not needed - implementation was clean on first pass

### Key Design Decisions

1. **Strict Frame Count Validation**
   - ImageSet raises ValueError if directories have different counts
   - Rationale: Safety - prevents subtle bugs from unmatched frame counts
   - VideoSet uses minimum count (different trade-off for video corruption)

2. **Natural Sort via natsort**
   - Added external dependency rather than custom implementation
   - Rationale: Well-tested, handles edge cases, industry standard
   - Performance: Negligible overhead for typical image counts (<1000)

3. **Protocol vs Abstract Base Class**
   - Used `typing.Protocol` with `@runtime_checkable`
   - Rationale: Structural subtyping allows VideoSet/ImageSet to work without explicit inheritance
   - Benefit: No changes needed to existing VideoSet class

4. **No Resources to Manage**
   - ImageSet context manager is no-op (open/close do nothing)
   - Rationale: Unlike VideoCapture, file paths don't need cleanup
   - Benefit: Simpler implementation, still API-compatible

5. **Auto-Detection Strategy**
   - Check first path, then validate all paths match type
   - Raises ValueError on mixed types (not silently pick one)
   - Rationale: Fail fast on user error rather than surprising behavior

## Testing

### Test Coverage
- **ImageSet Tests (10):**
  - Basic initialization and properties
  - Natural sort ordering (img_1, img_2, img_10 not lexicographic)
  - Mismatched frame counts raise ValueError
  - Empty directory raises ValueError
  - Mixed jpg/png extensions work
  - Frame iteration (start/stop/step)
  - Random access with get_frame()
  - Context manager protocol
  - Nonexistent directory raises FileNotFoundError
  - Case-insensitive extensions (JPG, PNG)

- **Auto-Detection Tests (3):**
  - Directory paths create ImageSet
  - File paths create VideoSet
  - Mixed types raise ValueError

- **Integration Test (1):**
  - detect_all_frames() works with image directories
  - Returns correct DetectionResult structure

### Test Results
- New tests: 14/14 passing
- Total tests: 586/586 passing (no regressions)
- Duration: 1:53 for full suite (non-slow)

## Deviations from Plan

None - plan executed exactly as written.

## Verification

```bash
python -m pytest tests/unit/test_images.py -v           # 14 passed
python -m pytest tests/unit/test_detection.py -v        # 20 passed (backward compat)
python -m pytest tests/ -m "not slow" -q                # 586 passed (full suite)
```

All verification criteria met:
- ✓ ImageSet loads JPEG/PNG images with natural sort ordering
- ✓ ImageSet raises ValueError on mismatched frame counts
- ✓ _create_frame_source auto-detects directory vs file
- ✓ detect_all_frames works with both VideoSet and ImageSet
- ✓ All existing tests pass (no regressions)
- ✓ natsort added to dependencies

## Impact

### For Users
- Can now pass image directories directly to detection pipeline
- No code changes needed - auto-detection handles it
- Enables `load_example('real-rig')` dataset flow (Phase 6 goal)

### For Developers
- FrameSet Protocol provides clean abstraction for future sources
- Easy to add new frame source types (e.g., network streams, generators)
- Type hints work correctly with Protocol-based duck typing

### For Pipeline
- Backward compatible - all existing VideoSet code works
- Forward compatible - easy to extend with new source types
- Validation happens early (at ImageSet construction)

## Files Changed

### Created
| File | Lines | Purpose |
|------|-------|---------|
| src/aquacal/io/frameset.py | 136 | FrameSet Protocol definition |
| src/aquacal/io/images.py | 226 | ImageSet implementation |
| tests/unit/test_images.py | 285 | Comprehensive test suite |

### Modified
| File | Changes | Purpose |
|------|---------|---------|
| src/aquacal/io/detection.py | +54 | Auto-detection and FrameSet integration |
| src/aquacal/io/__init__.py | +6 | Export new public symbols |
| pyproject.toml | +1 | Add natsort dependency |

### Total Impact
- Lines added: 708
- Lines modified: 61
- Tests added: 14
- Dependencies added: 1

## Commits

| Hash | Type | Description |
|------|------|-------------|
| d1ce779 | test | Add failing tests for ImageSet and auto-detection (RED) |
| c73aa71 | feat | Implement FrameSet protocol, ImageSet, and auto-detection (GREEN) |

## Next Steps

Plan 06-02 will implement example data loading utilities (`load_example()` function) that use ImageSet to load the real-rig dataset from cached directories.

## Self-Check: PASSED

**Created files exist:**
```bash
[ -f "src/aquacal/io/frameset.py" ] && echo "FOUND: frameset.py" || echo "MISSING: frameset.py"
# FOUND: frameset.py
[ -f "src/aquacal/io/images.py" ] && echo "FOUND: images.py" || echo "MISSING: images.py"
# FOUND: images.py
[ -f "tests/unit/test_images.py" ] && echo "FOUND: test_images.py" || echo "MISSING: test_images.py"
# FOUND: test_images.py
```

**Commits exist:**
```bash
git log --oneline --all | grep -q "d1ce779" && echo "FOUND: d1ce779" || echo "MISSING: d1ce779"
# FOUND: d1ce779
git log --oneline --all | grep -q "c73aa71" && echo "FOUND: c73aa71" || echo "MISSING: c73aa71"
# FOUND: c73aa71
```

**All tests pass:**
```bash
python -m pytest tests/unit/test_images.py -v
# 14 passed
python -m pytest tests/ -m "not slow" -q
# 586 passed
```

All claims verified.
