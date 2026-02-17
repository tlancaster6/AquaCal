---
phase: 11-documentation-visuals
plan: 01
subsystem: docs
tags: [matplotlib, palette, visualization, hero-image, diagrams]

# Dependency graph
requires:
  - phase: 10-documentation-audit
    provides: Documentation content complete, diagrams exist but use inconsistent colors
provides:
  - Blue/aqua color palette module (palette.py) importable by all diagram scripts
  - Style guide documenting the full palette with hex values and usage notes
  - New hero image (hero_ray_trace.png) showing 3-camera refractive scene in 2D
  - Updated ray_trace.png and coordinate_frames.png using palette colors
affects: [11-02, 11-03, any future diagram scripts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Centralized palette.py module — all diagram scripts import from docs/_static/scripts/palette.py"
    - "Path-based sys.path.insert for portable diagram script imports (no package installation required)"

key-files:
  created:
    - docs/_static/scripts/palette.py
    - docs/_static/scripts/style_guide.md
    - docs/_static/scripts/hero_image.py
    - docs/_static/hero_ray_trace.png
  modified:
    - docs/guide/_diagrams/ray_trace.py
    - docs/guide/_diagrams/coordinate_frames.py
    - docs/_static/diagrams/ray_trace.png
    - docs/_static/diagrams/coordinate_frames.png

key-decisions:
  - "Hero image uses 2D cross-section with 3 cameras, refracted rays, and tilted board — no equations or angle annotations"
  - "Palette constants defined as module-level strings in palette.py for maximum portability"
  - "Diagram scripts use Path(__file__).parent-based sys.path resolution for reliable imports regardless of working directory"

patterns-established:
  - "All AquaCal diagram scripts import colors from docs/_static/scripts/palette.py"
  - "New diagrams added in later plans should follow the same palette import pattern"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 11 Plan 01: Visual Foundation Summary

**Blue/aqua palette module, portable style guide, 3-camera 2D hero image, and palette-updated ray_trace/coordinate_frames diagrams**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-17T20:23:02Z
- **Completed:** 2026-02-17T20:26:48Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Created `docs/_static/scripts/palette.py` defining 13 color constants for the blue/aqua AquaCal theme
- Created `docs/_static/scripts/style_guide.md` documenting each color's hex value, usage, and rationale
- Created `docs/_static/scripts/hero_image.py` generating a professional 2D cross-section hero image with 3 cameras, refracted rays, and a calibration board; saved to `docs/_static/hero_ray_trace.png`
- Updated `ray_trace.py` and `coordinate_frames.py` to import from `palette.py` and regenerated both PNGs

## Task Commits

Each task was committed atomically:

1. **Task 1: Create style guide, palette module, and hero image** - `bfa1424` (feat)
2. **Task 2: Update existing diagrams to use shared palette** - `b18cd89` (feat)

**Plan metadata:** (final docs commit below)

## Files Created/Modified

- `docs/_static/scripts/palette.py` - 13 color constants for blue/aqua AquaCal theme
- `docs/_static/scripts/style_guide.md` - Human-readable palette reference with hex values and usage notes
- `docs/_static/scripts/hero_image.py` - Hero image generation script (3-camera 2D refractive scene, 1200x500px)
- `docs/_static/hero_ray_trace.png` - New hero image (21KB, replaces old single-ray diagram)
- `docs/guide/_diagrams/ray_trace.py` - Updated to import RAY_AIR, RAY_WATER, WATER_SURFACE, etc. from palette
- `docs/guide/_diagrams/coordinate_frames.py` - Updated to import AXIS_X/Y/Z, CAMERA_COLOR, BOARD_COLOR, etc. from palette
- `docs/_static/diagrams/ray_trace.png` - Regenerated with palette colors (70KB)
- `docs/_static/diagrams/coordinate_frames.png` - Regenerated with palette colors (246KB)

## Decisions Made

- Hero image is 2D (not 3D), showing a cross-section — simpler and more readable at a glance
- No equations or angle annotations on the hero image — "Camera", "Water surface", "Board" labels only
- Camera icons rendered as downward-pointing trapezoids (wider at top, narrower at lens end)
- `palette.py` uses module-level string constants (not a dict or dataclass) for direct `from palette import X` ergonomics

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Ruff pre-commit hook reformatted both commits (import ordering and line length). Re-staged and recommitted both times.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Palette established — Plan 02 (new diagrams: sparsity pattern, BFS pose graph) can import from `palette.py`
- `generate_all.py` ready to extend with new diagram generators in Plan 02
- Hero image in place at the canonical `docs/_static/hero_ray_trace.png` path

---
*Phase: 11-documentation-visuals*
*Completed: 2026-02-17*

## Self-Check: PASSED

All files confirmed present and all task commits verified in git log.
