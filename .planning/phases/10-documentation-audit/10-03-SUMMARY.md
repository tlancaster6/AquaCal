---
phase: 10-documentation-audit
plan: 03
subsystem: documentation
tags: [docs, cli, troubleshooting, glossary, camera-models, sphinx, markdown]

# Dependency graph
requires:
  - phase: 10-documentation-audit
    plan: 01
    provides: Audit report identifying documentation gaps
  - phase: 10-documentation-audit
    plan: 02
    provides: water_z terminology for consistent documentation
provides:
  - CLI reference documentation (calibrate, init, compare commands)
  - Troubleshooting guide with practical tips
  - Glossary with 18 domain terms
  - Camera model documentation (standard, rational, fisheye, auto-simplification)
affects: [end-users, documentation-site]

# Tech tracking
tech-stack:
  added: []
  patterns: [sphinx-myst, definition-lists, admonition-boxes]

key-files:
  created:
    - docs/guide/cli.md
    - docs/guide/troubleshooting.md
    - docs/guide/glossary.md
  modified:
    - docs/guide/optimizer.md
    - docs/guide/index.md
    - docs/index.md
    - docs/overview.md

key-decisions:
  - "CLI reference is reference-style (not workflow walkthrough)"
  - "Camera models section added to optimizer.md (not standalone page)"
  - "Troubleshooting uses practical 'if X, try Y' format"
  - "Glossary uses definition list format for clarity"
  - "Units convention ('all values in meters') added prominently to landing page"

patterns-established:
  - "Definition lists for glossary terms"
  - "Table format for CLI options"
  - "Cross-references between practical guides and theory pages"

# Metrics
duration: 4min
completed: 2026-02-16
---

# Phase 10 Plan 03: Apply Audit Fixes and Create Documentation Summary

**New CLI guide, troubleshooting section, glossary, camera models, and audit fixes applied — all 6 documentation todos resolved**

## Performance

- **Duration:** 3 min 39 sec
- **Started:** 2026-02-16T03:48:20Z
- **Completed:** 2026-02-16T03:51:59Z
- **Tasks:** 2 (1 checkpoint, 1 auto)
- **Files created:** 3
- **Files modified:** 4

## Accomplishments

### New Documentation Pages

**1. CLI Reference (`docs/guide/cli.md`)** — 6.3 KB
- Complete reference for all three commands: `calibrate`, `init`, `compare`
- All flags and options documented with descriptions
- Syntax examples for each command
- Exit codes and error handling
- Configuration file format overview

**2. Troubleshooting Guide (`docs/guide/troubleshooting.md`)** — 8.5 KB
- Reference camera choice guidance (use lowest Stage 1 RMS)
- High RMS troubleshooting (lower frame_step, verify board measurements)
- Camera model selection and overfitting detection
- Allowed camera combinations with examples (valid/invalid configs)
- Performance tips (max_calibration_frames)
- No detections troubleshooting

**3. Glossary (`docs/guide/glossary.md`)** — 3.9 KB
- 18 domain terms defined (water_z, extrinsics, intrinsics, board pose, ChArUco, reference camera, auxiliary camera, bundle adjustment, pose graph, refractive projection, Snell's law, Rodrigues vector, world frame, camera frame, interface, interface normal, refractive index, reprojection error, validation set)
- Definition list format for clarity
- Cross-references to theory pages

### Enhanced Existing Pages

**4. Camera Models Section in `docs/guide/optimizer.md`**
- Standard 5-parameter model with auto-simplification logic
- Rational 8-parameter model for wide-angle lenses
- Fisheye 4-parameter equidistant model
- When to use each model
- Signs of overfitting
- Allowed camera combinations (with constraint table)
- References to intrinsics.py implementation (lines 351-379)

**5. Guide Index (`docs/guide/index.md`)**
- Added "Practical Guides" section
- New pages in toctree: cli, troubleshooting, glossary
- Updated page descriptions

**6. Overview Page (`docs/overview.md`)**
- Removed placeholder text referencing "Plan 02" and "Plan 03" (DOC-INCONSISTENCY-001 from audit)
- Direct links to theory pages
- Added CLI reference and troubleshooting links

**7. Landing Page (`docs/index.md`)**
- Added prominent units statement: "All values are in meters."
- Clarifies that camera positions, water_z, board dimensions, and outputs all use meters

## Task Commits

Task 2 committed atomically:

1. **Task 2: Create new documentation sections and apply audit fixes** - `afdbbca` (docs)
   - 665 insertions, 10 deletions
   - 3 new files created
   - 4 existing files updated
   - All audit fixes applied

## Files Created

All new documentation pages created and linked into Sphinx toctree:

- `docs/guide/cli.md` — 6.3 KB, 250+ lines
- `docs/guide/troubleshooting.md` — 8.5 KB, 300+ lines
- `docs/guide/glossary.md` — 3.9 KB, 80+ lines

## Decisions Made

**1. CLI guide format: reference-style, not workflow walkthrough**
- Documented syntax, flags, and options for each command
- Included examples but focused on reference completeness
- Workflow tutorials belong in narrative guides or notebooks

**2. Camera models added to optimizer.md, not standalone page**
- Camera model selection is closely tied to calibration pipeline
- Keeps theory and practice together
- Cross-referenced from troubleshooting guide

**3. Troubleshooting format: practical "if X, try Y"**
- Problem → Why → Solution structure
- Real examples from actual calibration issues
- Cross-references to theory pages for deeper understanding

**4. Glossary uses definition list format**
- MyST Markdown definition lists (`term\n: definition`)
- Cleaner than headers for short definitions
- Easier to scan for quick reference

**5. Units convention prominently displayed**
- Landing page now states "all values in meters" upfront
- Prevents confusion for new users
- Reinforces convention established in coordinate documentation

## Deviations from Plan

None — plan executed exactly as written. All 6 documentation todos addressed:

1. ✅ CLI usage guide created
2. ✅ Camera model documentation created (in optimizer.md)
3. ✅ Troubleshooting section created
4. ✅ Allowed camera combinations documented
5. ✅ Glossary created
6. ✅ Audit fixes applied (placeholder text removed, units statement added)

## Issues Encountered

None — all documentation pages created successfully, audit fixes applied, and Sphinx toctree updated without issues.

## Next Phase Readiness

**Phase 10 Documentation Audit is now complete.**

All three plans in Phase 10:
- ✅ Plan 01: Comprehensive audit report
- ✅ Plan 02: Terminology update (interface_distance → water_z)
- ✅ Plan 03: New documentation sections and audit fixes

**Documentation coverage:**
- ✅ Theory pages (refractive geometry, coordinates, optimizer pipeline)
- ✅ Practical guides (CLI reference, troubleshooting, glossary)
- ✅ API reference (autodoc, existing)
- ✅ Tutorials (notebooks, existing from Phase 6)

**Ready for Phase 11 (Public Presence Enhancement)** — Documentation is publication-ready.

**No blockers.**

---

## Detailed Content Summary

### CLI Reference Coverage

**aquacal calibrate:**
- All flags: `-v`/`--verbose`, `-o`/`--output-dir`, `--dry-run`
- Exit codes: 0, 1, 2, 3, 130
- Examples: basic usage, verbose mode, dry-run validation, output override

**aquacal init:**
- Required: `--intrinsic-dir`, `--extrinsic-dir`
- Optional: `-o`/`--output`, `--pattern`
- Camera name extraction with regex examples
- Directory scanning logic explained
- Exit codes: 0, 1

**aquacal compare:**
- Required: 2+ directories
- Optional: `-o`/`--output-dir`, `--no-plots`
- Output files documented (CSV and PNG)
- Exit codes: 0, 1, 2

### Troubleshooting Topics Covered

1. Reference camera choice (use lowest Stage 1 RMS)
2. High RMS in Stage 1 (lower frame_step, verify board, check video quality, use max_calibration_frames)
3. Bad round-trip errors or diverging optimization (connectivity, initial_water_z, bounds)
4. Camera models and overfitting (standard auto-simplification, rational/fisheye manual downgrade)
5. Allowed camera combinations (constraints table, valid/invalid examples)
6. Memory and performance (max_calibration_frames)
7. No detections found (verify board config, improve video quality, lower min_corners)

### Glossary Terms Defined (18 total)

Core concepts: water_z, extrinsics, intrinsics, world frame, camera frame, reference camera, auxiliary camera

Calibration process: board pose, bundle adjustment, pose graph, ChArUco board, validation set

Refractive geometry: interface, interface normal, refractive index, refractive projection, Snell's law, reprojection error

Math representations: Rodrigues vector

### Camera Models Section

**Standard model:**
- 5 parameters (k1, k2, p1, p2, k3)
- Auto-simplification: full → fix k3 → fix k3+k2
- Implementation reference (intrinsics.py lines 351-379)

**Rational model:**
- 8 parameters (k1-k6, p1, p2)
- Rational polynomial distortion
- No auto-simplification
- Use for wide-angle lenses with 50+ frames

**Fisheye model:**
- 4 parameters (k1-k4 equidistant)
- Must be auxiliary cameras
- No auto-simplification
- Use for > 120° FOV lenses

**Constraints table:**
- cameras vs. auxiliary_cameras (no overlap)
- rational_model_cameras (can be primary or auxiliary, not fisheye)
- fisheye_cameras (must be auxiliary, not rational)

---

## Self-Check: PASSED

**Files created:**
- ✓ `docs/guide/cli.md` exists (6.3 KB)
- ✓ `docs/guide/troubleshooting.md` exists (8.5 KB)
- ✓ `docs/guide/glossary.md` exists (3.9 KB)

**Files modified:**
- ✓ `docs/guide/optimizer.md` contains camera models section (17 KB, expanded from 11 KB)
- ✓ `docs/guide/index.md` includes new pages in toctree
- ✓ `docs/overview.md` has placeholder text removed
- ✓ `docs/index.md` has units statement added

**Commit verified:**
- ✓ `afdbbca` — Task 2: create new documentation sections and apply audit fixes

**Content verification:**
- ✓ CLI guide documents all three commands with all flags
- ✓ Troubleshooting contains required tips (reference camera, frame_step, max_calibration_frames, camera models)
- ✓ Glossary contains 18+ domain terms
- ✓ Camera models section covers all three models + auto-simplification
- ✓ Allowed combinations documented with constraint table

All claimed artifacts verified.

---

*Phase: 10-documentation-audit*
*Completed: 2026-02-16*
