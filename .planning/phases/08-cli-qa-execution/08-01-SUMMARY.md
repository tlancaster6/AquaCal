---
phase: 08-cli-qa-execution
plan: 01
subsystem: cli
tags: [cli, qa, init, calibrate, compare]

requires:
  - phase: 06-interactive-tutorials
    provides: Complete CLI implementation and documentation
provides:
  - User-verified CLI workflows (init, calibrate, compare)
  - Bug fixes for config generation (init command)
  - Improved error messages and dry-run output
  - Documentation todos for Phase 10
affects: [09-bug-triage, 10-documentation-audit]

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - src/aquacal/cli.py
    - src/aquacal/calibration/intrinsics.py
    - src/aquacal/config/example_config.yaml
    - tests/unit/test_intrinsics.py

key-decisions:
  - "Raise roundtrip error threshold from 0.5 to 1.0 px"
  - "Don't add auto-simplification to rational model — users should downgrade to standard model if overfitting suspected"
  - "Config generator: auxiliary_cameras before rational/fisheye sections for logical grouping"

patterns-established:
  - "Config generation: show placeholder cam_name instead of listing all discovered cameras in commented-out sections"

duration: 15min
completed: 2026-02-15
---

# Phase 8: CLI QA Execution Summary

**User-verified all three CLI workflows (init, calibrate, compare) with real rig data; fixed config generation and error messages**

## Performance

- **Duration:** 15 min
- **Tasks:** 3 (all human-verify checkpoints)
- **Files modified:** 4

## Accomplishments
- All three CLI commands (init, calibrate, compare) verified working with real rig data
- Fixed config generation: added legacy_pattern, fisheye_cameras, refine_auxiliary_intrinsics; reordered sections; removed auto-populated camera lists
- Improved dry-run output with camera count summary
- Improved "no valid frames" error message to clarify board detection failure
- Raised roundtrip validation threshold from 0.5 to 1.0 px
- Captured 7 documentation todos for Phase 10

## Files Created/Modified
- `src/aquacal/cli.py` - Improved dry-run output, fixed config generation template
- `src/aquacal/calibration/intrinsics.py` - Better error message, raised roundtrip threshold
- `src/aquacal/config/example_config.yaml` - Matched config template improvements
- `tests/unit/test_intrinsics.py` - Updated error message match pattern

## Decisions Made
- Roundtrip threshold 0.5 → 1.0 px to reduce false warnings
- No auto-simplification for rational model; users should downgrade to standard if overfitting
- Config template uses generic placeholder instead of listing all cameras in optional sections

## Deviations from Plan
None - plan executed as written (human-verify checkpoints with fixes applied inline).

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CLI workflows verified, ready for Phase 9 (Bug Triage)
- Documentation todos captured in STATE.md for Phase 10
- Pending todo: reduce memory and CPU load during calibration

---
*Phase: 08-cli-qa-execution*
*Completed: 2026-02-15*
