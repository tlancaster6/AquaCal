# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-15)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 10 - Documentation Audit

## Current Position

Phase: 10 of 12 (Documentation Audit)
Plan: 01 of 03 complete
Status: In progress
Last activity: 2026-02-16 — Documentation audit complete, no critical errors found

Progress: [██████████████░░░░░░] 67% (8/12 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 22 (20 from v1.2, 2 from v1.4)
- Average duration: 5.4 min
- Total execution time: 2.03 hours

**By Phase (v1.2):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation and Cleanup | 3 | 21 min | 7 min |
| 2. CI/CD Automation | 3 | 18 min | 6 min |
| 3. Public Release | 3 | 14 min | 4.7 min |
| 4. Example Data | 3 | 15 min | 5 min |
| 5. Documentation Site | 4 | 22 min | 5.5 min |
| 6. Interactive Tutorials | 4 | 21 min | 5.25 min |

**By Phase (v1.4):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 7. Infrastructure Check | 1 | 8 min | 8 min |
| 10. Documentation Audit | 1 | 3 min | 3 min |

**Recent Trend:**
- Last 5 plans: 5.0, 5.3, 5.4, 8.0, 3.0 min
- Trend: Variable (infrastructure/audit phases faster than feature work)

*Updated after each plan completion*
| Phase 10 P01 | 3 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.4 Phase 10: All docstrings are accurate and complete — no critical errors or rewrites needed
- v1.4 Phase 10: interface_distance terminology update is extensive (55 files) but mechanical
- v1.4 Phase 10: Documentation gaps identified are user-facing guides (CLI, camera models, troubleshooting), not API docs
- v1.4 Phase 7: Phase 13 (Infrastructure Completion) should be removed from roadmap - all infrastructure work already complete
- v1.4 Phase 7: All three pending infrastructure todos (RTD, DOI, RELEASE_TOKEN) verified as done
- v1.2: Pre-execute notebooks for reproducible docs builds without runtime dependencies
- v1.2: Trusted Publishing (OIDC) for PyPI — no API tokens needed
- v1.2: python-semantic-release for automated version bumping from conventional commits

### Pending Todos

- Design better hero image for README (Phase 11)
- Detailed CLI usage guide — currently no documentation on CLI commands (Phase 10)
- Clarify allowed combinations of auxiliary_cameras, fisheye_cameras, and rational_model_cameras (Phase 10)
- Add usage note: extrinsic calibration is sensitive to reference camera choice. Set reference camera to whichever had lowest RMS in Stage 1 (printed to console during calibrate) (Phase 10)
- Check for misspellings of "auxiliary" (commonly misspelled "auxillary") across all docs (Phase 10)
- Add usage tip: bad RMS or high round-trip errors in Stage 1 → lower frame_step for more intrinsic data, optionally set max_calibration_frames to limit expensive Stage 3/4 optimization (Phase 10)
- Document the three camera models (standard 5-param, rational 8-param, fisheye 4-param) with a section on auto-simplification logic (git 0863fae). Note: only standard model has auto-simplification; rational/fisheye users should downgrade to standard if overfitting suspected. Include signs of overfitting (Phase 10)
- Reduce memory and CPU load during calibration

### Blockers/Concerns

None - all previously flagged infrastructure items (RTD, DOI, RELEASE_TOKEN) verified as complete during Phase 7.

## Session Continuity

Last session: 2026-02-16 (Phase 10 Plan 01 executed)
Stopped at: Completed 10-01-PLAN.md — audit report created, ready for Plan 02
Resume file: .planning/phases/10-documentation-audit/10-01-SUMMARY.md

**Next step:** Execute Plan 02 (Terminology Update) with `/gsd:execute-phase 10`
