# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-15)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 12 - Tutorial Verification

## Current Position

Phase: 12 of 12 (Tutorial Verification)
Plan: 01 complete (Plan 02 next)
Status: In progress
Last activity: 2026-02-17 — Phase 12 Plan 01 complete — tutorial restructure (3 to 2), diagnostics merged into tutorial 01

Progress: [████████████████████] 92% (11/12 phases complete, Phase 12 in progress)

## Performance Metrics

**Velocity:**
- Total plans completed: 24 (20 from v1.2, 4 from v1.4)
- Average duration: 5.2 min
- Total execution time: 2.08 hours

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
| 10. Documentation Audit | 3 | 21 min | 7 min |

**Recent Trend:**
- Last 5 plans: 5.4, 8.0, 3.0, 14.0, 4.0 min
- Trend: Variable (Plan 02 rename was comprehensive but mechanical)

*Updated after each plan completion*
| Phase 10 P01 | 3 | 2 tasks | 1 files |
| Phase 10 P02 | 14 | 2 tasks | 43 files |
| Phase 10 P03 | 4 | 2 tasks | 7 files |
| Phase 11 P01 | 3 | 2 tasks | 8 files |
| Phase 12 P01 | 12 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.4 Phase 10: All docstrings are accurate and complete — no critical errors or rewrites needed
- v1.4 Phase 10: interface_distance terminology update is extensive (55 files) but mechanical
- v1.4 Phase 10: Documentation gaps identified are user-facing guides (CLI, camera models, troubleshooting), not API docs
- v1.4 Phase 10: CLI reference is reference-style (not workflow walkthrough)
- v1.4 Phase 10: Camera models section added to optimizer.md (not standalone page)
- v1.4 Phase 10: Troubleshooting uses practical "if X, try Y" format
- v1.4 Phase 10: All 6 documentation todos from Phase 10 now resolved
- v1.4 Phase 7: Phase 13 (Infrastructure Completion) should be removed from roadmap - all infrastructure work already complete
- v1.4 Phase 7: All three pending infrastructure todos (RTD, DOI, RELEASE_TOKEN) verified as done
- v1.2: Pre-execute notebooks for reproducible docs builds without runtime dependencies
- v1.2: Trusted Publishing (OIDC) for PyPI — no API tokens needed
- v1.2: python-semantic-release for automated version bumping from conventional commits
- [Phase 11]: Centralized palette.py module — all diagram scripts import colors from docs/_static/scripts/palette.py
- [Phase 11]: Mermaid pipeline replaces ASCII diagram; intermediate data labels dropped for cleaner look
- [Phase 12 P01]: Diagnostics merged into tutorial 01 (not standalone) — calibrate-then-diagnose in single notebook
- [Phase 12 P01]: Tutorial 02 (was 03) content unchanged; rewrite deferred to Plan 02
- [Phase 12 P01]: Old 02_diagnostics.ipynb deleted; content subsumed by merged sections in tutorial 01

### Pending Todos

- Design better hero image for README (deferred from Phase 11 — user wants to rethink concept)
- Reduce memory and CPU load during calibration

### Blockers/Concerns

None - all previously flagged infrastructure items (RTD, DOI, RELEASE_TOKEN) verified as complete during Phase 7.

## Session Continuity

Last session: 2026-02-17 (Phase 12 Plan 01 complete — Tutorial Restructure)
Stopped at: Completed 12-01-PLAN.md (Tutorial Restructure)
Resume file: N/A

**Next step:** Execute Phase 12 Plan 02 (tutorial 01 re-execution and tutorial 02 rewrite)
