# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-15)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 7 - Infrastructure Check

## Current Position

Phase: 7 of 13 (Infrastructure Check)
Plan: 01 complete
Status: Phase complete
Last activity: 2026-02-15 — Infrastructure audit complete, all items verified

Progress: [█████████████░░░░░░░] 54% (7/13 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 21 (20 from v1.2, 1 from v1.4)
- Average duration: 5.6 min
- Total execution time: 1.98 hours

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

**Recent Trend:**
- Last 5 plans: 5.5, 5.0, 5.3, 5.4, 8.0 min
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.4 Phase 7: Phase 13 (Infrastructure Completion) should be removed from roadmap - all infrastructure work already complete
- v1.4 Phase 7: All three pending infrastructure todos (RTD, DOI, RELEASE_TOKEN) verified as done
- v1.2: Pre-execute notebooks for reproducible docs builds without runtime dependencies
- v1.2: Trusted Publishing (OIDC) for PyPI — no API tokens needed
- v1.2: python-semantic-release for automated version bumping from conventional commits

### Pending Todos

- Design better hero image for README (Phase 11)

### Blockers/Concerns

None - all previously flagged infrastructure items (RTD, DOI, RELEASE_TOKEN) verified as complete during Phase 7.

## Session Continuity

Last session: 2026-02-15 (Phase 7 complete)
Stopped at: Phase 07-01-PLAN.md complete - infrastructure audit verified all items done
Resume file: None

**Next step:** Continue v1.4 milestone with Phase 8 (Dataset Quality Check) or update roadmap to remove Phase 13
