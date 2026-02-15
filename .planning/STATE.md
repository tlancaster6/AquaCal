# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-15)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 7 - Infrastructure Check

## Current Position

Phase: 7 of 13 (Infrastructure Check)
Plan: Ready to plan
Status: Ready to plan
Last activity: 2026-02-15 — Roadmap created for v1.4 QA & Polish milestone

Progress: [████████████░░░░░░░░] 46% (6/13 phases complete from v1.2)

## Performance Metrics

**Velocity:**
- Total plans completed: 20 (v1.2 milestone)
- Average duration: 5.6 min
- Total execution time: 1.85 hours

**By Phase (v1.2):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation and Cleanup | 3 | 21 min | 7 min |
| 2. CI/CD Automation | 3 | 18 min | 6 min |
| 3. Public Release | 3 | 14 min | 4.7 min |
| 4. Example Data | 3 | 15 min | 5 min |
| 5. Documentation Site | 4 | 22 min | 5.5 min |
| 6. Interactive Tutorials | 4 | 21 min | 5.25 min |

**Recent Trend:**
- Last 5 plans: 5.2, 5.5, 5.0, 5.3, 5.4 min
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.2: Pre-execute notebooks for reproducible docs builds without runtime dependencies
- v1.2: Trusted Publishing (OIDC) for PyPI — no API tokens needed
- v1.2: python-semantic-release for automated version bumping from conventional commits

### Pending Todos

- Update DOI badge in README once Zenodo mints DOI (Phase 13)
- Set up Read the Docs deployment (Phase 13)
- Configure RELEASE_TOKEN PAT for semantic-release (Phase 13)
- Design better hero image for README (Phase 11)

### Blockers/Concerns

**Known from PROJECT.md:**
- Read the Docs deployment not yet configured (docs build locally)
- Zenodo DOI badge placeholder until webhook is set up
- RELEASE_TOKEN PAT needed for semantic-release to trigger publish workflow

These will be addressed in Phase 7 (Infrastructure Check) and Phase 13 (Infrastructure Completion).

## Session Continuity

Last session: 2026-02-15 (v1.2 completed)
Stopped at: Milestone v1.2 shipped, v1.4 roadmap created
Resume file: None

**Next step:** `/gsd:plan-phase 7` to begin v1.4 QA & Polish milestone
