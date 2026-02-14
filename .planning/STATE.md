# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 2 - CI/CD Automation

## Current Position

Phase: 2 of 6 (CI/CD Automation)
Plan: 2 of 3
Status: In progress
Last activity: 2026-02-14 — Completed 02-02: GitHub Actions test workflows

Progress: [████░░░░░░] 25%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 146 seconds
- Total execution time: 0.20 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 514s | 171s |
| 02 | 2 | 237s | 119s |

**Recent Executions:**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 02 | 02 | 157s | 2 | 2 |
| 02 | 01 | 80s | 2 | 2 |
| 01 | 03 | 5 min | 2 | 0 |
| 01 | 02 | 107s | 2 | 4 |
| 01 | 01 | 8 min | 2 | 6 |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyPI + GitHub distribution as standard for research Python libraries
- Real + synthetic example data to build trust and demonstrate correctness
- Jupyter notebooks for interactive, visual examples ideal for research audience
- License choice needed before public release (tracked for Phase 3)
- [Phase 01-foundation-and-cleanup]: Migrated dev/ documentation to .planning/ for GSD compatibility
- [Phase 01-foundation-and-cleanup]: Removed pre-GSD agent infrastructure to avoid confusion
- [Phase 01-foundation-and-cleanup]: Sdist build deferred to CI/CD due to Windows file locking
- [Phase 01-foundation-and-cleanup]: Cross-platform validation delegated to user checkpoint
- **Roadmap reorder**: CI/CD (Phase 2) and Public Release (Phase 3) moved ahead of Example Data, Docs, and Tutorials (Phases 4-6) to unblock dependent project
- [Phase 02-ci-cd-automation]: Use ruff instead of black/mypy (faster, all-in-one tool)
- [Phase 02-ci-cd-automation]: Lenient lint rules initially (E4, E7, E9, F, W, I) to avoid overwhelming noise
- [Phase 02-ci-cd-automation]: Pre-commit enforced in CI to catch contributors who skip local hooks
- [Phase 02-ci-cd-automation]: Slow tests separated to manual trigger to maintain fast PR feedback
- [Phase 02-ci-cd-automation]: Coverage upload graceful (fail_ci_if_error: false) until Codecov configured

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 2 (CI/CD Automation):**
- Trusted Publishing setup on PyPI (OIDC configuration)
- Codecov account and integration

**Phase 3 (Public Release):**
- License decision must be made before PyPI release
- Cross-platform installation validated (Windows automated, user confirmed Linux/macOS compatible)

**Phase 4 (Example Data):**
- Real calibration dataset availability and size constraints (<50MB)
- Zenodo account setup for larger dataset hosting

**Phase 5 (Documentation Site):**
- Read the Docs account and repository integration
- Docstring completeness verification for all public API

## Session Continuity

Last session: 2026-02-14 (plan execution)
Stopped at: Completed 02-02-PLAN.md (GitHub Actions test workflows)
Resume file: None
