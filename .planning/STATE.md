# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 3 complete — ready for Phase 4

## Current Position

Phase: 4 of 6 (Example Data)
Plan: 1 of 3
Status: In Progress
Last activity: 2026-02-15 — Completed 04-01: Synthetic data API with 3 presets

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 221 seconds
- Total execution time: 0.61 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 514s | 171s |
| 02 | 3 | 434s | 145s |
| 03 | 3 | 470s | 157s |
| 04 | 1 | 660s | 660s |

**Recent Executions:**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 04 | 01 | 660s | 2 | 5 |
| 03 | 03 | 299s | 2 | 1 |
| 03 | 02 | 175s | 2 | 3 |
| 03 | 01 | 120s | 2 | 6 |
| 02 | 03 | 197s | 3 | 8 |

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
- [Phase 02-ci-cd-automation]: Use Trusted Publishing (OIDC) for PyPI to avoid API tokens
- [Phase 02-ci-cd-automation]: Test gate before publish even though main should be green
- [Phase 02-ci-cd-automation]: Minimal Sphinx scaffolding now, full docs in Phase 5
- [Phase 02-ci-cd-automation]: Conventional commits for semantic versioning (feat→minor, fix→patch)
- [Phase 02-ci-cd-automation]: python-semantic-release automates version bumping and tag creation
- [Phase 03-public-release]: Badge ordering (Build|Coverage|PyPI|Python|License|DOI) for first impression quality signals
- [Phase 03-public-release]: DOI badge placeholder until Zenodo mints DOI post-release
- [Phase 03-public-release]: Non-binding citation request appropriate for open-source research software
- [Phase 03-public-release]: RELEASE_TOKEN PAT needed for semantic-release to trigger publish workflow
- [Phase 03-public-release]: TestPyPI stage with manual approval gate before real PyPI publish
- [Phase 03-public-release]: Codecov enabled and configured for coverage reporting

### Pending Todos

- Update DOI badge in README once Zenodo mints DOI

### Blockers/Concerns

**Phase 4 (Example Data):**
- Real calibration dataset availability and size constraints (<50MB)
- Zenodo account setup for larger dataset hosting

**Phase 5 (Documentation Site):**
- Read the Docs account and repository integration
- Docstring completeness verification for all public API

## Session Continuity

Last session: 2026-02-15 (phase execution)
Stopped at: Completed 04-01-PLAN.md — Synthetic data API
Resume file: None
