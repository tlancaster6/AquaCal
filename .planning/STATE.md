# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** Phase 5 complete — documentation site verified (20/20). Phase 6 next.

## Current Position

Phase: 6 of 6 (Interactive Tutorials)
Plan: 1 of 4
Status: In Progress
Last activity: 2026-02-15 — Completed 06-01: FrameSet protocol and ImageSet implementation

Progress: [██████████░░] 80% (16/20 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 335 seconds
- Total execution time: 1.49 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 514s | 171s |
| 02 | 3 | 434s | 145s |
| 03 | 3 | 470s | 157s |
| 04 | 3 | 1471s | 490s |
| 05 | 4 | 2367s | 592s |
| 06 | 1 | 463s | 463s |

**Recent Executions:**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 06 | 01 | 463s | 1 | 6 |
| 05 | 04 | 272s | 1 | 1 |
| 05 | 03 | 970s | 3 | 13 |
| 05 | 02 | 802s | 2 | 10 |
| 05 | 01 | 323s | 2 | 13 |

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
- [Phase 04-example-data]: Small preset ships in-package (17KB) for zero-download quick start
- [Phase 04-example-data]: Cache directory at ./aquacal_data/ with auto-generated .gitignore
- [Phase 04-example-data]: Zenodo datasets use manifest registry with null placeholders until upload
- [Phase 04-example-data]: NotImplementedError for non-available datasets with helpful message
- [Phase 04-example-data]: Download infrastructure with progress bars, checksums, and exponential backoff
- [Phase 04-example-data]: algorithm:hash checksum format supports both MD5 (Zenodo) and SHA256
- [Phase 04-example-data]: Real datasets return metadata and cache_path for user processing
- [Phase 04-example-data]: Nested ZIP directory structure handled automatically
- [Phase 05-documentation-site]: Sphinx with Furo theme and MyST Markdown for clean, modern docs
- [Phase 05-documentation-site]: OpenCV intersphinx removed (no valid inventory available)
- [Phase 06-01]: ImageSet validates strict frame count equality (not min) for safety
- [Phase 06-01]: Natural sort ordering via natsort library (industry standard)
- [Phase 06-01]: Auto-detection based on Path.is_dir() vs Path.is_file()
- [Phase 06-01]: FrameSet as Protocol (not ABC) for structural subtyping

### Pending Todos

- Update DOI badge in README once Zenodo mints DOI

### Blockers/Concerns

**Phase 5 (Documentation Site):**
- Read the Docs account and repository integration (required for live docs hosting)
- Human verification of visual appearance and navigation flow

## Session Continuity

Last session: 2026-02-15 (phase execution)
Stopped at: Completed 06-01-PLAN.md (FrameSet Protocol + ImageSet + Auto-Detection)
Resume file: None
