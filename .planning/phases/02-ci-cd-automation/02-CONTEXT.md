# Phase 2: CI/CD Automation - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Automated testing across Python versions and platforms with GitHub Actions workflows for tests, docs, and PyPI publishing. Pre-commit hooks for code quality. This phase creates the CI infrastructure; the actual PyPI release happens in Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Test matrix scope
- OS: Linux + Windows (no macOS — if Linux passes, macOS usually does too)
- Python: 3.10, 3.11, 3.12 — full 6-job matrix (3 versions x 2 OS)
- Slow tests (optimization-based, synthetic pipeline) run in a separate workflow with manual trigger only (`workflow_dispatch`)
- Fast tests (`-m "not slow"`) run in the main test workflow

### Workflow structure
- Separate workflow files: `test.yml`, `docs.yml`, `publish.yml`, `slow-tests.yml`
- **test.yml**: Triggers on push to main + PR to main. Runs fast tests across full matrix.
- **docs.yml**: Triggers on PR to main. Builds Sphinx docs to catch build errors.
- **publish.yml**: Triggers on git tag push (`v*`). Runs tests first, then publishes to PyPI via Trusted Publishing. Tests must pass before publish job runs.
- **slow-tests.yml**: Manual trigger only (`workflow_dispatch`). Runs full test suite including slow optimization tests.

### Pre-commit & linting
- Ruff replaces Black as formatter (`ruff format`) — one tool for both linting and formatting
- Ruff lint rules: start lenient with `E, F, W, I` (errors, pyflakes, warnings, import sorting). Tighten later.
- mypy: skip for now — numpy type stubs are still rough, would create friction without catching real bugs
- Pre-commit enforced in CI: a CI job runs `pre-commit run --all-files` to catch contributors who skip local hooks

### Claude's Discretion
- Coverage tooling and thresholds (user skipped this discussion area)
- Exact pre-commit hook versions and additional hooks (trailing whitespace, end-of-file, etc.)
- Docs workflow specifics (Sphinx config, build options)
- Whether to include a lint/format fix step or just check

</decisions>

<specifics>
## Specific Ideas

- Ruff recommendation: start lenient, tighten later — get to PyPI fast, don't bikeshed lint
- mypy skip rationale: heavy numpy usage makes mypy more friction than value right now
- Publish workflow safety: always run tests before publishing, even though main should be green

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-ci-cd-automation*
*Context gathered: 2026-02-14*
