# Phase 2: CI/CD Automation - Research

**Researched:** 2026-02-14
**Domain:** GitHub Actions CI/CD, Python package testing, PyPI publishing, pre-commit automation
**Confidence:** HIGH

## Summary

This phase implements automated testing and publishing infrastructure for a scientific Python package using GitHub Actions, pytest, Ruff, and PyPI Trusted Publishing. The key technical domains are well-established with mature tooling and official documentation.

The project requires testing across Python 3.10-3.12 on Linux and Windows, with special handling for slow optimization-based tests. Ruff has replaced Black/isort as the standard Python formatter/linter. PyPI Trusted Publishing via OIDC eliminates API token management. Codecov integration provides coverage reporting.

The primary technical challenges are: (1) Windows path handling in pytest, (2) separating fast unit tests from slow synthetic/optimization tests to keep PR feedback fast, (3) proper workflow job orchestration to ensure tests pass before publishing, and (4) scoping GitHub Actions permissions following principle of least privilege.

**Primary recommendation:** Use official GitHub Actions (setup-python@v5, pypa/gh-action-pypi-publish@release/v1, codecov/codecov-action@v5) with matrix strategy for testing, separate workflows for different concerns (test/docs/publish/slow-tests), and Ruff v0.15.1 for both linting and formatting.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Test matrix scope:**
- OS: Linux + Windows (no macOS — if Linux passes, macOS usually does too)
- Python: 3.10, 3.11, 3.12 — full 6-job matrix (3 versions x 2 OS)
- Slow tests (optimization-based, synthetic pipeline) run in a separate workflow with manual trigger only (`workflow_dispatch`)
- Fast tests (`-m "not slow"`) run in the main test workflow

**Workflow structure:**
- Separate workflow files: `test.yml`, `docs.yml`, `publish.yml`, `slow-tests.yml`
- **test.yml**: Triggers on push to main + PR to main. Runs fast tests across full matrix.
- **docs.yml**: Triggers on PR to main. Builds Sphinx docs to catch build errors.
- **publish.yml**: Triggers on git tag push (`v*`). Runs tests first, then publishes to PyPI via Trusted Publishing. Tests must pass before publish job runs.
- **slow-tests.yml**: Manual trigger only (`workflow_dispatch`). Runs full test suite including slow optimization tests.

**Pre-commit & linting:**
- Ruff replaces Black as formatter (`ruff format`) — one tool for both linting and formatting
- Ruff lint rules: start lenient with `E, F, W, I` (errors, pyflakes, warnings, import sorting). Tighten later.
- mypy: skip for now — numpy type stubs are still rough, would create friction without catching real bugs
- Pre-commit enforced in CI: a CI job runs `pre-commit run --all-files` to catch contributors who skip local hooks

### Claude's Discretion

- Coverage tooling and thresholds (user skipped this discussion area)
- Exact pre-commit hook versions and additional hooks (trailing whitespace, end-of-file, etc.)
- Docs workflow specifics (Sphinx config, build options)
- Whether to include a lint/format fix step or just check

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Standard Stack

### Core CI/CD Tools

| Library/Action | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| actions/setup-python | v5 | Python environment setup in CI | Official GitHub action, includes built-in pip caching |
| actions/checkout | v4 | Repository checkout | Required first step for all workflows |
| pypa/gh-action-pypi-publish | release/v1 | PyPI publishing via Trusted Publishing | Official PyPA action, OIDC-based auth, auto-generates attestations |
| codecov/codecov-action | v5 | Coverage upload to Codecov | Latest version (v5.5.2 as of Dec 2025), uses Codecov CLI wrapper |
| pytest | (current) | Test runner | Industry standard for Python testing |
| pytest-cov | (current) | Coverage reporting | Standard pytest plugin for coverage, generates XML for Codecov |

### Linting & Formatting

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ruff | 0.15.1 | Linter and formatter | Replaces Black/isort/Flake8 — 200x faster, single tool for both lint and format |
| ruff-pre-commit | v0.15.1 | Pre-commit integration | Official pre-commit hook for Ruff |

### Supporting Tools

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pre-commit | (current) | Git hook management | Enforce code quality before commit |
| sphinx | (existing) | Documentation building | Validate docs build in CI before merge |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Ruff | Black + isort + Flake8 | Ruff is faster and unified; Black+isort is more mature but requires 3 tools |
| Trusted Publishing | API tokens | API tokens work but require secret management; Trusted Publishing is more secure |
| Codecov | Coveralls, Codacy | All similar; Codecov is most popular for open-source Python |
| pytest-timeout | pytest-fail-slow | pytest-timeout kills hung tests (critical for CI); fail-slow just marks them |

**Installation:**

```bash
# Production dependencies (already in pyproject.toml)
# None — CI tools are actions, not package dependencies

# Dev dependencies to add to pyproject.toml
ruff  # Replaces black in dev dependencies

# Pre-commit config (separate .pre-commit-config.yaml file)
# Installed via: pip install pre-commit
```

---

## Architecture Patterns

### Recommended CI/CD Structure

```
.github/
├── workflows/
│   ├── test.yml           # Fast tests on push/PR (6-job matrix)
│   ├── docs.yml           # Doc build validation on PR
│   ├── publish.yml        # PyPI publish on tag (with test gate)
│   └── slow-tests.yml     # Full tests, manual trigger only
.pre-commit-config.yaml    # Pre-commit hooks (ruff)
codecov.yml                # Codecov configuration (optional)
pyproject.toml             # Updated: ruff replaces black in dev deps
```

### Pattern 1: Test Matrix with Dependency Caching

**What:** Run tests across multiple Python versions and OS platforms using GitHub Actions matrix strategy with built-in dependency caching.

**When to use:** All workflows that run tests (test.yml, slow-tests.yml, publish.yml).

**Example:**

```yaml
# Source: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false  # Show all failures, don't cancel on first fail
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]  # MUST quote: 3.10 as unquoted = 3.1

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'  # Built-in caching using requirements files

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run fast tests
        run: pytest tests/ -m "not slow" --cov=aquacal --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: true
```

**Critical details:**
- **Quote Python versions**: `"3.10"` not `3.10` (YAML parses unquoted as float 3.1)
- **fail-fast: false**: Run all matrix jobs even if one fails (see all platform issues)
- **cache: 'pip'**: setup-python automatically caches based on requirements.txt/pyproject.toml
- **Marker for fast tests**: `-m "not slow"` excludes optimization tests marked with `@pytest.mark.slow`

### Pattern 2: PyPI Trusted Publishing with Test Gate

**What:** Publish to PyPI on git tag using OIDC authentication (no API tokens), with mandatory test pass before publish.

**When to use:** publish.yml workflow only, triggered by version tags.

**Example:**

```yaml
# Source: https://docs.pypi.org/trusted-publishers/using-a-publisher/ and
# https://github.com/pypa/gh-action-pypi-publish
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'  # Trigger on version tags like v0.1.0

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -m "not slow"  # Fast tests only for release gate

  build:
    needs: test  # Wait for tests to pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Build package
        run: |
          python -m pip install --upgrade pip build
          python -m build
      - name: Upload dist artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build  # Wait for build to complete
    runs-on: ubuntu-latest
    environment: pypi  # Optional: GitHub environment for additional protection
    permissions:
      id-token: write  # MANDATORY for Trusted Publishing
    steps:
      - name: Download dist artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # No username/password/token needed — OIDC handles auth
```

**Critical details:**
- **Job dependencies**: `needs: test` ensures tests pass before build; `needs: build` ensures build succeeds before publish
- **permissions: id-token: write**: MANDATORY at job level (narrowest scope)
- **environment: pypi**: Optional GitHub environment for manual approval gates
- **No secrets needed**: OIDC authentication via GitHub's identity provider
- **Artifact passing**: Build job uploads dist/, publish job downloads it
- **PyPI config required**: Must configure Trusted Publisher on PyPI project settings BEFORE first publish

**PyPI Trusted Publisher Configuration:**
1. Go to PyPI project settings (or create "pending" publisher if project doesn't exist yet)
2. Add GitHub publisher with: repository owner, repository name, workflow filename (`publish.yml`), optional environment name (`pypi`)
3. First publish will transition pending publisher to active

### Pattern 3: Manual-Trigger Slow Test Workflow

**What:** Separate workflow for expensive/slow tests (optimization, synthetic pipeline) with manual trigger only.

**When to use:** When test suite has slow tests that would make PR feedback too slow if run on every commit.

**Example:**

```yaml
# Source: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
name: Slow Tests

on:
  workflow_dispatch:  # Manual trigger only — no push/PR triggers
    inputs:
      python-version:
        description: 'Python version to test'
        required: false
        default: '3.12'
        type: choice
        options:
          - '3.10'
          - '3.11'
          - '3.12'
      os:
        description: 'Operating system'
        required: false
        default: 'ubuntu-latest'
        type: choice
        options:
          - 'ubuntu-latest'
          - 'windows-latest'

jobs:
  slow-tests:
    runs-on: ${{ inputs.os || 'ubuntu-latest' }}
    timeout-minutes: 60  # Prevent runaway optimization tests

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version || '3.12' }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run full test suite (including slow tests)
        run: pytest tests/ --cov=aquacal --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

**Critical details:**
- **workflow_dispatch only**: No automatic triggers
- **inputs with type: choice**: UI dropdown for Python version and OS selection
- **timeout-minutes: 60**: Kill job if tests hang (default is 360 minutes = 6 hours)
- **Default values**: `|| 'ubuntu-latest'` provides fallback if input not specified

### Pattern 4: Pre-commit with Ruff (Linter + Formatter)

**What:** Use Ruff for both linting and formatting via pre-commit hooks.

**When to use:** All Python projects replacing Black/isort/Flake8 with single fast tool.

**Example (.pre-commit-config.yaml):**

```yaml
# Source: https://github.com/astral-sh/ruff-pre-commit
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.1
    hooks:
      # Run linter with auto-fix
      - id: ruff
        args: [--fix]
      # Run formatter (MUST come after ruff with --fix)
      - id: ruff-format

  # Standard quality-of-life hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**Ruff configuration (pyproject.toml):**

```toml
[tool.ruff]
line-length = 88  # Match Black
target-version = "py310"  # Minimum supported version

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "W", "I"]  # User decision: E, F, W, I
ignore = []
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

**Critical details:**
- **Hook order**: `ruff` (with --fix) MUST come before `ruff-format` — fixes may produce code that needs reformatting
- **Don't use isort**: Ruff includes isort functionality via `I` rules (import sorting)
- **Lenient rule selection**: User decision to start with `E, F, W, I` and tighten later
- **fixable = ["ALL"]**: Allow Ruff to auto-fix all fixable violations

### Pattern 5: Pre-commit Enforcement in CI

**What:** CI job that runs pre-commit hooks to catch contributors who skip local hooks.

**When to use:** Prevent formatting/linting issues from reaching main branch.

**Example (add to test.yml):**

```yaml
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Run pre-commit
        run: |
          pip install pre-commit
          pre-commit run --all-files
```

**Critical details:**
- **--all-files**: Check entire codebase, not just changed files
- **Separate job**: Runs in parallel with tests, fails fast on formatting issues
- **No fix attempt**: Just check — contributors must fix locally and re-push

### Pattern 6: Documentation Build Validation

**What:** Build Sphinx documentation in CI to catch build errors before merge.

**When to use:** All projects with Sphinx documentation.

**Example (docs.yml):**

```yaml
# Source: https://github.com/marketplace/actions/sphinx-build
name: Docs

on:
  pull_request:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install sphinx sphinx-rtd-theme  # Or whatever theme is used

      - name: Build documentation
        run: |
          cd docs
          make html

      - name: Check for warnings
        run: |
          cd docs
          make html SPHINXOPTS="-W --keep-going"
        # -W treats warnings as errors
        # --keep-going shows all warnings before failing
```

**Critical details:**
- **PR trigger only**: No need to build docs on every push to main
- **-W flag**: Treat Sphinx warnings as errors (catches broken refs, missing docstrings)
- **--keep-going**: Show all warnings in one run instead of failing on first

### Pattern 7: Coverage Reporting with pytest-cov

**What:** Generate XML coverage reports and upload to Codecov.

**When to use:** All test workflows.

**Example:**

```bash
# Generate coverage report
pytest tests/ --cov=aquacal --cov-report=xml --cov-report=term

# Upload to Codecov (in workflow)
- uses: codecov/codecov-action@v5
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    fail_ci_if_error: true
```

**Coverage configuration (pyproject.toml):**

```toml
[tool.coverage.run]
source = ["src/aquacal"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

**Critical details:**
- **--cov-report=xml**: Required for Codecov
- **--cov-report=term**: Optional, shows coverage in CI logs
- **fail_ci_if_error: true**: Fail build if coverage upload fails
- **Token requirement**: Public repos can use global upload token; private repos need CODECOV_TOKEN secret

### Anti-Patterns to Avoid

#### Anti-Pattern: Unquoted Python Versions in Matrix

**Bad:**
```yaml
matrix:
  python-version: [3.10, 3.11, 3.12]  # 3.10 parsed as 3.1
```

**Good:**
```yaml
matrix:
  python-version: ["3.10", "3.11", "3.12"]  # Quoted strings
```

**Why:** YAML parses `3.10` as float `3.1`, causing setup-python to fail.

#### Anti-Pattern: Secrets in Trusted Publishing

**Bad:**
```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    username: __token__
    password: ${{ secrets.PYPI_TOKEN }}
```

**Good:**
```yaml
permissions:
  id-token: write

- uses: pypa/gh-action-pypi-publish@release/v1
  # No username/password needed
```

**Why:** Trusted Publishing is more secure and eliminates secret management.

#### Anti-Pattern: Global id-token Permissions

**Bad:**
```yaml
permissions:
  id-token: write  # Top-level

jobs:
  test: ...
  publish: ...
```

**Good:**
```yaml
jobs:
  test: ...

  publish:
    permissions:
      id-token: write  # Job-level only
```

**Why:** Principle of least privilege — only publishing job needs OIDC token.

#### Anti-Pattern: Ruff-format Before Ruff with --fix

**Bad:**
```yaml
hooks:
  - id: ruff-format
  - id: ruff
    args: [--fix]
```

**Good:**
```yaml
hooks:
  - id: ruff
    args: [--fix]
  - id: ruff-format
```

**Why:** Ruff fixes may produce code that needs reformatting, causing endless hook loops.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PyPI authentication | Custom token rotation, secret management | PyPI Trusted Publishing via OIDC | OIDC is stateless, no secrets to rotate, officially supported |
| Python multi-version testing | Custom Docker containers for each version | actions/setup-python with matrix | Official action, built-in caching, supports all Python versions |
| Coverage upload | Custom curl/API calls to Codecov | codecov/codecov-action | Official action, handles retries, validates uploads, generates reports |
| Dependency caching | Manual actions/cache configuration | setup-python's built-in cache: 'pip' | Auto-detects requirements files, correct cache keys, faster |
| Lint/format orchestration | Shell scripts running Black, isort, Flake8 | Ruff (single tool) | 200x faster, single config, handles import sorting and formatting |
| Sphinx build checking | grep for "warning" in output | Sphinx -W flag | Native Sphinx feature, treats warnings as errors, more reliable |
| Test timeouts | Custom timeout wrappers | pytest-timeout plugin | Battle-tested, handles thread/subprocess cleanup, configurable per-test |

**Key insight:** GitHub Actions has mature official actions for Python CI/CD patterns. Custom solutions miss edge cases (pip cache invalidation, OIDC token rotation, multi-platform path handling). Ruff's speed comes from Rust — Python equivalents can't match performance.

---

## Common Pitfalls

### Pitfall 1: Windows Path Handling in pytest

**What goes wrong:** pytest on Windows can fail with path-related errors:
- Paths exceeding 260 characters cause FileNotFoundError during collection
- Backslash vs forward slash inconsistencies (`C:\src` vs `C:/src` treated as different files)
- Short paths like `C:\Users\RUNNER~1\...` fail test collection in pytest 8.0+

**Why it happens:** Windows has different path conventions than Unix, and pytest's path normalization doesn't always handle edge cases. GitHub Actions uses short paths in temp directories.

**How to avoid:**
- Use `pytest --import-mode=append` to reduce path sensitivity
- Keep test paths short (avoid deep nesting)
- Use forward slashes in pytest configuration even on Windows
- Test Windows locally or in CI before merging

**Warning signs:**
- `FileNotFoundError` during test collection on Windows only
- `ImportPathMismatchError` on Windows only
- Zero tests collected on Windows when Linux finds tests

**Sources:** [pytest #8999](https://github.com/pytest-dev/pytest/issues/8999), [pytest #4469](https://github.com/pytest-dev/pytest/issues/4469), [pytest #11895](https://github.com/pytest-dev/pytest/issues/11895)

### Pitfall 2: Trusted Publishing Configuration Mismatch

**What goes wrong:** PyPI publish fails with error "OIDC token doesn't match any known publisher" even though Trusted Publisher is configured.

**Why it happens:**
- Repository owner/name mismatch between PyPI config and actual repo
- Workflow filename mismatch (e.g., configured `publish.yml` but workflow is `release.yml`)
- Environment name mismatch (e.g., configured `pypi` environment but workflow doesn't use `environment: pypi`)
- Repository was renamed but PyPI config still uses old name

**How to avoid:**
- **Exact name matching**: Repository owner, repository name, workflow filename, and environment name must EXACTLY match on both GitHub and PyPI
- **Double-check after repo rename**: Update PyPI Trusted Publisher if repo is renamed
- **Use environment**: Specify `environment: pypi` in workflow AND in PyPI Trusted Publisher config (makes matching easier)
- **Pending publisher timing**: If creating pending publisher, register it before tagging release

**Warning signs:**
- Publish workflow reaches publish step but fails with OIDC error
- Error message mentions "no matching publisher" or "malformed token"
- Workflow runs but PyPI shows no publish attempt

**Sources:** [PyPI Troubleshooting](https://docs.pypi.org/trusted-publishers/troubleshooting/), [Trusted Publisher Pitfalls](https://dreamnetworking.nl/blog/2025/01/07/pypi-trusted-publisher-management-and-pitfalls/)

### Pitfall 3: Matrix fail-fast Hiding Issues

**What goes wrong:** First matrix job failure cancels remaining jobs, hiding platform-specific or version-specific issues.

**Why it happens:** GitHub Actions defaults `fail-fast: true` in matrix strategy — any job failure cancels all in-progress and queued jobs.

**How to avoid:**
- Set `fail-fast: false` in test matrix strategy
- See all platform/version failures in one CI run instead of fixing one at a time

**Warning signs:**
- CI shows "Job was cancelled" for some matrix jobs
- Issue appears on one platform, fixed, then appears on another platform

**Example:**
```yaml
strategy:
  fail-fast: false  # Show all failures
  matrix:
    os: [ubuntu-latest, windows-latest]
    python-version: ["3.10", "3.11", "3.12"]
```

**Sources:** [GitHub Actions Matrix Strategy](https://codefresh.io/learn/github-actions/github-actions-matrix/), [Matrix Failures Discussion](https://github.com/orgs/community/discussions/176052)

### Pitfall 4: Slow Tests Blocking PR Feedback

**What goes wrong:** Optimization-based tests or synthetic pipeline tests take 5-10+ minutes, making PR feedback too slow for productive development.

**Why it happens:** Scientific computing tests (bundle adjustment, large synthetic datasets) are inherently expensive. Running full suite on every commit creates bottleneck.

**How to avoid:**
- Mark slow tests with `@pytest.mark.slow` decorator
- Run fast tests only in main CI: `pytest -m "not slow"`
- Create separate manual-trigger workflow for slow tests
- Consider timeout for slow tests: `pytest --timeout=60` to catch hangs

**Warning signs:**
- PR feedback takes >5 minutes
- Developers skip running tests locally because they're too slow
- CI timeout issues

**Example marking slow tests:**
```python
import pytest

@pytest.mark.slow
def test_full_pipeline_optimization():
    # Expensive bundle adjustment test
    ...
```

**Sources:** [pytest-timeout docs](https://github.com/pytest-dev/pytest-timeout), [pytest-timeout best practices](https://lincc-ppt.readthedocs.io/en/v2.0.8/practices/pytest_timeout.html)

### Pitfall 5: Setup-python Cache Misses

**What goes wrong:** Dependencies reinstall on every CI run even with `cache: 'pip'` enabled.

**Why it happens:**
- Cache key based on requirements file hash — if no requirements.txt exists, caching doesn't work
- pyproject.toml changes don't invalidate cache (setup-python only watches requirements.txt by default)
- Cache miss on Windows if dependency installation uses different paths than Linux

**How to avoid:**
- If using pyproject.toml only, specify cache-dependency-path:
  ```yaml
  - uses: actions/setup-python@v5
    with:
      python-version: '3.12'
      cache: 'pip'
      cache-dependency-path: 'pyproject.toml'
  ```
- Or generate requirements.txt: `pip freeze > requirements.txt` and commit it

**Warning signs:**
- CI logs show "Cache not found" or "Post-cache" with no hit
- Dependencies reinstall every run

**Sources:** [setup-python caching docs](https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md), [Caching dependencies](https://oneuptime.com/blog/post/2025-12-20-github-actions-cache-dependencies/view)

### Pitfall 6: id-token Permission Too Broad

**What goes wrong:** Workflow grants `id-token: write` at top level, giving OIDC token access to all jobs including test/build jobs that don't need it.

**Why it happens:** Documentation examples sometimes show top-level permissions for simplicity, but this violates principle of least privilege.

**How to avoid:**
- **Only grant to publish job**:
  ```yaml
  jobs:
    test: ...  # No special permissions
    build: ...  # No special permissions
    publish:
      permissions:
        id-token: write  # Only publish needs this
  ```

**Warning signs:**
- Security audit flags excessive permissions
- All jobs have OIDC token access in logs

**Sources:** [GitHub Actions permissions](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/controlling-permissions-for-github_token), [Security best practices](https://blog.gitguardian.com/github-actions-security-cheat-sheet/)

### Pitfall 7: Codecov Token for Public Repos

**What goes wrong:** Setting up Codecov token in secrets for public repository when it's not needed.

**Why it happens:** Codecov documentation mentions tokens but doesn't clearly distinguish public vs private repo requirements.

**How to avoid:**
- **Public repos**: No token needed if global upload token is enabled in Codecov org settings
- **Private repos**: CODECOV_TOKEN secret is required
- **Forks**: Secrets not available to forks; public repo workflows on forks will fail if they require token

**Warning signs:**
- Codecov upload fails on fork PRs with "missing token" error
- Adding token secret doesn't solve fork issue

**Sources:** [codecov-action v5 docs](https://github.com/codecov/codecov-action)

---

## Code Examples

Verified patterns from official sources.

### Example 1: Complete test.yml Workflow

```yaml
# Source: Synthesized from https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
# and https://github.com/codecov/codecov-action
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
          cache-dependency-path: 'pyproject.toml'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -m "not slow" --cov=aquacal --cov-report=xml --cov-report=term

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: true

  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install pre-commit
        run: pip install pre-commit
      - name: Run pre-commit
        run: pre-commit run --all-files
```

### Example 2: Complete publish.yml Workflow

```yaml
# Source: https://docs.pypi.org/trusted-publishers/using-a-publisher/
# and https://github.com/pypa/gh-action-pypi-publish
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -m "not slow"

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install build
        run: python -m pip install --upgrade pip build
      - name: Build package
        run: python -m build
      - name: Upload dist
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - name: Download dist
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### Example 3: Complete .pre-commit-config.yaml

```yaml
# Source: https://github.com/astral-sh/ruff-pre-commit
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.1
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

### Example 4: Ruff Configuration in pyproject.toml

```toml
# Source: https://docs.astral.sh/ruff/configuration/
[tool.ruff]
line-length = 88
target-version = "py310"
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "W", "I"]  # User decision: lenient start
ignore = []
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports
"tests/**/*.py" = ["E501"]  # Allow long lines in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Example 5: Coverage Configuration

```toml
# Source: https://pytest-cov.readthedocs.io/en/latest/config.html
[tool.coverage.run]
source = ["src/aquacal"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Black + isort + Flake8 | Ruff (single tool) | 2023-2024 | 200x faster, single config, one tool to install |
| PyPI API tokens | Trusted Publishing (OIDC) | 2023 | No secrets to rotate, more secure, simpler setup |
| actions/setup-python v4 | actions/setup-python v5 | 2024 | Improved caching, better error messages |
| codecov-action v3 | codecov-action v5 | 2025 | Uses Codecov CLI wrapper, faster updates, better error handling |
| Separate lint CI job | Pre-commit enforcement in CI | 2024+ | Catches formatting issues earlier, faster feedback |
| mypy everywhere | Skip mypy for numpy-heavy code | 2024+ | numpy type stubs still immature, more friction than value |

**Deprecated/outdated:**
- **Black formatter alone**: Ruff includes formatting, no need for separate Black
- **isort**: Ruff's `I` rules handle import sorting
- **Flake8**: Ruff replaces with faster rust-based linting
- **PyPI tokens in GitHub secrets**: Trusted Publishing eliminates this
- **Unquoted Python versions in matrix**: YAML parsing breaks; always quote
- **gh-action-pypi-publish@master**: Branch sunset, use `@release/v1` instead

---

## Open Questions

### Question 1: Codecov Coverage Threshold

**What we know:** Codecov can enforce minimum coverage thresholds and fail PRs below threshold.

**What's unclear:** User skipped discussion of coverage thresholds. Should we:
- Set minimum coverage threshold (e.g., 80%)?
- Require coverage increase on PRs (no decrease allowed)?
- Just report coverage without enforcement?

**Recommendation:** Start with reporting only (no enforcement). Add thresholds later after establishing baseline coverage. This avoids blocking Phase 3 release on coverage goals.

### Question 2: Sphinx Build Options

**What we know:** Need to validate Sphinx docs build in CI on PR.

**What's unclear:** User skipped Sphinx workflow specifics:
- Should we fail on warnings (-W flag)?
- Should we check for broken links?
- Should we upload built docs as artifact for preview?

**Recommendation:** Use `-W` flag to treat warnings as errors (catches broken refs). Don't upload artifacts yet (adds complexity). Broken link checking can be added later if needed.

### Question 3: Pre-commit Additional Hooks

**What we know:** Need ruff and ruff-format hooks. User skipped discussion of additional hooks.

**What's unclear:** Which quality-of-life hooks to include:
- trailing-whitespace
- end-of-file-fixer
- check-yaml
- check-added-large-files
- debug-statements

**Recommendation:** Include standard set (trailing whitespace, EOF, YAML, large files) — low friction, high value. Skip debug-statements (can be overly aggressive with print debugging).

### Question 4: Windows-Only Test Failures

**What we know:** Windows path handling is a known pytest pitfall.

**What's unclear:** If Windows tests fail while Linux passes, should we:
- Block merge until Windows fixed?
- Allow merge with known Windows failure?
- Skip Windows for some test scenarios?

**Recommendation:** Block merge (fail-fast: false shows issue, but job must pass). Windows is supported Python platform per user decisions — failures indicate real bugs.

---

## Sources

### Primary Sources (HIGH confidence)

**GitHub Actions:**
- [Building and testing Python - GitHub Docs](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Workflow syntax - GitHub Docs](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Using jobs in workflows - GitHub Docs](https://docs.github.com/en/actions/using-jobs/using-jobs-in-a-workflow)
- [actions/setup-python advanced usage](https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md)

**PyPI Trusted Publishing:**
- [Publishing with Trusted Publisher - PyPI Docs](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [PyPI Trusted Publishing overview](https://docs.pypi.org/trusted-publishers/)
- [Configuring OIDC in PyPI - GitHub Docs](https://docs.github.com/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-pypi)
- [pypa/gh-action-pypi-publish GitHub](https://github.com/pypa/gh-action-pypi-publish)

**Ruff:**
- [Configuring Ruff - Official Docs](https://docs.astral.sh/ruff/configuration/)
- [Ruff Formatter - Official Docs](https://docs.astral.sh/ruff/formatter/)
- [Ruff pre-commit - Official Repo](https://github.com/astral-sh/ruff-pre-commit)
- [Integrations - Ruff Docs](https://docs.astral.sh/ruff/integrations/)

**Codecov:**
- [codecov-action v5 GitHub](https://github.com/codecov/codecov-action)
- [Python coverage - Codecov Docs](https://docs.codecov.com/docs/code-coverage-with-python)

**pytest-cov:**
- [pytest-cov reporting docs](https://pytest-cov.readthedocs.io/en/latest/reporting.html)
- [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/latest/config.html)

### Secondary Sources (MEDIUM confidence)

**Best Practices & Tutorials:**
- [GitHub Actions Matrix Strategy (Codefresh)](https://codefresh.io/learn/github-actions/github-actions-matrix/)
- [Python Coverage with GitHub Actions (Codecov blog)](https://about.codecov.io/blog/python-code-coverage-using-github-actions-and-codecov/)
- [Automate Python Formatting with Ruff (Medium)](https://medium.com/@kutayeroglu/automate-python-formatting-with-ruff-and-pre-commit-b6cd904b727e)
- [Manual Triggers in GitHub Actions (Medium)](https://poojabolla.medium.com/manual-triggers-in-github-actions-a-guide-to-workflow-dispatch-with-input-parameters-e127a0d39b11)

**Troubleshooting:**
- [PyPI Trusted Publishing Troubleshooting](https://docs.pypi.org/trusted-publishers/troubleshooting/)
- [Trusted Publisher Pitfalls (Dream Networking)](https://dreamnetworking.nl/blog/2025/01/07/pypi-trusted-publisher-management-and-pitfalls/)

### Tertiary Sources (LOW confidence — issue reports)

**Known Issues:**
- [pytest Windows path issues #8999](https://github.com/pytest-dev/pytest/issues/8999)
- [pytest Windows paths #4469](https://github.com/pytest-dev/pytest/issues/4469)
- [pytest short paths #11895](https://github.com/pytest-dev/pytest/issues/11895)
- [Ruff formatter conflicts with isort #12733](https://github.com/astral-sh/ruff/issues/12733)

---

## Metadata

**Confidence breakdown:**
- **Standard stack: HIGH** — Official actions (setup-python, pypi-publish, codecov-action) with clear versioning and docs. Ruff is mature and well-documented.
- **Architecture: HIGH** — GitHub Actions workflow patterns are well-established. Matrix strategy, job dependencies, and Trusted Publishing have official documentation and examples.
- **Pitfalls: MEDIUM-HIGH** — Common issues (Windows paths, OIDC config mismatch, fail-fast) verified through official issue trackers and troubleshooting docs. Some are issue reports (LOW confidence) but cross-referenced with multiple sources.
- **Code examples: HIGH** — All examples sourced from official documentation or verified against official docs.

**Research date:** 2026-02-14
**Valid until:** ~2026-03-14 (30 days) — GitHub Actions ecosystem is stable; Ruff is fast-moving but v0.15.1 is recent (current as of Feb 2026)

**Tool versions verified:**
- actions/setup-python: v5 (current)
- pypa/gh-action-pypi-publish: release/v1 (current, master sunset)
- codecov/codecov-action: v5 (v5.5.2 latest as of Dec 2025)
- ruff-pre-commit: v0.15.1 (current as of Feb 2026)
- pre-commit-hooks: v5.0.0 (current)
