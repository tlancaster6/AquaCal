---
phase: 02-ci-cd-automation
plan: 01
subsystem: dev-tooling
tags: [pre-commit, ruff, coverage, dev-deps]
dependencies:
  requires: []
  provides: [pre-commit-config, ruff-config, coverage-config]
  affects: [ci-workflows]
tech-stack:
  added: [ruff, pre-commit]
  patterns: [pre-commit-hooks, code-formatting, linting]
key-files:
  created: [.pre-commit-config.yaml]
  modified: [pyproject.toml]
decisions:
  - Use ruff instead of black for formatting (faster, all-in-one tool)
  - Skip mypy for now (numpy stubs too rough per user decision)
  - Lenient lint rules initially (E4, E7, E9, F, W, I) to avoid overwhelming noise
  - Include standard quality-of-life pre-commit hooks
metrics:
  tasks: 2
  commits: 2
  duration: 80
  completed: 2026-02-14
---

# Phase 02 Plan 01: Pre-commit Configuration and Tool Setup Summary

**One-liner:** Replaced black/mypy with ruff, added pre-commit configuration with ruff and quality-of-life hooks, configured coverage settings for pytest-cov.

## Objective

Create pre-commit configuration and update pyproject.toml with ruff (replacing black), coverage settings, and updated dev dependencies. Foundation for all CI workflows — ruff config is referenced by pre-commit CI job, coverage config by test workflows, and dev deps by pip install in CI.

## What Was Delivered

### Task 1: Update pyproject.toml with ruff config, coverage config, and dev deps
**Commit:** 8d65ef7

Updated pyproject.toml with three major changes:

1. **Dev dependencies**: Removed `black`, `mypy`, and `types-PyYAML`. Added `ruff` and `pre-commit`. Kept `pytest` and `pytest-cov`.

2. **Ruff configuration**: Added `[tool.ruff]` with:
   - Line length: 88 (black-compatible)
   - Target version: py310
   - Lint rules: E4, E7, E9, F, W, I (lenient by design)
   - Per-file ignores: F401 for `__init__.py`, E501 for tests
   - Format settings: double quotes, space indents

3. **Coverage configuration**: Added `[tool.coverage.run]` and `[tool.coverage.report]` with:
   - Source: `src/aquacal`
   - Omit: test files and pycache
   - Report precision: 2 decimals, show missing lines
   - Standard exclude patterns (pragma no cover, abstract methods, etc.)

**Files modified:** pyproject.toml

### Task 2: Create .pre-commit-config.yaml with ruff and quality-of-life hooks
**Commit:** 790fa1a

Created `.pre-commit-config.yaml` with two repo configurations:

1. **Ruff hooks** (v0.15.1):
   - `ruff` with `--fix` (BEFORE ruff-format, critical ordering)
   - `ruff-format` (after ruff fixes)

2. **Pre-commit standard hooks** (v5.0.0):
   - `trailing-whitespace`: Remove trailing whitespace
   - `end-of-file-fixer`: Ensure files end with newline
   - `check-yaml`: Validate YAML syntax
   - `check-added-large-files`: Prevent files >1000KB

**Files created:** .pre-commit-config.yaml

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. **Ruff replaces black and mypy**: Per user decision, use ruff as all-in-one tool. Mypy skipped due to rough numpy stubs causing noise.

2. **Lenient lint rules**: Started with E4, E7, E9, F, W, I (errors, warnings, imports). Avoids overwhelming noise on first run. Can tighten later (add N, B, UP, etc.).

3. **Hook ordering**: ruff before ruff-format is critical — fixes may produce code needing reformatting.

4. **Skipped debug-statements hook**: Per research, overly aggressive (false positives on legitimate debug patterns).

## Impact on Subsequent Plans

- **02-02 (GitHub Actions)**: Pre-commit CI workflow will reference `.pre-commit-config.yaml`
- **02-02 (GitHub Actions)**: Test workflow will use coverage config from pyproject.toml
- **02-02 (GitHub Actions)**: CI `pip install -e .[dev]` will install ruff and pre-commit
- **02-03 (Semantic Release)**: Release workflow depends on clean lint/format passing

## Technical Notes

### Ruff Configuration Rationale

**Line length 88**: Black-compatible default. Avoids churn when migrating from black.

**Lenient rules**: E4/E7/E9 (syntax errors), F (pyflakes), W (warnings), I (import order). Focused on correctness, not style bikeshedding. Future plans can add:
- N (naming conventions)
- B (bugbear - likely bugs)
- UP (pyupgrade - modern Python)
- ANN (annotations)

**Per-file ignores**:
- `__init__.py`: F401 (unused import) — standard pattern for re-exports
- `tests/**/*.py`: E501 (line too long) — test data/asserts often exceed 88 chars

### Pre-commit Hook Choices

**Included**:
- `trailing-whitespace`, `end-of-file-fixer`: Standard hygiene
- `check-yaml`: CI workflow validation
- `check-added-large-files`: Prevent accidental dataset commits

**Skipped**:
- `debug-statements`: Too aggressive (false positives)
- `check-json`, `check-toml`: Not needed (minimal JSON/TOML usage)
- `name-tests-test`: Unnecessary (pytest discovers tests fine)

### Coverage Configuration

**Source**: `src/aquacal` — only track library code, not tests.

**Omit patterns**: `*/tests/*`, `*/test_*.py`, `*/__pycache__/*` — standard exclusions.

**Exclude lines**: Standard patterns (pragma no cover, abstract methods, repr, if TYPE_CHECKING, if __name__ == .__main__.).

**Precision 2**: 99.99% vs 100.0% visibility matters for tracking small gaps.

## Verification Results

All verification criteria met:

- [x] pyproject.toml contains `[tool.ruff]` section with lenient rules (E4, E7, E9, F, W, I)
- [x] pyproject.toml contains `[tool.coverage.run]` and `[tool.coverage.report]` sections
- [x] Dev deps include ruff, pre-commit, pytest, pytest-cov
- [x] Dev deps do NOT include black or mypy
- [x] .pre-commit-config.yaml exists with 6 hooks in correct order
- [x] ruff hook comes before ruff-format hook

## Self-Check: PASSED

Verified all claimed artifacts exist and commits are valid:

- FOUND: pyproject.toml
- FOUND: .pre-commit-config.yaml
- FOUND: 8d65ef7 (Task 1 commit)
- FOUND: 790fa1a (Task 2 commit)
