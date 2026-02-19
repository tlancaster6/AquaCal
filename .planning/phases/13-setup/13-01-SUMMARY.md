---
phase: 13-setup
plan: 01
subsystem: infra
tags: [aquakit, torch, pytorch, dependency, ci, github-actions]

# Dependency graph
requires: []
provides:
  - aquakit declared as hard dependency of aquacal (>=1.0,<2)
  - Import-time torch presence check with clear actionable ImportError
  - CPU-only PyTorch installed in both CI workflows before aquacal
affects: [14-rewire, 15-test, 16-cleanup, 17-release]

# Tech tracking
tech-stack:
  added: [aquakit>=1.0,<2 (hard dependency)]
  patterns:
    - Guard pattern: _check_torch() called before all other imports in __init__.py
    - noqa: E402 suppression for intentional post-guard imports

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/aquacal/__init__.py
    - .github/workflows/test.yml
    - .github/workflows/slow-tests.yml

key-decisions:
  - "aquakit added as hard dep (>=1.0,<2) — users get it automatically via pip install aquacal"
  - "Torch check uses _check_torch() helper called before all other imports so missing torch fails fast at package boundary"
  - "CI uses CPU-only torch index URL (download.pytorch.org/whl/cpu) to minimize download size and install time"
  - "noqa: E402 used to suppress ruff lint for intentional post-guard imports — this is correct pattern"

patterns-established:
  - "Import guard pattern: define helper, call it, then imports — noqa: E402 suppresses ruff"

# Metrics
duration: 3min
completed: 2026-02-19
---

# Phase 13 Plan 01: AquaKit Dependency and Torch Guard Summary

**aquakit>=1.0,<2 declared as hard dependency with import-time PyTorch guard in __init__.py and CPU-only torch added to both CI workflows**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-19T16:21:38Z
- **Completed:** 2026-02-19T16:23:51Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `aquakit>=1.0,<2` to pyproject.toml dependencies so `pip install aquacal` pulls aquakit automatically
- Added `_check_torch()` guard in `__init__.py` that raises a clear `ImportError` with install instructions and pytorch.org link when torch is absent
- Updated both `.github/workflows/test.yml` and `.github/workflows/slow-tests.yml` to install CPU-only PyTorch before `pip install -e ".[dev]"`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add aquakit dependency and import-time torch check** - `77a5e95` (feat)
2. **Task 2: Update CI workflows to install CPU-only PyTorch** - `3bbd2d4` (ci)

**Plan metadata:** see final docs commit below

## Files Created/Modified
- `pyproject.toml` - Added `aquakit>=1.0,<2` to dependencies list
- `src/aquacal/__init__.py` - Added `_check_torch()` guard function called before all other imports; added `noqa: E402` to post-guard imports
- `.github/workflows/test.yml` - Added `pip install torch --index-url https://download.pytorch.org/whl/cpu` before editable install
- `.github/workflows/slow-tests.yml` - Added `pip install torch --index-url https://download.pytorch.org/whl/cpu` before editable install

## Decisions Made
- Used `_check_torch()` helper function (not inline try/except) for clean organization per plan spec
- Added `# noqa: E402` to all imports after the guard call to suppress ruff E402 lint errors — this is the correct pattern for intentional import ordering
- CPU-only torch index URL keeps CI fast (no CUDA downloads)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added noqa: E402 to post-guard imports to satisfy ruff pre-commit hook**
- **Found during:** Task 1 commit (pre-commit hook failure)
- **Issue:** Ruff flagged E402 (module level import not at top of file) for all imports after `_check_torch()` call
- **Fix:** Added `# noqa: E402` comment to each import line after the guard call
- **Files modified:** `src/aquacal/__init__.py`
- **Verification:** `python -m ruff check src/aquacal/__init__.py` passes with no errors; pre-commit hook passes
- **Committed in:** `77a5e95` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - lint error blocking commit)
**Impact on plan:** Necessary fix for code style compliance. No scope creep.

## Issues Encountered
- PyTorch is not installed in the local dev environment, so `import aquacal` correctly fails with the new guard. Tests cannot run locally without torch installed. This is expected behavior — CI will have torch installed via the updated workflows.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- aquakit is declared as a dependency; Phase 14 (rewire) can begin wiring call sites to aquakit functions
- The torch guard ensures any environment without torch gets a clear, actionable error immediately
- Blocker noted in STATE.md: `refractive_project` API change is non-trivial — Phase 14 needs careful call-site audit

---
*Phase: 13-setup*
*Completed: 2026-02-19*
