---
phase: 13-setup
verified: 2026-02-19T16:26:28Z
status: passed
score: 3/3 must-haves verified
re_verification: false
---

# Phase 13: Setup Verification Report

**Phase Goal:** AquaCal can declare and load AquaKit as a dependency with graceful PyTorch handling
**Verified:** 2026-02-19T16:26:28Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pip install aquacal` also installs `aquakit` as a transitive dependency | VERIFIED | `pyproject.toml` line 40: `"aquakit>=1.0,<2"` in `[project] dependencies` |
| 2 | Importing `aquacal` when PyTorch is not installed raises a clear, actionable error message | VERIFIED | `src/aquacal/__init__.py`: `_check_torch()` called at line 15 before all other imports; raises `ImportError` with exact message: "AquaCal requires PyTorch. Install it first: `pip install torch`. See https://pytorch.org/get-started/ for GPU variants." |
| 3 | Sphinx docs and README document the PyTorch prerequisite install step | VERIFIED | `README.md` Installation section starts with `pip install torch` + pytorch.org/get-started link; `docs/index.md` has MyST `:::{important}` admonition above Quick Start with `pip install torch` + pytorch.org link |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | aquakit dependency declaration | VERIFIED | Contains `"aquakit>=1.0,<2"` in `dependencies` list at line 40 |
| `src/aquacal/__init__.py` | Import-time torch check | VERIFIED | `_check_torch()` helper defined at line 4, called at line 15, before all other imports; raises `ImportError` with actionable message |
| `.github/workflows/test.yml` | CI torch install step | VERIFIED | Line 31: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -e ".[dev]"` |
| `.github/workflows/slow-tests.yml` | Slow-tests torch install step | VERIFIED | Line 42: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -e ".[dev]"` |
| `README.md` | Updated install instructions with torch prerequisite | VERIFIED | Installation section starts with PyTorch as first requirement; pytorch.org/get-started linked in three places |
| `docs/index.md` | Sphinx landing page with torch prerequisite | VERIFIED | MyST `:::{important}` admonition above Quick Start code block; `pip install torch` is first install command |
| `CHANGELOG.md` | v1.5 changelog entry for aquakit dependency | VERIFIED | `## Unreleased` section at top with two entries: aquakit dependency and import-time PyTorch check |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/aquacal/__init__.py` | `torch` | import check before aquakit imports | WIRED | `import torch` inside `_check_torch()` try/except; called at line 15 before any other package imports |
| `README.md` | https://pytorch.org/get-started/ | hyperlink in install section | WIRED | Three occurrences in Installation and Quick Start sections |

### Requirements Coverage

No REQUIREMENTS.md entries mapped to phase 13 found.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder patterns in any modified files. No stub implementations. All handlers and checks perform substantive work.

Notable code quality detail: `# noqa: E402` comments on all post-guard imports in `__init__.py` — this is intentional and correct per ruff rules for import-ordering guards.

### Human Verification Required

None required for automated functionality. One optional manual test could confirm the error message UX:

**Test:** Install aquacal in an environment without torch, then run `python -c "import aquacal"`.
**Expected:** `ImportError: AquaCal requires PyTorch. Install it first: \`pip install torch\`. See https://pytorch.org/get-started/ for GPU variants.`
**Why human:** Cannot run locally without uninstalling torch from the dev environment.

This is optional — the code path is simple and directly verifiable from source.

### Gaps Summary

No gaps. All three phase success criteria are fully implemented and wired:

- AquaKit is declared as a hard dependency in `pyproject.toml` — pip will install it automatically.
- The import guard in `__init__.py` checks for torch before any other import executes, ensuring users see a clear error immediately on `import aquacal` rather than a cryptic deep stack trace.
- Both CI workflows install CPU-only torch before the editable install so the guard does not break CI.
- README and Sphinx docs both surface the torch prerequisite prominently as the first step in install sections.
- CHANGELOG records the change for the upcoming v1.5 release.

---

_Verified: 2026-02-19T16:26:28Z_
_Verifier: Claude (gsd-verifier)_
