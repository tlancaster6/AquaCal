---
phase: 05-documentation-site
plan: 01
subsystem: documentation
tags: [sphinx, docs, infrastructure]
dependency_graph:
  requires: [phase-04-complete]
  provides: [sphinx-build, docs-skeleton, rtd-config]
  affects: [documentation-workflow]
tech_stack:
  added: [sphinx, furo, myst-parser, sphinx-copybutton, sphinx-design]
  patterns: [sphinx-autodoc, myst-markdown, rtd-deployment]
key_files:
  created:
    - docs/conf.py
    - docs/index.md
    - docs/overview.md
    - docs/guide/index.md
    - docs/api/index.md
    - docs/tutorials/index.md
    - docs/contributing.md
    - docs/changelog.md
    - docs/guide/_diagrams/generate_all.py
    - .readthedocs.yaml
  modified:
    - pyproject.toml
  deleted:
    - docs/index.rst
decisions:
  - title: "Removed OpenCV from intersphinx"
    rationale: "OpenCV docs don't provide valid intersphinx inventory"
    impact: "No cross-links to cv2 API, but builds cleanly"
metrics:
  duration_seconds: 323
  tasks_completed: 2
  files_created: 10
  files_modified: 2
  files_deleted: 1
  completed_at: "2026-02-15T03:42:58Z"
---

# Phase 05 Plan 01: Sphinx Infrastructure Summary

**One-liner:** Sphinx documentation site with Furo theme, MyST Markdown, rich landing page, site skeleton, and Read the Docs deployment config

## What Was Built

Set up complete Sphinx documentation infrastructure as the foundation for all documentation content. The site skeleton includes a rich landing page with grid cards, conceptual overview explaining refractive calibration, placeholder sections for theory pages (Plan 02) and API reference (Plan 03), and Read the Docs deployment configuration.

### Task 1: Sphinx Configuration, Dependencies, and RTD Config
- Added `docs` optional dependency group to `pyproject.toml` with sphinx, furo, myst-parser, sphinx-copybutton, sphinx-design
- Rewrote `docs/conf.py` with complete Sphinx configuration
- Extensions: autodoc, napoleon, viewcode, intersphinx, mathjax, myst_parser, copybutton, design
- Configured Furo theme with navigation settings
- Added intersphinx mapping for python, numpy, scipy (opencv removed due to missing inventory)
- Configured MyST parser with colon_fence and dollarmath extensions
- Added build-time diagram generation hook calling `docs/guide/_diagrams/generate_all.py`
- Created placeholder `generate_all.py` script (Plan 02 will populate with actual diagram generation)
- Created `.readthedocs.yaml` with Ubuntu 24.04, Python 3.10, docs extras installation
- Installed docs dependencies locally and verified imports

**Commit:** `8b094a8` - feat(05-01): add Sphinx configuration, docs dependencies, and RTD config

### Task 2: Landing Page, Overview, and Site Skeleton
- Deleted `docs/index.rst` (replaced by MyST markdown)
- Created `docs/index.md` with:
  - Badges (Build, Coverage, PyPI, Python, License)
  - One-paragraph description of AquaCal
  - Four sphinx-design grid cards linking to Overview, User Guide, API Reference, Tutorials
  - Quick Start code snippet showing `run_calibration` and `load_calibration`
  - Hidden toctree for sidebar navigation
- Created `docs/overview.md` explaining refractive calibration concept:
  - Problem description (cameras in air viewing underwater targets)
  - ASCII diagram of camera array, water surface, underwater target
  - Why standard calibration fails (ignores refraction)
  - What AquaCal does (models Snell's law)
  - Links to theory pages (coming in Plan 02)
  - Accessible tone, no equations, ~250 words
- Created `docs/guide/index.md` as User Guide landing page with toctree placeholder for theory pages
- Created `docs/tutorials/index.md` placeholder noting Phase 6 Jupyter notebook integration
- Created `docs/contributing.md` using MyST include directive to pull from `../CONTRIBUTING.md`
- Created `docs/changelog.md` using MyST include directive to pull from `../CHANGELOG.md`
- Created `docs/api/index.md` placeholder for Plan 03 API reference
- Verified Sphinx builds cleanly without errors or warnings
- Landing page renders with grid cards and navigable sidebar

**Commits:**
- `86e7161` - feat(05-01): add landing page, overview, and site skeleton
- `fde7a6c` - fix(05-01): remove opencv from intersphinx mapping

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification criteria met:

1. ✅ `pip install -e ".[docs]"` installed without errors
2. ✅ `python -m sphinx docs docs/_build -b html` builds without errors or warnings
3. ✅ Landing page (`docs/_build/index.html`) displays badges, grid cards, and quick start
4. ✅ Sidebar shows: Overview | User Guide | API Reference | Tutorials | Contributing | Changelog
5. ✅ Overview page explains refractive calibration concept with ASCII diagram
6. ✅ `.readthedocs.yaml` exists at repository root

## Key Decisions

### OpenCV Intersphinx Removed
**Decision:** Removed OpenCV from intersphinx mapping
**Rationale:** OpenCV documentation doesn't provide a valid intersphinx inventory at the expected URL, causing build warnings
**Impact:** No automatic cross-references to cv2 API documentation, but builds remain clean
**Alternatives considered:** Custom inventory file, but not worth the maintenance overhead for minimal benefit

## Success Criteria

✅ Sphinx documentation site skeleton builds cleanly with Furo theme
✅ Landing page is rich and navigable
✅ All section placeholders exist
✅ RTD configuration is ready for deployment

## Next Steps

**Plan 02:** Create three theory pages (refractive geometry, coordinate conventions, optimizer pipeline) with matplotlib-generated diagrams

**Plan 03:** Add complete API reference using sphinx-autodoc with cross-links to theory pages

## Self-Check: PASSED

All claimed files and commits verified:

**Files:**
- ✅ docs/conf.py
- ✅ docs/index.md
- ✅ docs/overview.md
- ✅ docs/guide/index.md
- ✅ docs/api/index.md
- ✅ docs/tutorials/index.md
- ✅ docs/contributing.md
- ✅ docs/changelog.md
- ✅ docs/guide/_diagrams/generate_all.py
- ✅ .readthedocs.yaml

**Commits:**
- ✅ 8b094a8 - feat(05-01): add Sphinx configuration, docs dependencies, and RTD config
- ✅ 86e7161 - feat(05-01): add landing page, overview, and site skeleton
- ✅ fde7a6c - fix(05-01): remove opencv from intersphinx mapping
