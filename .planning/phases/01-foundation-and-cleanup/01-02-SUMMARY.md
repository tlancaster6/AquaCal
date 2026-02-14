---
phase: 01-foundation-and-cleanup
plan: 02
subsystem: packaging
tags: [pypi, metadata, documentation, changelog]
dependency_graph:
  requires: []
  provides: [complete-package-metadata, changelog, contributing-guide]
  affects: [pypi-upload, documentation-site]
tech_stack:
  added: []
  patterns: [keep-a-changelog, pypi-metadata-completeness, deprecation-policy]
key_files:
  created: [CHANGELOG.md, CONTRIBUTING.md]
  modified: [pyproject.toml, .gitignore]
decisions: []
metrics:
  tasks_completed: 2
  duration_seconds: 107
  completed_at: 2026-02-14T16:11:34Z
---

# Phase 1 Plan 2: Package Metadata and Documentation Summary

**One-liner:** Complete PyPI metadata in pyproject.toml, create CHANGELOG.md following Keep a Changelog format, and establish CONTRIBUTING.md with deprecation policy

## Outcome

Successfully completed package metadata and foundational documentation files required for PyPI publication. The package now has all five standard project URLs (Homepage, Documentation, Repository, Issues, Changelog), updated classifiers reflecting Beta status, and standardized contributor documentation.

**Key changes:**
- Updated pyproject.toml description to highlight Snell's law modeling
- Changed Development Status from Alpha to Beta
- Added three new classifiers (Intended Audience :: Developers, Topic :: Scientific/Engineering, Topic :: Scientific/Engineering :: Image Processing)
- Added all five standard PyPI project URLs
- Created CHANGELOG.md with Keep a Changelog 1.1.0 format and Unreleased section documenting existing features
- Created CONTRIBUTING.md with development setup, code style guidelines, testing instructions, and comprehensive deprecation policy
- Cleaned .gitignore to remove obsolete dev/ reference

**Success criteria satisfied:**
- PKG-02: CHANGELOG.md exists following Keep a Changelog format
- PKG-03: Package metadata (description, classifiers, URLs) is complete
- Success criterion 5: Deprecation policy documented in CONTRIBUTING.md
- .gitignore cleaned of stale references

## Tasks Completed

### Task 1: Update pyproject.toml metadata and clean .gitignore
**Status:** Complete
**Commit:** bb16a42
**Files modified:** pyproject.toml, .gitignore

Updated package metadata for PyPI readiness:
- Changed description to "Refractive multi-camera calibration for underwater arrays with Snell's law modeling"
- Updated Development Status classifier from "3 - Alpha" to "4 - Beta"
- Added "Intended Audience :: Developers" classifier
- Added "Topic :: Scientific/Engineering" (broader category)
- Added "Topic :: Scientific/Engineering :: Image Processing" classifier
- Expanded project.urls from 1 to 5 entries: Homepage, Documentation, Repository, Issues, Changelog
- Removed dev/ line from .gitignore (directory no longer exists after Plan 01)

All metadata now follows scientific Python library conventions with minimal dependency pinning (only opencv-python>=4.6 lower bound, no upper bounds).

### Task 2: Create CHANGELOG.md and CONTRIBUTING.md
**Status:** Complete
**Commit:** 8d2adcb
**Files created:** CHANGELOG.md, CONTRIBUTING.md

Created foundational documentation files:

**CHANGELOG.md:**
- Follows Keep a Changelog 1.1.0 format with proper attribution and SemVer adherence note
- Includes [Unreleased] section documenting existing features:
  - Refractive multi-camera calibration pipeline (four-stage architecture)
  - Snell's law projection modeling for accurate refractive ray tracing
  - ChArUco board detection and pose estimation
  - BFS-based extrinsic initialization
  - Joint refractive bundle adjustment
  - Optional intrinsic refinement stage
  - Sparse Jacobian optimization
  - CLI interface
- Ready for v1.0.0 release preparation in Phase 6

**CONTRIBUTING.md:**
- Development setup section: clone, venv creation, editable install with dev dependencies, test verification
- Code style section: Black formatter, Google-style docstrings, numpy.typing.NDArray type hints, import ordering conventions
- Running tests section: all tests, skip slow tests, single file commands
- Submitting changes section: fork, branch, test, PR workflow basics
- Deprecation policy section (required by success criteria):
  1. Add warnings.warn() with DeprecationWarning including version deprecated and removal version
  2. Document in CHANGELOG.md under "Deprecated" category
  3. Maintain deprecated API for at least 2 minor versions
  4. Document replacement in function docstring
  5. Include code example showing the warning pattern with stacklevel=2

## Deviations from Plan

None - plan executed exactly as written.

## Technical Details

### PyPI Metadata Structure

The pyproject.toml now includes complete metadata for professional PyPI presentation:

**Project URLs (5 standard labels):**
- Homepage: GitHub repository root
- Documentation: aquacal.readthedocs.io (will be created in Phase 3)
- Repository: GitHub repository URL
- Issues: GitHub issues page
- Changelog: Direct link to CHANGELOG.md on main branch

**Classifiers (12 total):**
- Development Status: Beta (signals mature but pre-1.0 status)
- Intended Audience: Developers and Science/Research
- Topic hierarchy: Scientific/Engineering > Image Processing > Image Recognition
- License: MIT
- Programming Language: Python 3.10, 3.11, 3.12

**Dependencies:** Maintained minimal pinning approach (only opencv-python>=4.6) following research library best practices - no upper bounds to avoid version conflicts.

### Keep a Changelog Format

CHANGELOG.md structure follows standard conventions:
- Header with format attribution link
- SemVer adherence statement
- [Unreleased] section for v1.0.0 preparation
- Added category documenting existing features
- ISO 8601 date format (YYYY-MM-DD) for future releases
- Six standard categories available: Added, Changed, Deprecated, Removed, Fixed, Security

### Deprecation Policy Pattern

The documented deprecation workflow ensures backward compatibility:
1. Runtime warning (DeprecationWarning with stacklevel=2)
2. CHANGELOG.md documentation
3. Minimum 2 minor version maintenance window
4. Docstring migration guidance

Example pattern provided in CONTRIBUTING.md shows proper stacklevel usage to display caller location rather than warning location, improving developer experience.

## Verification Results

All verification steps passed:

**Task 1 verification:**
- pyproject.toml contains all 5 project URLs (Homepage, Documentation, Repository, Issues, Changelog)
- Development Status classifier shows "4 - Beta"
- .gitignore has no standalone dev/ reference (only valid dev/docs/_build/ pattern remains)

**Task 2 verification:**
- CHANGELOG.md header shows "# Changelog"
- Unreleased section present
- Keep a Changelog attribution link present
- CONTRIBUTING.md has Deprecation Policy section
- DeprecationWarning example code present
- Development setup pip install instructions present

**Overall verification:**
- All 5 project URLs validated via Python tomllib parsing
- CHANGELOG.md and CONTRIBUTING.md files exist at repository root
- Deprecation policy includes required DeprecationWarning pattern with stacklevel=2
- .gitignore cleaned of obsolete references

## Impact on Future Work

**Phase 3 (Documentation Site):**
- Documentation URL in pyproject.toml points to aquacal.readthedocs.io (will need Read the Docs setup)
- CHANGELOG.md provides content source for documentation changelog page
- Deprecation policy will guide API evolution documentation

**Phase 5 (CI/CD Automation):**
- CONTRIBUTING.md provides testing commands for CI workflow configuration
- Code style guidelines inform linter/formatter workflow setup

**Phase 6 (Public Release):**
- CHANGELOG.md ready for v1.0.0 release notes (move Unreleased to [1.0.0] section with date)
- Complete pyproject.toml metadata enables immediate PyPI upload
- Changelog URL in metadata will display correctly on PyPI project page

## Next Steps

Proceed to Plan 03 (Build package and validate installation) to verify:
1. Package builds successfully with new metadata
2. Installation works across Python 3.10, 3.11, 3.12
3. Metadata displays correctly on TestPyPI
4. Cross-platform installation validation (Windows/Linux)

## Self-Check: PASSED

**Created files verification:**
- FOUND: C:/Users/tucke/PycharmProjects/AquaCal/CHANGELOG.md
- FOUND: C:/Users/tucke/PycharmProjects/AquaCal/CONTRIBUTING.md

**Modified files verification:**
- FOUND: C:/Users/tucke/PycharmProjects/AquaCal/pyproject.toml (5 project URLs present)
- FOUND: C:/Users/tucke/PycharmProjects/AquaCal/.gitignore (dev/ line removed)

**Commits verification:**
- FOUND: bb16a42 (chore(01-02): update pyproject.toml metadata and clean .gitignore)
- FOUND: 8d2adcb (docs(01-02): create CHANGELOG.md and CONTRIBUTING.md)

All claimed files exist and all commits are present in git history.
