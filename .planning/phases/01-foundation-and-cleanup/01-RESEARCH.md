# Phase 1: Foundation and Cleanup - Research

**Researched:** 2026-02-14
**Domain:** Python package metadata, repository cleanup, and PyPI publication
**Confidence:** HIGH

## Summary

Phase 1 establishes AquaCal as a pip-installable package with complete PyPI metadata and a clean repository structure. The phase migrates legacy development documentation from `dev/` into `.planning/` (the GSD workflow home), removes pre-GSD agent infrastructure from `.claude/`, and validates cross-platform installation. This is package infrastructure work only -- no new features, no documentation site, no CI/CD.

Python packaging in 2026 relies on PEP 517/518 standards with `pyproject.toml` as the single source of configuration. For scientific research libraries, the standard stack is setuptools or hatchling as build backend, minimal dependency pinning (no upper bounds), semantic versioning, and Keep a Changelog format. The existing AquaCal `pyproject.toml` uses setuptools and is 90% complete -- missing only project URLs (Documentation, Issues, Changelog) and a richer package description.

**Primary recommendation:** Keep setuptools build backend (already configured), add missing PyPI metadata (project.urls), create CHANGELOG.md following Keep a Changelog format with Unreleased section for v1.0.0 preparation, migrate `dev/DESIGN.md` and `dev/GEOMETRY.md` to `.planning/architecture.md` and `.planning/geometry.md`, document deprecation policy in CONTRIBUTING.md, and test installation on fresh venvs across Python 3.10/3.11/3.12.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Migrate useful content from `dev/` into `.planning/` (architecture docs, geometry docs, knowledge base), then delete entire `dev/` folder
- Specifically: `dev/DESIGN.md` and `dev/GEOMETRY.md` go to `.planning/architecture.md` and `.planning/geometry.md`
- `dev/KNOWLEDGE_BASE.md` content should be migrated to `.planning/` as well
- `dev/CHANGELOG.md`, task files, handoffs -- discard (GSD workflow replaces these)
- `results/` folder: leave as-is (user is actively using it, will delete later). Do NOT add to .gitignore
- Clean up `.claude/` directory: remove confusing workflows, unneeded agent specs, bad config settings
- Clean up `CLAUDE.md`: bring up to GSD standard (Claude's discretion on specific changes)

### Claude's Discretion
- Package metadata completeness (PyPI URLs, classifiers, description) -- use scientific Python conventions
- Dependency pinning strategy -- choose what's standard for research libraries
- Versioning and CHANGELOG format -- follow Keep a Changelog as specified in success criteria
- Specific `.claude/` cleanup decisions (which workflows/agents/configs to remove)
- CLAUDE.md restructuring approach

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core Build Tools
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| setuptools | >=61.0 | Build backend (PEP 517) | Already configured; most mature; supports scientific community |
| wheel | latest | Binary distribution format | Standard for pip installs |
| twine | latest | PyPI upload tool | Official PyPA tool for secure uploads |

### Testing and Validation Tools
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| build | latest | Build sdist/wheel locally | Test package building before PyPI upload |
| pip | latest (in venv) | Install package in clean environment | Validate cross-platform installation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| setuptools | hatchling | Hatchling is PyPA-recommended for new projects (faster, better defaults), but setuptools already configured and working |
| setuptools | flit-core | Flit is minimalist and fast, but setuptools more familiar to scientific community |

**Installation:**
```bash
pip install build twine  # for package building and upload
```

**Note:** The existing `pyproject.toml` already specifies `setuptools>=61.0` and `wheel` in `[build-system]`. No changes needed to build backend.

## Architecture Patterns

### Recommended Repository Structure (Post-Cleanup)
```
AquaCal/
├── .planning/                    # GSD workflow home
│   ├── architecture.md           # Migrated from dev/DESIGN.md
│   ├── geometry.md               # Migrated from dev/GEOMETRY.md
│   ├── knowledge-base.md         # Migrated from dev/KNOWLEDGE_BASE.md
│   ├── phases/                   # Phase-specific planning
│   └── research/                 # Project research docs
├── .claude/                      # Minimal GSD infrastructure only
│   └── (no pre-GSD agents/workflows)
├── src/aquacal/                  # Package source code
├── tests/                        # Test suite
├── pyproject.toml                # Package metadata and config
├── CHANGELOG.md                  # Keep a Changelog format
├── CONTRIBUTING.md               # Contributor guidelines with deprecation policy
├── CLAUDE.md                     # Minimal project instructions
├── README.md                     # Package description
└── LICENSE                       # MIT license
```

### Pattern 1: Keep a Changelog Format
**What:** Standardized changelog format with semantic version sections and change categories
**When to use:** All version releases and preparation for v1.0.0
**Example:**
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Refractive multi-camera calibration pipeline
- Snell's law projection modeling
- ChArUco board detection and validation

### Changed
- (migration notes if converting from old system)

## [0.1.0] - YYYY-MM-DD

### Added
- Initial development release
```
**Source:** [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)

**Change categories (use these exactly):**
- **Added** - New features
- **Changed** - Modifications to existing functionality
- **Deprecated** - Features slated for removal
- **Removed** - Deleted features
- **Fixed** - Bug corrections
- **Security** - Vulnerability patches

**Date format:** ISO 8601 (YYYY-MM-DD), e.g., `2026-02-14`

### Pattern 2: PyPI Metadata Completeness
**What:** Complete `[project]` table in `pyproject.toml` with all recommended fields
**When to use:** Before first PyPI upload (Phase 1 deliverable)
**Example:**
```toml
[project]
name = "aquacal"
version = "0.1.0"
description = "Refractive multi-camera calibration for underwater arrays with Snell's law modeling"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Tucker Lancaster"}
]
keywords = ["calibration", "multi-camera", "underwater", "refraction", "computer-vision", "charuco"]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "numpy",
    "scipy",
    "opencv-python>=4.6",
    "pyyaml",
    "matplotlib",
    "pandas",
]

[project.urls]
Homepage = "https://github.com/tlancaster6/AquaCal"
Documentation = "https://aquacal.readthedocs.io"  # Phase 3
Repository = "https://github.com/tlancaster6/AquaCal"
Issues = "https://github.com/tlancaster6/AquaCal/issues"
Changelog = "https://github.com/tlancaster6/AquaCal/blob/main/CHANGELOG.md"
```
**Source:** [Python Packaging User Guide - Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

**Well-known URL labels (PyPI displays these specially):**
- Homepage, Documentation, Repository, Issues, Changelog

### Pattern 3: Dependency Specification for Research Libraries
**What:** Minimal lower-bound pinning, no upper bounds, let users control their environment
**When to use:** All library dependency specifications (applications can pin)
**Example:**
```toml
dependencies = [
    "numpy",              # No pins - let user control
    "scipy",              # No pins
    "opencv-python>=4.6", # Lower bound only (refractive functions need modern OpenCV)
    "pyyaml",
    "matplotlib",
    "pandas",
]
```
**Rationale:** Research libraries should not pin dependencies aggressively. Upper bound constraints cause real-world compatibility problems. If you pin everything, the end-user cannot choose their environment. Pinning reduces surprise breakage in applications but is anti-pattern for reusable libraries.

**Source:** [To pin or not to pin dependencies](http://blog.chrisgorgolewski.org/2017/12/to-pin-or-not-to-pin-dependencies.html)

### Pattern 4: CONTRIBUTING.md with Deprecation Policy
**What:** Contributor guide including development setup, style conventions, and API deprecation policy
**When to use:** Phase 1 (required by success criteria)
**Example:**
```markdown
# Contributing to AquaCal

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Install in editable mode: `pip install -e .[dev]`
4. Run tests: `pytest tests/`

## Code Style

- Formatter: Black
- Docstrings: Google style
- Type hints: Use numpy.typing.NDArray

## Deprecation Policy

When deprecating an API:

1. Add @deprecated decorator with version and alternative
2. Add DeprecationWarning in code
3. Document in CHANGELOG.md under "Deprecated" section
4. Maintain deprecated API for at least 2 minor versions
5. Document replacement in docstring

Example:
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated since v1.2.0 and will be removed in v1.4.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # implementation
```
```
**Source:** [PyAnsys Deprecation Best Practices](https://dev.docs.pyansys.com/coding-style/deprecation.html)

### Anti-Patterns to Avoid
- **Pinning upper bounds in library dependencies:** Causes compatibility conflicts for users
- **Using git diffs as changelog:** Changelogs are for humans, not machines
- **Ambiguous date formats:** Use ISO 8601 (YYYY-MM-DD) only
- **Missing project.urls in pyproject.toml:** PyPI displays these prominently; omitting them looks unprofessional
- **Skipping cross-platform installation testing:** Windows path handling, dependency availability differ

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Changelog generation | Custom git log parser | Keep a Changelog format (manual) | Standardized, human-readable, adopted by scientific community. Automation tools (semantic-release) available but overkill for Phase 1 |
| Package building | Custom setup.py scripts | `python -m build` (PEP 517) | Official PyPA tool, handles sdist and wheel correctly |
| PyPI upload | Direct API calls | `twine upload dist/*` | Handles authentication, retries, verification checks |
| Version management | Manual string replacement | Semantic versioning in pyproject.toml | Standard, tooling-compatible (bump2version, semantic-release) |
| Deprecation warnings | Print statements or custom logging | warnings.warn(DeprecationWarning) | Standard library, respected by pytest, filterable by users |

**Key insight:** Python packaging ecosystem has mature, well-tested tools for every step. Custom solutions introduce maintenance burden and edge cases (platform differences, authentication, encoding issues). The official toolchain (build + twine + pyproject.toml) handles all complexity.

## Common Pitfalls

### Pitfall 1: TestPyPI Database Pruning
**What goes wrong:** Upload package to TestPyPI for testing, it works, then weeks later the account or package disappears
**Why it happens:** TestPyPI database is periodically pruned; it's not a permanent repository
**How to avoid:** Use TestPyPI only for immediate pre-release validation, not long-term storage. Document in plan: upload to TestPyPI, test installation, then upload to real PyPI same day
**Warning signs:** Package not found on TestPyPI after successful upload weeks ago
**Source:** [Using TestPyPI - Python Packaging User Guide](https://packaging.python.org/guides/using-testpypi/)

### Pitfall 2: Missing project.urls After Upload
**What goes wrong:** Package uploads to PyPI successfully but sidebar shows only "Homepage" link, missing Documentation/Issues/Changelog
**Why it happens:** Forgot to add `[project.urls]` section to pyproject.toml before building package
**How to avoid:** Add all relevant URLs to pyproject.toml before building distribution. Rebuild package (increment version) if URLs missing after upload
**Warning signs:** PyPI project page lacks sidebar links to Issues, Documentation
**Source:** [Writing pyproject.toml - project.urls](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

### Pitfall 3: CHANGELOG.md vs Commit Messages
**What goes wrong:** Developer copies git log into CHANGELOG.md verbatim, producing low-value noise ("fix typo", "update tests", "WIP")
**Why it happens:** Misunderstanding that changelogs are for users, not developers
**How to avoid:** Keep a Changelog emphasizes "Changelogs are for humans, not machines." Summarize notable changes only (new features, breaking changes, important fixes). Group by category (Added, Changed, Fixed). Skip implementation details
**Warning signs:** CHANGELOG.md has 50+ entries per version, mentions internal variable names, includes "WIP" entries
**Source:** [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

### Pitfall 4: Stale .gitignore Patterns Breaking Package Build
**What goes wrong:** Package builds locally but files missing from PyPI-installed version
**Why it happens:** Overly broad `.gitignore` patterns exclude files needed in distribution (e.g., `*.md` excludes README.md in subdirectories)
**How to avoid:** Test installation from built wheel before PyPI upload: `python -m build && pip install dist/*.whl` in fresh venv. Verify file presence: `python -c "import aquacal; print(aquacal.__file__)"`
**Warning signs:** Import errors after pip install but not in local dev environment
**Source:** General Python packaging experience

### Pitfall 5: Legacy `.claude/` Confusing Future Contributors
**What goes wrong:** External contributor finds pre-GSD agent specifications, tries to follow outdated workflow, gets confused when GSD workflows behave differently
**Why it happens:** Incomplete cleanup of legacy agent infrastructure from `.claude/` directory
**How to avoid:** Remove all pre-GSD agents (`executor.md`, `planner.md`, `tasker.md`, `debugger.md`, `claude-expert.md`), outdated workflows, and restrictive permission settings from `.claude/settings.local.json`. Keep only GSD-compatible infrastructure
**Warning signs:** `.claude/agents/` contains agents not part of GSD workflow
**Source:** User's locked decision to "clean up `.claude/` directory: remove confusing workflows, unneeded agent specs"

### Pitfall 6: Moving dev/ Docs to docs/ Instead of .planning/
**What goes wrong:** Migrated `dev/DESIGN.md` and `dev/GEOMETRY.md` to `docs/` directory instead of `.planning/`, creating confusion about whether these are user-facing documentation or internal reference
**Why it happens:** Assumption that "documentation" belongs in `docs/`
**How to avoid:** Follow user's locked decision: `dev/DESIGN.md` and `dev/GEOMETRY.md` go to `.planning/architecture.md` and `.planning/geometry.md`. The `.planning/` directory is the GSD workflow home for project reference material, not user-facing docs. Phase 3 will create user-facing docs in `docs/` (Sphinx)
**Warning signs:** Migrated files appear in `docs/` directory during Phase 1
**Source:** User's locked decision and CONTEXT.md specific idea

## Code Examples

Verified patterns from official sources:

### Building and Testing Package Locally
```bash
# Build source distribution and wheel
python -m build

# Test installation in fresh venv (Linux/macOS)
python -m venv /tmp/test_aquacal
source /tmp/test_aquacal/bin/activate
pip install dist/aquacal-0.1.0-py3-none-any.whl
python -c "import aquacal; print(aquacal.__version__)"
deactivate

# Test installation in fresh venv (Windows Git Bash)
python -m venv /c/Users/tucke/tmp/test_aquacal
source /c/Users/tucke/tmp/test_aquacal/Scripts/activate
pip install dist/aquacal-0.1.0-py3-none-any.whl
python -c "import aquacal; print(aquacal.__version__)"
deactivate
```
**Source:** [Python Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)

### Uploading to TestPyPI
```bash
# Register account at https://test.pypi.org/account/register/
# Create API token at https://test.pypi.org/manage/account/token/

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aquacal
```
**Source:** [Using TestPyPI](https://packaging.python.org/guides/using-testpypi/)

**Important:** `--extra-index-url https://pypi.org/simple/` is required because TestPyPI doesn't host dependencies (numpy, scipy, etc.). This pulls dependencies from real PyPI.

### Checking PyPI Metadata Display
```bash
# After upload to TestPyPI, visit project page
# https://test.pypi.org/project/aquacal/

# Verify:
# - Description renders correctly (from README.md)
# - Classifiers show in sidebar
# - Project links appear (Homepage, Repository, Issues, Changelog, Documentation)
# - Supported Python versions listed
```

### Cross-Platform Installation Testing
```bash
# Test Python 3.10, 3.11, 3.12 on Linux/macOS/Windows
# Example for Python 3.11 on Windows Git Bash:
python3.11 -m venv /c/Users/tucke/tmp/test_py311
source /c/Users/tucke/tmp/test_py311/Scripts/activate
pip install aquacal
python -c "import aquacal; aquacal.cli.main(['--version'])"
pytest --pyargs aquacal  # if tests installed
deactivate
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setup.py with setup() call | pyproject.toml with [build-system] | PEP 517/518 (2017-2018) | Single config file, declarative metadata, build backend separation |
| requirements.txt for dependencies | dependencies in [project] table | PEP 621 (2020) | Unified metadata, tooling interoperability |
| Manual versioning in setup.py | version field in [project] | PEP 621 (2020) | Single source of truth (or dynamic via setuptools_scm) |
| Passwords for PyPI upload | API tokens and Trusted Publishing (OIDC) | 2019 (tokens), 2023 (OIDC) | Enhanced security, no stored credentials |
| Universal wheel format (py2.py3) | Python 3 only wheels | Python 2 EOL (2020) | Simpler builds, no compatibility shims |

**Deprecated/outdated:**
- **setup.py with distutils:** Deprecated in Python 3.10, removed in 3.12. Use setuptools with pyproject.toml
- **python setup.py upload:** Removed. Use twine for secure uploads
- **setup.cfg for metadata:** Still works but pyproject.toml is preferred standard (PEP 621)
- **Poetry [tool.poetry] table:** Poetry 2.0 (Jan 2025) now supports standard [project] table

## Open Questions

1. **Exact .claude/ files to remove**
   - What we know: User decided to "remove confusing workflows, unneeded agent specs, bad config settings"
   - What's unclear: Specific files to delete vs. keep
   - Recommendation: Remove pre-GSD agents (`executor.md`, `planner.md`, `tasker.md`, `debugger.md`, `claude-expert.md`), remove restrictive permissions from `settings.local.json`, keep GSD infrastructure. Document removals in plan tasks.

2. **CLAUDE.md restructuring specifics**
   - What we know: User decided to "bring up to GSD standard" with Claude's discretion on specific changes
   - What's unclear: What constitutes "GSD standard" for CLAUDE.md
   - Recommendation: Minimal project instructions only. Remove pre-GSD workflow references (dev/TASKS.md, dev/handoffs/, task file conventions). Keep domain conventions, architecture overview, key reference files, testing commands. Update references from `dev/DESIGN.md` to `.planning/architecture.md`.

3. **results/ folder handling**
   - What we know: User is actively using it, will delete later, do NOT add to .gitignore
   - What's unclear: Nothing -- explicit instruction to leave as-is
   - Recommendation: No action in Phase 1. Do not modify .gitignore, do not delete folder.

4. **TestPyPI upload timing**
   - What we know: Need to verify metadata displays correctly on PyPI test instance (success criterion 3)
   - What's unclear: Whether to do TestPyPI upload as part of Phase 1 or defer to Phase 6
   - Recommendation: Include TestPyPI upload in Phase 1 verification. Upload, verify metadata display, then delete test release. Real PyPI upload happens in Phase 6.

## Sources

### Primary (HIGH confidence)
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) - Official changelog format specification
- [Python Packaging User Guide - Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - Official PyPA guidance
- [PyPI Classifiers](https://pypi.org/classifiers/) - Complete classifier list
- [Using TestPyPI](https://packaging.python.org/guides/using-testpypi/) - Official TestPyPI docs
- [Semantic Versioning 2.0.0](https://semver.org/) - SemVer specification
- [PyPI Project Metadata](https://docs.pypi.org/project_metadata/) - PyPI metadata specification

### Secondary (MEDIUM confidence)
- [Python Packaging Best Practices: setuptools, Poetry, and Hatch in 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/) - Current tooling comparison
- [To pin or not to pin dependencies](http://blog.chrisgorgolewski.org/2017/12/to-pin-or-not-to-pin-dependencies.html) - Research library dependency strategy
- [PyAnsys Deprecation Best Practices](https://dev.docs.pyansys.com/coding-style/deprecation.html) - Scientific library deprecation policy
- [PyOpenSci Python Package Guide - CONTRIBUTING.md](https://www.pyopensci.org/python-package-guide/documentation/repository-files/contributing-file.html) - Contributor guide best practices
- [Scientific Python Development Guide - Simple packaging](https://learn.scientific-python.org/development/guides/packaging-simple/) - Scientific community standards

### Tertiary (LOW confidence)
None -- all findings verified with official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools from official PyPA, setuptools already configured
- Architecture: HIGH - Keep a Changelog and pyproject.toml are well-specified standards
- Pitfalls: HIGH - Common issues documented in official guides and community best practices
- .claude/ cleanup: MEDIUM - User locked decisions clear, but specific file decisions left to Claude's discretion
- CLAUDE.md changes: MEDIUM - "GSD standard" not explicitly defined, using inference from GSD workflow

**Research date:** 2026-02-14
**Valid until:** 2026-09-14 (6 months - packaging standards stable, but tooling recommendations evolve)
