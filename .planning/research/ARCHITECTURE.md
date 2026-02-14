# Architecture Research: Release Infrastructure

**Domain:** Scientific Python Library Release
**Researched:** 2026-02-14
**Confidence:** HIGH

## Standard Architecture

Scientific Python library release infrastructure follows a well-established pattern with five interconnected components: documentation site, CI/CD pipeline, packaging configuration, example/tutorial content, and data/artifact distribution.

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENTATION LAYER (Sphinx)                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ API Docs     │  │ Tutorials    │  │ Example      │              │
│  │ (autodoc)    │  │ (RST/MyST)   │  │ Gallery      │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         │                 │   ┌─────────────┴─────────────┐         │
│         │                 │   │ Jupyter Notebooks         │         │
│         │                 │   │ (executed & rendered)     │         │
│         │                 │   └───────────────────────────┘         │
├─────────┴─────────────────┴─────────────────────────────────────────┤
│                     CI/CD PIPELINE (GitHub Actions)                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ Test Matrix │  │ Build Docs  │  │ Build       │  │ Publish  │  │
│  │ (tox/nox)   │  │ (Sphinx)    │  │ Package     │  │ to PyPI  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    PACKAGING LAYER (pyproject.toml)                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  [build-system] → setuptools/hatch                         │    │
│  │  [project] → metadata, deps, scripts                       │    │
│  │  [tool.*] → pytest, mypy, black configs                    │    │
│  └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                    ARTIFACT DISTRIBUTION                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ PyPI         │  │ Read the     │  │ Data Repo    │              │
│  │ (wheel+sdist)│  │ Docs (HTML)  │  │ (Zenodo)     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Documentation Site | API reference, tutorials, examples rendered as HTML | Sphinx with autodoc, napoleon, sphinx-gallery extensions |
| CI/CD Pipeline | Test across Python versions, build docs, create releases, publish to PyPI | GitHub Actions workflows with tox/nox for test matrix |
| Packaging Config | Metadata, dependencies, build settings, tool configs | pyproject.toml with setuptools/hatch backend |
| Example Content | Executable demonstrations, Jupyter notebooks, gallery | Python scripts converted by sphinx-gallery, or .ipynb notebooks |
| Version Management | Semantic versioning, changelog generation, release tagging | python-semantic-release or manual with conventional commits |
| Data Repository | Example datasets, calibration files, benchmark data | Zenodo with DOI, or GitHub releases for smaller files |

## Recommended Project Structure

```
aquacal/
├── .github/
│   └── workflows/
│       ├── test.yml              # Multi-version test matrix
│       ├── docs.yml              # Build and deploy docs
│       ├── release.yml           # Publish to PyPI (on tag push)
│       └── pre-commit.yml        # Code quality checks
├── docs/
│   ├── source/
│   │   ├── conf.py               # Sphinx configuration
│   │   ├── index.rst             # Documentation homepage
│   │   ├── installation.rst      # Install instructions
│   │   ├── quickstart.rst        # Getting started guide
│   │   ├── tutorials/            # Step-by-step tutorials
│   │   ├── api/                  # API reference (autodoc)
│   │   ├── examples/             # Example Python scripts
│   │   └── _static/              # Custom CSS, images
│   ├── requirements.txt          # Doc build dependencies
│   └── Makefile                  # Sphinx build commands
├── examples/
│   ├── notebooks/                # Jupyter notebooks
│   │   ├── basic_calibration.ipynb
│   │   └── visualization.ipynb
│   └── datasets/                 # Small example data (or links)
│       ├── README.md             # Dataset descriptions + Zenodo links
│       └── synthetic_example.npz
├── src/aquacal/                  # Existing source code
├── tests/                        # Existing tests
├── pyproject.toml                # Package metadata + config
├── README.md                     # GitHub landing page
├── CHANGELOG.md                  # Release history
├── LICENSE                       # MIT license text
└── .readthedocs.yml              # Read the Docs config
```

### Structure Rationale

- **`.github/workflows/`:** GitHub Actions is the standard CI for scientific Python (tight GitHub integration, matrix strategies)
- **`docs/source/`:** Sphinx convention; separates source from built HTML
- **`docs/source/examples/`:** Python scripts here get executed and rendered by sphinx-gallery
- **`examples/notebooks/`:** Jupyter notebooks for interactive tutorials (can be executed by nbsphinx or sphinx-gallery)
- **`examples/datasets/`:** Small datasets live here; large datasets link to Zenodo
- **`.readthedocs.yml`:** Config for automated doc builds on Read the Docs (free for open source)

## Architectural Patterns

### Pattern 1: Sphinx + autodoc + napoleon

**What:** Auto-generate API documentation from docstrings using Sphinx extensions

**When to use:** All scientific Python libraries with Google/NumPy-style docstrings

**Trade-offs:**
- Pros: Single source of truth (docstrings), automatic cross-references, integrates with intersphinx
- Cons: Requires docstrings to be complete and well-formatted

**Example:**
```python
# docs/source/conf.py
extensions = [
    'sphinx.ext.autodoc',         # Extract docstrings
    'sphinx.ext.napoleon',        # Parse Google/NumPy docstrings
    'sphinx.ext.viewcode',        # Add source code links
    'sphinx.ext.intersphinx',     # Link to other projects (numpy, scipy)
    'sphinx.ext.mathjax',         # Render LaTeX math
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
```

### Pattern 2: sphinx-gallery for Example Gallery

**What:** Automatically execute Python scripts, capture outputs (plots, stdout), and render as HTML gallery

**When to use:** Libraries with visual output (matplotlib, images) or algorithmic demonstrations

**Trade-offs:**
- Pros: Examples guaranteed to work (they're executed), generates downloadable notebooks, auto-links to API docs
- Cons: Adds build time (executing examples), requires example scripts to be fast (<30s each)

**Example:**
```python
# docs/source/conf.py
extensions = ['sphinx_gallery.gen_gallery']

sphinx_gallery_conf = {
    'examples_dirs': '../examples',       # Source scripts
    'gallery_dirs': 'auto_examples',      # Generated gallery
    'filename_pattern': '/plot_',         # Only process plot_*.py
    'download_all_examples': False,
    'image_scrapers': ('matplotlib',),    # Scrape matplotlib figures
}
```

```python
# docs/source/examples/plot_basic_calibration.py
"""
Basic Calibration Example
==========================

This example demonstrates a simple refractive calibration.
"""
import aquacal
# ... code that generates plots ...
```

### Pattern 3: Trusted Publishing to PyPI

**What:** Use OpenID Connect (OIDC) to authenticate GitHub Actions to PyPI without storing API tokens

**When to use:** All new releases (PyPI recommends this over API tokens)

**Trade-offs:**
- Pros: No secrets management, tokens auto-expire, project-scoped
- Cons: Requires one-time PyPI configuration per project

**Example:**
```yaml
# .github/workflows/release.yml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  pypi-publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Required for trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Build package
        run: |
          pip install build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # No token needed - uses OIDC
```

### Pattern 4: Multi-Version Test Matrix with tox

**What:** Test package across multiple Python versions in isolated environments

**When to use:** Libraries claiming support for multiple Python versions (e.g., 3.10-3.12)

**Trade-offs:**
- Pros: Catches version-specific bugs, validates metadata claims
- Cons: Slower CI (serial by default unless using matrix strategies)

**Example:**
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ --cov=aquacal --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

### Pattern 5: Read the Docs Integration

**What:** Automatically build and host documentation on Read the Docs for every commit/PR

**When to use:** Open-source projects (free hosting), want versioned docs

**Trade-offs:**
- Pros: Free, versioned docs, PR previews, custom domains
- Cons: Public docs only (for free tier), must configure .readthedocs.yml

**Example:**
```yaml
# .readthedocs.yml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev

sphinx:
  configuration: docs/source/conf.py
```

## Data Flow

### Documentation Build Flow

```
Source Code (.py files)
    │
    ├─> Docstrings → autodoc → API Reference (RST) → Sphinx → HTML
    │
    ├─> Examples (.py scripts) → sphinx-gallery → Executed → Gallery (HTML)
    │
    └─> Tutorials (.rst/.md) → Sphinx → HTML
                                          │
                                          ├─> GitHub Pages (manual)
                                          └─> Read the Docs (automatic)
```

### Release Flow

```
Developer commits with conventional message (feat: / fix: / BREAKING CHANGE:)
    ↓
Push to main branch
    ↓
[Manual or semantic-release] Creates git tag (v1.2.3)
    ↓
GitHub Actions triggered on tag push
    ↓
Build package (python -m build)
    ↓
Publish to PyPI (pypa/gh-action-pypi-publish with OIDC)
    ↓
Create GitHub Release with CHANGELOG excerpt
    ↓
Update Read the Docs stable version
```

### Example Execution Flow

```
User downloads package from PyPI
    ↓
Visits documentation site
    ↓
Browses example gallery
    ↓
Clicks "Download Jupyter Notebook" on example
    ↓
Runs notebook locally
    ↓
Needs example datasets → Downloads from Zenodo (linked in docs)
```

### Key Data Flows

1. **Code → Docs:** Source code docstrings are the single source of truth for API documentation
2. **Examples → Gallery:** Python scripts in `docs/source/examples/` are executed during doc build and rendered as HTML + Jupyter notebooks
3. **Tests → CI → Badge:** Test results flow to coverage services (Codecov) and generate README badges
4. **Tag → PyPI:** Git tags trigger release workflow that builds and publishes to PyPI
5. **Datasets → Zenodo:** Large datasets uploaded to Zenodo, docs link to DOI for reproducibility

## Integration Points

### Documentation ↔ CI/CD

| Integration | Pattern | Notes |
|-------------|---------|-------|
| Doc build on PR | GitHub Actions runs `sphinx-build` on every PR | Catches doc build errors before merge |
| Read the Docs webhook | RTD builds docs automatically on push | Provides versioned docs and PR previews |
| Example execution | sphinx-gallery runs examples during doc build | Ensures examples stay up-to-date with API |

### Packaging ↔ CI/CD

| Integration | Pattern | Notes |
|-------------|---------|-------|
| Test installation | CI installs package with `pip install -e .` | Validates pyproject.toml dependencies |
| Build check | CI builds wheel/sdist to catch packaging errors | Runs on PRs, not just releases |
| PyPI publish | GitHub Actions publishes on tag push via OIDC | Uses pypa/gh-action-pypi-publish |

### Examples ↔ Documentation

| Integration | Pattern | Notes |
|-------------|---------|-------|
| sphinx-gallery | Executes .py scripts, renders as HTML gallery | Generates downloadable .ipynb for each example |
| nbsphinx | Renders .ipynb notebooks directly in docs | Alternative to sphinx-gallery for existing notebooks |
| Binder integration | Links to MyBinder for browser-based execution | Requires environment.yml or requirements.txt |

### Data ↔ Examples

| Integration | Pattern | Notes |
|-------------|---------|-------|
| Zenodo DOI | Link large datasets from docs/examples | Provides persistent, citable data location |
| GitHub releases | Attach small datasets to release assets | Good for <100MB files, version-pinned |
| In-repo data | Store tiny datasets (<1MB) in `examples/datasets/` | Bundled with package for quick start |

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Pre-release | Minimal: docs/ with Sphinx, basic CI with pytest, manual PyPI publish |
| Initial release (v0.1-v1.0) | Add: Read the Docs, test matrix (3+ Python versions), CHANGELOG.md, example gallery |
| Growing (multiple releases) | Add: semantic-release for automation, codecov, pre-commit hooks, contributor docs |
| Mature (10+ releases) | Add: Binder for interactive examples, citation.cff for citations, benchmark tracking |

### Scaling Priorities

1. **First bottleneck:** Documentation build time (from executing examples)
   - **Solution:** Use `sphinx_gallery_conf['first_notebook_cell']` to skip heavy imports or cache data
   - **Alternative:** Move slow examples to a separate "advanced" gallery built less frequently

2. **Second bottleneck:** CI test matrix time (many Python versions × many tests)
   - **Solution:** Parallelize test matrix, use `pytest-xdist` for parallel test execution
   - **Alternative:** Run full matrix only on releases, fast matrix (latest Python only) on PRs

3. **Third bottleneck:** Example dataset download time (large datasets)
   - **Solution:** Host datasets on Zenodo, provide small synthetic datasets for quick start
   - **Alternative:** Use pooch library to cache downloaded datasets

## Anti-Patterns

### Anti-Pattern 1: Documentation Separate from Code

**What people do:** Write documentation in a separate wiki or Google Docs

**Why it's wrong:** Docs drift from code, no version control, can't review doc changes in PRs

**Do this instead:** Use Sphinx with autodoc, co-locate tutorials with code, review docs in PRs

### Anti-Pattern 2: Manual Version Bumping

**What people do:** Manually edit version strings in multiple files (setup.py, __init__.py, docs/conf.py)

**Why it's wrong:** Easy to forget a file, version inconsistencies, error-prone

**Do this instead:** Single source of truth in pyproject.toml version field, or use python-semantic-release for automation

### Anti-Pattern 3: Large Datasets in Git

**What people do:** Commit multi-megabyte calibration datasets to git repository

**Why it's wrong:** Bloats repository, slow clones, GitHub has 100MB file size limit

**Do this instead:** Upload datasets to Zenodo (with DOI), link from docs, provide small synthetic examples in-repo

### Anti-Pattern 4: Notebooks as Source of Truth

**What people do:** Write library code in Jupyter notebooks, copy-paste to .py files

**Why it's wrong:** Notebooks not easily diffable, can't be imported, hard to test

**Do this instead:** Write library in .py files, write example notebooks that import the library

### Anti-Pattern 5: Skipping Test Matrix

**What people do:** Claim "Python 3.10+" support but only test on developer's Python 3.11

**Why it's wrong:** Code may break on 3.10 or 3.12 due to syntax/API differences, violates user expectations

**Do this instead:** Test on min (3.10) and max (3.12) supported versions at minimum, full matrix for releases

## Build Order Dependencies

Release infrastructure components have a clear dependency order:

```
1. pyproject.toml (packaging metadata)
   └─> Defines package name, dependencies, entry points

2. Basic CI (test.yml)
   └─> Validates pyproject.toml, runs tests

3. Documentation (docs/)
   └─> Depends on: package installable (from pyproject.toml)
   └─> Imports library to extract docstrings

4. Example gallery (docs/source/examples/)
   └─> Depends on: library working, docs configured
   └─> Executes example code

5. Release workflow (release.yml)
   └─> Depends on: tests passing, docs building
   └─> Publishes to PyPI

6. Read the Docs (.readthedocs.yml)
   └─> Depends on: docs/ configured, dependencies in pyproject.toml
   └─> Auto-builds on every commit
```

**Recommended implementation order for AquaCal:**

1. **Phase 1: Packaging Foundation**
   - Enhance pyproject.toml (add URLs, classifiers, optional-dependencies)
   - Add CHANGELOG.md, LICENSE already present

2. **Phase 2: Basic CI/CD**
   - Test matrix workflow (test.yml) for Python 3.10, 3.11, 3.12
   - Pre-commit hooks for black, mypy

3. **Phase 3: Documentation Site**
   - docs/ structure with Sphinx
   - conf.py with autodoc, napoleon for NumPy docstrings
   - API reference, installation, quickstart pages

4. **Phase 4: Example Content**
   - Jupyter notebooks in examples/notebooks/
   - Integrate with sphinx-gallery OR nbsphinx
   - Small synthetic datasets in examples/datasets/

5. **Phase 5: Deployment**
   - Read the Docs configuration (.readthedocs.yml)
   - docs.yml workflow (build docs on PR)
   - release.yml workflow (PyPI publish on tag)

6. **Phase 6: Data & Polish**
   - Upload calibration datasets to Zenodo
   - Add citation.cff for citing library
   - Contributor guidelines, issue templates

## Sources

- [Scientific Python Development Guide](https://learn.scientific-python.org/development/) - Comprehensive guide for research software engineering (HIGH confidence)
- [Python Packaging User Guide: Publishing with GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) - Official PyPI publishing guide (HIGH confidence)
- [Python Packaging User Guide: Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - Official packaging standards (HIGH confidence)
- [Sphinx-Gallery Documentation](https://sphinx-gallery.github.io/stable/index.html) - Example gallery automation tool (HIGH confidence)
- [python-semantic-release Documentation](https://python-semantic-release.readthedocs.io/) - Automated versioning and releases (HIGH confidence)
- [Read the Docs Documentation](https://docs.readthedocs.com/platform/latest/reference/git-integration.html) - Documentation hosting integration (HIGH confidence)
- [Scientific Python Cookiecutter: Publishing Releases](https://nsls-ii.github.io/scientific-python-cookiecutter/publishing-releases.html) - Release checklist patterns (MEDIUM confidence)
- [Zenodo Developer Documentation](https://developers.zenodo.org/) - Dataset repository API (MEDIUM confidence)

---
*Architecture research for: AquaCal Release Infrastructure*
*Researched: 2026-02-14*
