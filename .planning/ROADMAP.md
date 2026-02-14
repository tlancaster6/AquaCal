/# Roadmap: AquaCal

## Overview

Transform AquaCal from a working calibration library into a pip-installable PyPI package with comprehensive documentation, example datasets, Jupyter tutorials, and automated CI/CD. This roadmap delivers the v1.0 public release by following scientific Python community standards: Sphinx documentation on Read the Docs, GitHub Actions testing across Python 3.10-3.12, example notebooks demonstrating end-to-end workflows, and Trusted Publishing for secure PyPI releases.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation and Cleanup** - Package infrastructure and repository cleanup
- [ ] **Phase 2: CI/CD Automation** - Multi-platform testing and release workflows
- [ ] **Phase 3: Public Release** - PyPI v1.0.0 and community files
- [ ] **Phase 4: Example Data** - Synthetic and real calibration datasets
- [ ] **Phase 5: Documentation Site** - Sphinx API reference and user guide
- [ ] **Phase 6: Interactive Tutorials** - Jupyter notebooks demonstrating workflows

## Phase Details

### Phase 1: Foundation and Cleanup
**Goal**: Package metadata is complete, repository is clean of legacy artifacts, and installation is validated across platforms
**Depends on**: Nothing (first phase)
**Requirements**: CLEAN-01, CLEAN-02, PKG-01, PKG-02, PKG-03, PKG-04
**Success Criteria** (what must be TRUE):
  1. User can install AquaCal via pip on fresh virtual environments (Linux/macOS/Windows, Python 3.10/3.11/3.12)
  2. Repository contains no legacy dev artifacts that would confuse external contributors
  3. Package metadata (description, classifiers, URLs) displays correctly on PyPI test instance
  4. CHANGELOG.md exists following Keep a Changelog format with v1.0.0 preparation section
  5. Deprecation policy is documented in CONTRIBUTING.md
**Plans:** 3 plans

Plans:
- [ ] 01-01-PLAN.md -- Migrate dev/ docs to .planning/ and clean .claude/ directory
- [ ] 01-02-PLAN.md -- Complete pyproject.toml metadata, create CHANGELOG.md and CONTRIBUTING.md
- [ ] 01-03-PLAN.md -- Build package and validate installation

### Phase 2: CI/CD Automation
**Goal**: Automated testing across Python versions and platforms with GitHub Actions workflows for tests, docs, and PyPI publishing
**Depends on**: Phase 1
**Requirements**: CI-01, CI-02, CI-03
**Success Criteria** (what must be TRUE):
  1. GitHub Actions workflow runs pytest on push/PR across Python 3.10, 3.11, 3.12 on Linux and Windows
  2. GitHub Actions workflow builds Sphinx documentation on PR to catch doc build errors before merge
  3. GitHub Actions workflow publishes to PyPI on git tag via Trusted Publishing (OIDC, no API tokens)
  4. Pre-commit configuration exists with ruff (linter/formatter) -- mypy skipped per user decision
  5. Codecov integration reports test coverage on pull requests
**Plans:** 3 plans

Plans:
- [ ] 02-01-PLAN.md -- Pre-commit config and pyproject.toml updates (ruff, coverage, dev deps)
- [ ] 02-02-PLAN.md -- Test workflows (test.yml with matrix + pre-commit CI, slow-tests.yml)
- [ ] 02-03-PLAN.md -- Docs workflow (docs.yml + Sphinx scaffolding) and publish workflow (publish.yml)

### Phase 3: Public Release
**Goal**: AquaCal v1.0.0 is live on PyPI with community files, README badges, and Zenodo DOI
**Depends on**: Phase 2
**Requirements**: PKG-01 (final validation)
**Success Criteria** (what must be TRUE):
  1. User can `pip install aquacal` and import the library on any supported platform
  2. PyPI package page displays correct metadata with links to GitHub, docs, and changelog
  3. GitHub Release exists for v1.0.0 with CHANGELOG excerpt and installation instructions
  4. Zenodo DOI is minted for v1.0.0 release with citation metadata
  5. README includes badges for build status, coverage, PyPI version, license, DOI
  6. CONTRIBUTING.md provides development setup instructions and PR guidelines
  7. CODE_OF_CONDUCT.md exists using PSF Code of Conduct
**Plans**: TBD

Plans:
- [ ] 03-01-PLAN: TBD

### Phase 4: Example Data
**Goal**: Researchers have access to both synthetic calibration datasets with known ground truth and real-world example data
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. User can generate synthetic calibration scenarios via `aquacal.examples.generate_sample_data()` with configurable rig size
  2. User can download a real calibration dataset (<50MB total) from GitHub Releases or examples/datasets/
  3. User can load example datasets via convenience function `aquacal.datasets.load_example()`
  4. Larger real datasets (>10MB) are hosted on Zenodo with DOI and download instructions
**Plans**: TBD

Plans:
- [ ] 04-01-PLAN: TBD

### Phase 5: Documentation Site
**Goal**: Comprehensive documentation site with auto-generated API reference and user guide hosted on Read the Docs
**Depends on**: Phase 4
**Requirements**: THEO-01, THEO-02, THEO-03
**Success Criteria** (what must be TRUE):
  1. User can read refractive geometry explanation covering Snell's law, ray tracing, and projection model
  2. User can read coordinate convention documentation covering world frame (Z-down), camera frame (OpenCV), interface normal
  3. User can read optimizer documentation covering four-stage pipeline, parameter layout, sparse Jacobian strategy, loss functions
  4. API reference is auto-generated from docstrings via Sphinx autodoc with napoleon extension
  5. CITATION.cff file exists in repository root with BibTeX-compatible metadata
  6. README includes "How to Cite" section with DOI and BibTeX entry
**Plans**: TBD

Plans:
- [ ] 05-01-PLAN: TBD

### Phase 6: Interactive Tutorials
**Goal**: Jupyter notebook tutorials demonstrate end-to-end calibration, visualization, and comparison workflows
**Depends on**: Phase 5
**Requirements**: TUT-01, TUT-02, TUT-03, NB-01, NB-02, NB-03
**Success Criteria** (what must be TRUE):
  1. User can run a notebook demonstrating full pipeline: config creation, intrinsic calibration, extrinsic estimation, joint optimization, validation
  2. User can run a notebook demonstrating calibration result visualization and diagnostics interpretation
  3. User can run a notebook demonstrating synthetic data generation and ground-truth validation
  4. Tutorial explains common failure modes (insufficient overlap, degenerate board poses, interface distance convergence)
  5. All notebooks execute end-to-end without manual data preparation using Phase 4 datasets
  6. Notebooks are integrated into Sphinx documentation via nbsphinx for rendered HTML display
**Plans**: TBD

Plans:
- [ ] 06-01-PLAN: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation and Cleanup | 3/3 | Complete | 2026-02-14 |
| 2. CI/CD Automation | 0/3 | Not started | - |
| 3. Public Release | 0/TBD | Not started | - |
| 4. Example Data | 0/TBD | Not started | - |
| 5. Documentation Site | 0/TBD | Not started | - |
| 6. Interactive Tutorials | 0/TBD | Not started | - |
