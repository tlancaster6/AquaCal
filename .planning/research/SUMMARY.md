# Project Research Summary

**Project:** AquaCal PyPI Public Release
**Domain:** Scientific Python Camera Calibration Library (Underwater Computer Vision)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Executive Summary

AquaCal is a refractive multi-camera calibration library for underwater computer vision, targeting the scientific Python research community. The recommended release approach follows established scientific Python ecosystem patterns: PyPI distribution via setuptools with Trusted Publishing, Sphinx documentation hosted on Read the Docs, comprehensive API reference with autodoc, and Jupyter notebook tutorials. The library should prioritize research reproducibility through citation metadata (CITATION.cff, Zenodo DOI), example datasets, and rigorous testing across Python versions (3.10-3.12).

The recommended stack centers on modern Python packaging best practices (setuptools >=61, build + twine, GitHub Actions CI/CD), documentation infrastructure (Sphinx with autodoc/napoleon, pydata-sphinx-theme, Read the Docs hosting), and code quality tooling (ruff replacing black+flake8+isort, mypy, pytest with coverage). This stack is already partially in place (pytest, setuptools), requiring primarily additive work rather than migration. The key differentiators for AquaCal are refractive geometry modeling (unique to this library), comprehensive diagnostic visualizations, and synthetic data generation for validation.

Critical risks center on backward compatibility management (no deprecation policy yet established), cross-platform installation verification (OpenCV/NumPy dependencies), and missing example datasets preventing tutorial execution. The highest-priority mitigation is establishing a deprecation policy and testing installation on fresh environments across all platforms before the v1.0 PyPI release. Secondary risks include documentation completeness (ensuring all public API has docstrings) and dependency pinning (avoiding overly strict upper bounds that cause conflicts). Following the recommended six-phase roadmap addresses these risks systematically while maintaining momentum toward public release.

## Key Findings

### Recommended Stack

Scientific Python library packaging in 2025/2026 follows a standardized stack centered on setuptools for build backend, Sphinx for documentation, and GitHub Actions for CI/CD. The recommended stack leverages AquaCal's existing infrastructure (setuptools, pytest) while adding modern tooling to meet community expectations.

**Core technologies:**
- **setuptools >=61.0**: Build backend already in use, supports pyproject.toml [project] table, no migration needed, widest compatibility
- **Sphinx >=7.0 + Read the Docs**: Documentation generator and hosting, scientific Python standard, autodoc for API reference from docstrings, free hosting for open source
- **ruff >=0.8.0**: Linter and formatter, replaces black+flake8+isort, 10-100x faster (Rust-based), 800+ rules, adopted by SciPy/Pandas/FastAPI
- **GitHub Actions**: CI/CD platform with free tier for public repos, Trusted Publishing support for PyPI, official PyPA-documented workflows
- **pytest + pytest-cov**: Testing framework already in use, coverage.py integration for code coverage reporting
- **Zenodo**: Dataset hosting with DOI support, versioning, scientific credibility, free 50GB/dataset

**Critical version constraints:**
- Python 3.10+ (current minimum, maintain for 3 years per SPEC 0)
- NumPy >=1.20 (permissive lower bound, test against 1.24 and 2.0+ in CI)
- OpenCV >=4.6 (lower bound only, avoid exact pinning)

### Expected Features

Research identified 30 table-stakes features and 17 differentiators. The MVP for v1.0 public release should focus on packaging fundamentals, comprehensive documentation, and minimal example content. Advanced features (JOSS publication, Docker, HDF5 export) should wait until v1.x after community validation.

**Must have (table stakes):**
- PyPI package with `pip install aquacal` working on all platforms
- Sphinx API documentation + user guide hosted on Read the Docs
- Installation instructions for both pip and conda (conda-forge later)
- Docstrings (Google/NumPy style) for all public API
- README badges (build status, coverage, PyPI version, license)
- Example datasets (2-3 small calibration scenarios, <10MB or Zenodo-hosted)
- Jupyter tutorial notebook (end-to-end calibration walkthrough)
- CITATION.cff for academic attribution + Zenodo DOI
- CHANGELOG.md following Keep a Changelog format
- Contributing guide (CONTRIBUTING.md) and Code of Conduct

**Should have (competitive):**
- Refractive geometry modeling (unique core value, already implemented)
- Diagnostic visualizations (reprojection plots, heatmaps, already exists)
- Synthetic data generator (ground-truth validation, already implemented)
- Comparison utilities (`aquacal compare` CLI, already exists)
- Multi-camera support (already implemented)
- Fisheye camera support (already implemented)
- Interface tilt estimation (already implemented)

**Defer (v2+):**
- JOSS publication (wait for 3+ external users to validate)
- Docker container (add when reproducibility requests arrive)
- HDF5 export (only if JSON proves too slow for >1000 frames)
- Performance benchmarks (document after performance questions arise)
- Multi-language bindings (C++/MATLAB, only if consistent demand)

### Architecture Approach

Scientific Python library release infrastructure follows a five-component pattern: documentation site (Sphinx), CI/CD pipeline (GitHub Actions), packaging configuration (pyproject.toml), example content (notebooks + datasets), and artifact distribution (PyPI + Read the Docs + Zenodo). The standard architecture uses autodoc to auto-generate API reference from docstrings, sphinx-gallery or nbsphinx to execute and render examples, Trusted Publishing for secure PyPI releases, and multi-version test matrices (tox/nox) for cross-Python-version validation.

**Major components:**
1. **Documentation Layer**: Sphinx with autodoc/napoleon/intersphinx extensions, pydata-sphinx-theme, renders API reference + tutorials + examples as HTML on Read the Docs
2. **CI/CD Pipeline**: GitHub Actions workflows for test matrix (Python 3.10-3.12), doc builds (on PR), package builds (on tag), PyPI publish (Trusted Publishing via OIDC)
3. **Packaging Configuration**: pyproject.toml with setuptools backend, [project] metadata, [tool.*] configs for pytest/mypy/ruff, permissive dependency bounds
4. **Example Content**: Jupyter notebooks in examples/notebooks/, small datasets in examples/datasets/ or Zenodo-hosted with DOI, executed by nbsphinx
5. **Artifact Distribution**: Wheels + sdist on PyPI, HTML docs on Read the Docs (versioned), datasets on Zenodo (citable DOI)

**Key patterns:**
- **Sphinx + autodoc + napoleon**: Single source of truth in docstrings, auto-generate API docs, parse NumPy-style docstrings
- **Trusted Publishing**: OIDC authentication to PyPI, no API token storage, project-scoped, PyPA-recommended 2025+ practice
- **Multi-version test matrix**: GitHub Actions matrix strategy for Python 3.10/3.11/3.12, validates version support claims
- **Read the Docs integration**: Automated builds on every commit, PR previews, versioned docs, free for open source

### Critical Pitfalls

Research identified eight critical pitfalls and numerous technical debt patterns. The top five risks for AquaCal's release are backward compatibility, installation failures, missing example data, dependency conflicts, and citation metadata gaps.

1. **Breaking Backward Compatibility Without Deprecation Warnings** — Implement two-release deprecation cycle: warn in version N, remove in N+2. Use `warnings.warn(..., DeprecationWarning)` for any API changes. Document deprecations in CHANGELOG and release notes. Never break API in patch versions (X.Y.Z).

2. **Undocumented or Missing Installation Instructions** — Test installation in fresh virtual environments on Linux/macOS/Windows. Document both pip and conda paths. List system dependencies explicitly (video codec support for OpenCV). Add GitHub Actions to test installation on all platforms.

3. **Example Data and Datasets Missing or Inaccessible** — Provide synthetic data generation functions (`aquacal.examples.generate_sample_data()`). Host small datasets (<10MB) on GitHub Releases or Zenodo with DOI. Include minimal test data in package. Add `aquacal.datasets.load_example()` convenience function.

4. **Overly Strict Dependency Pinning (Upper Bounds)** — For libraries, use lower bounds only (`numpy>=1.20`) or compatible release (`numpy~=1.24.0`). Only add upper bounds for known incompatibilities. Test against wide range of dependency versions in CI matrix. Follow "Libraries should be permissive, applications can be strict" guidance.

5. **No Citation Metadata for Academic Credit** — Create CITATION.cff file in repository root. Link GitHub to Zenodo for automated DOI assignment. Include "How to Cite" section in README. Provide BibTeX entry in documentation. Update CITATION.cff for major releases (versioned DOIs).

**Secondary pitfalls:**
- Binary wheel platform incompatibility (use cibuildwheel for multi-platform wheels targeting manylinux2014)
- README/PyPI metadata formatting failures (use `twine check dist/*` before upload, test on TestPyPI)
- Missing migration guides for version updates (write migration guide before releasing breaking changes)

## Implications for Roadmap

Based on research, suggested six-phase structure prioritizes packaging fundamentals, documentation infrastructure, example content, and deployment automation. Phase ordering follows build-order dependencies discovered in architecture research.

### Phase 1: Package Infrastructure Foundation
**Rationale:** pyproject.toml is the foundation for all other work (documentation imports the library, CI tests the package, release builds from metadata). Must be complete and correct before proceeding. Addresses critical pitfall of installation failures by validating cross-platform installation early.

**Delivers:**
- Enhanced pyproject.toml with complete metadata (URLs, classifiers, optional-dependencies for [dev] and [docs])
- CHANGELOG.md following Keep a Changelog format
- Deprecation policy documented in CONTRIBUTING.md
- Validated installation on fresh environments (Linux/macOS/Windows, Python 3.10-3.12)

**Addresses:**
- PyPI package (table stakes feature)
- Semantic versioning (table stakes feature)
- Backward compatibility (critical pitfall #1)
- Installation failures (critical pitfall #2)
- Dependency conflicts (critical pitfall #4)

**Avoids:**
- Breaking backward compatibility without deprecation warnings
- Undocumented installation instructions
- Overly strict dependency pinning (use permissive bounds)

### Phase 2: Example Datasets and Sample Data
**Rationale:** Documentation and tutorials (Phase 3-4) depend on example datasets being available. Creating datasets before writing docs ensures all examples are executable. Addresses critical pitfall #3.

**Delivers:**
- 2-3 synthetic calibration scenarios (small rig 2-3 cameras, medium rig 5-7 cameras, large rig 10+ cameras)
- Small example dataset (<10MB) in examples/datasets/ or GitHub Releases
- Larger datasets hosted on Zenodo with DOI
- Synthetic data generation utility (`aquacal.examples.generate_sample_data()`)
- Dataset loading convenience function (`aquacal.datasets.load_example()`)

**Addresses:**
- Example datasets (table stakes feature)
- Example data missing (critical pitfall #3)

**Avoids:**
- Tutorials that can't be executed
- Datasets too large for git (>10MB goes to Zenodo)

### Phase 3: Documentation Site (Sphinx + Read the Docs)
**Rationale:** Documentation is the primary user entry point for scientific libraries. Sphinx API documentation is a table-stakes expectation. Read the Docs provides free hosting and versioning. Depends on Phase 1 (package must be importable) and Phase 2 (examples need data).

**Delivers:**
- docs/ structure with Sphinx configuration (conf.py)
- API reference auto-generated from docstrings (autodoc + napoleon)
- User guide (installation, quickstart, concepts, advanced topics)
- .readthedocs.yml configuration for automated builds
- CITATION.cff for academic attribution
- "How to Cite" section in README

**Addresses:**
- API documentation (table stakes feature)
- User guide (table stakes feature)
- Installation instructions (table stakes feature)
- Citation metadata (table stakes + critical pitfall #5)
- Docstring coverage (table stakes feature)

**Avoids:**
- No citation metadata for academic credit (pitfall #5)
- README/PyPI metadata formatting failures (validate with twine check)

**Uses:**
- Sphinx >=7.0 with autodoc, napoleon, intersphinx, mathjax extensions
- pydata-sphinx-theme (NumPy/SciPy standard)
- Read the Docs (free hosting, versioned docs)

### Phase 4: Example Notebooks and Tutorials
**Rationale:** Jupyter notebooks are how researchers learn new libraries. Depends on Phase 2 (datasets) and Phase 3 (documentation infrastructure). Notebooks should be integrated into Sphinx docs via nbsphinx for rendering.

**Delivers:**
- examples/notebooks/basic_calibration.ipynb (end-to-end walkthrough)
- examples/notebooks/visualization.ipynb (diagnostic plots)
- examples/notebooks/comparison.ipynb (multi-calibration comparison)
- nbsphinx integration in Sphinx docs
- Notebooks execute end-to-end without manual data preparation

**Addresses:**
- Jupyter notebook tutorials (table stakes feature)
- Example usage (table stakes feature)

**Avoids:**
- Example-only documentation (include conceptual user guide)
- Notebooks that require unavailable data (use Phase 2 datasets)

**Implements:**
- Example Content component (Architecture: notebooks + datasets)

### Phase 5: CI/CD and Quality Automation
**Rationale:** Automated testing across Python versions and platforms is required before public release. GitHub Actions workflows for test matrix, doc builds, and PyPI publishing. Depends on Phase 1-4 (tests need working package, docs, examples).

**Delivers:**
- .github/workflows/test.yml (multi-version matrix: Python 3.10, 3.11, 3.12)
- .github/workflows/docs.yml (build docs on PR, catch doc build errors)
- .github/workflows/release.yml (PyPI publish on tag via Trusted Publishing)
- pre-commit configuration with ruff, mypy
- codecov integration for coverage reporting

**Addresses:**
- Basic tests (table stakes feature)
- GitHub Actions CI/CD (recommended stack)
- Multi-platform testing (installation pitfall #2)
- Dependency version testing (dependency conflict pitfall #4)

**Avoids:**
- Single-platform testing (test on Linux/macOS/Windows)
- Overly strict dependencies (test against min and max versions)
- Binary wheel issues (configure cibuildwheel for multi-platform wheels)

**Uses:**
- GitHub Actions (CI/CD platform, Trusted Publishing support)
- pytest + pytest-cov (already in use)
- ruff (replaces black+flake8+isort)
- mypy (static type checking)

### Phase 6: PyPI Release and Community Files
**Rationale:** Final polish before v1.0 public release. Create GitHub Release, publish to PyPI via Trusted Publishing, announce to community. Depends on all prior phases (package, docs, examples, CI must be complete).

**Delivers:**
- PyPI v1.0.0 release (built via GitHub Actions)
- GitHub Release with CHANGELOG excerpt
- Zenodo DOI for v1.0.0
- CONTRIBUTING.md (development setup, PR guidelines)
- CODE_OF_CONDUCT.md (PSF Code of Conduct)
- README badges (build status, coverage, PyPI version, license, DOI)
- TestPyPI trial upload (validate before production)

**Addresses:**
- PyPI package (table stakes feature)
- Contributing guide (table stakes feature)
- Code of Conduct (table stakes feature)
- README badges (table stakes feature)
- Zenodo DOI (citation metadata)

**Avoids:**
- README/PyPI metadata formatting failures (test on TestPyPI first)
- Breaking compatibility in first release (establish deprecation policy)

**Uses:**
- pypa/gh-action-pypi-publish (Trusted Publishing, no API tokens)
- twine check (validate metadata before upload)
- Zenodo GitHub integration (automated DOI)

### Phase Ordering Rationale

- **Phase 1 must come first**: pyproject.toml is the foundation; documentation imports the library, CI tests the package, release builds from metadata. Can't proceed without correct packaging.
- **Phase 2 before 3-4**: Documentation and notebooks depend on example datasets being available. Writing tutorials without data to test them is error-prone.
- **Phase 3 before 4**: Sphinx infrastructure (docs/) must exist before integrating notebooks via nbsphinx. API reference provides context for tutorials.
- **Phase 5 before 6**: CI/CD validation (test matrix, doc builds) must pass before public PyPI release. Can't release without cross-platform testing.
- **Phase 6 is the gate**: Public release only after all quality gates (tests, docs, examples, CI) are passing.

**Parallelization opportunities:**
- Phase 2 and Phase 3 can partially overlap (start Sphinx setup while datasets are being created)
- Phase 4 can start as soon as Phase 2 completes (doesn't strictly need full Sphinx docs)
- Phase 5 workflows can be drafted in parallel with Phase 3-4 (but not validated until those complete)

### Research Flags

**Phases with well-documented patterns (skip research-phase):**
- **Phase 1: Package Infrastructure**: Standard pyproject.toml patterns, well-documented in Python Packaging Guide
- **Phase 3: Documentation Site**: Sphinx is extremely well-documented, scientific-python.org has comprehensive guides
- **Phase 5: CI/CD**: GitHub Actions for Python is extensively documented, PyPA provides official workflows
- **Phase 6: PyPI Release**: Trusted Publishing is official PyPA pattern, step-by-step guides available

**Phases potentially needing targeted research:**
- **Phase 2: Example Datasets**: May need research into optimal synthetic data generation for refractive calibration (domain-specific), but can defer to execution phase with domain knowledge available in CLAUDE.md/MEMORY.md
- **Phase 4: Example Notebooks**: nbsphinx vs sphinx-gallery decision may need brief research (but both are well-documented)

**Recommendation:** All phases have sufficient documentation to proceed without `/gsd:research-phase`. Domain knowledge already captured in dev/DESIGN.md, dev/GEOMETRY.md, and MEMORY.md. Execution phase can use standard web search for specific tool questions (e.g., "how to configure nbsphinx for Sphinx 7.0").

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All recommendations from official PyPA documentation, Scientific Python Development Guide, and widely-adopted tools (Sphinx, pytest, GitHub Actions). NumPy/SciPy/pandas use identical stack. |
| Features | HIGH | Table stakes derived from JOSS requirements, scientific Python ecosystem analysis (scikit-image, scipy), and pyOpenSci packaging guide. Differentiators validated against domain (refractive calibration is unique). |
| Architecture | HIGH | Standard patterns documented in Scientific Python Development Guide, Read the Docs official docs, PyPA guides. Trusted Publishing is official PyPA recommendation for 2025+. |
| Pitfalls | MEDIUM-HIGH | Pitfalls derived from official PEP documents (PEP 387 backward compatibility, PEP 440 versioning), real-world issue trackers (OpenCV compatibility issues), and scientific Python best practices. Some are inferred from common complaints but well-validated. |

**Overall confidence:** HIGH

All four research files drew from official documentation (PyPA, Scientific Python, Sphinx, Read the Docs) and established community standards (JOSS requirements, pyOpenSci guides). The recommended stack is proven by adoption in major scientific Python libraries (NumPy, SciPy, pandas, scikit-image). Pitfalls are documented in PEPs and observable in real-world issue trackers.

### Gaps to Address

**Conda-forge distribution:** Research focused on PyPI release. Conda-forge is mentioned as "later" but not detailed. This is acceptable for v1.0 (PyPI is sufficient), but should be added to v1.x roadmap if user demand emerges.

**Performance benchmarking methodology:** Research mentions documenting runtime characteristics but doesn't specify how. This can be addressed during execution with simple timing scripts (not blocking for v1.0).

**JOSS publication timeline:** Research recommends "after 3+ external users" but doesn't specify how to track this. Can use PyPI download stats and GitHub issues/stars as proxies.

**Notebook execution environment:** nbsphinx vs sphinx-gallery decision deferred. Both are viable; nbsphinx is simpler for existing .ipynb files. Decision can be made during Phase 4 based on notebook complexity.

**Dataset licensing:** Research mentions "licensing unclear for distributing calibration board images." ChArUco board patterns are OpenCV-provided (BSD-licensed), safe to distribute. Synthetic data is self-generated (no licensing issue). Can document this in Phase 2.

**Versioning automation:** setuptools-scm and python-semantic-release mentioned as options but not required for v1.0. Manual versioning is acceptable for initial releases. Automation can be added post-v1.0 if release frequency increases.

## Sources

### Primary (HIGH confidence)

**Packaging and Distribution:**
- [Python Packaging User Guide - Publishing with GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) — Official PyPA guide for Trusted Publishing
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) — Official PyPA GitHub Action
- [Python Packaging User Guide - Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) — Official packaging standards
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/) — Versioning specification
- [PEP 513 - manylinux Platform Tags](https://peps.python.org/pep-0513/) — Binary wheel compatibility

**Documentation:**
- [Scientific Python Development Guide - Documentation](https://learn.scientific-python.org/development/guides/docs/) — Scientific Python community standards
- [Sphinx Documentation](https://www.sphinx-doc.org/) — Official Sphinx docs (autodoc, napoleon extensions)
- [Read the Docs Documentation](https://docs.readthedocs.com/) — Official RTD hosting guide
- [Sphinx-Gallery Documentation](https://sphinx-gallery.github.io/stable/index.html) — Example gallery automation

**Testing and Quality:**
- [Scientific Python Development Guide - Coverage](https://learn.scientific-python.org/development/guides/coverage/) — Testing best practices
- [Ruff Documentation](https://docs.astral.sh/ruff/) — Official Ruff linter/formatter docs
- [Scientific Python Development Guide - GitHub Actions](https://learn.scientific-python.org/development/guides/gha-basic/) — CI/CD patterns

**Academic Standards:**
- [JOSS Documentation](https://joss.readthedocs.io/) — Journal of Open Source Software submission requirements
- [Citation File Format](https://citation-file-format.github.io/) — CITATION.cff specification
- [Zenodo](https://zenodo.org/) — Dataset/software archival with DOI

**Backward Compatibility:**
- [PEP 387 - Backwards Compatibility Policy](https://peps.python.org/pep-0387/) — Official Python backward compatibility policy
- [Scientific Python SPEC 0 - Minimum Supported Dependencies](https://scientific-python.org/specs/spec-0000/) — Dependency support timeline

### Secondary (MEDIUM confidence)

**Community Best Practices:**
- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/) — Community packaging best practices
- [Scientific Python Cookiecutter](https://nsls-ii.github.io/scientific-python-cookiecutter/) — Project template and patterns
- [Semantic Versioning 2.0.0](https://semver.org/) — Versioning specification (community standard, not Python-specific)

**Dependency Management:**
- [Should You Use Upper Bound Version Constraints?](https://iscinumpy.dev/post/bound-version-constraints/) — Analysis of dependency pinning practices
- [Dependency management - Python for Scientific Computing](https://aaltoscicomp.github.io/python-for-scicomp/dependencies/) — Scientific computing dependency patterns

**Real-world Examples:**
- [OpenCV Compatibility Issues](https://github.com/opencv/opencv-python/issues) — Real-world platform compatibility issues
- [NumPy Documentation Standards](https://numpy.org/doc/1.19/docs/howto_document.html) — Example of mature scientific Python docs

### Tertiary (LOW confidence)

**Dataset Hosting:**
- [NeurIPS 2025 Data Hosting Guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines) — Academic dataset hosting (NeurIPS-specific, but general principles apply)
- [Zenodo Developer Documentation](https://developers.zenodo.org/) — API details (not needed for basic upload)

**Advanced Automation:**
- [python-semantic-release Documentation](https://python-semantic-release.readthedocs.io/) — Automated versioning (optional, not required for v1.0)
- [setuptools-scm Documentation](https://setuptools-scm.readthedocs.io/) — Version from git tags (optional alternative)

---
*Research completed: 2026-02-14*
*Ready for roadmap: yes*
