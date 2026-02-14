# Feature Landscape: Open-Source Python Scientific/CV Libraries

**Domain:** Scientific Python camera calibration library for underwater computer vision
**Researched:** 2026-02-14
**Confidence:** HIGH

## Table Stakes

Features users expect from a mature scientific Python library. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| PyPI package | Standard distribution method for Python | LOW | `pip install aquacal` must work |
| API documentation (Sphinx) | All scientific Python libs use Sphinx + ReadTheDocs | MEDIUM | Auto-generated from docstrings, hosted on ReadTheDocs |
| Installation instructions | Users need to get started quickly | LOW | README with conda/pip commands |
| Example usage | Researchers learn by example | LOW | Quick-start code snippet in README |
| Docstrings (Google/NumPy style) | Inline documentation standard | MEDIUM | Already present, ensure complete coverage |
| User guide | Explains concepts, not just API | MEDIUM | Tutorial-style narrative documentation |
| Semantic versioning | Expected for API stability signaling | LOW | Already using semver, document policy |
| Open source license (MIT/BSD) | Academic/research community standard | LOW | Already MIT licensed |
| README badges | Quick health indicators | LOW | Build status, coverage, PyPI version, license |
| GitHub repository | Code hosting and issue tracking | LOW | Already exists |
| Basic tests | Quality assurance expectation | MEDIUM | Already exists (pytest suite) |
| Changelog | Track what changed between versions | LOW | HISTORY.md or CHANGELOG.md |
| Contributing guide | How to contribute code/bugs | LOW | CONTRIBUTING.md with dev setup |
| Code of Conduct | Community interaction guidelines | LOW | Standard PSF Code of Conduct |
| CLI interface | Researchers expect command-line tools | MEDIUM | Already exists (`aquacal calibrate`) |
| Example datasets | Users need test data | MEDIUM | Downloadable sample calibration data |
| Citation metadata (CITATION.cff) | Academic attribution | LOW | Enables proper citation in papers |
| JSON/YAML serialization | Standard data interchange | LOW | Already implemented for results |

## Differentiators

Features that set AquaCal apart. Not expected, but valued for underwater CV research.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Refractive geometry modeling | Unique: handles air-water interface | HIGH | Core innovation, already implemented |
| Jupyter notebook tutorials | Interactive learning for researchers | MEDIUM | Step-by-step calibration walkthrough |
| Synthetic data generator | Validation without hardware | MEDIUM | Generate ground-truth test cases |
| Diagnostic visualizations | Debug calibration quality | MEDIUM | Already exists (reprojection plots, heatmaps) |
| Multi-camera support | Enables array calibration | HIGH | Already implemented |
| Comparison utilities | Quantify calibration quality | MEDIUM | Already exists (`aquacal compare`) |
| Detailed error analysis | Research-grade validation | MEDIUM | Per-camera, per-frame, spatial breakdown |
| Flat interface modeling | General underwater case | HIGH | Already implemented (water_z parameter) |
| HDF5 export option | Large dataset archival | LOW | Common in scientific Python (h5py) |
| Docker container | Reproducibility for papers | MEDIUM | Pre-configured environment |
| Performance benchmarks | Transparency about speed/scale | LOW | Document runtime for N cameras/frames |
| Integration tests (synthetic) | End-to-end validation | MEDIUM | Already exists |
| JOSS publication | Academic credibility | HIGH | Peer-reviewed software paper |
| Fisheye camera support | Wide-angle underwater rigs | MEDIUM | Already implemented |
| Interface tilt estimation | Handles non-horizontal surfaces | MEDIUM | Already implemented |

## Anti-Features

Features to explicitly NOT build (prevent scope creep).

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| GUI application | "Easier for non-programmers" | Maintenance burden, not target audience | Good CLI + Jupyter notebooks |
| Real-time calibration | "Live camera feeds" | Not needed for research workflows | Post-processing focus |
| Camera driver integration | "One-stop solution" | Hardware-specific, fragile | Users provide image files |
| Video processing pipeline | "Automatic frame extraction" | Scope creep, ffmpeg exists | Document how to extract frames |
| Cloud-based processing | "No local setup needed" | Infrastructure cost, privacy concerns | Docker for reproducibility |
| Multi-OS GUI installer | "Click to install" | Testing matrix explosion | pip/conda is standard |
| Built-in target generation | "Print calibration boards" | PDFs widely available | Link to external resources |
| Automatic target detection | "No manual annotation" | Already in OpenCV, not core value | Use OpenCV's detector |
| Database backend | "Store all calibrations" | Complexity, JSON/YAML sufficient | File-based results |
| Web dashboard | "Visualize online" | Maintenance burden | Static plots + notebooks |

## Feature Dependencies

```
PyPI Package
    └──requires──> Semantic Versioning
    └──requires──> LICENSE file
    └──requires──> README with install instructions

API Documentation (Sphinx)
    └──requires──> Docstrings (all public functions)
    └──requires──> ReadTheDocs hosting

User Guide
    └──requires──> Example datasets
    └──enhances──> Jupyter notebooks

Jupyter Notebooks
    └──requires──> Example datasets
    └──requires──> Installation instructions

Citation (CITATION.cff)
    └──enhances──> JOSS publication
    └──requires──> DOI (Zenodo)

JOSS Publication
    └──requires──> Documentation (user guide + API)
    └──requires──> Tests
    └──requires──> Example usage
    └──requires──> Installation instructions
    └──requires──> Community guidelines (CONTRIBUTING + CODE_OF_CONDUCT)
    └──requires──> Statement of need
    └──requires──> Citation metadata

Docker Container
    └──requires──> Reproducible environment spec
    └──enhances──> Paper reproducibility

HDF5 Export
    └──conflicts──> "Simple formats only" philosophy
    └──justified-by──> Large multi-camera datasets
```

### Dependency Notes

- **PyPI requires LICENSE + README:** Standard packaging expectations, already met
- **Sphinx requires complete docstrings:** Need audit to ensure all public API documented
- **Jupyter notebooks require example data:** Need hosted downloadable datasets
- **JOSS requires comprehensive docs + community files:** Multiple prerequisites for submission
- **Docker enhances reproducibility:** Not required but valuable for paper submissions
- **HDF5 is optional complexity:** Only add if JSON/YAML proves insufficient for large datasets

## MVP Recommendation

### Launch With (v1.0 - Public Release)

Prioritize for initial PyPI release:

1. **PyPI package** - Must be installable via pip
2. **Sphinx documentation** - API reference + user guide on ReadTheDocs
3. **Example datasets** - 2-3 downloadable calibration scenarios (small/medium/large rigs)
4. **Jupyter tutorial notebook** - End-to-end walkthrough
5. **README badges** - Build, coverage, PyPI version, license
6. **CITATION.cff** - Academic attribution file
7. **Changelog** - CHANGELOG.md following Keep a Changelog format
8. **Community files** - CONTRIBUTING.md + CODE_OF_CONDUCT.md
9. **Zenodo DOI** - Archive first release for citation

### Add After Initial Release (v1.x)

Features to add once core is validated:

10. **JOSS publication** - After docs stabilize and community validates (trigger: 3+ external users)
11. **Docker container** - When reproducibility requests arrive (trigger: paper submissions)
12. **HDF5 export** - If JSON proves too slow for large datasets (trigger: >1000 frames)
13. **Performance benchmarks** - Document runtime characteristics (trigger: performance questions)
14. **Additional synthetic tests** - Expand test coverage (trigger: bug reports)

### Future Consideration (v2+)

Features to defer until product-market fit established:

15. **Multi-language bindings** - C++/MATLAB wrappers (trigger: consistent requests from non-Python users)
16. **Alternative optimization backends** - Ceres, g2o integration (trigger: performance issues)
17. **Advanced camera models** - Omnidirectional, catadioptric (trigger: user demand)
18. **Temporal calibration** - Rolling shutter support (trigger: underwater video requests)
19. **Online calibration** - Incremental updates (trigger: long-term deployment use cases)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Phase |
|---------|------------|---------------------|----------|-------|
| PyPI package | HIGH | LOW | P1 | v1.0 |
| Sphinx docs (API) | HIGH | MEDIUM | P1 | v1.0 |
| User guide | HIGH | MEDIUM | P1 | v1.0 |
| Example datasets | HIGH | MEDIUM | P1 | v1.0 |
| Jupyter tutorial | HIGH | MEDIUM | P1 | v1.0 |
| CITATION.cff | MEDIUM | LOW | P1 | v1.0 |
| README badges | MEDIUM | LOW | P1 | v1.0 |
| Changelog | MEDIUM | LOW | P1 | v1.0 |
| CONTRIBUTING.md | MEDIUM | LOW | P1 | v1.0 |
| CODE_OF_CONDUCT.md | MEDIUM | LOW | P1 | v1.0 |
| Zenodo DOI | MEDIUM | LOW | P1 | v1.0 |
| JOSS publication | HIGH | HIGH | P2 | v1.x |
| Docker container | MEDIUM | MEDIUM | P2 | v1.x |
| Performance benchmarks | MEDIUM | LOW | P2 | v1.x |
| HDF5 export | LOW | LOW | P3 | v2+ |
| Multi-language bindings | LOW | HIGH | P3 | v2+ |
| Alternative optimizers | MEDIUM | HIGH | P3 | v2+ |

**Priority key:**
- P1: Must have for v1.0 public release (table stakes + key differentiators)
- P2: Should have post-release (community validation + adoption drivers)
- P3: Nice to have when demand proven (avoid premature optimization)

## Ecosystem Comparison

Comparing AquaCal to similar libraries in scientific Python ecosystem:

| Feature | OpenCV | Kalibr | scikit-image | SciPy | AquaCal Target |
|---------|--------|--------|--------------|-------|----------------|
| PyPI package | Yes | No (ROS) | Yes | Yes | **Yes** |
| Conda package | Yes | No | Yes | Yes | **Yes** (later) |
| Sphinx docs | Yes | Wiki only | Yes | Yes | **Yes** |
| Example gallery | Yes | No | Yes | No | **Notebooks** |
| CLI tool | Limited | Yes (ROS) | No | No | **Yes** |
| Jupyter tutorials | No | No | Yes | Partial | **Yes** |
| Unit tests | Yes | Yes | Yes | Yes | **Yes** |
| Integration tests | Limited | No | Yes | Yes | **Yes** (synthetic) |
| Code coverage | ~80% | Unknown | >90% | >85% | **>80%** |
| CITATION.cff | No | No | Yes | No | **Yes** |
| JOSS paper | No | No | No | No | **Yes** (planned) |
| Refractive modeling | No | No | No | No | **Yes** (unique) |

**Key insights:**
- **OpenCV:** Comprehensive but lacks refractive modeling, weak documentation structure
- **Kalibr:** Excellent calibration but ROS-dependent, not PyPI-installable, limited docs
- **scikit-image:** Gold standard for scientific Python packaging (follow this model)
- **SciPy:** Exemplary API documentation structure (emulate their user guide + API split)
- **AquaCal:** Combines Kalibr's calibration rigor with scikit-image's packaging quality

## Research Community Expectations (2026)

Based on JOSS submission requirements and scientific Python best practices:

### Documentation Requirements
1. **Statement of need** - Why this software exists (addressed in docs introduction)
2. **Installation instructions** - Clear pip/conda commands
3. **Example usage** - Runnable code snippets
4. **API documentation** - All public functions/classes
5. **Community guidelines** - How to contribute, report issues

### Quality Standards
1. **Tests** - Comprehensive suite with >80% coverage
2. **CI/CD** - GitHub Actions for automated testing
3. **Deprecation policy** - Follow SPEC 0 (support Python versions for 3 years, dependencies for 2 years)
4. **Versioning** - Semantic versioning with clear changelog
5. **Code coverage reporting** - Codecov integration

### Attribution Standards
1. **CITATION.cff** - Machine-readable citation metadata
2. **Zenodo DOI** - Persistent identifier for each release
3. **Authors file** - Credit contributors
4. **License** - MIT/BSD/Apache (permissive for research)

### Reproducibility Standards
1. **Example datasets** - Publicly accessible test data
2. **Synthetic tests** - Ground-truth validation
3. **Environment spec** - requirements.txt / pyproject.toml
4. **Docker** - Optional but increasingly expected for papers

## Sources

### Tool Strategy and Verification
Research followed the hierarchy: Official Docs → GitHub → WebSearch (verified)

**Scientific Python Ecosystem:**
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/)
- [SciPy Lecture Notes](https://scipy-lectures.org/intro/intro.html)
- [SPEC 0 - Minimum Supported Dependencies](https://scientific-python.org/specs/spec-0000/)

**Library Documentation Standards:**
- [scikit-image Documentation](https://scikit-image.org/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [NumPy Documentation Standards](https://numpy.org/doc/1.19/docs/howto_document.html)

**Packaging and Distribution:**
- [Python Packaging Guide - PyPI Release](https://pythonpackaging.info/07-Package-Release.html)
- [Scientific Python - Code Coverage](https://learn.scientific-python.org/development/guides/coverage/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [Python Package Versioning Guide](https://inventivehq.com/blog/python-package-versioning-guide)

**Academic Software Standards:**
- [JOSS Documentation](https://joss.readthedocs.io/)
- [Citation File Format](https://citation-file-format.github.io/)
- [Research Software Citation](https://cite.research-software.org/developers/)
- [Zenodo](https://zenodo.org/)

**Community Guidelines:**
- [Python Software Foundation Code of Conduct](https://policies.python.org/python.org/code-of-conduct/)
- [PyOpenSci Python Package Guide - Contributing Files](https://www.pyopensci.org/python-package-guide/documentation/repository-files/contributing-file.html)
- [PyOpenSci - License Files](https://www.pyopensci.org/python-package-guide/documentation/repository-files/license-files.html)

**Documentation Tools:**
- [Sphinx + ReadTheDocs Tutorial](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/)
- [Python Package Documentation Guide](https://inventivehq.com/blog/python-package-documentation-guide)
- [Sphinx AutoAPI](https://github.com/readthedocs/sphinx-autoapi)

**Testing and Quality:**
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Python Performance Benchmark Suite](https://pyperformance.readthedocs.io/)

**Calibration Library Comparison:**
- [Wide-Angle Camera Calibration Comparative Study](https://arxiv.org/html/2306.09014v2) (Kalibr, OpenCV comparison)
- [Kalibr GitHub Repository](https://github.com/ethz-asl/kalibr)
- [Underwater Refractive Vision Calibration Tool](https://arxiv.org/html/2405.18018v1)

**Data Formats:**
- [Scientific Python Data Formats](https://aaltoscicomp.github.io/python-for-scicomp/data-formats/)
- [HDF5 for Python](https://pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/)

---
*Feature research for: AquaCal public release to underwater CV research community*
*Researched: 2026-02-14*
*Confidence: HIGH (verified with official docs, JOSS requirements, scientific Python standards)*
