# Pitfalls Research

**Domain:** Python scientific library release (computer vision/calibration)
**Researched:** 2026-02-14
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Breaking Backward Compatibility Without Deprecation Warnings

**What goes wrong:**
API-breaking changes (parameter renames, function signature changes, removed features) are released without prior warning, causing user code to immediately break on minor version updates. This is especially damaging for research reproducibility where users pin to "library>=X.Y" expecting SemVer semantics.

**Why it happens:**
Developers focus on improving the API and fixing design mistakes, but don't realize that researchers rely on exact reproducibility of published results. The "it's alpha/beta" mindset leads to breaking changes in minor versions even after 1.0.

**How to avoid:**
- Implement two-release deprecation cycle: warn in version N, remove in N+2
- Use `warnings.warn(..., DeprecationWarning)` for any API that will change
- Document deprecations prominently in CHANGELOG and release notes
- Consider using `@deprecated` decorator from the `Deprecated` library for automatic warnings
- Never break API in patch versions (X.Y.Z), only in minor (X.Y) or major (X)

**Warning signs:**
- GitHub issues titled "code broke after update"
- Users asking "how do I pin to old version?"
- Complaints about "working yesterday, broken today"
- High number of issues opened immediately after release

**Phase to address:**
Phase 1 (Package Infrastructure) - establish deprecation policy and tooling before first PyPI release

---

### Pitfall 2: Undocumented or Missing Installation Instructions for Scientific Ecosystem

**What goes wrong:**
Installation instructions assume `pip install` works universally, but fail to document:
- Platform-specific binary dependencies (OpenCV, scipy compiled components)
- Conda vs pip installation differences
- Python version constraints not reflected in `requires-python`
- Missing system dependencies (e.g., ffmpeg for video processing)
- Incompatibility with common scientific stacks (older NumPy, specific OpenCV versions)

Users waste hours debugging installation, give up, or create GitHub issues that could have been prevented.

**Why it happens:**
Developers test on their own machine where dependencies are already installed. Documentation written after the fact. Assumption that "it works on my machine" means it works everywhere.

**How to avoid:**
- Test installation in fresh virtual environments on all three major platforms (Linux, macOS, Windows)
- Document both pip and conda installation paths
- List system dependencies explicitly (e.g., "Requires video codec support")
- Add "Known Issues" section for platform-specific gotchas
- Use GitHub Actions to test installation on multiple platforms/Python versions
- Consider providing conda-forge recipe in addition to PyPI package
- Document minimum versions for heavy dependencies (NumPy, OpenCV, scipy)

**Warning signs:**
- Installation issues dominate GitHub issue tracker
- Users asking basic "how to install" questions
- CI passes but users report installation failures
- Issues like "ModuleNotFoundError" or "ImportError: DLL load failed"

**Phase to address:**
Phase 1 (Package Infrastructure) - verify cross-platform installation before public release
Phase 3 (Documentation) - document all installation paths and requirements

---

### Pitfall 3: Binary Wheel Platform Incompatibility (manylinux)

**What goes wrong:**
Package uploads source distribution to PyPI but no binary wheels, forcing users to compile from source. Or, binary wheels target wrong manylinux platform (e.g., manylinux_2_28 not available on Ubuntu 20.04), causing cryptic "no matching distribution" errors.

For packages with C/C++ extensions or dependencies like OpenCV, NumPy, scipy, this causes:
- Installation failures on older Linux distributions
- 10+ minute compile times for users
- Platform-specific compilation errors users can't debug

**Why it happens:**
PyPI accepts source distributions without wheels. Developers don't realize their package has compiled dependencies (transitively through OpenCV, scipy). manylinux versioning is confusing (manylinux1 vs 2010 vs 2014 vs 2_17 vs 2_28).

**How to avoid:**
- Use `cibuildwheel` to build multi-platform wheels in CI
- Target manylinux2014 (widest compatibility) unless specific need for newer
- Test wheel installation on older LTS distributions (Ubuntu 20.04, not just 24.04)
- Run `auditwheel show <wheel>` to verify manylinux compliance
- For pure Python packages, verify no compiled dependencies sneak in
- Document whether package is pure Python or requires compilation

**Warning signs:**
- Installation works on developer machine but not user systems
- Issues titled "no matching distribution found"
- Users reporting 10+ minute pip install times (compiling from source)
- Errors mentioning "legacy-install-failure" or "setup.py failed"

**Phase to address:**
Phase 1 (Package Infrastructure) - configure wheel building before first PyPI release
Phase 5 (CI/CD) - automate wheel builds for all releases

---

### Pitfall 4: README/PyPI Metadata Formatting Failures

**What goes wrong:**
Package uploads successfully to PyPI, but the project description page shows:
- Raw unformatted text instead of rendered Markdown/RST
- Error messages like "long_description has syntax errors in markup"
- Missing images (relative paths don't work on PyPI)
- Sphinx directives like `:py:func:` causing rendering errors

Users see an unprofessional, unreadable package page and assume the library is low-quality or unmaintained.

**Why it happens:**
PyPI has stricter markup requirements than GitHub. Developers write README for GitHub, don't test PyPI rendering. Forgot to set `long_description_content_type = "text/markdown"` in setup configuration.

**How to avoid:**
- Set `long_description_content_type = "text/markdown"` in pyproject.toml (or setup.py)
- Use `twine check dist/*` before uploading to verify markup is valid
- Test on TestPyPI before uploading to production PyPI
- Avoid Sphinx-specific directives in README (they work on ReadTheDocs, not PyPI)
- Use absolute URLs for images, not relative paths
- Keep summary (description field) under 512 characters

**Warning signs:**
- PyPI page shows raw markup instead of formatted text
- Images missing on PyPI but present on GitHub
- Error during `twine upload`: "long_description has syntax errors"

**Phase to address:**
Phase 1 (Package Infrastructure) - validate metadata before first release
Phase 3 (Documentation) - ensure README renders on PyPI

---

### Pitfall 5: Overly Strict Dependency Pinning (Upper Bounds)

**What goes wrong:**
Package specifies dependencies like `numpy>=1.24,<2.0` or `opencv-python==4.6.0.66` (exact pin). This causes:
- Dependency conflicts when users have newer NumPy installed
- Installation failures in environments with other packages requiring newer versions
- Package becomes unusable in newer ecosystems (Python 3.12+, NumPy 2.0)
- "Dependency hell" where resolver can't find compatible versions

**Why it happens:**
Fear that new versions will break code. Copying patterns from application development (where pinning is appropriate). Not understanding the difference between library and application dependency management.

**How to avoid:**
- For libraries: use lower bounds only (`numpy>=1.20`) or compatible release (`numpy~=1.24.0`)
- Only add upper bounds for *known* incompatibilities (e.g., `numpy<2.0` only if actually incompatible)
- Test against wide range of dependency versions in CI (use matrix testing)
- Pin dependencies in lock files for reproducibility, not in package metadata
- Follow guidance: "Libraries should be permissive, applications can be strict"
- Document tested version ranges in README, but don't enforce in requirements

**Warning signs:**
- Installation fails with "could not find compatible versions"
- Users asking "how do I use this with NumPy 2.0?"
- Package works alone but conflicts with other libraries in environment
- Issues titled "dependency conflict with [common library]"

**Phase to address:**
Phase 1 (Package Infrastructure) - set permissive dependencies before first release
Phase 5 (CI/CD) - test against multiple dependency versions in CI matrix

---

### Pitfall 6: Example Data and Datasets Missing or Inaccessible

**What goes wrong:**
Documentation shows "Quick Start" tutorials that require example datasets, but:
- Datasets not included in package (too large)
- Download links broken or require authentication
- No instructions for obtaining sample data
- Examples use proprietary/private data users can't access
- Synthetic data generation not documented

Users can't run examples, can't verify installation works, can't learn the API. Library appears unusable.

**Why it happens:**
Developers use their own real datasets during development. Example data is an afterthought. Concern about package size if data included. Licensing unclear for distributing calibration board images.

**How to avoid:**
- Provide synthetic data generation functions (e.g., `aquacal.examples.generate_sample_data()`)
- Host small example datasets (<10MB) on GitHub Releases or Zenodo with DOI
- Document how to create test data from scratch (e.g., print ChArUco board PDF)
- Include minimal test data in package for smoke tests (excluded from source dist if too large)
- For calibration libraries: provide board generation utilities and recommend print services
- Add `aquacal.datasets.load_example()` convenience function

**Warning signs:**
- Issues asking "where can I get example data?"
- Users unable to run Quick Start tutorial
- High bounce rate on documentation (users read intro, can't proceed)
- Forum questions "how do I test this?"

**Phase to address:**
Phase 2 (Example Datasets) - create small synthetic/real example datasets
Phase 3 (Documentation) - tutorials use only provided example data
Phase 4 (Example Notebooks) - notebooks download data automatically or generate synthetically

---

### Pitfall 7: Missing or Incomplete Migration Guides for Version Updates

**What goes wrong:**
New version changes API (even with deprecation warnings), but no migration guide explaining:
- What changed and why
- How to update existing code
- Side-by-side comparison of old vs new API
- What functionality was removed and alternatives

Users see deprecation warnings, don't know how to fix them, stay on old version indefinitely. Library fragments into "old version" and "new version" user bases.

**Why it happens:**
Developers know the changes (they implemented them), assume users will figure it out from CHANGELOG. Migration guides feel like extra work after coding is done.

**How to avoid:**
- Write migration guide before releasing breaking changes
- Include in documentation (docs/migration/v1-to-v2.md)
- Link to migration guide in deprecation warning messages
- Use code examples showing before/after
- List removed features with recommended alternatives
- Provide automated migration tools if feasible (e.g., script to update config files)

**Warning signs:**
- Users asking "how do I upgrade from v0.X to v1.0?"
- Many users staying on old versions despite bugs/missing features
- GitHub issues requesting "bring back old API"
- Confusion about which tutorial/example applies to which version

**Phase to address:**
Phase 6+ (Future versions) - required for any backward-incompatible release
Phase 3 (Documentation) - establish template for migration guides

---

### Pitfall 8: No Citation Metadata for Academic Credit

**What goes wrong:**
Researchers use the library in published work, want to cite it, but:
- No CITATION.cff file in repository
- No DOI assigned (Zenodo)
- No BibTeX entry in documentation
- Unclear how to cite in papers
- No academic paper/preprint describing methodology

Result: library gets used but not cited, or cited incorrectly (just GitHub URL). Developers don't get academic credit for research software work.

**Why it happens:**
Developers from software background don't realize citation is expected in research. Zenodo/CITATION.cff are unknown. Feels like self-promotion.

**How to avoid:**
- Create `CITATION.cff` file in repository root
- Link GitHub repository to Zenodo for automated DOI assignment
- Include "How to Cite" section in README
- Provide BibTeX entry in documentation
- Consider writing methodology paper (JOSS, SoftwareX) for citable publication
- Update CITATION.cff for major releases (versioned DOIs)

**Warning signs:**
- Users asking "how should I cite this?"
- No Google Scholar citations despite usage
- Work used in papers but only mentioned in acknowledgments, not cited

**Phase to address:**
Phase 3 (Documentation) - add CITATION.cff and citation instructions
Phase 6+ (Optional) - publish methodology paper for enhanced citability

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded paths in examples | Examples run immediately | Examples don't work on other machines, Windows vs Unix paths | Never - use pathlib, relative paths, or programmatic path construction |
| Unpinned dev dependencies | Simpler pyproject.toml | Tests fail unpredictably when new pytest/mypy released | Never - pin dev dependencies in optional group |
| No type hints | Faster initial development | Users have poor IDE autocomplete, runtime errors | Only for rapid prototyping, add before v1.0 |
| Single-platform testing | Faster CI, simpler setup | Linux-only users, Windows users can't install | Never for public release - test all platforms |
| Global state in modules | Simpler API | Thread-unsafe, testing difficulties, conflicts in multi-use | Never - use class instances or context managers |
| Example-only documentation | Quick to write initial docs | Users confused about real API, can't generalize | Acceptable for v0.1, inadequate for v1.0 |

## Integration Gotchas

Common mistakes when integrating scientific Python ecosystem.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenCV | Requiring exact opencv-python version | Use `opencv-python>=4.6` (lower bound only), test against 4.6, 4.7, 4.8+ |
| NumPy | Not testing against NumPy 2.0 | Test in CI matrix: NumPy 1.24 (oldest), 1.26, 2.0+ (newest) |
| Matplotlib | Assuming interactive backend available | Use `matplotlib.use('Agg')` for non-interactive contexts, document backend requirements |
| scipy | Using deprecated scipy.optimize functions | Check scipy deprecation warnings, migrate to current API |
| Video codecs | Assuming H.264 works everywhere | Document codec requirements, test with multiple formats, provide troubleshooting for missing codecs |
| YAML parsing | Using unsafe PyYAML loader | Use `yaml.safe_load()`, never `yaml.load()` (security issue) |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading all frames into memory | Simple code for video processing | Out-of-memory errors, slow startup | >500 frames, high-resolution video |
| Nested Python loops for optimization | Easy to prototype | Optimization takes hours | >50 parameters, >1000 residuals |
| Generating Jacobian without sparsity | Works for small problems | Slow convergence, excessive FD evaluations | >100 parameters |
| Dense matrix operations on sparse data | NumPy/SciPy easy to use | Memory explosion, slow solve | >10000x10000 Jacobian |
| Re-detecting in every calibration run | Ensures fresh data | User waits 10 min each run | >100 frames across multiple videos |
| No frame budget limit | Uses all available data | Optimizer too slow, diminishing returns | >500 calibration frames |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Using `yaml.load()` instead of `yaml.safe_load()` | Arbitrary code execution from config files | Always use `yaml.safe_load()`, document in code review checklist |
| Loading pickled data from untrusted sources | Code execution from malicious .pkl files | Use JSON for serialization, warn users about pickle risks |
| Shell injection in video path handling | User-provided paths executed in shell | Use subprocess with list arguments, never shell=True with string interpolation |
| Logging sensitive paths | Calibration paths contain usernames, private info | Sanitize logs, use relative paths in error messages |

## UX Pitfalls

Common user experience mistakes in scientific libraries.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Cryptic error messages | User has no idea what failed or how to fix | Validate inputs early, provide actionable error messages with examples |
| Silent failures | Calibration completes with nonsense results | Add sanity checks, warn if RMS error > threshold, fail loudly on degeneracy |
| No progress indication | User doesn't know if optimizer is running or hung | Use tqdm for long operations, log optimizer iterations if --verbose |
| Requiring manual config file creation | High barrier to entry | Provide `aquacal init` to generate template config from video directories |
| Unclear coordinate conventions | User gets Z flipped, coordinates don't match expectation | Document coordinate system prominently with diagrams, provide validation utilities |
| No way to visualize results | User can't tell if calibration is good | Save diagnostic plots automatically, provide visualization utilities |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Package uploaded to PyPI:** Often missing wheel builds — verify binary wheels exist for Linux/macOS/Windows on PyPI download page
- [ ] **Documentation online:** Often missing API reference — verify all public functions documented with examples, not just tutorials
- [ ] **Example notebooks:** Often missing runnable data — verify notebooks execute end-to-end without manual data preparation
- [ ] **CI passing:** Often missing platform matrix — verify tests run on all three OS, multiple Python versions (3.10, 3.11, 3.12)
- [ ] **Tests at 100% coverage:** Often missing edge cases — verify error handling, invalid inputs, boundary conditions tested
- [ ] **README has examples:** Often missing dependencies — verify example code actually runs in fresh environment
- [ ] **CHANGELOG updated:** Often missing migration notes — verify breaking changes have before/after code examples
- [ ] **Version tagged:** Often missing GitHub release — verify tag has release notes, not just version number

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Backward incompatibility released | HIGH | 1. Issue patch release reverting change, 2. Deprecate in next minor version, 3. Post apology/explanation in GitHub announcement, 4. Update SemVer policy documentation |
| Broken wheels on PyPI | MEDIUM | 1. Yank broken release from PyPI, 2. Fix wheel build config, 3. Release patch version, 4. Update installation docs with workaround for affected versions |
| README not rendering | LOW | 1. Fix markup, 2. Release patch version (metadata-only), 3. PyPI updates automatically |
| Overly strict dependencies | MEDIUM | 1. Relax version constraints in patch release, 2. Add CI matrix testing to prevent recurrence, 3. Document tested version ranges |
| Missing example data | MEDIUM | 1. Upload dataset to GitHub Releases/Zenodo, 2. Update docs with download links, 3. Consider adding synthetic data generation |
| Security vulnerability (YAML) | HIGH | 1. Patch immediately, 2. Release emergency update, 3. Post security advisory on GitHub, 4. Notify downstream users via mailing list/community channels |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Breaking compatibility | Phase 1 - establish deprecation policy | PR template includes deprecation checklist |
| Installation failures | Phase 1, 5 - CI tests installation on fresh VMs | CI matrix includes Ubuntu 20.04/22.04, macOS, Windows |
| Binary wheel issues | Phase 1, 5 - configure cibuildwheel | PyPI page shows wheels for cp310-cp312, all platforms |
| PyPI metadata | Phase 1 - validate with twine check | TestPyPI upload before production |
| Dependency conflicts | Phase 1, 5 - permissive bounds, matrix testing | CI tests with oldest and newest compatible versions |
| Missing example data | Phase 2 - create datasets | Documentation examples run without manual data setup |
| No migration guide | Phase 3 - template created | Major version releases include migration docs |
| No citation metadata | Phase 3 - add CITATION.cff | Repository has Zenodo badge, citation widget on GitHub |
| Cryptic errors | Phase 4 - improve error messages in notebooks | User testing feedback session |
| Silent failures | Phase 5 - add validation checks | Test suite includes known-bad inputs that should fail |

## Sources

### Official Documentation and Guides
- [Python Packaging User Guide - Making a PyPI-friendly README](https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
- [PEP 387 - Backwards Compatibility Policy](https://peps.python.org/pep-0387/)
- [PEP 440 - Version Identification and Dependency Specification](https://peps.python.org/pep-0440/)
- [PEP 513 - manylinux Platform Tags](https://peps.python.org/pep-0513/)
- [Scientific Python Development Guide - Writing Documentation](https://learn.scientific-python.org/development/guides/docs/)
- [Scientific Python Development Guide - GitHub Actions](https://learn.scientific-python.org/development/guides/gha-basic/)

### Backward Compatibility and Deprecation
- [Deprecated library documentation](https://deprecated.readthedocs.io/en/latest/introduction.html)
- [pyDeprecate - Deprecation management](https://borda.github.io/pyDeprecate/)
- [PEP 702 - Marking deprecations using the type system](https://peps.python.org/pep-0702/)

### Dependency Management and Reproducibility
- [Should You Use Upper Bound Version Constraints?](https://iscinumpy.dev/post/bound-version-constraints/)
- [Dependency management - Python for Scientific Computing](https://aaltoscicomp.github.io/python-for-scicomp/dependencies/)
- [Python's Reproducibility Crisis](https://www.leahwasser.com/blog/2025/2025-09-15-reproducibility-python-environments/)
- [Reproducible and upgradable Conda environments with conda-lock](https://pythonspeed.com/articles/conda-dependency-management/)

### Binary Wheels and Platform Compatibility
- [PyPA manylinux repository](https://github.com/pypa/manylinux)
- [Controlling Python Wheel Compatibility](https://www.vinnie.work/blog/2020-10-12-python-wheel-manylinux)
- [PEP 600 - Future manylinux Platform Tags](https://peps.python.org/pep-0600/)

### OpenCV Compatibility Issues (Real-world Examples)
- [OpenCV 4.7.0.68 ZLIB backward compatibility issue](https://github.com/opencv/opencv-python/issues/765)
- [ChArUco pattern breaking change in OpenCV 4.6.0](https://github.com/opencv/opencv/issues/23152)
- [OpenCV 4.0 binary compatibility changes](https://github.com/opencv/opencv/wiki/OE-4.-OpenCV-4)

### Academic Citation and Credit
- [Zenodo CITATION.cff file guide](https://help.zenodo.org/docs/github/describe-software/citation-file/)
- [Citation File Format (CFF)](https://citation-file-format.github.io/)
- [pyOpenSci - How to Add a Citation to Your Code](https://www.pyopensci.org/lessons/package-share-code/publish-share-code/cite-code.html)

### Reproducibility and Documentation
- [Ten Simple Rules for Reproducible Research in Jupyter Notebooks](https://arxiv.org/pdf/1810.08055)
- [Research Software Engineering with Python](https://third-bit.com/py-rse/)
- [NYU Research Guides - Software Reproducibility](https://guides.nyu.edu/software-reproducibility/documentation)

### Installation and Environment Management
- [setuptools Development Mode documentation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
- [PEP 660 - Editable installs for pyproject.toml](https://peps.python.org/pep-0660/)

---

*Pitfalls research for: Python scientific library release (AquaCal refractive multi-camera calibration)*
*Researched: 2026-02-14*
