# Stack Research: PyPI Release

**Domain:** Python scientific library packaging and distribution
**Researched:** 2026-02-14
**Confidence:** HIGH

## Recommended Stack

### Packaging Core

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| setuptools | >=61.0 (current >=80) | Build backend | Most mature, widest compatibility. Already in use. Supports pyproject.toml [project] table. No migration needed. |
| build | latest | Build tool CLI | PyPA-blessed frontend for creating distributions. Replaces `python setup.py`. Standard practice 2025+. |
| twine | latest | PyPI upload tool | Secure upload over HTTPS. Required if not using Trusted Publishing. Verification before upload. |

### Documentation

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Sphinx | >=7.0 | Documentation generator | Scientific Python standard. Autodoc support. Cross-referencing. LaTeX/PDF output. Used by NumPy, SciPy, Astropy. |
| sphinx.ext.autodoc | built-in | API doc generation | Auto-generates docs from docstrings. Essential for API reference. |
| sphinx.ext.napoleon | built-in | Docstring parser | Parses NumPy/Google style docstrings. Standard for scientific Python. |
| sphinx.ext.intersphinx | built-in | Cross-documentation links | Links to NumPy, SciPy, OpenCV docs. Critical for scientific libraries. |
| sphinx-autodoc-typehints | latest | Type hint support | Renders type annotations in docs. Modern Python best practice. |
| sphinx-copybutton | latest | Code snippet UX | Copy button for code blocks. Improves user experience. |
| myst-parser | latest | Markdown support | Write docs in Markdown instead of reST. Easier for contributors. |
| pydata-sphinx-theme | latest | Documentation theme | Modern, responsive theme. Used by NumPy, Pandas, SciPy. |
| Read the Docs | - | Documentation hosting | Free, automated builds, versioning, PR previews. Scientific Python standard. |

### Code Quality

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ruff | >=0.8.0 | Linter + formatter | Replaces Black, Flake8, isort, pyupgrade. 10-100x faster (Rust). 800+ rules. Used by FastAPI, Pandas, SciPy. |
| mypy | latest | Static type checker | Type safety verification. Ruff doesn't do type checking. Catch bugs before runtime. |
| pre-commit | latest | Git hook framework | Enforce quality checks before commit. Prevents CI failures. Standard practice 2025. |

### Testing & Coverage

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pytest | >=7.0 | Test framework | Already in use. Scientific Python standard. Plugin ecosystem. |
| pytest-cov | latest | Coverage plugin | Integrates coverage.py with pytest. Single command for test + coverage. |
| coverage.py | >=7.13.4 | Coverage measurement | Industry standard. Python 3.10-3.15 support. Latest release Feb 2026. |

### CI/CD

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| GitHub Actions | - | CI/CD platform | Free for public repos. Trusted Publishing support. PyPA-documented. |
| actions/checkout | v6 | Repo checkout | Official GitHub action. Latest stable version. |
| actions/setup-python | v6 | Python setup | Multi-version matrix testing. Cache support. |
| pypa/gh-action-pypi-publish | release/v1 | PyPI publishing | Official PyPA action. Trusted Publishing. Automatic PEP 740 attestations. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| setuptools-scm | latest | Version from git tags | Automate versioning. Derive from git tags. No manual version bumps. |
| python-semantic-release | latest | Automated releases | Automate version bumps + changelogs. Parse commit messages. Full release automation. |
| nbsphinx | latest | Jupyter notebook docs | Render notebooks in Sphinx. For tutorial/example notebooks. |
| jupyter-book | latest | Alternative doc system | If prefer notebook-first docs. Uses Sphinx under hood. |

## Installation

```bash
# Packaging tools
pip install build twine

# Documentation
pip install sphinx sphinx-autodoc-typehints sphinx-copybutton myst-parser pydata-sphinx-theme nbsphinx

# Code quality
pip install ruff mypy pre-commit

# Testing (already in dev deps)
pip install pytest pytest-cov coverage

# Optional automation
pip install setuptools-scm python-semantic-release
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Build backend | setuptools | hatch, flit, pdm, poetry | Already using setuptools. Migration effort not justified. Setuptools most mature and compatible. |
| Formatter/linter | ruff | black + flake8 + isort | Ruff consolidates all three, 10-100x faster, single config. 2025 best practice. |
| Documentation | Sphinx | MkDocs | Sphinx standard for scientific Python. Better autodoc, cross-refs. NumPy/SciPy use Sphinx. |
| Doc hosting | Read the Docs | GitHub Pages | RTD automates everything. Versioning, builds, PR previews built-in. Scientific Python standard. |
| CI/CD | GitHub Actions | GitLab CI, CircleCI | Free, well-documented, Trusted Publishing support, already on GitHub. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| python setup.py sdist/bdist_wheel | Deprecated. Security issues. Inconsistent behavior. | build package (`python -m build`) |
| Manual API token storage | Security risk. Token exposure. | Trusted Publishing with GitHub Actions |
| Black + Flake8 + isort separately | Slower, multiple configs, 2024 practice. | Ruff (single tool, single config) |
| Manual version bumping | Error-prone. Forget to update. Inconsistent. | setuptools-scm or semantic-release |
| Hard-coded version in pyproject.toml | Out of sync with git tags. Manual maintenance. | setuptools-scm (version from git) |

## Stack Patterns by Use Case

**Minimal release (current state):**
- setuptools + build + twine
- Sphinx + Read the Docs
- GitHub Actions with Trusted Publishing
- Ruff for quality

**Full automation (recommended target):**
- Add setuptools-scm for version from git tags
- Add python-semantic-release for automated releases
- Add pre-commit hooks with ruff
- Add nbsphinx for Jupyter notebook examples
- Test + coverage enforced in CI

**If documentation-heavy:**
- Add jupyter-book if notebook-first docs preferred
- Add nbsphinx for mixed Sphinx + notebook docs
- Consider gallery extension (sphinx-gallery) for example scripts

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| setuptools >=80 | Python 3.10+ | Recommended for best pyproject.toml support |
| ruff >=0.8.0 | Python 3.10+ | No Python 2 support. Type checking NOT included. |
| Sphinx >=7.0 | Python 3.10+ | Modern extension compatibility |
| pytest-cov latest | pytest >=7.0 | Requires coverage.py >=7.0 |
| setuptools-scm | setuptools >=61 | Requires >=80 for best results |

## Dataset Hosting (for example data)

| Platform | Free Tier | DOI Support | Best For |
|----------|-----------|-------------|----------|
| Zenodo | 50GB/dataset | Yes | Versioned datasets, citations, CERN-backed |
| Harvard Dataverse | 1TB/dataset | Yes | Large datasets, academic credibility |
| Hugging Face | 300GB | No | ML/CV datasets, programmatic access |
| Kaggle | 200GB | No | Public visibility, community engagement |

**Recommendation for AquaCal:** Zenodo (DOI support, versioning, scientific credibility) or GitHub Releases (small datasets <50MB, simple integration).

## Sources

**HIGH confidence (official documentation):**
- [Python Packaging User Guide - Publishing with GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) - Official PyPA guide
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) - Official PyPA GitHub Action
- [Scientific Python Development Guide - Documentation](https://learn.scientific-python.org/development/guides/docs/) - Scientific Python community standards
- [Ruff Documentation](https://docs.astral.sh/ruff/) - Official Ruff docs
- [pyOpenSci Python Package Guide - Documentation Hosting](https://www.pyopensci.org/python-package-guide/documentation/hosting-tools/publish-documentation-online.html) - Scientific Python packaging guide
- [NeurIPS 2025 Data Hosting Guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines) - Academic dataset hosting standards

**MEDIUM confidence (community resources, verified):**
- [pyOpenSci Python Packaging Tools Guide](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-build-tools.html) - Community best practices
- [Scientific Python Development Guide - Packaging](https://learn.scientific-python.org/development/guides/packaging-simple/) - Community standards
- [Python Packaging User Guide - Versioning](https://packaging.python.org/en/latest/discussions/versioning/) - Official versioning discussion
- [pyOpenSci - Versioning Guide](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-versions.html) - Community versioning practices
- [Sphinx napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) - Official Sphinx extension docs
- [Scientific Python Development Guide - Coverage](https://learn.scientific-python.org/development/guides/coverage/) - Community testing practices

---
*Stack research for: AquaCal PyPI Release*
*Researched: 2026-02-14*
*Focus: Modern 2025/2026 Python scientific library packaging best practices*
